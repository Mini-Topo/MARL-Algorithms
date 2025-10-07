import torch
import os
from network.base_net import RNN
from network.commnet import CommNet
from network.g2anet import G2ANet


class Reinforce:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape  # アクター（actor）ネットワークの入力次元は、VDNやQMIXのRNNの入力次元と同じであり、同じネットワーク構造を使用している。
        # パラメータに応じて RNN の入力次元を決定する。
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args

        # ニューラルネットワーク
        # 各エージェントが行動を選択するためのネットワークであり、現在のエージェントに対するすべての行動の確率を出力する。
        # この確率を使って実際に行動を選ぶ際には、さらに softmax を一度かけて計算する必要がある。
        if self.args.alg == 'reinforce':
            print('Init alg reinforce')
            self.eval_rnn = RNN(actor_input_shape, args)
        elif self.args.alg == 'reinforce+commnet':
            print('Init alg reinforce+commnet')
            self.eval_rnn = CommNet(actor_input_shape, args)
        elif self.args.alg == 'reinforce+g2anet':
            print('Init alg reinforce+g2anet')
            self.eval_rnn = G2ANet(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")

        if self.args.cuda:
            self.eval_rnn.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # モデルが存在する場合は、そのモデルを読み込む。
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_params.pkl'):
                path_rnn = self.model_dir + '/rnn_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        self.rnn_parameters = list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.lr_actor)
        self.args = args

        # 実行中は、各エージェントごとに eval_hidden を保持する必要がある。
        # 学習中は、各エピソード内の各エージェントごとに eval_hidden を保持する必要がある。
        self.eval_hidden = None

    # 可視化用の状態（EMA）
    _ema_W = None
    _ema_beta = 0.8
    _thr_weight = 0.0   # Visualizer側で arrow_on_threshold を使うなら 0.0 でOK
    _thr_hard = 0.5     # hard のON確率の最低閾値（G2ANet側で使ってもOK）

    def links(self, world):
        """Visualizer が期待する [(i, j, w), ...] を返す。
        i: 受け手（矢印の先）, j: 送り手（矢印の元）, w: 重み [0,1]
        """
        net = self.eval_rnn

        # G2ANet 以外（RNN/CommNet）はリンク無し
        if not hasattr(net, "link_matrix"):
            return []

        W = net.link_matrix()  # shape [n, n] or None
        if W is None:
            return []

        links = []
        n = W.shape[0]
        thr = getattr(self, "_thr_weight", 0.0)  # 例: 可視化の最小表示しきい値
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = float(W[i, j])
                # hard=0 の箇所は W が 0 のはず → 自然に外れる
                if w >= thr:
                    links.append((i, j, w))
        return links


    # ★ 追加: Runner などから学習対象パラメータを標準的に取得できるようにする
    def parameters(self):
        return self.eval_rnn.parameters()

    # ★ 追加: 名前はそのまま残したい場合の互換（callableにする）
    def rnn_parameters(self):
        return self.eval_rnn.parameters()

    def learn(self, batch, max_episode_len, train_step, epsilon):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # to tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        u, r, avail_u, terminated = batch['u'], batch['r'],  batch['avail_u'], batch['terminated']
        mask = (1 - batch["padded"].float())  # (B,T,1)
        if self.args.cuda:
            r = r.cuda(); u = u.cuda(); mask = mask.cuda(); terminated = terminated.cuda()

        # returns: (B,T,1) → expand to (B,T,n_agents)
        n_return = self._get_returns(r, mask, terminated, max_episode_len)  # (B,T,n_agents)

        # action prob: (B,T,n_agents,n_actions)
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        # mask を (B,T,n_agents) に
        mask_agents = mask.repeat(1, 1, self.n_agents)

        # 取った行動の確率 (B,T,n_agents,1)→(B,T,n_agents)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        pi_taken[mask_agents == 0] = 1.0  # log(1)=0 → パディング無影響
        log_pi_taken = torch.log(pi_taken + 1e-12)

        loss = - ((n_return * log_pi_taken) * mask_agents).sum() / (mask_agents.sum() + 1e-12)

        self.rnn_optimizer.zero_grad()
        loss.backward()

        # ★ grad_norm を測る
        total_norm = 0.0
        for p in self.eval_rnn.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5

        if self.args.alg == 'reinforce+g2anet':
            torch.nn.utils.clip_grad_norm_(self.eval_rnn.parameters(), self.args.grad_norm_clip)

        self.rnn_optimizer.step()

        # ★ 参考メトリクス: エントロピー（平均）
        with torch.no_grad():
            p = torch.clamp(action_prob, min=1e-12)
            ent = -(p * torch.log(p)).sum(dim=-1)  # (B,T,n_agents)
            entropy = (ent * mask_agents).sum() / (mask_agents.sum() + 1e-12)

        # ★ メトリクスを返す（Runnerが受け取ってログ可）
        return {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": float(total_norm),
            "entropy": float(entropy.detach().cpu().item()),
            "return_mean": float(n_return.detach().mean().cpu().item()),
            "reward_mean": float(r.detach().mean().cpu().item()),
        }

    def _get_returns(self, r, mask, terminated, max_episode_len):
        r = r.squeeze(-1)
        mask = mask.squeeze(-1)
        terminated = terminated.squeeze(-1)
        terminated = 1 - terminated
        n_return = torch.zeros_like(r)
        n_return[:, -1] = r[:, -1] * mask[:, -1]
        for transition_idx in range(max_episode_len - 2, -1, -1):
            n_return[:, transition_idx] = (r[:, transition_idx] + self.args.gamma * n_return[:, transition_idx + 1] * terminated[:, transition_idx]) * mask[:, transition_idx]
        return n_return.unsqueeze(-1).expand(-1, -1, self.n_agents)

    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        # 给inputs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            # inputs维度为(episode_num * n_agents,inputs_shape)，得到的outputs维度为(episode_num * n_agents, n_actions)
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            # 把q_eval维度重新变回(8, 5,n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        # 得的action_prob是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        action_prob = torch.stack(action_prob, dim=1).cpu()

        action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])   # 可以选择的动作的个数
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        action_prob[avail_actions == 0] = 0.0
        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


# すべてのエージェントの観測（obs）を入力し、すべてのエージェントの行動の確率分布を出力する
class G2ANet(nn.Module):
    def __init__(self, input_shape, args):
        super(G2ANet, self).__init__()

        # Encoding
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)  # すべてのエージェントの観測（obs）をデコードする
        self.h = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # 各エージェントは自分の観測（obs）に基づいてエンコードを行い、hidden_state を得る。これは過去の観測を記憶するために用いられる。
        
        # Hard
        # GRU の入力 [[h_i,h_1],[h_i,h_2],...[h_i,h_n]] と [0,...,0] を与えると、出力は [[h_1],[h_2],...,[h_n]] と [h_n] になる。ここで h_j はエージェント j とエージェント i の関係を表す。
        # 入力の inputs の次元は (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2) であり、
        # つまり batch_size 本のデータに対して、各エージェントと他の n_agents - 1 個のエージェントの hidden_state を結合したものを入力する。
        self.hard_bi_GRU = nn.GRU(args.rnn_hidden_dim * 2, args.rnn_hidden_dim, bidirectional=True)
        # h_j を解析して、エージェント j がエージェント i に対して持つ重みを得る。出力は2次元で、gumble_softmax を通した後、そのうちの1次元のみを取ればよい。もし 0 であればエージェント j を考慮せず、1 であれば考慮する。
        self.hard_encoding = nn.Linear(args.rnn_hidden_dim * 2, 2)  # 2倍するのは双方向GRUだからであり、hidden_state の次元は 2 * hidden_dim となる。

        # Soft
        self.q = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.rnn_hidden_dim, args.attention_dim)

        # デコーディングでは、自分の h_i と x_i を入力し、自分の行動の確率分布を出力する。
        self.decoding = nn.Linear(args.rnn_hidden_dim + args.attention_dim, args.n_actions)
        self.args = args
        self.input_shape = input_shape

        # attention value
        self.last_soft = None
        self.last_hard = None
        self.col_index = None


    def forward(self, obs, hidden_state):
        size = obs.shape[0]  # batch_size * n_agents

        # まず obs をエンコードする
        obs_encoding = f.relu(self.encoding(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        # 自分のGRUを通して h を得る
        h_out = self.h(obs_encoding, h_in)  # (batch_size * n_agents, args.rnn_hidden_dim)

        # Hard Attention，GRU と GRUCell は異なり、入力の次元は (系列の長さ, batch_size, dim) である。
        if self.args.hard:

            # Hard Attention 前の準備
            h = h_out.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)  # h を変換して n_agents 次元にし、(batch_size, n_agents, rnn_hidden_dim) の形にする。
            input_hard = []

            for i in range(self.args.n_agents):
                h_i = h[:, i]  # (batch_size, rnn_hidden_dim)
                h_hard_i = []

                for j in range(self.args.n_agents):  # エージェント i に対して、自分の h_i を他のエージェントの h とそれぞれ結合する。
                    if j != i:
                        h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))

                # j のループが終了した後、h_hard_i は、(batch_size, rnn_hidden_dim * 2) のテンソルを n_agents - 1 個格納したリストとなる。
                h_hard_i = torch.stack(h_hard_i, dim=0)
                input_hard.append(h_hard_i)

            # i のループが終了した後、input_hard は、(n_agents - 1, batch_size, rnn_hidden_dim * 2) のテンソルを n_agents 個格納したリストとなる。
            input_hard = torch.stack(input_hard, dim=-2)

            # 最終的に (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2) の次元となり、入力できる状態になる。
            input_hard = input_hard.view(self.args.n_agents - 1, -1, self.args.rnn_hidden_dim * 2)

            h_hard = torch.zeros((2 * 1, size, self.args.rnn_hidden_dim))  # 双方向GRUであり、各GRUは1層だけなので、最初の次元は 2 * 1 となる。
            if self.args.cuda:
                h_hard = h_hard.cuda()
            h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)  # (n_agents - 1,batch_size * n_agents,rnn_hidden_dim * 2)
            h_hard = h_hard.permute(1, 0, 2)  # (batch_size * n_agents, n_agents - 1, rnn_hidden_dim * 2)
            h_hard = h_hard.reshape(-1, self.args.rnn_hidden_dim * 2)  # (batch_size * n_agents * (n_agents - 1), rnn_hidden_dim * 2)

            # hard の重みを得ると、(n_agents, batch_size, 1, n_agents - 1) となり、1つ余分な次元がある。これは後で加重和を取るときに使う。
            hard_weights = self.hard_encoding(h_hard)
            hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)

            hard_weights = hard_weights[:, 1].view(-1, self.args.n_agents, 1, self.args.n_agents - 1)
            hard_weights = hard_weights.permute(1, 0, 2, 3)

        else:
            hard_weights = torch.ones((self.args.n_agents, size // self.args.n_agents, 1, self.args.n_agents - 1))
            if self.args.cuda:
                hard_weights = hard_weights.cuda()


        # Soft Attention
        q = self.q(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        k = self.k(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        x = []

        B = q.shape[0]
        soft_list = []
        for i in range(self.args.n_agents):
            q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # エージェント i の q，(batch_size, 1, args.attention_dim)
            k_i = [k[:, j] for j in range(self.args.n_agents) if j != i]  # エージェント i にとって他のエージェントの k
            v_i = [v[:, j] for j in range(self.args.n_agents) if j != i]  # エージェント i にとって他のエージェントの v

            k_i = torch.stack(k_i, dim=0)  # (n_agents - 1, batch_size, args.attention_dim)
            k_i = k_i.permute(1, 2, 0)  # 3つの次元を入れ替えて (batch_size, args.attention_dim, n_agents - 1) にする
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0)

            # (batch_size, 1, attention_dim) * (batch_size, attention_dim，n_agents - 1) = (batch_size, 1，n_agents - 1)
            score = torch.matmul(q_i, k_i)

            # 正規化
            scaled_score = score / np.sqrt(self.args.attention_dim)

            # softmax で重みを得る
            soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size，1, n_agents - 1)
            # print("soft: ", soft_weight)

            soft_list.append(soft_weight)

            # 加重和を計算。注意：3つの行列の最後の次元は n_agents - 1 であるため、結果は (batch_size, args.attention_dim) になる
            x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)
            x.append(x_i)

        # 各エージェントの h と x を結合する
        x = torch.stack(x, dim=1).reshape(-1, self.args.attention_dim)  # (batch_size * n_agents, args.attention_dim)

        final_input = torch.cat([h_out, x], dim=-1)
        output = self.decoding(final_input)

        # forward() の最後の方で（B=batch_size）
        if B == 1:
            # soft: List[[1, n-1]] -> [n, 1, n-1] -> squeeze(1) -> [n, n-1]
            self.last_soft = torch.stack(soft_list, dim=0).squeeze(1).detach().cpu()   # [n, n-1]
            # hard: [n, B, 1, n-1] -> squeeze(1)->[n,1,n-1] -> squeeze(1)->[n,n-1]
            self.last_hard = hard_weights.squeeze(1).squeeze(1).detach().cpu()            # [n, n-1]
                 # [n, n-1]

            # 列→元 j の対応を作る
            n = self.args.n_agents
            self.col_index = [ [j for j in range(n) if j != i] for i in range(n) ]    # List[List[int]]
        else:
            self.last_soft = None
            self.last_hard = None
            self.col_index = None
        # print(B)
        # print("soft: ", self.last_soft)
        # print("hard: ", self.last_hard)
        # print("col_index: ", self.col_index)

        return output, h_out
    
    def link_matrix(self):
        """
        直近 forward の soft/hard から w_{i<-j} を [n,n] 行列で返す（対角0）。
        可視化向け：batch=1 前提。未計算なら None。
        方針：hard は 0/1 マスク。hard=1 の位置だけ soft を採用。
        """
        if self.last_soft is None or self.last_hard is None or self.col_index is None:
            return None

        soft = self.last_soft
        hard = self.last_hard

        # 形状の安全確保（例: [n,1,n-1] -> [n,n-1]）
        if soft.dim() == 3:
            soft = soft.squeeze(1)
        if hard.dim() == 3:
            hard = hard.squeeze(1)

        n = self.args.n_agents
        # dtype/デバイスを soft に合わせる
        W = torch.zeros((n, n), dtype=soft.dtype, device=soft.device)

        # 必要なら binarize（true 0/1 を保証）
        # （forward で既に 0/1 ならこの2行は不要だが保険として残す）
        hard_bin = (hard > 0.5).to(soft.dtype)

        # 行 i（受け手）、列 k -> 送り手 j を復元
        for i in range(n):
            # soft[i]: shape [n-1], hard_bin[i]: shape [n-1]
            masked = soft[i] * hard_bin[i]   # hard=0 の位置は 0、hard=1 の位置は soft が残る
            # 対応する送り手 j を col_index から復元
            for k, j in enumerate(self.col_index[i]):
                W[i, j] = masked[k]

            # 対角は 0（自分自身へのリンクはなし）
            W[i, i] = 0.0

            # もし「可視化の都合で行正規化したい」場合はここで：
            row_sum = W[i].sum()
            if row_sum > 0:
                W[i] = W[i] / row_sum
            # 行内が全部 0（= hard=1 が一つもない）ならそのまま 0 行

        # CPU numpy で返す（可視化側が numpy 前提なら）
        return W.detach().cpu().numpy()




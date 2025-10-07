# env/pz_simple_tag_adapter.py
from mpe2 import simple_tag_v3
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class PZSimpleTagAdapter:
    """
    SMAC 互換の API を満たす PettingZoo simple_tag_v3 アダプタ。
    - control_team: "adversary" または "good"（good は名前に 'agent' を含む）
    - opponent_policy: "random" / "scripted"（必要に応じて実装）
    """
    def __init__(
        self,
        num_good=2,
        num_adversaries=5,
        num_obstacles=1,
        episode_limit=25,
        control_team="adversary",   # or "good"
        opponent_policy="random",
        seed=0,
        on_render=None,            # レンダリングの追加描画用
        render_mode=None,           # "human" は可視化用だけに
    ):
        self.env = simple_tag_v3.parallel_env(
            continuous_actions=False,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=episode_limit,
            render_mode=render_mode,
            dynamic_rescaling=True,
        )
        self.episode_limit = episode_limit
        self.control_team = control_team
        self.opponent_policy = opponent_policy
        self.rng = np.random.default_rng(seed)

        # 初期化して空間を確定
        obs, info = self.env.reset(seed=seed)
        self.last_obs = obs
        self.all_agents = list(self.env.possible_agents)

        # チーム振り分け
        self.ctrl_agents = [a for a in self.all_agents
                            if ("adversary" in a and control_team=="adversary")
                            or ("agent" in a and control_team=="good")]
        self.opp_agents  = [a for a in self.all_agents if a not in self.ctrl_agents]

        # ID マップ（SMAC は 0..n-1 を期待）
        self.ctrl_id = {a:i for i,a in enumerate(sorted(self.ctrl_agents))}
        self.opp_id  = {a:i for i,a in enumerate(sorted(self.opp_agents))}

        # 形状
        some_agent = self.all_agents[0]
        self._n_actions = self.env.action_space(some_agent).n
        self._obs_dim   = self.env.observation_space(some_agent).shape[0]

        # グローバル状態は全エージェント観測の連結
        self._state_dim = self._obs_dim * len(self.all_agents)
        self._obs_zero  = np.zeros(self._obs_dim, dtype=np.float32)

        self._t = 0
        self._terminated = False
        self._truncated  = False

        # 表示追加用
        self.on_render = on_render
        self.render_mode = render_mode
        self._frame_idx = 0

        # 汎用定数
        self._avail_all_ones  = np.ones(self._n_actions, dtype=np.int32)
        self._avail_all_zeros = np.zeros(self._n_actions, dtype=np.int32)

        self._d_prev = None           # 前ステップの距離（最小距離のベクトル）
        self._shaping_coef = 0

        self.adv_reward_mode = "at_least_k_goods"  # "env" / "at_least_k_goods"
        self.adv_reward_k = 2         # k >= 1
        self.adv_bonus = 100.0
        self.adv_scale_by_goods = True # k倍ボーナスにするか
        self.adv_combine = "add"      # "add" / "replace"


    def _get_world(self):
        # parallel_env でも raw_env でも拾えるようにフォールバック
        try:
            return self.env.aec_env.unwrapped.world
        except AttributeError:
            return self.env.unwrapped.world
        
    @property
    def world(self):
        """Runner 等の外部から world にアクセスするための公開プロパティ。"""
        return self._get_world()

    # （必要なら）外部がエージェント列を欲しがるケースに備えて
    @property
    def agent_names(self):
        """ワールド内エージェント名の固定並び（描画やログ用）"""
        try:
            return [ag.name for ag in self._get_world().agents]
        except Exception:
            return list(self.all_agents)  # フォールバック

    def _place_goods_with_spacing(self, spacing=0.8, center=0.0, y=0.0, clamp=True):
        """
        good（非adversary）を y=y に固定し、x を center を中心に spacing 間隔で等間隔配置。
        clamp=True のときは座標範囲 [-1,1] に収まるようクリップ。
        """
        import numpy as np

        world = self._get_world()
        goods = [a for a in world.agents if not getattr(a, "adversary", False)]
        n = len(goods)
        if n == 0:
            return

        start = center - spacing * (n - 1) / 2.0
        for i, ag in enumerate(goods):
            x = start + i * spacing
            if clamp:
                x = float(np.clip(x, -1.0, 1.0))
                yy = float(np.clip(y, -1.0, 1.0))
            else:
                yy = y
            ag.state.p_pos[:] = [x, yy]
            ag.state.p_vel[:] = 0.0  # 初速はゼロに
        
        # 反力をゼロにする
        world.contact_force = 0.01
        world.contact_margin = 0.01

    def _min_ctrl_to_opp_dists(self):
        """
        自チーム各体から相手チームへの最小距離ベクトルを返す。
        取れない場合は None（ラッパ差異に備えた try/except）。
        """
        try:
            # parallel_env -> AEC -> raw_env の中に world がいます
            aec = getattr(self.env, "aec_env", self.env)
            raw = getattr(aec, "unwrapped", aec)
            world = raw.world
        except Exception:
            return None

        advs  = [ag for ag in world.agents if 'adversary' in ag.name]
        goods = [ag for ag in world.agents if 'agent'     in ag.name]

        if self.control_team == "adversary":
            ctrl = advs;  opp = goods
        else:
            ctrl = goods; opp = advs

        if not ctrl or not opp:
            return None

        dists = []
        for c in ctrl:
            dmin = min(np.linalg.norm(c.state.p_pos - o.state.p_pos) for o in opp)
            dists.append(dmin)
        return np.asarray(dists, dtype=np.float32)


    # ===== SMAC 互換メソッド群 =====
    def get_env_info(self):
        return {
            "n_actions": self._n_actions,
            "n_agents":  len(self.ctrl_agents),
            "state_shape": self._state_dim,
            "obs_shape":   self._obs_dim,
            "episode_limit": self.episode_limit,
        }

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.last_obs = obs
        self._t = 0
        self._terminated = False
        self._truncated  = False
        self._tag_count = 0
        self._d_prev = None
        # Good の初期位置を固定
        self._place_goods_with_spacing(spacing=0.4, center=0.0, y=0.0, clamp=True)

        return  # SMAC 互換：返り値なし

    def get_obs(self):
        if not getattr(self, "_printed_team", False):
            # print(f"[team] control_team = {self.control_team}")  # "adversary" or "good"
            self._printed_team = True
        # 終了エージェントがいてもゼロで埋めて形を保つ
        return [np.asarray(self.last_obs.get(a, self._obs_zero), dtype=np.float32)
                for a in sorted(self.ctrl_agents)]

    def get_state(self):
        # 全エージェント観測の連結（終了済みはゼロ）
        vecs = [np.asarray(self.last_obs.get(a, self._obs_zero), dtype=np.float32)
                for a in sorted(self.all_agents)]
        return np.concatenate(vecs, axis=-1)

    def get_avail_actions(self):
        """SMAC互換の“全エージェント分”の可用アクション行列を返す。"""
        live = set(self.env.agents)  # いまアクティブなagent
        mat = np.zeros((len(self.ctrl_agents), self._n_actions), dtype=np.int32)
        for i, a in enumerate(sorted(self.ctrl_agents)):
            mat[i, :] = self._avail_all_ones if a in live else self._avail_all_zeros
        return mat

    def get_avail_agent_actions(self, agent_id: int):
        """
        SMAC互換の“単一エージェント用”API。
        rollout.py からはこっちが呼ばれる。
        戻り値: shape=(n_actions,), dtype=int32
        """
        a = sorted(self.ctrl_agents)[agent_id]  # agent_id は自チーム内の 0..n-1
        if a in self.env.agents:
            return self._avail_all_ones.copy()
        else:
            return self._avail_all_zeros.copy()
        
    def _is_collision(self, a, b):
        # PettingZoo/MPEの衝突ロジック準拠
        delta = a.state.p_pos - b.state.p_pos
        dist = float(np.sqrt(np.sum(np.square(delta))))
        dist_min = a.size + b.size
        return dist < dist_min
    
    def _count_tagged_goods(self, world):
        goods = [ag for ag in world.agents if not getattr(ag, "adversary", False)]
        advs  = [ag for ag in world.agents if getattr(ag, "adversary", False)]
        # このステップで「少なくとも1人のadversaryに触られた」goodのユニーク数を数える
        tagged = 0
        for g in goods:
            if any(self._is_collision(g, a) for a in advs):
                tagged += 1
        return tagged


    def step(self, actions):
        """
        SMAC 流儀:
          入力: actions = [a_0, a_1, ..., a_{n_agents-1}]  自チームのみ
          返り値: reward(float), terminated(bool), info(dict)
        """
        if self._terminated or self._truncated:
            return 0.0, True, {}

        # "現在生存している" 自チーム/相手チームだけに行動を出す
        live = set(self.env.agents)  # 現在アクティブなエージェント
        live_ctrl = [a for a in self.ctrl_agents if a in live]
        live_opp  = [a for a in self.opp_agents  if a in live]

        joint_act = {}
        for a in live_ctrl:
            idx = self.ctrl_id[a]
            joint_act[a] = int(actions[idx])

        for a in live_opp:
            if self.opponent_policy == "random":
                joint_act[a] = self.env.action_space(a).sample()
            elif self.opponent_policy == "still":
                joint_act[a] = 0  # noop
            elif self.opponent_policy == "up":
                joint_act[a] = 4  # up
            else:
                # TODO: 追う/逃げる簡易スクリプトなど
                joint_act[a] = self.env.action_space(a).sample()

        next_obs, rewards, term, trunc, infos = self.env.step(joint_act)
        
        self.last_obs = next_obs
        self._t += 1

        # 終端判定（空dictでの all(True) を避ける & 誰も生きていない場合も終端）
        self._terminated = (len(self.env.agents) == 0) or (bool(term) and all(term.values()))
        self._truncated  = bool(trunc) and all(trunc.values())


        # === ここからチーム報酬カスタム ===
        if self.adv_reward_mode != "env":
            # adversary側の設定を取得
            mode = self.adv_reward_mode
            k = self.adv_reward_k
            base_bonus = self.adv_bonus
            scale = self.adv_scale_by_goods
            combine = self.adv_combine

            if self.control_team == "adversary":
                # 同時に触れたgood数 m をカウント（同一step）
                m = self._count_tagged_goods(self.world)

                # ボーナス計算
                bonus = 0.0
                if mode == "at_least_k_goods" and m >= k:
                    bonus = base_bonus * (m if scale else 1.0)
                


                if bonus != 0.0:
                    # adversary全員に同額を付与（既存のenv仕様に合わせる）
                    for a in self.ctrl_agents:  # adversary名のリスト
                        curr = float(rewards.get(a, 0.0))
                        rewards[a] = (curr + bonus) if combine == "add" else bonus


        # チーム報酬（キー欠損に備えて get）
        team_reward = float(sum(rewards.get(a, 0.0) for a in self.ctrl_agents))
        if self.control_team == "adversary":
            n = len(self.ctrl_agents)
            if n > 0:
                team_reward /= n  # 平均化
        if m > 1:
            print(f"[step {self._t}] tagged {m} goods, bonus {bonus:.1f}")
            print(rewards)
            print(team_reward)

        # --- shaping: 近づけば(adv)プラス / 離れれば(good)プラス ---
        d_now = self._min_ctrl_to_opp_dists()
        if d_now is not None:
            if self._d_prev is not None:
                # adversary学習→距離が減る(=追い詰める)と +、good学習→距離が増えると +
                sign = +1.0 if self.control_team == "adversary" else -1.0
                # 前-今 が正なら「近づいた」: その分だけ加点（goodなら符号反転）
                shaping_raw = sign * float(np.sum(self._d_prev - d_now))
                # スケール調整（人数で平均化。もとの +10/ヒット と桁が違いすぎないように）
                shaping = (self._shaping_coef * shaping_raw) / max(1, len(self.ctrl_agents))
                team_reward += shaping
                # デバッグしたければ:
                # print(f"[shape] t={self._t} raw={shaping_raw:.3f} add={shaping:.4f}")
            # 次回用に保存
            self._d_prev = d_now
        # --- shaping ここまで ---


        # タグ数カウント（デバッグ用）
        # step_tags = int(sum(1 for a in self.ctrl_agents if rewards.get(a, 0.0) > 0.0))
        step_tags = int(any(rewards.get(a, 0.0) > 0.0 for a in self.ctrl_agents))
        ctrl_rewards_dict = {a: float(rewards.get(a, 0.0)) for a in self.ctrl_agents}
        if step_tags > 0:
            # print(f"[step {self._t}] rewards {ctrl_rewards_dict} tags {step_tags}")
            pass
        self._tag_count += step_tags

        terminated_flag = (self._terminated or self._truncated)
        info = {
            "terminated": self._terminated,
            "truncated":  self._truncated,
            "t": self._t,
            "step_tags": step_tags,
            "tag_count": self._tag_count,
        }
        return team_reward, terminated_flag, info

    def render(self):

        # PettingZooの標準描画
        ret = None
        try:
            ret = self.env.render()
        except Exception:
            pass

        # Axesを現在のFigureから取得
        try:
            ax = plt.gca()
        except Exception:
            ax = None

        # worldを取得
        try:
            world = self._get_world()
        except Exception:
            world = None

        if self.on_render is not None:
            try:
                self.on_render(
                    ax=ax,
                    world=world,
                    info={
                        "t": getattr(self, "_t", None),
                        "tag_count": getattr(self, "_tag_count", None),
                    }, 
                    frame_idx=self._frame_idx, 
                )
            except Exception as e:
                print("[on_render error]", repr(e))

        self._frame_idx += 1

        return ret

    def close(self):
        self.env.close()

import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from common.visualizer import Visualizer
from common.recorder_wrapper import Recorder
from common.config import Config

from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import time


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)

        # on-policy 以外は ReplayBuffer を使う
        self.on_policy = (
            args.alg.find('coma') > -1 or
            args.alg.find('central_v') > -1 or
            args.alg.find('reinforce') > -1
        )
        if not args.evaluate and not self.on_policy:
            self.buffer = ReplayBuffer(args)

        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # 観戦用GUIの使い回し（必要なら）
        self._human_env = None
        self._human_worker = None

    def run(self):
        print(f"[cfg] n_steps={self.args.n_steps}, "
              f"n_episodes={self.args.n_episodes}, "
              f"episode_limit={self.args.episode_limit}, "
              f"evaluate_cycle={self.args.evaluate_cycle}")

        time_steps, train_steps, evaluate_steps = 0, 0, -1

        while time_steps < self.args.n_steps:
            print(f'time_steps {time_steps}')
            # 周期評価
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt()
                evaluate_steps += 1

            # ===== ロールアウト収集 =====
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps

            # 複数エピソードを結合（axis=0 に沿って）
            episode_batch = episodes[0]
            episodes.pop(0)
            for ep in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], ep[key]), axis=0)

            # ===== 学習 =====
            if self.on_policy:
                # on-policy（COMA / CentralV / REINFORCE）
                # episode_batch 全体を1回学習
                self.agents.train(episode_batch, train_steps, getattr(self.rolloutWorker, "epsilon", 0.0))
                train_steps += 1
            else:
                # off-policy（VDN / QMIX / CommNet / G2ANet など）
                # 収集分をバッファへ
                self.buffer.store_episode(episode_batch)

                # 学習ステップ数だけサンプリング学習
                for _ in range(self.args.train_steps):
                    # ウォームアップ: 十分に溜まるまでスキップ
                    batch_size = min(self.buffer.current_size, self.args.batch_size)
                    if batch_size < self.args.batch_size:
                        break
                    mini_batch = self.buffer.sample(batch_size)
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

        # 最終評価
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt()

    def evaluate(self):
        # 評価はヘッドレス（描画なし）で実行
        worker = self.rolloutWorker

        total_tags = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            # evaluate=True → ε=0 で実行
            _, episode_reward, tag_count, _ = worker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            total_tags += int(tag_count)

        avg_tags = total_tags / self.args.evaluate_epoch
        return avg_tags, episode_rewards / self.args.evaluate_epoch

    def watch(self, n_episodes=1, render_interval=1, video_filename=None, fps=20):
        # 保存先
        os.makedirs(self.save_path, exist_ok=True)
        if video_filename is None:
            video_filename = f"watch_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(self.save_path, video_filename)

        # 既存 runner/env の構造に合わせて渡す
        env_adapter = getattr(self.rolloutWorker, "env", None)
        assert hasattr(env_adapter, "world"), "EnvAdapter(world) が必要です"

        cfg = Config()
        viz = Visualizer(cfg, env_adapter, self.agents.policy if hasattr(self.agents, "policy") else self.agents)

        with Recorder(cfg, viz.scene.fig) as rec:
            # エピソードごと
            for ep in range(n_episodes):
                # evaluate=True で ε=0、描画は on_step で
                def _on_step():
                    viz.on_step()
                    rec.grab()

                # 初期状態を1枚
                viz.scene.draw()
                rec.grab()
                _, episode_reward, tag_count, _ = self.rolloutWorker.generate_episode(
                    ep, evaluate=True, on_step=_on_step
                )
                print(f"[watch] ep={ep} reward={episode_reward:.3f} tags={tag_count}")

        print(f"[watch] saved: {os.path.abspath(video_path)}")

    def close_watch(self):
        """観戦用の human 環境を閉じる（必要なら実行の最後で呼ぶ）"""
        if self._human_env is not None:
            try:
                self._human_env.close()
            except Exception:
                pass
            self._human_env = None
            self._human_worker = None

    def plt(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('avg tags / episode')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt.png', format='png')
        np.save(self.save_path + '/win_rates', self.win_rates)
        np.save(self.save_path + '/episode_rewards', self.episode_rewards)
        plt.close()

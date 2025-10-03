#!/usr/bin/env python3
"""
main.py
- Predator-Prey 環境での MARL アルゴリズム実行

usage example:
python3 main.py \
    --episode_limit 1000 \
    --num_good 2 \
    --num_adversaries 2 \
    --num_obstacles 0 \
    --opponent_policy 'still' \
    --n_steps 2000000 \
    --evaluate_cycle 10000
"""
from __future__ import annotations

from runner import Runner
from common.arguments import (
    get_common_args, get_coma_args, get_mixer_args,
    get_centralv_args, get_reinforce_args,
    get_commnet_args, get_g2anet_args
)
from env.pz_simple_tag_adapter import PZSimpleTagAdapter

if __name__ == '__main__':
    args = get_common_args()

    # ==== アルゴ別オプション ====
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    # ==== 環境生成（PettingZoo simple_tag_v3 用）====
    env = PZSimpleTagAdapter(
        num_good=getattr(args, 'num_good', 2),
        num_adversaries=getattr(args, 'num_adversaries', 5),
        num_obstacles=getattr(args, 'num_obstacles', 1),
        episode_limit=getattr(args, 'episode_limit', 25),
        control_team=getattr(args, 'control_team', 'adversary'),   # or 'good'
        opponent_policy=getattr(args, 'opponent_policy', 'still'),
        seed=getattr(args, 'seed', 0),
        render_mode=None,
    )

    # ==== SMAC互換のenv_infoを取得 ====
    env_info = env.get_env_info()
    args.n_actions     = env_info["n_actions"]
    args.n_agents      = env_info["n_agents"]
    args.state_shape   = env_info["state_shape"]
    args.obs_shape     = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    runner = Runner(env, args)

    if not args.evaluate:
        runner.run()
    else:
        win_rate, _ = runner.evaluate()

    runner.watch(n_episodes=1, render_interval=1)
    runner.close_watch()
    env.close()

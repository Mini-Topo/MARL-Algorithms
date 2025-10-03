import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()

    # ===== 保存用ラベル（Runnerのsave_path互換のため残す） =====
    parser.add_argument('--map', type=str, default='simple_tag_v3_adversary',
                        help='used only for saving dirs (label)')

    # ===== PettingZoo simple_tag_v3 用 =====
    parser.add_argument('--episode_limit', type=int, default=1000)
    parser.add_argument('--num_good', type=int, default=2)
    parser.add_argument('--num_adversaries', type=int, default=5)
    parser.add_argument('--num_obstacles', type=int, default=1)
    parser.add_argument('--control_team', type=str, default='adversary',
                        choices=['adversary', 'good'])
    parser.add_argument('--opponent_policy', type=str, default='random',
                        choices=['random', 'still', 'up', 'scripted'])
    parser.add_argument('--seed', type=int, default=123)

    # 可視化（学習時は基本オフ）
    parser.add_argument('--render', action='store_true', help='use human render')
    parser.add_argument('--render_eval', action='store_true',
                        help='show a human-rendered window during evaluation')

    # ===== アルゴリズム共通 =====
    parser.add_argument('--alg', type=str, default='reinforce+g2anet')
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--n_episodes', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--optimizer', type=str, default='RMS')

    parser.add_argument('--evaluate_cycle', type=int, default=10000)
    parser.add_argument('--evaluate_epoch', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--result_dir', type=str, default='./result')

    # ===== フラグ類は action で =====
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--last_action', action='store_true', help='append last action to obs')
    parser.add_argument('--no-last_action', dest='last_action', action='store_false')
    parser.set_defaults(last_action=True)

    parser.add_argument('--reuse_network', action='store_true', help='share params across agents')
    parser.add_argument('--no-reuse_network', dest='reuse_network', action='store_false')
    parser.set_defaults(reuse_network=True)

    # ===== オプション（任意の評価規則） =====
    parser.add_argument('--timeout_win_for', type=str, default='good',
                        choices=['good', 'adversary'])
    parser.add_argument('--capture_win_for', type=str, default='adversary',
                        choices=['good', 'adversary'])

    args = parser.parse_args()
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args


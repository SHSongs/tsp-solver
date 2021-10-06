import os

import or_gym

import torch
from util import VisualData, visualization, make_pointer_network
from config import args_parser
from gym_util import play_tsp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(actor, actor_dir, seq_len, result_dir):
    # env setup
    env_config = {'N': seq_len}
    env = or_gym.make('TSP-v1', env_config=env_config)

    # load actor
    actor.load_state_dict(torch.load(actor_dir))
    actor.eval()

    visual_data = VisualData()

    coords = torch.FloatTensor(env.coords).transpose(1, 0).unsqueeze(0)
    total_reward = play_tsp(env, coords, actor, device)

    log_probs, actions = actor.result()

    visual_data.add(coords, actions, "test")
    c, a, e = visual_data.get()
    visualization(result_dir, c, a, e)

    print('total length', total_reward)


def main():
    args = args_parser()

    seq_len = args.seq_len
    result_dir = args.result_dir
    actor_dir = args.actor_dir

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Pointer network hyper parameter
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    n_glimpses = args.n_glimpses
    tanh_exploration = args.tanh_exploration

    print("args: ")
    print("embedding size: %d" % embedding_size)
    print("hidden size: %d" % hidden_size)
    print("num glimpses: %d" % n_glimpses)
    print("tanh exploration: %d" % tanh_exploration)

    print("")
    print("sequence length: %d" % seq_len)
    print("result dir: %s" % result_dir)
    print("actor dir: %s" % actor_dir)

    ptr_net = make_pointer_network(embedding_size, hidden_size, n_glimpses, tanh_exploration, seq_len, device)
    test(ptr_net, actor_dir, seq_len, result_dir)


if __name__ == "__main__":
    main()
    print('end tsp')

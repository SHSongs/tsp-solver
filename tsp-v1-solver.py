import argparse
import os

import or_gym

import torch
import torch.optim as optim
from pointer_network import PointerNetwork
import matplotlib.pyplot as plt

from util import rotate_actions, args_parser, visualization, VisualData


def play_tsp(env, actions):
    """
    Play TSP in Gym and return reward
    """
    total_reward = 0
    cnt = 0
    done = False
    while not done:
        a = actions[cnt]
        next_state, reward, done, _ = env.step(a)
        total_reward += reward
        cnt += 1

    # return home
    total_reward += env.distance_matrix[actions[-2], actions[-1]]
    return total_reward


def main(embedding_size, hidden_size, grad_clip, learning_rate, n_glimpses, tanh_exploration, train_mode, episode,
         seq_len, beta, result_dir, result_graph_dir):
    env_config = {'N': seq_len}
    env = or_gym.make('TSP-v1', env_config=env_config)

    model = PointerNetwork(embedding_size, hidden_size, seq_len, n_glimpses=n_glimpses,
                           tanh_exploration=tanh_exploration)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    episodes_length = []

    moving_avg = torch.zeros(1)
    first_step = True

    visual_data = VisualData()

    for i in range(episode):
        s = env.reset()

        coords = torch.FloatTensor(env.coords).transpose(1, 0).unsqueeze(0)

        log_probs, actions = model(coords)

        if i % 10 == 9:
            visual_data.add(coords, actions, i)
        if i % 100 == 99:
            c, a, e = visual_data.get()
            visualization(result_graph_dir, c, a, e)
            visual_data.clear()

        actions = rotate_actions(actions.squeeze(0).tolist(), s[0])

        total_reward = play_tsp(env, actions)
        episodes_length.append(total_reward)
        print('total length', total_reward)

        if first_step:  # generating first baseline
            moving_avg = total_reward
            first_step = False
            continue

        moving_avg = moving_avg * beta + total_reward * (1.0 - beta)
        advantage = total_reward - moving_avg
        log_probs = torch.sum(log_probs)
        log_probs[log_probs < -100] = - 100
        loss = advantage * log_probs

        losses.append(loss.item())
        print('loss : ', loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.close('all')
    plt.plot(range(len(losses)), losses, color="blue")
    plt.title(train_mode + " losses")
    plt.xlabel("episode")
    plt.ylabel("loss")
    filename = train_mode + 'loss.png'
    plt.savefig(os.path.join(result_dir, filename))

    plt.close('all')
    plt.plot(range(len(episodes_length)), episodes_length, color="blue")
    plt.title("Episode length")
    plt.xlabel("episode")
    plt.ylabel("length")
    plt.savefig(os.path.join(result_dir, 'episode_length.png'))


if __name__ == "__main__":
    args = args_parser()

    # Pointer network hyper parameter
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    grad_clip = args.grad_clip
    learning_rate = args.lr
    n_glimpses = args.n_glimpses
    tanh_exploration = args.tanh_exploration

    # train mode active-search or actor-critic
    train_mode = args.mode

    episode = args.episode
    seq_len = args.seq_len

    # Active search hyper parameter
    beta = args.beta

    result_dir = args.result_dir

    result_graph_dir = os.path.join(result_dir, 'graph')

    if not os.path.exists(result_dir):
        os.makedirs(result_graph_dir)

    main(embedding_size, hidden_size, grad_clip, learning_rate,
         n_glimpses, tanh_exploration, train_mode, episode, seq_len, beta, result_dir, result_graph_dir)

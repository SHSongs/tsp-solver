import os

import or_gym

import torch
import torch.optim as optim

from pointer_network import PointerNetwork
from critic_network import CriticNetwork
from util import rotate_actions, visualization, VisualData, draw_list_graph
from config import args_parser
import torch.nn as nn


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


def stack_visualization_data(visual_data, coords, actions, episode, result_graph_dir):
    """
    data를 특정 주기마다 쌓고 시각화합니다.
    Args:
        visual_data: data를 쌓을 class (VisualData class)
        coords: 현재 episode의 coords
        actions: 현재 episode의 actions
        episode: 현재 episode
        result_graph_dir: file 저장 경로
    """
    if episode % 10 == 9:
        visual_data.add(coords, actions, episode)
    if episode % 100 == 99:
        c, a, e = visual_data.get()
        visualization(result_graph_dir, c, a, e)
        visual_data.clear()


def train(embedding_size, hidden_size, grad_clip, learning_rate, n_glimpses, tanh_exploration, train_mode, episode_num,
          seq_len, beta, result_dir, result_graph_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # env setup
    env_config = {'N': seq_len}
    env = or_gym.make('TSP-v1', env_config=env_config)

    # actor setup
    actor = PointerNetwork(embedding_size, hidden_size, seq_len, n_glimpses, tanh_exploration)
    actor.to(device)
    optimizer = optim.Adam(actor.parameters(), lr=learning_rate)

    # result data
    losses = []
    episodes_length = []
    visual_data = VisualData()

    if train_mode == "active-search":

        # Active search
        moving_avg = torch.zeros(1)
        first_step = True

        for i in range(episode_num):
            s = env.reset()

            coords = torch.FloatTensor(env.coords).transpose(1, 0).unsqueeze(0)

            log_probs, actions = actor(coords.to(device))

            stack_visualization_data(visual_data, coords, actions, i, result_graph_dir)

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
            actor_loss = advantage * log_probs

            losses.append(actor_loss.item())
            print('loss : ', actor_loss.item())

            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()

    elif train_mode == "actor-critic":

        # critic
        critic = CriticNetwork(embedding_size, hidden_size, seq_len, n_glimpses, tanh_exploration)
        critic.to(device)
        critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
        l2Loss = nn.MSELoss()

        for i in range(episode_num):
            s = env.reset()

            coords = torch.FloatTensor(env.coords).transpose(1, 0).unsqueeze(0).to(device)

            log_probs, actions = actor(coords)
            value = critic(coords)

            stack_visualization_data(visual_data, coords, actions, i, result_graph_dir)

            actions = rotate_actions(actions.squeeze(0).tolist(), s[0])

            total_reward = play_tsp(env, actions)

            episodes_length.append(total_reward)
            print('total length', total_reward)

            log_probs = torch.sum(log_probs)
            log_probs[log_probs < -100] = - 100

            advantage = value - total_reward
            total_reward = torch.FloatTensor([total_reward]).to(device)

            critic_loss = l2Loss(value.squeeze(0), total_reward)
            actor_loss = advantage * -log_probs

            loss = actor_loss + critic_loss

            losses.append(loss.item())
            print('actor loss : ', actor_loss.item())
            print('critic loss : ', critic_loss.item())

            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)

            optimizer.zero_grad()
            critic_optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            critic_optimizer.step()

    draw_list_graph(losses, result_dir, train_mode + " loss", xlabel="episode", ylabel="loss")
    draw_list_graph(episodes_length, result_dir, train_mode + " episode length", xlabel="episode", ylabel="length")


def main():
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

    print("args: ")
    print("embedding size: %d" % embedding_size)
    print("hidden size: %d" % hidden_size)
    print("grad clip: %f" % grad_clip)
    print("learning rate: %f" % learning_rate)
    print("num glimpses: %d" % n_glimpses)
    print("tanh exploration: %d" % tanh_exploration)

    print("")
    print("mode: %s" % train_mode)
    print("episode: %d" % episode)
    print("sequence length: %d" % seq_len)
    print("beta: %f" % beta)
    print("result dir: %s" % result_dir)

    train(embedding_size, hidden_size, grad_clip, learning_rate,
          n_glimpses, tanh_exploration, train_mode, episode, seq_len, beta, result_dir, result_graph_dir)


if __name__ == "__main__":
    main()

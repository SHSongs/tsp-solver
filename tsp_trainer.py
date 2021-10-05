import os

import or_gym

import torch
import torch.optim as optim

from util import rotate_actions, VisualData, draw_list_graph, \
    stack_visualization_data, make_pointer_network, make_critic_network
from gym_util import play_tsp
from config import args_parser
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(actor, critic, grad_clip, decay, learning_rate, train_mode, episode_num,
          seq_len, beta, result_dir, result_graph_dir):
    # env setup
    env_config = {'N': seq_len}
    env = or_gym.make('TSP-v1', env_config=env_config)

    optimizer = optim.Adam(actor.parameters(), lr=learning_rate, weight_decay=decay)

    # result data
    losses = []
    episodes_length = []
    visual_data = VisualData()

    if train_mode == "active-search":

        # Active search
        moving_avg = torch.zeros(1)
        first_step = True

        for i in range(episode_num):
            env.reset()
            coords = torch.FloatTensor(env.coords).transpose(1, 0).unsqueeze(0)

            actor.prepare(coords.to(device))

            done = False
            total_reward = 0

            while not done:
                visited = env.visit_log
                log_prob, action = actor.one_step(visited, env.step_count)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

            log_probs, actions = actor.result()
            stack_visualization_data(visual_data, coords, actions, i, result_graph_dir)

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

    torch.save(actor.state_dict(), os.path.join(result_dir, 'actor.pth'))

    draw_list_graph(losses, result_dir, train_mode + " loss", xlabel="episode", ylabel="loss")
    draw_list_graph(episodes_length, result_dir, train_mode + " episode length", xlabel="episode", ylabel="length")


def main():
    args = args_parser()

    # Pointer network hyper parameter
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    grad_clip = args.grad_clip
    decay = args.decay
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
    print("weight decay: %f" % decay)
    print("learning rate: %f" % learning_rate)
    print("num glimpses: %d" % n_glimpses)
    print("tanh exploration: %d" % tanh_exploration)

    print("")
    print("mode: %s" % train_mode)
    print("episode: %d" % episode)
    print("sequence length: %d" % seq_len)
    print("beta: %f" % beta)
    print("result dir: %s" % result_dir)

    ptr_net = make_pointer_network(embedding_size, hidden_size, n_glimpses, tanh_exploration, seq_len, device)

    critic_net = None
    if train_mode == "actor-critic":
        critic_net = make_critic_network(embedding_size, hidden_size, n_glimpses, tanh_exploration, seq_len, device)

    train(ptr_net, critic_net, grad_clip, decay, learning_rate, train_mode, episode,
          seq_len, beta, result_dir, result_graph_dir)


if __name__ == "__main__":
    main()

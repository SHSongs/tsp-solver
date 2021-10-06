import os

import matplotlib.pyplot as plt
import torch

from pointer_network import PointerNetwork
from critic_network import CriticNetwork


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_pointer_network(embedding_size, hidden_size, n_glimpses, tanh_exploration, seq_len, device):
    actor = PointerNetwork(embedding_size, hidden_size, seq_len, n_glimpses, tanh_exploration)
    actor.to(device)
    return actor


def make_critic_network(embedding_size, hidden_size, n_glimpses, tanh_exploration, seq_len, device):
    critic = CriticNetwork(embedding_size, hidden_size, seq_len, n_glimpses, tanh_exploration)
    critic.to(device)
    return critic


class VisualData:
    def __init__(self):
        self.coords = []
        self.actions = []
        self.episode = []

    def add(self, c, a, e):
        self.coords.append(c.squeeze(0))
        self.actions.append(a.squeeze(0))
        self.episode.append(str(e))

    def get(self):
        return torch.stack(self.coords), torch.stack(self.actions), self.episode

    def clear(self):
        self.coords.clear()
        self.actions.clear()
        self.episode.clear()


def visualization(result_graph_dir, coords, tour_indices, episodes):
    """
    Args:
        result_graph_dir: plot save path
        coords: [ data_num x seq_num x 2 ]
        tour_indices: [ data_num x seq_num ]
        episodes: [ data_num ]
    """
    plt.close('all')

    num_plots = 3
    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]  # 2dim -> 1dim

    for i, ax in enumerate(axes):
        # idx 의 좌표 가져오기
        idx = tour_indices[i].unsqueeze(0)
        idx = idx.expand(2, -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)
        data = coords[i].transpose(1, 0).cpu()
        data = data.gather(1, idx.cpu()).cpu().numpy()

        # draw graph
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        # limit 설정
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.title.set_text(episodes[i])

        if len(tour_indices) == 1:
            break

    filename = episodes[0] + '-' + episodes[-1] + '_episode_result.png'
    plt.savefig(os.path.join(result_graph_dir, filename))


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


def draw_list_graph(lst, result_dir, title, xlabel='episode', ylabel=''):
    plt.close('all')
    plt.plot(range(len(lst)), lst, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    filename = title + '.png'
    plt.savefig(os.path.join(result_dir, filename))

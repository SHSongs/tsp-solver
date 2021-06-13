import os

import matplotlib.pyplot as plt
import torch


def rotate_actions(actions, start):
    """
    Examples::
        Args:
            actions: [3, 2, 0, 1, 6, 4, 5]
            start: 2
        return
            [0, 1, 6, 4, 5, 3, 2]
    """
    start_idx = actions.index(start)
    a_1 = actions[start_idx + 1:]
    a_2 = actions[0:start_idx + 1]
    return a_1 + a_2


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

    filename = episodes[0] + '-' + episodes[-1] + '_episode_result.png'
    plt.savefig(os.path.join(result_graph_dir, filename))


def draw_list_graph(lst, result_dir, title, xlabel='episode', ylabel=''):
    plt.close('all')
    plt.plot(range(len(lst)), lst, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    filename = title + '.png'
    plt.savefig(os.path.join(result_dir, filename))

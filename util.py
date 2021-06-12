import argparse

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


def args_parser():
    parser = argparse.ArgumentParser(description="Solve TSP",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lr", default=3e-4, type=float, dest="lr")
    parser.add_argument("--embedding_size", default=128, type=int, dest="embedding_size")
    parser.add_argument("--hidden_size", default=128, type=int, dest="hidden_size")
    parser.add_argument("--grad_clip", default=1.5, type=float, dest="grad_clip")

    parser.add_argument("--n_glimpses", default=2, type=int, dest="n_glimpses")
    parser.add_argument("--tanh_exploration", default=10, type=int, dest="tanh_exploration")

    parser.add_argument("--beta", default=0.99, type=float, dest="beta")

    parser.add_argument("--episode", default=1000, type=int, dest="episode")
    parser.add_argument("--seq_len", default=10, type=int, dest="seq_len")

    parser.add_argument("--mode", default="active-search", choices=["active-search", "actor-critic"], type=str,
                        dest="mode", help="mode is active-search or actor-critic")

    args = parser.parse_args()
    return args


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


def visualization(coords, tour_indices, episodes):
    """
    Args:
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
        data = coords[i].transpose(1, 0)
        data = data.gather(1, idx).cpu().numpy()

        # draw graph
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        # limit 설정
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.title.set_text(episodes[i])
    plt.show()

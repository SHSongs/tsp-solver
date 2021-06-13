import argparse


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

    parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

    args = parser.parse_args()
    return args

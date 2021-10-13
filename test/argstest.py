import os
from subprocess import *
import unittest


def execute_cmd(cmd):
    cmd = cmd.split()
    out = check_output(cmd, universal_newlines=True, stderr=STDOUT)
    out = list(filter(lambda x: len(x) > 0, out.split('\n')))
    return out[-1]


class TestArgs(unittest.TestCase):

    def test_pointer_network_in_out(self):
        os.chdir('../')
        print(os.getcwd())

        cmd_1 = "python tsp_trainer.py  \
        --lr 3e-4 \
        --embedding_size 10 \
        --hidden_size 10 \
        --grad_clip 1.5 \
        --decay 0.01 \
        --n_glimpses 2 \
        --tanh_exploration 10 \
        --beta 0.99 \
        --episode 100 \
        --seq_len 4 \
        --mode active-search \
        --result_dir ./result"

        cmd_2 = "python tsp_trainer.py  \
        --lr 3e-4 \
        --embedding_size 10 \
        --hidden_size 10 \
        --grad_clip 1.5 \
        --decay 0.01 \
        --n_glimpses 2 \
        --tanh_exploration 10 \
        --beta 0.99 \
        --episode 10 \
        --seq_len 4 \
        --mode actor-critic \
        --result_dir ./result"

        cmd_3 = "python tsp_tester.py \
        --embedding_size 10 \
        --hidden_size 10 \
        --seq_len 20 \
        --actor_dir ./result/actor.pth"

        cmd_lst = [cmd_1, cmd_2, cmd_3]

        for i in cmd_lst:
            print(i)
            self.assertTrue(execute_cmd(i) == "end tsp")


if __name__ == "__main__":
    unittest.main()

import or_gym

import torch
import torch.optim as optim
from pointer_network import PointerNetwork

episode = 1000
seq_len = 7

env_config = {'N': seq_len}
env = or_gym.make('TSP-v1', env_config=env_config)

embedding_size = 128
hidden_size = 128

learning_rate = 3e-4
model = PointerNetwork(embedding_size, hidden_size, seq_len, n_glimpses=2, tanh_exploration=10)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

episodes_length = []

for i in range(episode):
    s = env.reset()

    coords = torch.FloatTensor(env.coords).transpose(1, 0).unsqueeze(0)

    log_probs, actions = model(coords)

    actions = actions.squeeze(0).tolist()

    start_idx = actions.index(s[0])
    a_1 = actions[start_idx + 1:]
    a_2 = actions[0:start_idx + 1]
    actions = a_1 + a_2

    print('first state', s)

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

    episodes_length.append(total_reward)
    print('total length', total_reward)

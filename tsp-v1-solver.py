import or_gym

import torch
import torch.optim as optim
from pointer_network import PointerNetwork
import matplotlib.pyplot as plt

train_mode = "active-search"

episode = 1000
seq_len = 7

env_config = {'N': seq_len}
env = or_gym.make('TSP-v1', env_config=env_config)

# Pointer network hyper parameter
embedding_size = 128
hidden_size = 128
grad_clip = 1.5
learning_rate = 3e-4
model = PointerNetwork(embedding_size, hidden_size, seq_len, n_glimpses=2, tanh_exploration=10)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
episodes_length = []

# Active search hyper parameter
beta = 0.99
moving_avg = torch.zeros(1)
first_step = True

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

plt.plot(range(len(losses)), losses, color="blue")
plt.title(train_mode + " losses")
plt.xlabel("episode")
plt.ylabel("loss")
plt.show()

plt.plot(range(len(episodes_length)), episodes_length, color="blue")
plt.title("Episode length")
plt.xlabel("episode")
plt.ylabel("length")
plt.show()

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

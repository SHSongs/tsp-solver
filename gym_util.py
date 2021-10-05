def play_tsp(env, coords, actor, device):
    env.reset()

    actor.prepare(coords.to(device))

    done = False
    total_reward = 0

    while not done:
        visited = env.visit_log
        log_prob, action = actor.one_step(visited, env.step_count)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward

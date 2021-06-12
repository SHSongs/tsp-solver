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


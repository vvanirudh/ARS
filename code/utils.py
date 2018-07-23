# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np

def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)



def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def restructure_episodes(episodes):
    '''
    episodes is a list of dictionaries with 2D numpy arrays
    Need to return a dictionary of 3D numpy arrays
    '''
    num_episodes = len(episodes)
    keys = episodes[0].keys()
    shapes = {key: episodes[0][key].shape for key in keys}
    re_episodes = {key: np.zeros((num_episodes, shapes[key][0], shapes[key][1])) for key in keys}
    c = 0
    for episode in episodes:
        for key in keys:
            re_episodes[key][c, :, :] = episode[key]
        c += 1
    return re_episodes

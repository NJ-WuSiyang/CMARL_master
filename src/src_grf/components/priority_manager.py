import numpy as np
import math


class PriorityManager:
    def __init__(self, size, alpha):
        self.segment_tree_depth = math.ceil(np.log2(size)) + 1
        self.segment_tree_size = 1 << self.segment_tree_depth
        self.segment_tree_half_size = self.segment_tree_size >> 1
        self.segment_tree = np.zeros(self.segment_tree_size, dtype=np.float64)
        self.alpha = alpha

    def update(self, indices, priorities):
        idx = indices + self.segment_tree_half_size
        proportional = priorities ** self.alpha
        self.segment_tree[idx] = proportional
        for _ in range(self.segment_tree_depth - 1):
            idx >>= 1
            self.segment_tree[idx] = self.segment_tree[idx << 1] + self.segment_tree[(idx << 1) | 1]

    def sample(self, batch_size):
        cdf = np.random.rand(batch_size) * self.segment_tree[1]
        idx = np.ones(batch_size, dtype=np.int32)
        for _ in range(self.segment_tree_depth - 1):
            go_right = cdf > self.segment_tree[idx << 1]
            cdf -= go_right * self.segment_tree[idx << 1]
            idx = (idx << 1) | go_right
        return idx - self.segment_tree_half_size


if __name__ == '__main__':
    seg = PriorityManager(5, 1)
    for i in range(4):
        seg.update(np.array([i]), np.array([(i + 1) / 10]))
        print(seg.segment_tree)
    for i in range(4):
        print(seg.sample(i + 1))

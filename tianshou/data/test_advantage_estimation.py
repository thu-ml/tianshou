

from advantage_estimation import *

class ReplayBuffer(object):
    def __init__(self):
        self.index = [
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1],
        ]
        self.data = [
            [(0, 0, 10, False), (0, 0, 1, False), (0, 0, -100, True)],
            [(0, 0, 1, False), (0, 0, 1, False), (0, 0, 1, False), (0, 0, 5, False)],
            [(0, 0, 9, False), (0, 0, -2, True)],
        ]


buffer = ReplayBuffer()
sample_index = [
    [0, 2, 0],
    [1, 2, 1, 3],
    [],
]

data = full_return(buffer)
print(data['return'])
data = full_return(buffer, sample_index)
print(data['return'])
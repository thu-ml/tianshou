import numpy as np

from data_buffer.vanilla import VanillaReplayBuffer

capacity = 12
nstep = 3
buffer = VanillaReplayBuffer(capacity=capacity, nstep=nstep)

for i in range(capacity):
    s = np.random.randint(10)
    a = np.random.randint(3)
    r = np.random.randint(5)
    done = np.random.rand() > 0.6

    buffer.add((s, a, r, done))

    if i % 5 == 0:
        print('i = {}:'.format(i))
        print(buffer.index)
        print(buffer.data)

print('Now buffer with size {}:'.format(buffer.size))
print(buffer.index)
print(buffer.data)
buffer.clear()
print('Cleared buffer with size {}:'.format(buffer.size))
print(buffer.index)
print(buffer.data)

for i in range(20):
    s = np.random.randint(10)
    a = np.random.randint(3)
    r = np.random.randint(5)
    done = np.random.rand() > 0.6

    buffer.add((s, a, r, done))
    print('added frame {}, {}:'.format(i, (s, a, r, done)))
    print(buffer.index)
    print(buffer.data)

print('sampling from buffer:')
print(buffer.index)
print(buffer.data)
print(buffer.sample(8))

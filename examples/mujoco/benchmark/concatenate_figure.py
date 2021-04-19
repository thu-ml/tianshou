import matplotlib.pyplot as plt
import matplotlib.image as gImage

import os
# plot all figure
# path="/home/huayu/git/tianshou/examples/mujoco/benchmark"
# dirs = os.listdir(path)
# algorithms=["ppo","a2c","reinforce","ddpg","td3","sac"]
# # algorithms=[""]
# for algorithm in algorithms:
#     aa = []
#     i=1
#     for dir in dirs:
#         if '.' in dir:
#             continue
#         aa.append(dir)

#     fig = plt.figure(figsize=(42,42))   #定义整个画布
#     for i in range(9):
#         ax = fig.add_subplot(331 + i, title=aa[i])
#         ax.imshow(gImage.imread(os.path.join(path,aa[i],algorithm,"figure.png")))
#         ax.set_title(aa[i], fontsize=50)

#     plt.tight_layout()
#     plt.savefig(algorithm + ".png")


path="/home/huayu/git/tianshou/"
algorithms=["PPO","A2C","REINFORCE","DDPG","TD3","SAC"]
# algorithms=[""]
fig = plt.figure(figsize=(42,34))   #定义整个画布
for i in range(6):
    algorithm = algorithms[i]
    ax = fig.add_subplot(321 + i)
    ax.imshow(gImage.imread(os.path.join(path,algorithm+".png")))

plt.tight_layout()
plt.savefig("all.png")



hidden_dim = 512

max_step = 300 # 2000 for 100100 # 300 for 5050 700 for 7070
GAMMA = 0.99
n_episode = 50000
i_episode = 0
capacity = 50000
batch_size = 256
n_epoch = 25
epsilon = 0.9 # 0.9
score = 0

tau = 0.96
lamb = 0.001

sensor_range = 21
monitor_range = 3
map_size = 50

episode_length=max_step
target_max_health = 10

entropy_coef =0.06
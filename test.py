import numpy as np
import random
#
# size=200
# sensor_range=10
# monitor_range=5
# landmarks=[]
# if size % (sensor_range - 2) > 0:
#     n_landmark_in_length = size // (sensor_range - 2) + 1
# else:
#     n_landmark_in_length = size // (sensor_range - 2)
# the_distance=size/n_landmark_in_length
# the_distance=the_distance/2
# for i in range(n_landmark_in_length):
#     landmarks.append([])
#     for ii in range(n_landmark_in_length):
#         landmarks[i].append(np.array([the_distance * (2 * ii + 1), the_distance * (2 * i + 1)]))
# landmark_visited = [[False for _ in range(n_landmark_in_length)] for _ in range(n_landmark_in_length)]
#
# landmarks = np.array(landmarks)
# # landmarks_position = np.array(landmarks[0])
# # for i in range(len(landmarks) - 1):
# #     landmarks_position = np.concatenate([landmarks_position, np.array(landmarks[i + 1])])
# # print(landmarks_position)
# state_n=[10,10]
# x_min_range = max(0, state_n[0] // (sensor_range - 2) - 1)
# y_min_range = max(0, state_n[1] // (sensor_range - 2) - 1)
# x_max_range = min(n_landmark_in_length -1, state_n[0] // (sensor_range - 2) + 1)
# y_max_range = min(n_landmark_in_length -1, state_n[1] // (sensor_range - 2) + 1)
# for x_i in range(x_min_range, x_max_range + 1):
#     for y_i in range(y_min_range, y_max_range + 1):
#         print(landmarks[y_i][x_i], landmarks[y_i][x_i][0] - state_n[0], landmarks[y_i][x_i][1] - state_n[1])
#         if np.abs(landmarks[y_i][x_i][0] - state_n[0]) <= (monitor_range // 2) and np.abs(
#                 landmarks[y_i][x_i][1] - state_n[1]) <= (monitor_range // 2):
#             if landmark_visited[y_i][x_i] == False:
#                 landmark_visited[y_i][x_i] = True
# print(landmark_visited)
#
# # ans_landmarks=landmarks[0]
# # for i in range(len(landmarks)-1):
# #     ans_landmarks = np.concatenate([ans_landmarks,landmarks[i+1]])
# # print(landmarks)
#
# # print(target)
# # for i in range(n_landmark_in_length):
# #     for ii in range(n_landmark_in_length):
# #         print(landmarks[i][ii])
# # x=24
# # print(n_landmark_in_length )
# # print(landmarks[x//n_landmark_in_length][x%n_landmark_in_length])
# # print(landmarks[0][24])
#
# # c=np.array([1,1])
# # b=np.array([[1,2],[2,3]])
# # wolf_info = np.concatenate([[wolf[0], wolf[1]]
# # 		                            for i, wolf in enumerate(landmarks)], axis=0)
# # print(wolf_info)
# # np.concatenate([wolf_info,c], axis=0)

import torch
import torch.nn.functional as F
import torch.nn as nn

# rnn = nn.GRUCell(10, 20)
# input = torch.randn(4, 2)
# print(input)
# input = input.reshape(-1, 4)
# print(input)
# output = []
# for i in range(6):
#         hx = rnn(input[i], hx)
#         output.append(hx)

# x = torch.tensor([[1,2],[3,4]],dtype=torch.float)
# # x = x[None, :]
# x = F.softmax(x,dim=1)
# print(x)

# index = random.sample([0, 10, 100, 101], 2)
# print(index)

myarray = np.array([[[0., 4.]],
                    [[1., 5.]],
                    [[2., 6.]],
                    [[3., 7.]]])
maxes = np.max(myarray,axis=2)
mins = np.min(myarray,axis=2)
print(maxes,mins)
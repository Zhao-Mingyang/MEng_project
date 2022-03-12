import numpy as np
from common.tools import pooling
import copy
import gym
from config import *
# relation ship between the observation and map size

class multi_agent_env():
    def __init__(self, n_agent):
        super(multi_agent_env, self).__init__()
        self.stage = 0
        self.n_agent = n_agent

        self.sensor_range = sensor_range
        self.monitor_range = monitor_range

        self.episode_length = episode_length
        self.map_size = map_size
        self.map_pos_len = int(np.log2(self.map_size)) + 1

        if self.map_size % (self.sensor_range - 2) > 0:
            self.n_landmark_in_length = self.map_size // (self.sensor_range - 2) + 1
        else:
            self.n_landmark_in_length = self.map_size // (self.sensor_range - 2)
        self.n_landmark = self.n_landmark_in_length * self.n_landmark_in_length
        self.n_action_move = 4
        self.n_dim = 2
        self.n_target = 1  # change to n_targets


        self.landmarks = []
        the_distance = self.map_size / self.n_landmark_in_length
        the_distance = the_distance / 2
        for i in range(self.n_landmark_in_length):
            self.landmarks.append([])
            for ii in range(self.n_landmark_in_length):
                self.landmarks[i].append(np.array([the_distance * (2 * ii + 1), the_distance * (2 * i + 1)]))

        self.landmarks = np.around(np.array(self.landmarks))
        self.landmarks = self.landmarks.astype(int)
        # print(self.landmarks)
        self.previous_landmark_counter = 0

        self.landmark_visited = np.array([[False for _ in range(self.n_landmark_in_length)] for _ in
                                          range(self.n_landmark_in_length)])
        self.landmark_visited_times = np.array(
            [[0 for _ in range(self.n_landmark_in_length)] for _ in range(self.n_landmark_in_length)])
        self.agent_landmark_id = np.array([[-1, -1] for _ in range(self.n_agent)])
        self.agent_monitor_target = np.array([-1 for _ in range(self.n_agent)])

        self.mask_dim = self.n_agent #self.n_landmark + self.n_target

        # Movable Agent
        self.state_n = [np.array([0, 0, 0]) for _ in
                        range(self.n_agent)]  # 0 for searching, 1 for monitoring, else for tracking
        self.pre_state_n = self.state_n
        self.agent_searching_n = [True for _ in range(self.n_agent)]  # change to agents in searching
        self.agent_monitor_n = [False for _ in range(self.n_agent)]  # change to agent monitored

        # Movable target
        self.target_n = [self.random_target() for _ in range(self.n_target)]
        self.target_info = np.array([[-1, -1] for _ in range(self.n_target)])
        self.target_monitored_n = [False for _ in range(self.n_target)]  # change to targets not monitored
        self.target_detected_n = np.array(
            [False for _ in range(self.n_target)])  # change to targets detected not monitored

        # observation
        self.map_eye = np.array(
            [[0 for _ in range(self.map_size)] for _ in range(self.map_size)])
        self.landmarks_eye = np.array(
            [[0 for _ in range(self.map_size)] for _ in range(self.map_size)])
        for i in range(self.n_landmark_in_length):
            for ii in range(self.n_landmark_in_length):
                x = self.landmarks[i][ii][0] - 1
                y = self.landmarks[i][ii][1] - 1
                self.landmarks_eye[y][x] = 1

        self.targets_eye = np.array(
            [[0 for _ in range(self.map_size)] for _ in range(self.map_size)])
        for t_i, thetarget in enumerate(self.target_n):
            x = thetarget[0] - 1
            y = thetarget[1] - 1
            self.targets_eye[y][x] = 1

        # health
        self.target_max_health = target_max_health

        self.target_health_eye = np.eye(self.target_max_health)
        self.target_health = np.array([np.random.randint(self.target_max_health) for _ in range(self.n_target)])
        self.target_current_health = self.target_health

        self.detected_reward = 50
        self.goal_reward = 20
        self.agent_score = np.array([0 for _ in range(self.n_agent)])

        # Used by OpenAI baselines
        # self.action_space = gym.spaces.Discrete(self.n_action)
        # self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.n_action_move), gym.spaces.Discrete(self.n_landmark + self.n_target)))
        self.action_space = gym.spaces.Discrete(self.n_action_move)
        self.n_action = self.n_action_move
        self.view_range = self.sensor_range // 2 * 2
        # self.observation_space = gym.spaces.Box(low=-1, high=1,
        #                                         shape=[self.sensor_range ** 2 // (3 ** 2) * self.n_agent + 3 +
        #                                                # self.n_target * 1 +
        #                                                self.n_landmark * 1 +
        #                                                self.n_agent * 2])
        # self.observation_space = gym.spaces.Box(low=-1, high=1,
        #                                         shape=[ 5 * 2 + 1])
        # self.observation_space = gym.spaces.Box(low=-1, high=1,
        #                                         shape=[(self.sensor_range // 3) ** 2 + 5*2 + 1])

        # self.len_obs = sum(self.observation_space.shape)
        self.len_obs =  (self.sensor_range) ** 2 + self.map_pos_len * 2 + 1
        # print(self.len_obs)
        self.metadata = {'render.modes': []}
        # self.reward_range = (-200., 20000.)
        self.spec = self.n_agent

        self.t_step = 0

    def reset(self):
        self.stage = 0
        self.landmark_visited = np.array([[False for _ in range(self.n_landmark_in_length)] for _ in
                                          range(self.n_landmark_in_length)])
        self.landmark_visited_times = np.array(
            [[0 for _ in range(self.n_landmark_in_length)] for _ in range(self.n_landmark_in_length)])
        self.agent_landmark_id = np.array([[-1, -1] for _ in range(self.n_agent)])
        self.previous_landmark_counter = 0
        self.agent_monitor_target = np.array([-1 for _ in range(self.n_agent)])

        self.state_n = [np.array([0, 0,  0]) for _ in
                        range(self.n_agent)]  # 0 for searching, 1 for monitoring, else for tracking
        self.pre_state_n = self.state_n
        self.agent_searching_n = [True for _ in range(self.n_agent)]  # change to agents in searching
        self.agent_monitor_n = [False for _ in range(self.n_agent)]  # change to agent monitored

        self.target_n = [self.random_target() for _ in range(self.n_target)]
        self.target_info = np.array([[-1, -1] for _ in range(self.n_target)])
        self.target_monitored_n = [False for _ in range(self.n_target)]  # change to targets not monitored
        self.target_detected_n = np.array(
            [False for _ in range(self.n_target)])  # change to targets detected not monitored

        self.target_health = np.array([np.random.randint(self.target_max_health) for _ in range(self.n_target)])

        self.map_eye = np.array(
            [[0 for _ in range(self.map_size)] for _ in range(self.map_size)])
        self.landmarks_eye = np.array(
            [[0 for _ in range(self.map_size)] for _ in range(self.map_size)])
        for i in range(self.n_landmark_in_length):
            for ii in range(self.n_landmark_in_length):
                x = self.landmarks[i][ii][0] - 1
                y = self.landmarks[i][ii][1] - 1
                self.landmarks_eye[y][x] = 1

        self.target_health_eye = np.eye(self.target_max_health)
        self.target_health = np.array([np.random.randint(self.target_max_health) for _ in range(self.n_target)])
        self.target_current_health = self.target_health

        self.agent_score = np.array([0 for _ in range(self.n_agent)])

        self.t_step = 0

        return self.obs_n()

    def random_target(self):
        target = np.random.randint(0, self.map_size, [self.n_dim])
        return target

    def step(self, action_n):

        self.t_step += 1
        self.agent_score = np.array([0 for _ in range(self.n_agent)])
        # detect and monitor target

        for a_i, agent in enumerate(self.state_n):
            if action_n[a_i]== 5 and self.agent_monitor_n[a_i] == True and self.target_monitored_n[
                self.agent_monitor_target[a_i]] == False:
                self.target_monitored_n[self.agent_monitor_target[a_i]] = False
                self.agent_monitor_target[a_i] = -1
                self.agent_monitor_n[a_i] = False
            if self.agent_monitor_n[a_i] and self.agent_monitor_target[a_i] > -1 and self.target_monitored_n[
                self.agent_monitor_target[a_i]] == True:
                self.target_current_health[self.agent_monitor_target[a_i]] = min(self.target_max_health,
                                                                                 self.target_current_health[
                                                                                     self.agent_monitor_target[
                                                                                         a_i]] + 0.01)
                # self.agent_score[a_i] += self.target_max_health - self.target_current_health[
                #     self.agent_monitor_target[a_i]]

        # Move Agents # problems with landmark update
        for i, action in enumerate(action_n):
            if not self.agent_monitor_n[i]:
                # new_row = -1
                # new_column = -1
                serch_monitor = 0
                if action == 0:
                    new_row = max(self.state_n[i][0] - 1, 0)
                    new_column = self.state_n[i][1]
                elif action == 1:
                    new_row = min(self.state_n[i][0] + 1, self.map_size - 1)
                    new_column = self.state_n[i][1]
                elif action == 2:
                    new_row = self.state_n[i][0]
                    new_column = max(self.state_n[i][1] - 1, 0)
                elif action == 3:
                    new_row = self.state_n[i][0]
                    new_column = min(self.state_n[i][1] + 1, self.map_size - 1)
                elif action == 4 or 5:
                    new_row = self.state_n[i][0]
                    new_column = self.state_n[i][1]
                    if action ==4:
                        self.agent_searching_n[i] = False
                        serch_monitor = 1
                    else:
                        self.agent_searching_n[i] = True

                self.state_n[i] = np.array([new_row, new_column, serch_monitor])
                x_min_range = max(0, self.state_n[i][0] // (self.sensor_range + 2) - 1)
                y_min_range = max(0, self.state_n[i][1] // (self.sensor_range + 2) - 1)
                x_max_range = min(self.n_landmark_in_length, self.state_n[i][0] // self.sensor_range + 2)
                y_max_range = min(self.n_landmark_in_length, self.state_n[i][1] // self.sensor_range  + 2)
                # print(x_min_range,x_max_range,y_min_range,y_max_range)
                for x_i in range(x_min_range, x_max_range):
                    for y_i in range(y_min_range, y_max_range):
                        if np.abs(self.landmarks[y_i][x_i][0] - self.state_n[i][0] -1) <= (
                                self.monitor_range // 2) and np.abs(
                                self.landmarks[y_i][x_i][1] - self.state_n[i][1] -1) <= (self.monitor_range // 2):
                            if self.landmark_visited[y_i][x_i] == False:
                                # print('myindex:',x_i,y_i,self.agent_landmark_id[i][0],self.agent_landmark_id[i][1])
                                self.landmark_visited[y_i][x_i] = True
                                self.agent_score[i] += self.detected_reward
                            if self.agent_landmark_id[i][0] != x_i or self.agent_landmark_id[i][1] != y_i:
                                self.agent_landmark_id[i][0] = x_i
                                self.agent_landmark_id[i][1] = y_i
                                self.landmark_visited_times[y_i][x_i] += 1
        for c_i in range(len(self.landmark_visited_times[0])):
            for c_ii in range(len(self.landmark_visited_times[0])):
                if self.landmark_visited_times[c_i][c_ii] == True and self.landmark_visited_times[c_i][c_ii]==0:
                    print('error')
                    print(self.landmark_visited_times)
                    print( self.landmark_visited)
        for t_i, target in enumerate(self.target_n):
            for a_i, agent in enumerate(self.state_n):
                if self.agent_monitor_n[a_i] == False:
                    if np.abs(target[0] - agent[0]) <= (self.sensor_range // 2) and np.abs(target[1] - agent[1]) <= (
                            self.sensor_range // 2):
                        if self.target_detected_n[t_i] == False:
                            # self.agent_score[a_i] += self.detected_reward
                            self.target_detected_n[t_i] = True
                            self.target_info[t_i] = [target[0], target[1]]
                        # elif self.state_n[a_i][2] != 0 and self.stage == 1:
                        #     if t_i == self.state_n[a_i][2]:
                        #         self.agent_score[a_i] += self.detected_reward * 3
                    if np.abs(target[0] - agent[0]) <= (self.monitor_range // 2) and np.abs(
                                target[1] - agent[1]) <= (self.monitor_range // 2):
                        if action_n[a_i] == 4 and self.target_monitored_n[t_i] == False:  # into the monitor state
                            self.agent_monitor_target[a_i] = t_i
                            self.target_monitored_n[t_i] = True
                            self.agent_monitor_n[a_i] = True

        # state_target_info = np.concatenate([[target[0], target[1]]
        #                                     for i, target in enumerate(self.target_info)], axis=0)
        info_state_n = []
        for i, state in enumerate(self.state_n):
            info_state_n.append(state)

        info_landmark = np.concatenate([thelandmark for thelandmark in self.landmarks])

        info = {'target': self.target_info, 'landmark': info_landmark, 'state': info_state_n}

        return_obs = self.obs_n()
        return_adj = self.get_adj()
        return_rew, info_r = self.reward()
        return_done = self.done()

        if return_done and self.stage == 0:
            print("NONO", info_r['landmark_counter'])

        info['rew'] = info_r
        self.pre_state_n = self.state_n

        # if return_done:
        #     self.reset()

        return return_obs, return_rew, return_done

    def obs_n(self):
        # ans_obs = [self.obs(a_i) for a_i in range(self.n_agent)]
        ans_obs = []
        for agent in range(self.n_agent):
            landmarks_view = self.get_obs(agent)
            a_state = self.state_n[agent]

            a_bin = []
            a_x = a_state[0]
            a_y = a_state[1]
            for i in range(self.map_pos_len):
                a_bin.insert(0, np.mod(a_x, 2))
                a_x = int(a_x / 2)
            for i in range(self.map_pos_len):
                a_bin.insert(0, np.mod(a_y, 2))
                a_y = int(a_y / 2)
            # a_bin.append(a_state[2])
            # ans_landmark_visited_times = np.array(self.landmark_visited_times[0])
            # for i in range(len(self.landmarks) - 1):
            #     ans_landmark_visited_times = np.concatenate(
            #         [ans_landmark_visited_times, np.array(self.landmark_visited_times[i + 1])])
            temp = [landmarks_view,a_bin]
            # print(temp)
            # temp.append(ans_landmark_visited_times)
            # temp.append(self.agent_searching_n)
            # temp.append(self.agent_monitor_n)
            # print(temp)
            temp = np.concatenate(temp)
            temp = np.append(temp,agent)
            ans_obs.append(temp)
            # print(len(ans_obs[0]))
        return np.array(ans_obs)

    def get_obs(self, agent_id):
        self.landmarks_eye = np.array(
            [[0 for _ in range(self.map_size)] for _ in range(self.map_size)])
        mark_x_zero = False
        mark_x_edge = False
        mark_y_zero = False
        mark_y_edge = False
        x_min_range = max(0, self.state_n[agent_id][0] - self.sensor_range // 2)
        y_min_range = max(0, self.state_n[agent_id][1] - self.sensor_range // 2)
        x_max_range = min(self.map_size, self.state_n[agent_id][0] + self.sensor_range // 2 + 1)
        y_max_range = min(self.map_size, self.state_n[agent_id][1] + self.sensor_range // 2 + 1)
        if x_min_range == 0 and (x_max_range - x_min_range) < self.sensor_range:
            mark_x_zero = True
        elif x_max_range == self.map_size and (x_max_range - x_min_range) < self.sensor_range:
            mark_x_edge = True
        if y_min_range == 0 and (y_max_range - y_min_range) < self.sensor_range:
            mark_y_zero = True
        elif y_max_range == self.map_size and (y_max_range - y_min_range) < self.sensor_range:
            mark_y_edge = True
        # target_view=[]
        # for y_i in range(y_min_range, y_max_range + 1):
        #     target_view.append(self.targets_eye[y_i][x_min_range:x_max_range+1])
        # target_view = np.concatenate([target_view], axis=0)
        # target_view = np.concatenate([view for view in target_view])
        # target_view = np.array(target_view)
        #
        #
        # a_state = []
        # for y_i in range(y_min_range, y_max_range + 1):
        #     a_state.append(self.map_eye[y_i][x_min_range:x_max_range+1])
        # a_state = np.concatenate([a_state], axis=0)
        # a_state = np.concatenate([view for view in a_state])
        # a_state = np.array(a_state)33
        #
        #
        ans_landmark_visited_times = np.array(self.landmark_visited_times[0])
        for i in range(len(self.landmarks) - 1):
            ans_landmark_visited_times = np.concatenate(
                [ans_landmark_visited_times, np.array(self.landmark_visited_times[i + 1])])
        visited_max = max(ans_landmark_visited_times)
        visited_min = min(ans_landmark_visited_times)
        norm_ans_landmark_visited_times = np.zeros(len(ans_landmark_visited_times))

        for i in range(len(ans_landmark_visited_times)):
            if ans_landmark_visited_times[i]==0:
                norm_ans_landmark_visited_times[i] = 1

        # norm_ans_landmark_visited_times = [(1 - (value - visited_min) / (visited_max - visited_min + 0.1)) for
        #                                    value in ans_landmark_visited_times]

        # print(norm_ans_landmark_visited_times)
        #
        for y_i, landmarkrow in enumerate(self.landmarks):
            for x_i, thelandmark in enumerate(landmarkrow):
                # print(thelandmark)
                self.landmarks_eye[thelandmark[1] - 1][thelandmark[0] - 1] = norm_ans_landmark_visited_times[y_i * self.n_landmark_in_length + x_i]
                # self.landmarks_eye[thelandmark[1]-1][thelandmark[0]-1] = np.round(
                #     norm_ans_landmark_visited_times[y_i * self.n_landmark_in_length + x_i])

        # for t_i in range(self.n_target):
        #     coff = (self.target_max_health-self.target_health[t_i])*0.5*(1-self.target_detected_n[t_i])
        #     self.landmarks_eye[self.target_n[t_i][1]][self.target_n[t_i][0]] += coff + 1
        #     print(self.landmarks_eye[self.target_n[t_i][1]][self.target_n[t_i][0]] )

        # print(self.landmarks_eye)
        landmarks_view = []  # max pooling
        for y_i in range( y_min_range,y_max_range):
            landmarks_view.append(self.landmarks_eye[y_i][x_min_range:x_max_range])
            if mark_x_zero:
                # print(self.sensor_range-x_max_range -1)
                landmarks_view[-1] = np.concatenate([np.zeros(self.sensor_range - x_max_range), landmarks_view[-1]],
                                                    axis=0)
            elif mark_x_edge:
                landmarks_view[-1] = np.concatenate(
                    [landmarks_view[-1], np.zeros(x_min_range + self.sensor_range - x_max_range)], axis=0)
        # with open('landmarks_eye.txt', 'w') as f:
        #     for i in range(len(self.landmarks_eye[0])):
        #         # print(i)
        #         f.write(str(self.landmarks_eye[i]) + '\n')

        if mark_y_zero:
            # print(x_min_range,x_max_range)
            temp = np.zeros((self.sensor_range - y_max_range, self.sensor_range))
            landmarks_view = np.concatenate([temp, landmarks_view])
        elif mark_y_edge:
            temp = np.zeros((y_min_range + self.sensor_range - y_max_range, self.sensor_range))
            landmarks_view = np.concatenate([landmarks_view, temp])
        landmarks_view = np.concatenate([landmarks_view], axis=1)
        # with open('landmarks_eye.txt', 'w') as f:
        #     for i in range(len(landmarks_view[0])):
        #         # print(i)
        #         f.write(str(landmarks_view[i]) + '\n')
        # print(landmarks_view)
        # landmarks_view = pooling(landmarks_view, (3, 3))
        # with open('landmarks_eye.txt', 'a') as f:
        #     f.write(str(self.landmark_visited) + '\n')
        #     f.write(str(self.landmarks) + '\n')
        #     f.write(str(self.state_n[agent_id]) + '\n')
        #     for i in range(len(landmarks_view[0])):
        #         # print(i)
        #         f.write(str(landmarks_view[i]) + '\n')
        # print(landmarks_view)
        landmarks_view = np.concatenate([view for view in landmarks_view])
        landmarks_view = np.array(landmarks_view)
        if len( landmarks_view) != 49 and len( landmarks_view) !=  21**2:
            print('error')
            print( landmarks_view.shape)
        # print(len(landmarks_view))
        return landmarks_view

    def obs(self, agent_id):


        landmarks_view = np.concatenate([self.get_obs(agent) for agent in range(self.n_agent)])

        a_state = self.state_n[agent_id]
        # landmarks_position = np.array(self.landmarks[0])
        # for i in range(len(self.landmarks) - 1):
        #     landmarks_position = np.concatenate([landmarks_position, np.array(self.landmarks[i + 1])])
        # landmarks_position = np.concatenate([position for position in landmarks_position])
        #
        # targets_position = np.concatenate([position for position in self.target_info])

        ans_landmark_visited_times = np.array(self.landmark_visited_times[0])
        for i in range(len(self.landmarks) - 1):
            ans_landmark_visited_times = np.concatenate(
                [ans_landmark_visited_times, np.array(self.landmark_visited_times[i + 1])])
        ans = [landmarks_view, a_state]


        ans.append(ans_landmark_visited_times)
        ans.append(self.agent_searching_n)
        ans.append(self.agent_monitor_n)
            # ans.append(self.target_monitored_n)
        # for item in ans:
        #     print(len(item))

        ans = np.concatenate(ans).copy()

        return ans

    def get_adj_reset(self): # dim num_agent * num_landmarks + num_target for multipying
        adj = np.zeros((self.n_agent + self.n_landmark + self.n_target, self.n_agent + self.n_landmark + self.n_target))
        for a_i in range(self.n_agent):
            x = self.state_n[a_i][0]
            y = self.state_n[a_i][1]
            for i in range(self.n_landmark):
                row = i // self.n_landmark_in_length
                col = i % self.n_landmark_in_length
                x1= self.landmarks[row][col][0]
                y1 = self.landmarks[row][col][1]
                if (np.abs(x-x1)<=self.sensor_range//2) and (np.abs(y-y1)<=self.sensor_range//2):
                    adj[a_i][i]=1
            for t_i in range(self.n_target):
                x1 = self.target_n[t_i][0]
                y1 = self.target_n[t_i][1]
                if (np.abs(x-x1)<=self.sensor_range//2) and (np.abs(y-y1)<=self.sensor_range//2):
                    adj[a_i][self.n_landmark+t_i]=1
        return np.array(adj)

    def get_adj(self): # dim num_agent * num_landmarks + num_target for multipying
        adj = np.zeros((self.n_agent, self.n_agent))
        for a_i in range(self.n_agent):
            x = self.state_n[a_i][0]
            y = self.state_n[a_i][1]
            for i in range(self.n_agent):
                x1= self.state_n[i][0]
                y1 = self.state_n[i][1]
                if (np.abs(x-x1)<=self.sensor_range//2) and (np.abs(y-y1)<=self.sensor_range//2):
                    adj[a_i][i]=1
        return np.array(adj)

    def reward(self):

        rew = self.agent_score.copy()
        # rew = np.array([agent_score for _ in range(self.n_agent)]).astype(float)
        landmark_counter = 0

        if self.stage == 0:
            if self.landmark_visited.all() and self.target_detected_n.all():
                temp_rew = 0
                print('yes', self.t_step)
                for t_i in range(self.n_target):
                    if self.target_monitored_n[t_i] == True:
                        temp_rew = temp_rew + (3 - min(3, self.target_health[t_i])) * self.goal_reward
                for r_i in range(len(rew)):
                    rew[r_i] = self.goal_reward * self.episode_length // self.t_step + temp_rew
            # for agent_index in range(self.n_agent):
            #     if self.pre_state_n[agent_index].all() == self.state_n[agent_index].all():
            #         rew[agent_index] -= 1

            # for agent_index in range(self.n_agent):
            #     if self.state_n[agent_index][2] > self.n_landmark - 1:
            #         if self.target_info[self.state_n[agent_index][2] - self.n_landmark][0] in range(
            #                 self.map_size) and self.stage == 1:
            #             rew += 1
            #     else:
            #         if self.stage == 0:
                        # row = self.state_n[agent_index][2] // self.n_landmark_in_length
                        # col = self.state_n[agent_index][2] % self.n_landmark_in_length
                        # pre_row = self.pre_state_n[agent_index][2] // self.n_landmark_in_length
                        # pre_col = self.pre_state_n[agent_index][2] % self.n_landmark_in_length
                        # if self.state_n[agent_index][2] == self.pre_state_n[agent_index][2] and not \
                        # self.landmark_visited[row][col]:
                        #
                        #     if np.abs(self.landmarks[row][col][0] - self.state_n[agent_index][0]) + np.abs(
                        #             self.landmarks[row][col][1] - self.state_n[agent_index][1]) < \
                        #             np.abs(self.landmarks[row][col][0] - self.pre_state_n[agent_index][0]) + np.abs(
                        #         self.landmarks[row][col][1] - self.pre_state_n[agent_index][1]):
                        #         rew += 1
                        # elif self.state_n[agent_index][2] != self.pre_state_n[agent_index][2] and not \
                        # self.landmark_visited[pre_row][pre_col]:
                        #     rew -= 1
            for thelandmark in self.landmark_visited:
                for item_res in thelandmark:
                    if item_res:
                        landmark_counter += 1
            # print(landmark_counter)
            # rew += (landmark_counter - self.previous_landmark_counter)
            # rew += landmark_counter
            self.previous_landmark_counter = landmark_counter



        time_length = []
        for agent_index in range(self.n_agent):
            if self.agent_searching_n[agent_index]:
                time_length.append(1.)
            else:
                time_length.append(0.)

        info = {}
        info['landmark'] = self.landmarks
        info['monitor'] = self.agent_monitor_n
        info['time_length'] = time_length
        info['landmark_counter'] = landmark_counter

        return np.round(rew).astype(int), info

    def done(self):
        if self.t_step >= self.episode_length:
            print(self.landmark_visited)
            return 1
        if self.stage == 0:
            if self.landmark_visited.all() and self.target_detected_n.all():
                return 1
        return 0

    def close(self):
        self.reset()

    def render(self):
        # print(len(self.obs_n()[0]))
        # print(self.len_obs)
        print(self.state_n)
        # print(self.obs_n())
        print(self.landmark_visited)
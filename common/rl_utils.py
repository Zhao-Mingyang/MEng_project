import torch as th


def build_td_lambda_targets(rewards, terminated, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    # target_qs = target_qs.squeeze()
    ret = target_qs.new_zeros(*target_qs.shape)
    # test = th.sum(terminated, dim=1)
    # print(target_qs.size())
    # print(terminated.size(), ret.size())
    # terminated_sum = (1 - th.sum(terminated, dim=1))
    terminated_sum = (1 - terminated)
    # print(terminated_sum.size(),ret[:,-1].size(),target_qs[:,-1].size())
    # ret = target_qs * terminated_sum
    ret[:, -1] = target_qs[:, -1] * terminated_sum
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + \
                    (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
        # ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
        #             * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


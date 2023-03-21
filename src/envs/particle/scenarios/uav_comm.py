import numpy as np
from envs.particle.core import World, Agent, Landmark
from envs.particle.scenario import BaseScenario
import utils.uav_utils as utils
import math
import random


class Scenario(BaseScenario):
    height = 14
    env_a = 27.23  # 环境参数 a
    env_b = 0.08  # 环境参数 b
    # bandwidth = 200  # 带宽（单位MHz）
    frequency = 2000  # 载波频率（单位MHz）
    yita_l = 2.3  # 环境参数
    yita_n = 34  # 环境参数
    noise_power = 1e-13  # 噪声 -100 dBm


    # transmit_power = 100

    def make_world(self, args=None):
        world = World()
        world.user = np.loadtxt('/home/ykzhao/code/ykworkspace/facmac/users_location_20.txt')
        # world.dim_c = 2
        world.num_uav = 4
        world.num_rb = 5
        world.num_user = len(world.user)
        world.num_k = 1  # 与k个agent建立超边
        world.agents = [Agent() for i in range(world.num_uav)]  # 所有agent集合
        # world.power = np.random.normal(400, 5, world.num_uav)
        # world.user = []
        # for i in range(world.num_user):
        #    a = np.random.randint(-50, 50)
        #    b = np.random.randint(-50, 50)
        #    world.user.append([a, b])
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            # agent.silent = True
            agent.accel = 3.0  # if agent.adversary else 4.0 加速度
            agent.max_speed = 1.0  # if agent.adversary else 1.3 最大速度
            agent.action_callback = None  # if i < (num_agents - num_good_agents) else self.prey_policy  # action_callback？
            agent.view_radius = getattr(args, "agent_view_radius", -1)
            agent.dist_min = 0.001

        world.rb = np.zeros([world.num_uav, world.num_user, world.num_rb])
        for j in range(world.num_user):
            m = np.random.randint(0, world.num_uav)  # 每个用户仅关联一个UAV
            r = np.random.randint(0, world.num_rb)  # 每个用户仅由一个RB服务
            world.rb[m][j][r] = 1

            # print("AGENT VIEW RADIUS set to: {}".format(agent.view_radius))
        # print(world.user)
        self.reset_world(world)
        self.score_function = getattr(args, "score_function", "sum")
        return world

    # todo reset 智能体状态：位置、功率、RB分配
    def reset_world(self, world):
        for agent in world.agents:
            # agent.state.p_pos = np.array([0.4,0.4],dtype="float64")
            agent.state.p_pos = np.random.uniform(0.5, 0.5001, world.dim_p)  # 随机生成agent的位置
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.power = np.ones([world.num_user, world.num_rb])  # 生成agent的功率分配矩阵：UAV对各用户在各RB上的功率分配
            # agent.state.rb = agent.rb
            # for j in range(world.num_user):
            #    r = np.random.randint(0, self.num_rb)  # 每个用户仅由一个RB服务
            #    agent.state.rb[j][r] = 1
            # print(agent.state.rb)
            # agent.state.p_pos = pos[count]
            # count=count+1
            # agent.state.p_vel = np.zeros(world.dim_p)  # agent的速度初始化为0
            # agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #    if not landmark.boundary:
        #        landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
        #        landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        '''碰撞判断'''
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # dist_min = agent1.size + agent2.size
        return True if dist < agent1.dist_min else False

    def get_path_loss(self, world):
        # 计算distance
        user = world.user
        xiang1 = np.zeros((world.num_uav, world.num_user))
        xiang2 = np.zeros((world.num_uav, world.num_user))
        xiang3 = np.zeros((world.num_uav, 1))
        distance = np.zeros((world.num_uav, world.num_user))
        i = -1
        for agent in world.agents:
            i += 1
            for j in range(world.num_user):
                xiang1[i][j] = math.pow((agent.state.p_pos[0] * 1000 - user[j][0]), 2)
                xiang2[i][j] = math.pow((agent.state.p_pos[1] * 1000 - user[j][1]), 2)
                xiang3[i] = math.pow(self.height, 2)
                distance[i][j] = math.sqrt(xiang1[i][j] + xiang2[i][j] + xiang3[i])

        # 计算pro_los
        theta = np.zeros((world.num_uav, world.num_user))
        pro_los = np.zeros((world.num_uav, world.num_user))
        for i in range(world.num_uav):
            for j in range(world.num_user):
                theta[i][j] = math.asin(
                    self.height / distance[i][j]) * 180 / math.pi
                pro_los[i][j] = 1 / (1 + self.env_a * math.exp(-self.env_b * (theta[i][j] - self.env_a)))

        # 计算path_loss
        free_space = np.zeros((world.num_uav, world.num_user))
        free_space_xiang = 20 * math.log10(self.frequency) + 32.44  # 单位为dB
        pl_los = np.zeros((world.num_uav, world.num_user))
        pl_nlos = np.zeros((world.num_uav, world.num_user))
        path_loss = np.zeros((world.num_uav, world.num_user))
        r = np.random.rand(world.num_uav, world.num_user)
        for i in range(world.num_uav):
            for j in range(world.num_user):
                free_space[i][j] = 20 * math.log10(distance[i][j] / 1000) + free_space_xiang  # 因为d的单位是km，路损单位为dB
                pl_los[i][j] = free_space[i][j] + self.yita_l  # 单位为dB
                pl_los[i][j] = math.pow(10, pl_los[i][j] / 10)  # 转化为W
                pl_nlos[i][j] = free_space[i][j] + self.yita_n  # 单位为dB
                pl_nlos[i][j] = math.pow(10, pl_nlos[i][j] / 10)  # 转化为W
                if r[i][j] < pro_los[i][j]:
                    path_loss[i][j] = 1 / pl_los[i][j]
                else:
                    path_loss[i][j] = 1 / pl_nlos[i][j]
                # path_loss[i][j] = 1 / (pl_los[i][j] * pro_los[i][j] + pl_nlos[i][j] * (1 - pro_los[i][j]))
        return path_loss

    # 计算reward: data rate
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        channel_gain = self.get_path_loss(world)  # 计算UAV与所有用户间的pathloss
        # print(channel_gain)
        power = []
        sigma = world.rb
        for agent in world.agents:
            power.append(agent.state.power)
        power = power * sigma

        # print(power)
        # print(sigma)

        # received_power = np.zeros([world.num_uav, world.num_user, self.num_rb])
        # for i in range(world.num_uav):
        #    for j in range(world.num_user):
        #        for k in range(self.num_rb):
        #            received_power[i][j][k] = power[i][j][k] * channel_gain[i][j]
        # print(received_power)

        # 计算各UAV在RB k上的功率
        power_rb = np.zeros([world.num_uav, world.num_rb])
        for k in range(world.num_rb):
            for i in range(world.num_uav):
                tmp = 0
                for j in range(world.num_user):
                    tmp += power[i][j][k]
                power_rb[i][k] = tmp
        # print(power_rb)

        # 计算簇外干扰：来自在第k个RB上发送信号的其他空中基站
        inter_interference = np.zeros([world.num_uav, world.num_user, world.num_rb])
        for k in range(world.num_rb):
            for j in range(world.num_user):
                for i in range(world.num_uav):
                    tmp = 0
                    for m in range(world.num_uav):
                        if i == m:
                            continue
                        else:
                            tmp += power_rb[i][k] * channel_gain[m][j]
                    inter_interference[i][j][k] = tmp
        # print(intra_interference)

        # 计算簇内干扰：为在小区i内使用同样RB的NOMA通信链路具有SIC无法消除的较高干扰
        intra_interference = np.zeros([world.num_uav, world.num_user, world.num_rb])
        for k in range(world.num_rb):
            for i in range(world.num_uav):
                for j in range(world.num_user):
                    tmp = 0
                    for m in range(world.num_user):
                        if channel_gain[i][m] > channel_gain[i][j]:
                            tmp += power[i][m][k] * channel_gain[i][j]
                    intra_interference[i][j][k] = tmp
        # print(inter_interference)

        rate = np.zeros([world.num_uav, world.num_user, world.num_rb])
        for k in range(world.num_rb):
            for i in range(world.num_uav):
                for j in range(world.num_user):
                    # print(sigma[i][j][k] * power[i][j][k] * channel_gain[i][j])
                    # print(intra_interference[i][j][k])
                    # print(inter_interference[i][j][k])
                    rate[i][j][k] = sigma[i][j][k] * power[i][j][k] * channel_gain[i][j] / (
                            self.noise_power + intra_interference[i][j][k] + inter_interference[i][j][k])

        rew = np.sum(rate)
        # print(capability / self.num_users)
        # print(rew)

        # 碰撞约束
        if agent.collide:
            for other in world.agents:
                if other is agent: continue
                if self.is_collision(agent, other):
                    rew -= 10

        return rew

    def observation(self, agent, world):
        # comm = []
        other_pos = []\
        # other_vel = []
        for other in world.agents:
            if other is agent: continue
            dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            # 观测范围约束
            if agent.view_radius >= 0 and dist <= agent.view_radius:
                # comm.append(other.state.c)
                other_pos.append(other.state.p_pos)  # 距离差
                # if not other.adversary:
                # other_vel.append(other.state.p_vel)
            else:
                other_pos.append(np.array([0., 0.]))
                # if not other.adversary:
                # other_vel.append(np.array([0., 0.]))
        # print("part obs: ", np.concatenate([agent.state.p_pos] + other_pos))
        return np.concatenate([agent.state.p_pos] + other_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def full_observation(self, agent, world):
        # comm = []
        other_pos = []
        # other_vel = []
        for other in world.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append(other.state.p_pos)
            # if not other.adversary:
            # other_vel.append(other.state.p_vel)
        # print("full obs: ", np.concatenate([agent.state.p_pos] + other_pos))
        return np.concatenate([agent.state.p_pos] + other_pos)

    def adj_kmax(self, agent, world):
        '''输出adjacent matrix (与干扰最大的K个智能体相邻)
               adjacent= N × 1
               feature= K × 1
               '''
        adj = np.zeros(world.num_uav)
        other_intra = []  # 小区内干扰
        other_interference = []  # 小区外干扰
        all_other_interference = []
        for other in world.agents:
            if other is agent:
                all_other_interference.append(np.array([0.]))
            else:
                dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
                if agent.view_radius >= 0 and dist <= agent.view_radius:  # 观测范围约束
                    # comm.append(other.state.c)
                    all_other_interference.append(other.state.p_pos[0] - agent.state.p_pos[0])  # todo 根据干扰关系得到超图关联矩阵
                    # if not other.adversary:
                else:
                    all_other_interference.append(np.array([0.]))
        # print(other_interference)
        # 选出K个interference最大的other的索引
        a = np.array(all_other_interference)
        a.argsort()
        b = a.argsort()[-world.num_k:]
        for index in b:
            adj[index] = 1
            # view_other = world.agents[index]
            # view_other_interference = view_other.state.p_pos - agent.state.p_pos  # todo 计算other对当前agent的干扰（簇外干扰）
            # other_interference.append(view_other_interference)
        # todo 计算当前agent的簇内干扰
        # feature = np.concatenate(other_interference + other_intra)
        # max_data = max(other_interference)
        # max_index=other_interference.index(max_data)
        # adj[max_index]=1
        # return np.concatenate([agent.state.p_pos] + other_pos)
        return adj

    def adj(self, agent, world):

        '''输出adjacent matrix (与最近的K个智能体相邻)
        adjacent= N × 1
        feature= K × 1
        '''
        adj = np.zeros(world.num_uav)
        other_dis = []
        # other_vel = []
        for other in world.agents:
            dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            other_dis.append(dist)
        # print(other_dis)
        a = np.array(other_dis)
        # print("a: ", a)
        a.argsort()  # 将距离从小到大排列 返回索引
        # print(a.argsort())
        # k = 2  # 前k个
        b1 = a.argsort()[0]  # 返回0的索引(自己)
        # print(b1)
        adj[b1] = 1
        b2 = a.argsort()[-world.num_k:]  # 返回最大K个的索引
        # print(b2)
        for index in b2:
            adj[index] = 1
        # print("adj: ", adj)
        return adj

    def feature(self, agent, world):
        '''输出feature matrix
        adjacent= N × 1
        feature= 2N × 1
        '''
        other_pos = []
        adj = self.adj(agent, world)
        # print(adj)
        for index in range(len(adj)):
            if adj[index] == 0:  # 只观测邻接agent
                other_pos.append(np.array([0., 0.]))
            else:
                view_other = world.agents[index]
                view_other_pos = view_other.state.p_pos  # todo 计算other对当前agent的干扰（簇外干扰），目前保存的是位置
                other_pos.append(view_other_pos)

        # print("feature: ", np.concatenate([agent.state.p_pos] + other_pos))
        # print(np.concatenate(other_pos))

        return np.concatenate(other_pos)

    # def feature_old(self, agent, world):
    #    '''输出feature matrix
    #    adjacent= N × 1
    #    feature= K × 1
    #    '''
    #    other_interference = []
    #    adj = self.adj(agent, world)
    #    for index in range(len(adj)):
    #        if adj[index] == 0:
    #            continue
    #        else:
    #            view_other = world.agents[index]
    #            view_other_interference = view_other.state.p_pos - agent.state.p_pos
    #            other_interference.append(view_other_interference)
    # 计算当前agent的簇内干扰
    #    other_intra = []
    # feature维度不对
    #    feature = np.concatenate(other_interference + other_intra)
    #    return feature

import numpy as np
from envs.particle.core import World, Agent, Landmark
from envs.particle.scenario import BaseScenario
import utils.uav_utils as utils
import math
import random


class Scenario(BaseScenario):
    def make_world(self, args=None):
        world = World()
        world.user = np.loadtxt('/home/ykzhao/code/ykworkspace/facmac/users_location.txt')
        # world.dim_c = 2
        world.num_k = 2  # 与k个agent建立超边
        world.agents = [Agent() for i in range(world.num_uav)]  # 所有agent集合
        world.noise = getattr(args, "noise", -1)
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            # agent.silent = True
            # agent.accel = 3.0  # if agent.adversary else 4.0 加速度
            # agent.max_speed = 1.0  # if agent.adversary else 1.3 最大速度
            agent.action_callback = None  # if i < (num_agents - num_good_agents) else self.prey_policy  # action_callback？
            agent.view_radius = getattr(args, "agent_view_radius", -1)
            agent.dist_min = 0.001
            print("AGENT VIEW RADIUS set to: {}".format(agent.view_radius))
        self.reset_world(world)
        self.score_function = getattr(args, "score_function", "sum")
        return world

    # todo reset 智能体状态：位置、功率、RB分配
    def reset_world(self, world):
        for agent in world.agents:
            # agent.state.p_pos = np.array([0.4,0.4],dtype="float64")
            agent.state.p_pos = np.random.uniform(0.5, 0.5001, world.dim_p)  # 随机生成agent的位置
            agent.state.power = np.ones(world.num_user)  # 生成agent的功率分配矩阵：UAV对所有用户的功率分配
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

    # todo 计算reward: data rate 需要考虑RB（目前是对某个RB k来说的data rate）
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        uav_pos = agent.state.p_pos * 1000
        #print(uav_pos)
        other_pos = []
        other_power = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos * 1000)
            other_power.append(other.state.power)
        # print(other_pos)
        # print(other_power)
        other_pos = np.concatenate(other_pos, axis=0)
        other_power = np.concatenate(other_power, axis=0)
        other_power_RB = other_power  # todo 不对
        # print(other_pos)
        # print(other_power)
        #    inter_interference = utils.get_inter_interference(other.state.p_pos, world.user, other_power_RB)
        #    intra_interference = utils.get_intra_interference(agent.state.p_pos, world.user, agent.state.power)
        channel_gain = []
        desired_signal = []
        for j in range(len(world.user)):
            h = utils.get_channel_gain(uav_pos, world.user[j])
            x = utils.get_desired_signal(uav_pos, world.user[j], agent.state.power[j])
            channel_gain.append(h)
            desired_signal.append(x)

        intra_interference = []
        for j in range(len(world.user)):
            for j_ in range(len(world.user)):
                if channel_gain[j_] > channel_gain[j]:
                    interference = agent.state.power[j_] * channel_gain[j]
                else:
                    interference = 0
            intra_interference.append(interference)

        inter_interference = np.zeros([len(other_pos), len(world.user)])
        for i_ in range(len(other_pos)):
            for j in range(len(world.user)):
                inter_interference[i_][j] = other_power_RB[i_] * channel_gain[j]

        sum_inter_interference = sum(sum(inter_interference))
        sum_intra_interference = sum(intra_interference)

        # sinr for uav i to user j via RB k
        sinr = []
        rate = []
        for j in range(len(world.user)):
            gamma = desired_signal[j] / (world.noise + sum_inter_interference + sum_intra_interference)
            r = math.log(1 + gamma)
            sinr.append(gamma)
            rate.append(r)
        # print(sinr)
        # print(rate)
        rew = sum(rate)
        #print(rew)

        # 碰撞约束
        if agent.collide:
            for other in world.agents:
                if other is agent: continue
                if self.is_collision(agent, other):
                    rew -= 10

        return rew

    # todo 修改observation agent添加视野范围内干扰最大的K的other的信息
    def observation(self, agent, world):
        # comm = []
        other_pos = []
        # other_vel = []
        for other in world.agents:
            if other is agent: continue
            dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            # 观测范围约束
            if agent.view_radius >= 0 and dist <= agent.view_radius:
                # comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)  # 距离差
                # if not other.adversary:
                # other_vel.append(other.state.p_vel)
            else:
                other_pos.append(np.array([0., 0.]))
                # if not other.adversary:
                # other_vel.append(np.array([0., 0.]))
        return np.concatenate([agent.state.p_pos] + other_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    # todo (没有部分可观约束)
    def full_observation(self, agent, world):
        # comm = []
        other_pos = []
        # other_vel = []
        for other in world.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary:
            # other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_pos] + other_pos)

    def adj(self, agent, world):
        '''输出adjacent matrix和feature matrix
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
        k = 3  # 前k个
        b = a.argsort()[-k:]
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

    def feature(self, agent, world):
        '''输出adjacent matrix和feature matrix
        adjacent= N × 1
        feature= K × 1
        '''
        other_interference = []
        adj = self.adj(agent, world)
        # print(adj)
        for index in range(len(adj)):
            if adj[index] == 0:
                continue
            else:
                view_other = world.agents[index]
                view_other_interference = view_other.state.p_pos - agent.state.p_pos  # todo 计算other对当前agent的干扰（簇外干扰）
                other_interference.append(view_other_interference)
        # todo 计算当前agent的簇内干扰
        other_intra = []
        # todo feature维度不对
        feature = np.concatenate(other_interference + other_intra)
        # print(adj)
        # print(feature)
        # max_data = max(other_interference)
        # max_index=other_interference.index(max_data)
        # adj[max_index]=1
        # return np.concatenate([agent.state.p_pos] + other_pos)
        return feature

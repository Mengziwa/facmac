import numpy as np
from envs.particle.core import World, Agent, Landmark
from envs.particle.scenario import BaseScenario
import math
import random


class Scenario(BaseScenario):
    def make_world(self, args=None):
        world = World()
        world.user = np.loadtxt('/home/ykzhao/code/ykworkspace/facmac/users_location.txt')
        # set any world properties first
        world.dim_c = 2
        # num_good_agents = 0
        # num_adversaries = 3
        world.num_agents = 6  # agent总数
        k = 2  # 与k个agent建立超边
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]  # 所有agent集合
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            # agent.adversary = True if i < num_adversaries else False
            # agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0  # if agent.adversary else 4.0 加速度
            agent.max_speed = 1.0  # if agent.adversary else 1.3 最大速度
            agent.action_callback = None  # if i < (num_agents - num_good_agents) else self.prey_policy  # action_callback？
            agent.view_radius = getattr(args, "agent_view_radius", -1)
            agent.dist_min = 0.001
            print("AGENT VIEW RADIUS set to: {}".format(agent.view_radius))
        # add landmarks
        # world.landmarks = [Landmark() for i in range(num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #    landmark.name = 'landmark %d' % i
        #    landmark.collide = True
        #    landmark.movable = False
        #    landmark.size = 0.001
        #    landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        self.score_function = getattr(args, "score_function", "sum")
        return world

    def reset_world(self, world):
        # random properties for agents
        # for i, agent in enumerate(world.agents):
        #    agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #    landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        # pos=np.array([[-0.6,-0.6],
        # [-0.6,0.6],
        # [0.6,-0.6],
        # [0.6,0.6]],dtype="float64"
        # )
        # count=0
        for agent in world.agents:
            # agent.state.p_pos = np.array([0.4,0.4],dtype="float64")
            agent.state.p_pos = np.random.uniform(0.5, 0.5001, world.dim_p)  # 随机生成agent的位置
            # agent.state.p_pos = pos[count]
            # count=count+1  
            agent.state.p_vel = np.zeros(world.dim_p)  # agent的速度初始化为0
            # agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #    if not landmark.boundary:
        #        landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
        #        landmark.state.p_vel = np.zeros(world.dim_p)

    # def benchmark_data(self, agent, world):
    #    # returns data for benchmarking purposes
    #    if agent.adversary:
    #        collisions = 0
    #        for a in self.good_agents(world):
    #            if self.is_collision(a, agent):
    #                collisions += 1
    #        return collisions
    #    else:
    #        return 0

    def is_collision(self, agent1, agent2):
        '''碰撞判断'''
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # dist_min = agent1.size + agent2.size
        return True if dist < agent1.dist_min else False

    # return all agents that are not adversaries
    # def good_agents(self, world):
    #    return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    # def adversaries(self, world):
    #    return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        s = agent.state.p_pos * 1000
        QoS_current = 0
        SINR = 0
        S = 0
        reward = 0
        user = world.user
        Nu = 10 ** (-11)
        for i in range(len(user)):
            l = ((s[0] - user[i][0]) ** 2 + (s[1] - user[i][1]) ** 2) ** 0.5
            if (l < 1000):
                d = (20 ** 2 + l ** 2) ** 0.5
                if l == 0:
                    angle = 90
                else:
                    angle = math.atan(20 / l) * 180 / math.pi
                P_LOS = 1 / (1 + 11.95 * math.exp(-0.136 * (angle - 11.95)))
                P_NLOS = 1 - P_LOS
                S_LOS = 180 * 10 ** (-4.11 - 2.09 * math.log10(d))
                S_NLOS = 180 * 10 ** (-3.3 - 3.75 * math.log10(d))
                SINR_LOS = S_LOS / Nu
                SINR_NLOS = S_NLOS / Nu
                SINR += P_LOS * 10 * math.log10(SINR_LOS) + P_NLOS * 10 * math.log10(SINR_NLOS)
                S += P_LOS * 10 * math.log10(S_LOS) + P_NLOS * 10 * math.log10(S_NLOS)
                QoS_current += P_LOS * math.log2(1 + SINR_LOS) + P_NLOS * math.log2(1 + SINR_NLOS)
            # print('QoS_current',P_LOS * math.log2(1 + SINR_LOS) + P_NLOS * math.log2(1 + SINR_NLOS))

        # l = ((s[0] - user[i][0]) ** 2 + (s[1] - user[i][1]) ** 2) ** 0.5
        # print('l',l)
        # if(l<500):
        #     reward=10
        # if(l<100):
        #     reward=50
        return QoS_current

    # todo 修改observation agent添加视野范围内干扰最大的K的other的信息
    def observation(self, agent, world):
        # comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if agent.view_radius >= 0 and dist <= agent.view_radius:  # 观测范围约束
                # comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)  # 距离差
                # if not other.adversary:
                other_vel.append(other.state.p_vel)
            else:
                other_pos.append(np.array([0., 0.]))
                # if not other.adversary:
                other_vel.append(np.array([0., 0.]))
        return np.concatenate([agent.state.p_pos] + other_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    # todo (没有部分可观约束)
    def full_observation(self, agent, world):
        # comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary:
            other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_pos] + other_pos)

    def adj(self, agent, world):
        '''输出adjacent matrix和feature matrix
        adjacent= N × 1
        feature= K × 1
        '''
        adj = np.zeros(world.num_agents)
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
        #print(adj)
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
        #print(adj)
        #print(feature)
        # max_data = max(other_interference)
        # max_index=other_interference.index(max_data)
        # adj[max_index]=1
        # return np.concatenate([agent.state.p_pos] + other_pos)
        return feature

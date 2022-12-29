# --------------------------------------------------------
# Utility functions for UAV environment

# --------------------------------------------------------
import numpy as np
import math as math


def get_dis(uav_loc, user_loc):
    """
    Calculate the distance between the current UAV and all users
    :param uav_loc: 2 X 1; UAV水平位置
    :param user_loc: 2 X 1; 用户位置矩阵
    :return: distance
    """
    dis = np.linalg.norm(uav_loc - user_loc)
    return dis


def get_channel_gain(uav_loc, user_loc):
    '''
    Calculate the channel gain between the current UAV and all users
    :return: 1 X N channel gain matrix
    '''
    dis = get_dis(uav_loc, user_loc)
    # 参考增益
    rho = 9.999999999999982e-9
    h = rho * (dis ** (-2))
    return h


def get_desired_signal(uav_loc, user_loc, power_user):
    '''
    Calculate the desired signal from the current UAV to all users via given RB k
    :param power_user: 1 X N; 当前无人机对所有用户的发射功率
    :return: 1 X N desired signal matrix
    '''
    channel_gain = get_channel_gain(uav_loc, user_loc)
    x = power_user * channel_gain
    return x


# ___________________________________________________________________________________________

def get_dis_2(uav_loc, user_loc):
    """
    Calculate the distance between the current UAV and all users
    :param uav_loc: 2 X 1; UAV水平位置
    :param uav_h: scalar; UAV高度
    :param user_loc: 2 X N; 用户位置矩阵
                N: the user number
    :return: 1 X N distance matrix
    """
    dis_mat = []
    for i in range(len(user_loc)):
        dis = np.linalg.norm(uav_loc - user_loc[i])
        # dis = np.sqrt(uav_h ** 2 + dis_2d ** 2)
        dis_mat.append(dis)
    return dis_mat


def get_channel_gain_2(uav_loc, user_loc):
    '''
    Calculate the channel gain between the current UAV and all users
    :return: 1 X N channel gain matrix
    '''
    dis = get_dis_2(uav_loc, user_loc)
    # todo 参考增益
    rho = 1
    channel_gain = []
    for i in range(len(user_loc)):
        h = rho * (dis[i] ** (-2))
        channel_gain.append(h)
    return channel_gain


def get_desired_signal_2(uav_loc, user_loc, power_user):
    '''
    Calculate the desired signal from the current UAV to all users via given RB k
    :param power_user: 1 X N; 当前无人机对所有用户的发射功率
    :return: 1 X N desired signal matrix
    '''
    desired_signal = []
    channel_gain = get_channel_gain_2(uav_loc, user_loc)
    for i in range(len(user_loc)):
        x = power_user[i] * channel_gain[i]
        desired_signal.append(x)
    return desired_signal


def get_intra_interference(uav_loc, user_loc, power_user):
    '''
    Calculate the intra-interference from the current UAV to its users via given RB k
    小区内干扰: 在小区i内使用同样RB的NOMA通信链路具有SIC无法消除的较高干扰;
    NOMA 接收用户在解调自身信号时，已经检测并去除了比自身功率高的其他 NOMA 用户的信号，并将比自身功率低的 NOMA 用户信号视为干扰信号。
    :param power_user: 1 X N; 当前无人机对所有用户的发射功率
    :return: 1 X N intra_interference matrix
    '''
    intra_interference = []
    channel_gain = get_channel_gain(uav_loc, user_loc)
    for j in range(len(user_loc)):
        for j_ in range(len(user_loc)):
            if channel_gain[j_] > channel_gain[j]:
                interference = power_user[j_] * channel_gain[j]
            else:
                interference = 0
        intra_interference.append(interference)

    return intra_interference


def get_inter_interference(other_loc, user_loc, other_power_RB):
    '''
    Calculate the inter-interference from all other UAV to all users via given RB k
    小区外干扰:在第k个RB上发送信号的其他空中基站引起
    :param power_RB: 1 X N; 其他空中基站在RB k上的发射功率。
    :return: 1 X N inter_interference matrix
    '''
    inter_interference = np.zeros([len(other_loc), len(user_loc)])
    for i_ in range(len(other_loc)):
        for j in range(len(user_loc)):
            channel_gain = get_channel_gain(other_loc[i_], user_loc)
            inter_interference[i_][j] = other_power_RB[i_] * channel_gain[j]
    return inter_interference


def get_sinr(uav_loc, power_user, other_loc, user_loc, other_power_RB):
    '''
    Calculate the sinr from UAV i to all users via given RB k
    小区外干扰:在第k个RB上发送信号的其他空中基站引起
    :param power_RB: 1 X N; 其他空中基站在RB k上的发射功率。
    :return: 1 X N inter_interference matrix
    '''
    inter_interference = get_inter_interference(other_loc, user_loc, other_power_RB)
    sum_inter_interference = sum(sum(inter_interference))
    intra_interference = get_intra_interference(uav_loc, user_loc, power_user)
    sum_intra_interference = sum(intra_interference)
    desired_signal = get_desired_signal(uav_loc, user_loc, power_user)
    # sinr for uav i to user j via RB k
    sinr = []
    noise = 1  # todo 噪声args
    for j in range(len(user_loc)):
        gamma = desired_signal[j] / (noise + sum_inter_interference + sum_intra_interference)
        sinr.append(gamma)
    return sinr


def get_rate(uav_loc, power_user, other_loc, user_loc, other_power_RB):
    '''
    Calculate the inter-interference from all other UAV to all users via given RB k
    小区外干扰:在第k个RB上发送信号的其他空中基站引起
    :param power_RB: 1 X N; 其他空中基站在RB k上的发射功率。
    :return: 1 X N inter_interference matrix
    '''
    rate = []
    sinr = get_sinr(uav_loc, power_user, other_loc, user_loc, other_power_RB)
    for j in range(len(user_loc)):
        r = math.log(1 + sinr[j])
        rate.append(r)
    return rate

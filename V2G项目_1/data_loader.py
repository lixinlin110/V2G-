"""
IEEE 33节点配电网数据读取脚本
用于V2G双向互动调控平台

使用方法:
    from data_loader import load_ieee33_data
    nodes, branches, ev_profile = load_ieee33_data()
"""

import pandas as pd
import os

def load_ieee33_data():
    """加载IEEE 33节点配电网数据"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 加载节点数据
    nodes = pd.read_csv(os.path.join(base_path, 'ieee33_nodes.csv'))
    
    # 加载支路数据
    branches = pd.read_csv(os.path.join(base_path, 'ieee33_branches.csv'))
    
    # 加载电动汽车充电负荷数据
    ev_profile = pd.read_csv(os.path.join(base_path, 'ev_charging_profile.csv'))
    
    return nodes, branches, ev_profile


def get_branch_admittance(branches):
    """
    计算支路导纳矩阵
    返回: dict {支路编号: (G, B)}  G=电导, B=电纳
    """
    admittance = {}
    for _, row in branches.iterrows():
        r = row['电阻R(标幺)']
        x = row['电抗X(标幺)']
        y = 1 / complex(r, x)  # 导纳 Y = 1/(R + jX)
        admittance[row['支路编号']] = (y.real, y.imag)
    return admittance


def get_load_profile(nodes):
    """
    获取节点负荷数据
    返回: list [(节点编号, P_kW, Q_kVar), ...]
    """
    loads = []
    for _, row in nodes.iterrows():
        if row['类型'] == '负荷节点':
            loads.append((row['节点编号'], row['有功负荷P(kW)'], row['无功负荷Q(kVar)']))
    return loads


def get_ev_scheduling_data(ev_profile):
    """
    获取电动汽车调度数据
    返回24小时充放电计划
    """
    return ev_profile


if __name__ == '__main__':
    # 测试数据加载
    nodes, branches, ev_profile = load_ieee33_data()
    
    print("=== IEEE 33节点配电网数据 ===")
    print(f"\n节点数量: {len(nodes)}")
    print(f"支路数量: {len(branches)}")
    print(f"\n节点数据前5行:\n{nodes.head()}")
    print(f"\n支路数据前5行:\n{branches.head()}")
    print(f"\n电动汽车充电负荷数据:\n{ev_profile}")

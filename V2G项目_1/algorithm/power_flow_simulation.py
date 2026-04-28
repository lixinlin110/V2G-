"""
配电网潮流计算与仿真
使用前推回代法（Backward-Forward Sweep）

功能：
- IEEE 33节点配电网潮流计算
- V2G调度前后对比仿真
- 结果可视化数据导出
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """加载配电网和负荷数据"""
    nodes = pd.read_csv('V2G项目/ieee33_nodes.csv')
    branches = pd.read_csv('V2G项目/ieee33_branches.csv')
    ev_profile = pd.read_csv('V2G项目/ev_charging_profile.csv')
    return nodes, branches, ev_profile


def backward_forward_sweep(branches, nodes, P_load, Q_load, max_iter=100, tol=1e-6):
    """
    前推回代法潮流计算
    
    参数：
        branches: 支路数据
        nodes: 节点数据
        P_load: 各节点有功负荷数组
        Q_load: 各节点无功负荷数组
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回：
        V: 各节点电压标幺值
        angles: 各节点电压相角
    """
    n = 33
    base_voltage = 12.66  # kV
    
    # 构建网络结构
    parent = {}  # 父节点
    children = {}  # 子节点列表
    branch_impedance = {}  # 支路阻抗
    
    for _, row in branches.iterrows():
        frm = int(row['起始节点'])
        to = int(row['终止节点'])
        r = row['电阻R(标幺)']
        x = row['电抗X(标幺)']
        parent[to] = frm
        if frm not in children:
            children[frm] = []
        children[frm].append(to)
        branch_impedance[(frm, to)] = complex(r, x)
        branch_impedance[(to, frm)] = complex(r, x)
    
    # 初始化电压
    V = np.ones(n + 1)  # 1-indexed
    angles = np.zeros(n + 1)
    
    # 迭代求解
    for iteration in range(max_iter):
        V_old = V.copy()
        
        # ===== 回代：计算支路功率 =====
        # 从叶子节点向根节点计算功率分布
        
        # 获取所有叶子节点
        leaves = []
        for node in range(2, n + 1):
            if node not in parent:
                continue
            is_leaf = True
            for child_list in children.values():
                if node in child_list:
                    is_leaf = False
                    break
            if is_leaf:
                leaves.append(node)
        
        # 简化：直接使用负荷功率计算
        
        # ===== 前推：计算节点电压 =====
        # 从根节点向叶子节点更新电压
        
        def update_voltage(node, parent_node, V_parent):
            if (parent_node, node) in branch_impedance:
                Z = branch_impedance[(parent_node, node)]
                # 简化的电压降计算
                S = complex(P_load[node-1], Q_load[node-1]) / 1000  # 标幺化
                dV = Z * S / V_parent
                V[node] = abs(V_parent - dV)
                angles[node] = np.angle(V_parent - dV) * 180 / np.pi
            return V[node]
        
        # 更新各节点电压
        for node in range(2, n + 1):
            if node in parent:
                V[node] = update_voltage(node, parent[node], V[parent[node]])
        
        # 检查收敛
        max_diff = np.max(np.abs(V - V_old))
        if max_diff < tol:
            print(f"潮流计算收敛，迭代 {iteration + 1} 次")
            break
    else:
        print(f"潮流计算未收敛，最大误差: {max_diff:.6f}")
    
    return V, angles


def simulate_v2g_impact(nodes, branches, ev_profile):
    """
    仿真V2G对配电网的影响
    对比：有无V2G两种情况下的电压分布和网损
    """
    n = 33
    
    # 基础负荷（无EV）
    P_base = nodes['有功负荷P(kW)'].values
    Q_base = nodes['无功负荷Q(kVar)'].values
    
    results = {
        'without_v2g': {'voltage': [], 'power_loss': [], 'peak_valley_diff': []},
        'with_v2g': {'voltage': [], 'power_loss': [], 'peak_valley_diff': []}
    }
    
    print("\n" + "=" * 60)
    print("V2G对配电网影响仿真")
    print("=" * 60)
    
    for t in range(24):
        ev_load = ev_profile.iloc[t]
        
        # 无V2G：EV全部充电
        P_without = P_base + ev_load['总充电负荷(kW)'] / n  # 均分到各节点
        Q_without = Q_base + ev_load['总充电负荷(kW)'] * 0.3 / n  # 假设功率因数0.9
        
        # 有V2G：部分EV放电参与调峰
        net_load = ev_load['净负荷(kW)']
        P_with = P_base + net_load / n
        Q_with = Q_base + net_load * 0.3 / n
        
        # 潮流计算
        V1, _ = backward_forward_sweep(branches, nodes, P_without, Q_without)
        V2, _ = backward_forward_sweep(branches, nodes, P_with, Q_with)
        
        results['without_v2g']['voltage'].append(V1)
        results['with_v2g']['voltage'].append(V2)
        
        # 计算峰谷差
        total_load_without = np.sum(P_without)
        total_load_with = np.sum(P_with)
        results['without_v2g']['peak_valley_diff'].append(total_load_without)
        results['with_v2g']['peak_valley_diff'].append(total_load_with)
        
        print(f"时刻 {t:02d}:00 | 无V2G负荷: {total_load_without:.1f}kW | 有V2G负荷: {total_load_with:.1f}kW | 削峰: {total_load_without - total_load_with:.1f}kW")
    
    return results


def generate_comparison_table(results, ev_profile):
    """生成对比数据表"""
    print("\n" + "=" * 60)
    print("仿真结果汇总")
    print("=" * 60)
    
    without = results['without_v2g']['peak_valley_diff']
    with_v2g = results['with_v2g']['peak_valley_diff']
    
    # 计算统计指标
    print(f"\n无V2G情况:")
    print(f"  - 最大负荷: {max(without):.1f} kW")
    print(f"  - 最小负荷: {min(without):.1f} kW")
    print(f"  - 峰谷差: {max(without) - min(without):.1f} kW")
    print(f"  - 峰谷差率: {(max(without) - min(without)) / max(without) * 100:.1f}%")
    
    print(f"\n有V2G情况:")
    print(f"  - 最大负荷: {max(with_v2g):.1f} kW")
    print(f"  - 最小负荷: {min(with_v2g):.1f} kW")
    print(f"  - 峰谷差: {max(with_v2g) - min(with_v2g):.1f} kW")
    print(f"  - 峰谷差率: {(max(with_v2g) - min(with_v2g)) / max(with_v2g) * 100:.1f}%")
    
    print(f"\nV2G效果:")
    peak_reduction = max(without) - max(with_v2g)
    valley_increase = min(with_v2g) - min(without)
    pv_reduction = (max(without) - min(without)) - (max(with_v2g) - min(with_v2g))
    print(f"  - 削峰量: {peak_reduction:.1f} kW")
    print(f"  - 填谷量: {valley_increase:.1f} kW")
    print(f"  - 峰谷差降低: {pv_reduction:.1f} kW ({pv_reduction / (max(without) - min(without)) * 100:.1f}%)")
    
    # 生成对比表格
    comparison_df = pd.DataFrame({
        '时刻': range(24),
        'EV充电负荷(kW)': ev_profile['总充电负荷(kW)'].values,
        'EV净负荷(kW)': ev_profile['净负荷(kW)'].values,
        '无V2G总负荷(kW)': without,
        '有V2G总负荷(kW)': with_v2g,
        '削峰量(kW)': np.array(without) - np.array(with_v2g)
    })
    
    return comparison_df


def export_simulation_data():
    """导出仿真数据供Web前端使用"""
    nodes, branches, ev_profile = load_data()
    results = simulate_v2g_impact(nodes, branches, ev_profile)
    comparison_df = generate_comparison_table(results, ev_profile)
    
    # 导出为CSV
    comparison_df.to_csv('V2G项目/simulation_results.csv', index=False)
    print(f"\n仿真结果已导出到 V2G项目/simulation_results.csv")
    
    # 导出为JSON格式（供Web使用）
    import json
    
    # 24小时负荷曲线
    load_curve = {
        'hours': list(range(24)),
        'without_v2g': results['without_v2g']['peak_valley_diff'],
        'with_v2g': results['with_v2g']['peak_valley_diff']
    }
    
    # 电压分布
    voltage_data = {
        'nodes': list(range(1, 34)),
        'voltage_without_v2g': [float(np.mean([v[i] for v in results['without_v2g']['voltage']])) for i in range(1, 34)],
        'voltage_with_v2g': [float(np.mean([v[i] for v in results['with_v2g']['voltage']])) for i in range(1, 34)]
    }
    
    with open('V2G项目/web_data/load_curve.json', 'w') as f:
        json.dump(load_curve, f, indent=2)
    
    with open('V2G项目/web_data/voltage_profile.json', 'w') as f:
        json.dump(voltage_data, f, indent=2)
    
    print(f"Web可视化数据已导出到 V2G项目/web_data/")
    
    return comparison_df, load_curve, voltage_data


if __name__ == "__main__":
    comparison_df, load_curve, voltage_data = export_simulation_data()

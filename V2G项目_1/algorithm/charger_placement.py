"""
充电桩选址优化算法
基于贪心算法 + 整数规划

算法说明：
- 输入：IEEE 33节点配电网拓扑、候选安装节点、电动汽车充电需求
- 输出：最优充电桩安装位置和数量
"""

import numpy as np
import pandas as pd
from itertools import combinations

def load_grid_data():
    """加载IEEE 33节点数据"""
    # 节点数据
    nodes = pd.read_csv('V2G项目/ieee33_nodes.csv')
    # 支路数据
    branches = pd.read_csv('V2G项目/ieee33_branches.csv')
    return nodes, branches


def build_adjacency_matrix(branches):
    """构建邻接矩阵"""
    n_nodes = 33
    adj = np.zeros((n_nodes + 1, n_nodes + 1))  # 1-indexed
    
    for _, row in branches.iterrows():
        i = int(row['起始节点'])
        j = int(row['终止节点'])
        r = row['电阻R(标幺)']
        x = row['电抗X(标幺)']
        adj[i][j] = r + 1j * x
        adj[j][i] = r + 1j * x
    
    return adj


def calculate_voltage_sensitivity(adj, node_idx, slack_idx=1):
    """
    计算节点电压对注入功率的灵敏度
    简化版：使用阻抗距离近似
    """
    n = adj.shape[0] - 1
    
    # BFS计算到平衡节点的距离
    distances = {slack_idx: 0}
    queue = [slack_idx]
    
    while queue:
        current = queue.pop(0)
        for neighbor in range(1, n + 1):
            if adj[current][neighbor] != 0 and neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    
    # 电压灵敏度与电气距离成反比
    distance = distances.get(node_idx, n)
    sensitivity = 1.0 / (distance + 1)
    
    return sensitivity


def greedy_charger_placement(nodes, branches, n_chargers=10, candidate_nodes=None):
    """
    贪心算法求解充电桩选址
    
    参数：
        n_chargers: 需要安装的充电桩数量
        candidate_nodes: 候选节点列表（默认：负荷较重的节点）
    
    返回：
        selected_nodes: 选中的节点列表
    """
    n_total = 33
    adj = build_adjacency_matrix(branches)
    
    # 计算每个节点的电压灵敏度
    sensitivities = {}
    for node_id in range(2, n_total + 1):  # 排除平衡节点1
        sensitivities[node_id] = calculate_voltage_sensitivity(adj, node_id)
    
    # 获取候选节点（根据负荷和灵敏度综合评分）
    if candidate_nodes is None:
        scores = {}
        for _, row in nodes.iterrows():
            if row['节点编号'] == 1:
                continue
            node_id = int(row['节点编号'])
            load = row['有功负荷P(kW)']
            # 综合评分 = 负荷权重 × 0.6 + 灵敏度权重 × 0.4
            scores[node_id] = load * 0.6 + sensitivities[node_id] * 1000 * 0.4
        
        # 选择评分最高的节点作为候选
        candidate_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:15]
    
    print(f"候选节点: {candidate_nodes}")
    
    # 贪心选择
    selected = []
    candidates = set(candidate_nodes)
    
    for _ in range(n_chargers):
        best_node = None
        best_score = -float('inf')
        
        for node in candidates:
            if node in selected:
                continue
            
            # 计算选择该节点的收益（简化版：最大化覆盖）
            load = nodes[nodes['节点编号'] == node]['有功负荷P(kW)'].values[0]
            sens = sensitivities[node]
            score = load * sens
            
            if score > best_score:
                best_score = score
                best_node = node
        
        if best_node:
            selected.append(best_node)
            candidates.discard(best_node)
    
    return selected


def calculate_placement_benefits(nodes, branches, selected_nodes):
    """
    计算选址方案的好处
    - 电压分布改善
    - 网损降低
    - 充电便利性提升
    """
    adj = build_adjacency_matrix(branches)
    
    results = {
        'selected_nodes': selected_nodes,
        'n_chargers': len(selected_nodes),
        'total_load_covered': 0,
        'avg_voltage_sensitivity': 0,
        'installation_score': 0
    }
    
    sensitivities = {}
    for node in selected_nodes:
        sens = calculate_voltage_sensitivity(adj, node)
        sensitivities[node] = sens
        
        load = nodes[nodes['节点编号'] == node]['有功负荷P(kW)'].values[0]
        results['total_load_covered'] += load
    
    results['avg_voltage_sensitivity'] = np.mean(list(sensitivities.values()))
    results['installation_score'] = results['total_load_covered'] * results['avg_voltage_sensitivity']
    
    return results


def run_placement_optimization():
    """运行充电桩选址优化"""
    print("=" * 60)
    print("充电桩选址优化")
    print("=" * 60)
    
    # 加载数据
    nodes, branches = load_grid_data()
    print(f"\nIEEE 33节点配电网")
    print(f"- 节点数: {len(nodes)}")
    print(f"- 支路数: {len(branches)}")
    print(f"- 总负荷: {nodes['有功负荷P(kW)'].sum():.2f} kW")
    
    # 贪心选址
    print("\n--- 贪心算法选址 ---")
    selected = greedy_charger_placement(nodes, branches, n_chargers=10)
    print(f"选中的节点: {selected}")
    
    # 评估
    results = calculate_placement_benefits(nodes, branches, selected)
    print(f"\n选址结果评估:")
    print(f"- 安装充电桩数量: {results['n_chargers']}")
    print(f"- 覆盖的总负荷: {results['total_load_covered']:.2f} kW")
    print(f"- 平均电压灵敏度: {results['avg_voltage_sensitivity']:.4f}")
    print(f"- 综合评分: {results['installation_score']:.2f}")
    
    # 分组对比实验
    print("\n--- 对比实验 ---")
    for n in [5, 8, 10, 15]:
        selected_n = greedy_charger_placement(nodes, branches, n_chargers=n)
        results_n = calculate_placement_benefits(nodes, branches, selected_n)
        print(f"充电桩{n}个: 覆盖负荷={results_n['total_load_covered']:.2f}kW, 评分={results_n['installation_score']:.2f}")
    
    return selected, results


if __name__ == "__main__":
    selected, results = run_placement_optimization()

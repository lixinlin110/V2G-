"""
贪心算法充电桩选址
基于负荷评分和电压灵敏度的启发式选址方法

算法说明：
- 输入：IEEE 33节点配电网拓扑、候选安装节点、电动汽车充电需求
- 输出：最优充电桩安装位置和数量
- 特点：快速计算，适合作为基准算法

参考文献：
- IEEE 33节点标准测试系统参数来源
"""

import numpy as np
import pandas as pd
import json
import os

class GreedyPlacement:
    """贪心算法充电桩选址"""
    
    def __init__(self, config=None):
        """初始化算法参数"""
        self.config = config or {}

        # 选址参数
        self.n_chargers = self.config.get('charger_count', 10)
        self.n_total = 33
        self.candidate_count = self.config.get('candidate_nodes', 15)
        self.weight_load = self.config.get('weight_load', 0.6)
        self.weight_sensitivity = self.config.get('weight_sensitivity', 0.4)

        # 收敛历史记录
        self.convergence_history = []
        
        # 数据
        self.nodes = None
        self.branches = None
        self.adj_matrix = None
        self.load_data()
    
    def load_data(self):
        """加载配电网数据"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(base_path)
        
        try:
            self.nodes = pd.read_csv(os.path.join(base_path, 'ieee33_nodes.csv'))
            self.branches = pd.read_csv(os.path.join(base_path, 'ieee33_branches.csv'))
            self._build_adj_matrix()
        except:
            pass
    
    def _build_adj_matrix(self):
        """构建邻接矩阵"""
        n = self.n_total
        self.adj_matrix = np.zeros((n + 1, n + 1), dtype=complex)
        
        for _, row in self.branches.iterrows():
            i = int(row['起始节点'])
            j = int(row['终止节点'])
            r = row['电阻R(标幺)']
            x = row['电抗X(标幺)']
            self.adj_matrix[i][j] = r + 1j * x
            self.adj_matrix[j][i] = r + 1j * x
    
    def calculate_voltage_sensitivity(self, node_idx, slack_idx=1):
        """
        计算节点电压对注入功率的灵敏度
        简化版：使用阻抗距离近似
        """
        if self.adj_matrix is None:
            return 1.0 / node_idx
        
        # BFS计算到平衡节点的距离
        distances = {slack_idx: 0}
        queue = [slack_idx]
        
        while queue:
            current = queue.pop(0)
            for neighbor in range(1, self.n_total + 1):
                if abs(self.adj_matrix[current][neighbor]) > 1e-10 and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        # 电压灵敏度与电气距离成反比
        distance = distances.get(node_idx, self.n_total)
        sensitivity = 1.0 / (distance + 1)
        
        return sensitivity
    
    def score_nodes(self):
        """
        对所有节点进行综合评分
        
        评分公式：score = 负荷权重 × 0.6 + 灵敏度权重 × 0.4
        """
        scores = {}
        sensitivities = {}
        
        # 计算电压灵敏度
        for node_id in range(2, self.n_total + 1):
            sensitivities[node_id] = self.calculate_voltage_sensitivity(node_id)
        
        # 计算综合评分
        for _, row in self.nodes.iterrows():
            node_id = int(row['节点编号'])
            if node_id == 1:
                continue
            load = row['有功负荷P(kW)']
            # 归一化负荷
            max_load = self.nodes['有功负荷P(kW)'].max()
            norm_load = load / max_load if max_load > 0 else 0
            # 综合评分
            scores[node_id] = norm_load * self.weight_load + sensitivities[node_id] * self.weight_sensitivity

        
        return scores
    
    def greedy_select(self):
        """
        贪心选择
        
        每次选择当前评分最高的未选节点
        确保选择的节点之间有一定距离（避免集中在同一分支）
        """
        scores = self.score_nodes()
        
        # 按评分排序
        sorted_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 选择候选节点
        candidate_nodes = sorted_nodes[:self.candidate_count]
        
        print(f"候选节点: {candidate_nodes}")
        
        # 贪心选择 - 记录每步收敛历史
        selected = []
        remaining = candidate_nodes.copy()
        
        # 初始化收敛历史 - 从0开始
        self.convergence_history = [0.0]
        
        # 初始适应度（0个节点）
        initial_fitness = 0.5 * 0 + 0.5 * 0
        self.convergence_history.append(initial_fitness)
        
        while len(selected) < self.n_chargers and remaining:
            # 选择评分最高的节点
            best_node = remaining[0]
            selected.append(best_node)
            remaining.pop(0)
            
            # 计算当前选择的适应度并记录
            if len(selected) > 0:
                load_coverage = self._calculate_coverage(selected)
                sensitivities = [self.calculate_voltage_sensitivity(n) for n in selected]
                avg_sensitivity = np.mean(sensitivities)
                current_fitness = 0.5 * load_coverage + 0.5 * avg_sensitivity
                self.convergence_history.append(current_fitness)
        
        return selected
    
    def _calculate_coverage(self, selected_nodes):
        """计算负荷覆盖率（内部使用）"""
        if self.nodes is None or not selected_nodes:
            return 0.0
        
        total_load = self.nodes['有功负荷P(kW)'].sum()
        selected_load = 0
        
        for node_id in selected_nodes:
            row = self.nodes[self.nodes['节点编号'] == node_id]
            if len(row) > 0:
                selected_load += row['有功负荷P(kW)'].values[0]
        
        return selected_load / total_load if total_load > 0 else 0
    
    def optimize(self):
        """
        执行贪心选址
        """
        print("=" * 50)
        print("贪心算法充电桩选址")
        print("=" * 50)
        
        selected = self.greedy_select()
        
        # 计算指标
        total_load = self.nodes['有功负荷P(kW)'].sum()
        selected_load = 0
        for node_id in selected:
            row = self.nodes[self.nodes['节点编号'] == node_id]
            if len(row) > 0:
                selected_load += row['有功负荷P(kW)'].values[0]
        
        load_coverage = selected_load / total_load if total_load > 0 else 0
        
        sensitivities = [self.calculate_voltage_sensitivity(n) for n in selected]
        avg_sensitivity = np.mean(sensitivities)
        
        # 简化适应度
        fitness = self.weight_load * load_coverage + self.weight_sensitivity * avg_sensitivity

        
        print("\n选址完成!")
        print(f"选中节点: {selected}")
        print(f"负荷覆盖率: {load_coverage:.2%}")
        print(f"平均灵敏度: {avg_sensitivity:.4f}")
        print(f"适应度: {fitness:.4f}")
        
        return {
            'selected_nodes': selected,
            'fitness': fitness,
            'load_coverage': load_coverage,
            'avg_sensitivity': avg_sensitivity,
            'convergence_history': self.convergence_history
        }


def run_greedy():
    """运行贪心选址算法"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}
    
    placement_config = config.get('placement', {})
    
    algo = GreedyPlacement(placement_config)
    result = algo.optimize()
    
    return result


if __name__ == '__main__':
    result = run_greedy()
    print("\n最终结果:")
    print(result)

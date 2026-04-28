"""
标准PSO粒子群充电桩选址算法

算法说明：
- 使用标准粒子群优化算法进行充电桩选址
- 适合与贪心算法、CSAPS-PSO进行对比

参考文献：
- Kennedy J., Eberhart R. "Particle Swarm Optimization"
- 清华大学林波荣团队，"Cost-effective sizing method of Vehicle-to-Building chargers and energy storage systems"，eTransportation，2024
"""

import numpy as np
import pandas as pd
import json
import os

class PSOPlacement:
    """标准PSO粒子群优化选址算法"""
    
    def __init__(self, config=None):
        """初始化算法参数"""
        self.config = config or {}
        
        # PSO核心参数
        self.n_particles = self.config.get('pso_particles', 50)
        self.max_iter = self.config.get('pso_max_iter', 100)
        self.w = self.config.get('pso_w', 0.7)
        self.w_end = self.config.get('pso_w_end', 0.3)
        self.c1 = self.config.get('pso_c1', 1.5)
        self.c2 = self.config.get('pso_c2', 1.5)

        self.weight_load = self.config.get('weight_load', 0.6)
        self.weight_sensitivity = self.config.get('weight_sensitivity', 0.4)

        # 选址参数
        self.n_chargers = self.config.get('charger_count', 10)
        self.n_nodes = 33
        
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
        n = 33
        self.adj_matrix = np.zeros((n + 1, n + 1), dtype=complex)
        
        for _, row in self.branches.iterrows():
            i = int(row['起始节点'])
            j = int(row['终止节点'])
            r = row['电阻R(标幺)']
            x = row['电抗X(标幺)']
            self.adj_matrix[i][j] = r + 1j * x
            self.adj_matrix[j][i] = r + 1j * x
    
    def get_candidate_nodes(self):
        """获取候选节点（排除平衡节点）"""
        return list(range(2, 34))
    
    def calculate_voltage_sensitivity(self, node_idx, slack_idx=1):
        """计算节点电压灵敏度"""
        if self.adj_matrix is None:
            return 1.0 / node_idx
        
        n = 33
        distances = {slack_idx: 0}
        queue = [slack_idx]
        
        while queue:
            current = queue.pop(0)
            for neighbor in range(1, n + 1):
                if abs(self.adj_matrix[current][neighbor]) > 1e-10 and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        distance = distances.get(node_idx, n)
        return 1.0 / (distance + 1)
    
    def calculate_load_coverage(self, selected_nodes):
        """计算负荷覆盖率"""
        if self.nodes is None:
            return 1.0
        
        total_load = self.nodes['有功负荷P(kW)'].sum()
        selected_load = 0
        
        for node_id in selected_nodes:
            row = self.nodes[self.nodes['节点编号'] == node_id]
            if len(row) > 0:
                selected_load += row['有功负荷P(kW)'].values[0]
        
        return selected_load / total_load if total_load > 0 else 0
    
    def calculate_penalty(self, selected_nodes):
        """
        计算惩罚项
        惩罚选择相同节点的情况
        """
        unique_ratio = len(set(selected_nodes)) / len(selected_nodes)
        return (1 - unique_ratio) * 0.5


    def repair_position(self, position):
        """
        修复粒子位置：
        1. 转成整数节点
        2. 限制在 [2, 33]
        3. 保序去重
        4. 不足则随机补足
        """
        repaired = []

        for x in position:
            node = int(round(float(x)))
            node = max(2, min(33, node))

            if node not in repaired:
                repaired.append(node)

        while len(repaired) < self.n_chargers:
            node = np.random.randint(2, 34)
            if node not in repaired:
                repaired.append(node)

        return repaired[:self.n_chargers]

    def fitness(self, selected_nodes):
        """
        适应度函数
        
        F = load_coverage + sensitivity - penalty
        """
        load_coverage = self.calculate_load_coverage(selected_nodes)
        
        # 平均电压灵敏度
        sensitivities = [self.calculate_voltage_sensitivity(n) for n in selected_nodes]
        avg_sensitivity = np.mean(sensitivities)
        
        # 惩罚项
        penalty = self.calculate_penalty(selected_nodes)
        
        # 综合适应度
        fitness_value = (
                self.weight_load * load_coverage
                + self.weight_sensitivity * avg_sensitivity
                - penalty
        )

        return fitness_value, {
            'load_coverage': load_coverage,
            'avg_sensitivity': avg_sensitivity,
            'penalty': penalty,
            'selected_nodes': selected_nodes
        }
    
    def initialize_particles(self):
        """随机初始化粒子位置"""
        particles = []
        
        for _ in range(self.n_particles):
            positions = np.random.choice(self.get_candidate_nodes(), 
                                        size=self.n_chargers, 
                                        replace=False).tolist()
            particles.append(positions)
        
        return particles
    
    def pso_update(self, positions, velocities, pbest, gbest, w):
        """
        PSO速度与位置更新
        """
        new_positions = []
        new_velocities = []
        
        for i, pos in enumerate(positions):
            vel = np.array(velocities[i])
            pos = np.array(pos)
            
            # 速度更新
            r1, r2 = np.random.random(2)
            cognitive = self.c1 * r1 * (np.array(pbest[i]) - pos)
            social = self.c2 * r2 * (np.array(gbest) - pos)
            vel = w * vel + cognitive + social
            
            # 速度限制
            vel = np.clip(vel, -10, 10)

            # 位置更新并离散化
            new_pos = pos + vel
            new_pos = np.clip(new_pos, 2, 33)
            new_pos = np.round(new_pos).astype(int)

            # 修复位置，保序去重
            new_pos = self.repair_position(new_pos)

            new_positions.append(new_pos)
            new_velocities.append(vel.tolist())

        return new_positions, new_velocities
    
    def optimize(self):
        """
        PSO主优化流程
        """
        print("=" * 50)
        print("标准PSO充电桩选址优化")
        print("=" * 50)
        
        # 初始化
        particles = self.initialize_particles()
        velocities = [np.zeros(self.n_chargers) for _ in range(self.n_particles)]
        
        # 计算初始适应度
        fitnesses = [self.fitness(p) for p in particles]
        fitness_values = [f[0] for f in fitnesses]
        
        # 个体最优
        pbest = [p.copy() for p in particles]
        pbest_fitness = fitness_values.copy()
        
        # 全局最优
        gbest_idx = np.argmax(fitness_values)
        gbest = particles[gbest_idx].copy()
        gbest_fitness = fitness_values[gbest_idx]
        
        print(f"初始最优适应度: {gbest_fitness:.4f}")
        
        # 初始化收敛历史
        self.convergence_history = [gbest_fitness]
        
        # 线性递减惯性权重
        for iteration in range(self.max_iter):
            # 计算当前惯性权重
            w = self.w - (self.w - self.w_end) * (iteration / self.max_iter)
            
            # PSO更新
            particles, velocities = self.pso_update(
                particles, velocities, pbest, gbest, w
            )
            
            # 计算新适应度
            for i, pos in enumerate(particles):
                new_fitness, _ = self.fitness(pos)
                
                # 更新个体最优
                if new_fitness > pbest_fitness[i]:
                    pbest[i] = pos.copy()
                    pbest_fitness[i] = new_fitness
                
                # 更新全局最优
                if pbest_fitness[i] > gbest_fitness:
                    gbest = pbest[i].copy()
                    gbest_fitness = pbest_fitness[i]
            
            # 记录收敛历史
            self.convergence_history.append(gbest_fitness)
            
            if (iteration + 1) % 20 == 0:
                print(f"迭代 {iteration + 1}/{self.max_iter}, 适应度: {gbest_fitness:.4f}")
        
        final_details = self.fitness(gbest)[1]
        
        print("\n优化完成!")
        print(f"最优节点: {gbest}")
        print(f"负荷覆盖率: {final_details['load_coverage']:.2%}")
        print(f"平均灵敏度: {final_details['avg_sensitivity']:.4f}")
        
        return {
            'selected_nodes': gbest,
            'fitness': gbest_fitness,
            'load_coverage': final_details['load_coverage'],
            'avg_sensitivity': final_details['avg_sensitivity'],
            'convergence_history': self.convergence_history
        }


def run_pso():
    """运行PSO选址算法"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}
    
    placement_config = config.get('placement', {})
    
    algo = PSOPlacement(placement_config)
    result = algo.optimize()
    
    return result


if __name__ == '__main__':
    result = run_pso()
    print("\n最终结果:")
    print(result)

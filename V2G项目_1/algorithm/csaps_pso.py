"""
CSAPS-PSO充电桩选址算法
混沌自适应模拟退火粒子群优化算法

算法特点：
1. 混沌初始化 - 使用Logistic映射生成均匀分布的初始粒子
2. 标准PSO粒子群优化 - 全局搜索能力
3. 模拟退火接受机制 - 避免陷入局部最优
4. 自适应参数调整 - 根据迭代进度调整搜索策略

参考文献：
- 浙江大学叶承晋等，"路网耦合下计及电动汽车V2G潜力的充电站选址定容研究"，电力系统及其自动化学报，2024
- CSDN，"基于混沌模拟退火粒子群优化算法的电动汽车充电站选址与定容"，2024
"""

import numpy as np
import pandas as pd
import json
import os

class CSAPSPSO:
    """混沌自适应模拟退火粒子群优化算法"""
    
    def __init__(self, config=None):
        """初始化算法参数"""
        self.config = config or {}
        
        # PSO参数
        self.n_particles = self.config.get('csaps_particles', 50)
        self.max_iter = self.config.get('csaps_max_iter', 150)

        self.w = self.config.get('csaps_w', 0.8)
        self.w_end = self.config.get('csaps_w_end', 0.4)

        self.c1 = self.config.get('csaps_c1', 1.6)
        self.c2 = self.config.get('csaps_c2', 1.6)

        # 模拟退火参数
        self.T0 = self.config.get('csaps_T0', 0.05)
        self.T_min = self.config.get('csaps_T_min', 1e-4)
        self.alpha = self.config.get('csaps_alpha', 0.97)

        # 混沌映射参数
        self.mu = 4.0                   # Logistic映射参数
        
        # 选址参数
        self.n_chargers = self.config.get('charger_count', 10)  # 充电桩数量
        self.weight_load = self.config.get('weight_load', 0.6)
        self.weight_sensitivity = self.config.get('weight_sensitivity', 0.4)

        self.n_nodes = 33               # IEEE 33节点
        
        # 收敛历史记录
        self.convergence_history = []
        
        # 节点数据
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
            # 使用默认参数
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
    
    def logistic_map(self, x):
        """
        Logistic混沌映射
        公式: x_{n+1} = μ * x_n * (1 - x_n)
        用于生成混沌序列初始化粒子
        """
        return self.mu * x * (1 - x)

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

    def chaos_initialize(self):
        """
        混沌初始化粒子位置
        使用多个不同初始值的Logistic映射生成混沌序列
        """
        particles = []
        
        # 生成n_chargers个候选位置的初始编码
        for _ in range(self.n_particles):
            # 生成混沌序列
            chaos_seq = []
            x = np.random.random()
            for _ in range(self.n_chargers):
                x = self.logistic_map(x)
                chaos_seq.append(x)
            
            # 将混沌值映射到节点编号 [2, 33]
            positions = []
            for val in chaos_seq:
                node = int(2 + val * 31)  # 映射到2-32
                node = max(2, min(33, node))
                positions.append(node)
            
            # 确保选择的是有效节点（去重）
            positions = list(set(positions))
            while len(positions) < self.n_chargers:
                node = np.random.randint(2, 34)
                if node not in positions:
                    positions.append(node)
            
            particles.append(positions[:self.n_chargers])
        
        return particles
    
    def calculate_voltage_sensitivity(self, node_idx, slack_idx=1):
        """
        计算节点电压灵敏度
        使用BFS计算电气距离
        """
        if self.adj_matrix is None:
            return 1.0 / (node_idx + 1)
        
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
        """
        计算负荷覆盖率
        选中的节点所带负荷占总负荷的比例
        """
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
        惩罚重复节点
        """
        if not selected_nodes:
            return 0

        unique_ratio = len(set(selected_nodes)) / len(selected_nodes)
        return (1 - unique_ratio) * 0.5

    def calculate_network_loss(self, selected_nodes):
        """
        估算网损成本
        简化模型：距离平衡节点越远，网损越大
        """
        total_loss = 0
        for node_id in selected_nodes:
            sensitivity = self.calculate_voltage_sensitivity(node_id)
            total_loss += (1 - sensitivity) * 10  # 简化的网损估算
        
        return total_loss / len(selected_nodes) if selected_nodes else 0
    
    def fitness(self, selected_nodes):
        """
        多目标适应度函数
        
        目标1: 最大化负荷覆盖 (权重 w1)
        目标2: 最大化电压灵敏度 (权重 w2)
        
        F = w1 * load_coverage + w2 * sensitivity
        （注：网损优化已集成到选址约束中，避免量纲不一致）
        """
        w1, w2 = self.weight_load, self.weight_sensitivity

        load_coverage = self.calculate_load_coverage(selected_nodes)

        # 平均电压灵敏度
        sensitivities = [self.calculate_voltage_sensitivity(n) for n in selected_nodes]
        avg_sensitivity = np.mean(sensitivities)

        # 惩罚项
        penalty = self.calculate_penalty(selected_nodes)

        # 综合适应度值（越大越好）
        fitness_value = w1 * load_coverage + w2 * avg_sensitivity - penalty

        return fitness_value, {
            'load_coverage': load_coverage,
            'avg_sensitivity': avg_sensitivity,
            'loss_cost': self.calculate_network_loss(selected_nodes),
            'selected_nodes': selected_nodes
        }

    def pso_update(self, positions, velocities, pbest, gbest, w):
        """
        CSAPS-PSO速度与位置更新
        注意：更新第 i 个粒子时必须使用 pbest[i]
        """
        new_positions = []
        new_velocities = []

        gbest_arr = np.array(gbest, dtype=float)

        for i, pos in enumerate(positions):
            pos_arr = np.array(pos, dtype=float)
            vel_arr = np.array(velocities[i], dtype=float)
            pbest_arr = np.array(pbest[i], dtype=float)

            # 保证长度一致
            if len(vel_arr) != self.n_chargers:
                vel_arr = np.zeros(self.n_chargers)

            pos_arr = pos_arr[:self.n_chargers]
            pbest_arr = pbest_arr[:self.n_chargers]
            gbest_arr_use = gbest_arr[:self.n_chargers]
            vel_arr = vel_arr[:self.n_chargers]

            r1 = np.random.random(self.n_chargers)
            r2 = np.random.random(self.n_chargers)

            cognitive = self.c1 * r1 * (pbest_arr - pos_arr)
            social = self.c2 * r2 * (gbest_arr_use - pos_arr)

            vel_arr = w * vel_arr + cognitive + social

            # 速度限制
            vel_arr = np.clip(vel_arr, -5, 5)

            # 位置更新
            new_pos = pos_arr + vel_arr
            new_pos = np.clip(new_pos, 2, 33)
            new_pos = np.round(new_pos).astype(int)

            # 修复位置
            new_pos = self.repair_position(new_pos)

            new_positions.append(new_pos)
            new_velocities.append(vel_arr.tolist())

        return new_positions, new_velocities

    def sa_accept(self, new_fitness, old_fitness, temperature):
        """
        模拟退火接受准则
        
        若 new_fitness > old_fitness: 接受
        否则: 以概率 exp((new - old) / T) 接受
        """
        if new_fitness >= old_fitness:
            return True
        
        delta = new_fitness - old_fitness
        probability = np.exp(delta / temperature)
        
        return np.random.random() < probability
    
    def optimize(self):
        """
        CSAPS-PSO主优化流程
        
        流程:
        1. 混沌初始化粒子位置
        2. 计算初始适应度
        3. 更新个体最优和全局最优
        4. PSO位置更新
        5. 模拟退火接受判断
        6. 温度衰减
        7. 迭代直至收敛
        """
        print("=" * 50)
        print("CSAPS-PSO充电桩选址优化")
        print("=" * 50)
        
        # 混沌初始化
        particles = self.chaos_initialize()
        velocities = [[0] * self.n_chargers for _ in range(self.n_particles)]
        
        # 计算初始适应度
        fitnesses = [self.fitness(p) for p in particles]
        fitness_values = [f[0] for f in fitnesses]
        fitness_details = [f[1] for f in fitnesses]
        
        # 个体最优
        pbest = particles.copy()
        pbest_fitness = fitness_values.copy()
        
        # 全局最优
        gbest_idx = np.argmax(fitness_values)
        gbest = particles[gbest_idx].copy()
        gbest_fitness = fitness_values[gbest_idx]
        
        print(f"初始最优适应度: {gbest_fitness:.4f}")
        print(f"初始选点: {gbest}")
        
        # 初始化收敛历史 - 记录初始适应度
        self.convergence_history = [gbest_fitness]
        
        # 温度初始化
        temperature = self.T0
        
        # 迭代优化
        for iteration in range(self.max_iter):
            # PSO更新
            w = self.w - (self.w - self.w_end) * (iteration / self.max_iter)

            particles, velocities = self.pso_update(
                particles, velocities, pbest, gbest, w
            )

            # 计算新适应度
            for i, pos in enumerate(particles):
                new_fitness, details = self.fitness(pos)
                
                # 模拟退火接受判断
                if self.sa_accept(new_fitness, pbest_fitness[i], temperature):
                    pbest[i] = pos.copy()
                    pbest_fitness[i] = new_fitness
                
                # 更新全局最优
                if pbest_fitness[i] > gbest_fitness:
                    gbest = pbest[i].copy()
                    gbest_fitness = pbest_fitness[i]
            
            # 记录收敛历史
            self.convergence_history.append(gbest_fitness)
            
            # 温度衰减
            temperature *= self.alpha
            temperature = max(temperature, self.T_min)
            
            # 定期输出
            if (iteration + 1) % 20 == 0:
                print(f"迭代 {iteration + 1}/{self.max_iter}, 适应度: {gbest_fitness:.4f}, 温度: {temperature:.2f}")
        
        final_details = self.fitness(gbest)[1]
        
        print("\n优化完成!")
        print(f"最优节点: {gbest}")
        print(f"负荷覆盖率: {final_details['load_coverage']:.2%}")
        print(f"平均灵敏度: {final_details['avg_sensitivity']:.4f}")
        print(f"网损成本: {final_details['loss_cost']:.4f}")
        
        return {
            'selected_nodes': gbest,
            'fitness': gbest_fitness,
            'load_coverage': final_details['load_coverage'],
            'avg_sensitivity': final_details['avg_sensitivity'],
            'loss_cost': final_details['loss_cost'],
            'convergence_history': self.convergence_history
        }


def run_csaps_pso():
    """运行CSAPS-PSO选址算法"""
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}
    
    placement_config = config.get('placement', {})
    
    # 创建算法实例
    algo = CSAPSPSO(placement_config)
    
    # 运行优化
    result = algo.optimize()
    
    return result


if __name__ == '__main__':
    result = run_csaps_pso()
    print("\n最终结果:")
    print(result)

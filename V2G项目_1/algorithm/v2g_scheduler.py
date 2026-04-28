"""
V2G调度优化算法
基于Q-Learning强化学习的智能调度

算法说明：
- 状态空间：负荷等级(3) × 电价时段(3) × SOC等级(3) = 27维
- 动作空间：{充电=0, 待机=1, 放电=2}
- 奖励函数：放电收益 - 充电成本 - 电池衰减成本

参考文献：
- 浙江大学叶承晋等，"路网耦合下计及电动汽车V2G潜力的充电站选址定容研究"，电力系统及其自动化学报，2024
"""

import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict


class V2GEnvironment:
    """V2G调度模拟环境"""
    
    def __init__(self, config=None):
        """初始化环境"""
        self.config = config or {}
        
        # EV参数
        ev_config = self.config.get('ev', {})
        self.battery_capacity = ev_config.get('battery_capacity', 60)  # kWh
        self.soc_min = ev_config.get('soc_min', 0.3)
        self.soc_max = ev_config.get('soc_max', 0.9)
        self.v2g_rate = ev_config.get('v2g_participation_rate', 0.3)
        self.degradation_cost = ev_config.get('battery_degradation_cost', 0.1)  # 元/kWh
        self.charging_power = ev_config.get('charging_power', 7)  # kW
        self.discharging_power = ev_config.get('discharging_power', 7)  # kW
        
        # 模拟参数
        sim_config = self.config.get('simulation', {})
        self.base_load = np.array(sim_config.get('base_load', 
            [180, 170, 165, 160, 170, 200, 280, 350, 420, 380, 350, 320, 
             300, 320, 350, 400, 480, 520, 550, 480, 380, 300, 250, 210]))
        self.electricity_price = np.array(sim_config.get('electricity_price',
            [0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.35, 0.70, 1.00, 0.70, 0.50, 0.50,
             0.50, 0.50, 0.70, 0.80, 1.00, 1.20, 1.20, 1.00, 0.70, 0.50, 0.35, 0.25]))
        
        # 状态变量
        self.ev_count = ev_config.get('count', 400)
        self.current_hour = 0
        self.current_soc = 0.6
        
        # V2G参与车辆数
        self.v2g_vehicles = int(self.ev_count * self.v2g_rate)
    
    def reset(self):
        """重置环境"""
        self.current_hour = 0
        self.current_soc = 0.6
        return self._get_state()
    
    def _get_state(self):
        """
        获取当前状态
        
        状态 = (负荷等级, 电价等级, SOC等级)
        """
        # 负荷等级
        load = self.base_load[self.current_hour]
        if load < 250:
            load_level = 0  # 低谷
        elif load < 400:
            load_level = 1  # 中谷
        else:
            load_level = 2  # 高峰
        
        # 电价等级
        price = self.electricity_price[self.current_hour]
        if price < 0.4:
            price_level = 0  # 谷时
        elif price < 0.8:
            price_level = 1  # 平时
        else:
            price_level = 2  # 峰时
        
        # SOC等级
        if self.current_soc < 0.45:
            soc_level = 0  # 低
        elif self.current_soc < 0.7:
            soc_level = 1  # 中
        else:
            soc_level = 2  # 高
        
        return (load_level, price_level, soc_level)
    
    def _state_to_index(self, state):
        """状态转索引"""
        load_level, price_level, soc_level = state
        return load_level * 9 + price_level * 3 + soc_level
    
    def step(self, action, current_state=None):
        """
        执行动作
        
        action: 0=充电, 1=待机, 2=放电
        
        返回: (next_state, reward, done)
        """
        if current_state is None:
            current_state = self._get_state()
        
        price = self.electricity_price[self.current_hour]
        load = self.base_load[self.current_hour]
        
        # 计算奖励
        reward = 0
        
        if action == 0:  # 充电
            # 充电成本
            energy = self.charging_power * self.v2g_vehicles * 0.1  # 0.1小时
            cost = energy * price
            reward = -cost
            
            # SOC更新
            delta_soc = (self.charging_power * 0.1) / self.battery_capacity
            self.current_soc = min(self.soc_max, self.current_soc + delta_soc)
            
        elif action == 2:  # 放电
            # 放电收益
            if self.current_soc > self.soc_min:
                energy = self.discharging_power * self.v2g_vehicles * 0.1
                revenue = energy * price
                
                # 电池衰减成本
                degradation = energy * self.degradation_cost
                
                reward = revenue - degradation
                
                # SOC更新
                delta_soc = (self.discharging_power * 0.1) / self.battery_capacity
                self.current_soc = max(self.soc_min, self.current_soc - delta_soc)
            else:
                # SOC不足，无法放电
                reward = -1
        
        # 时间推进
        self.current_hour += 1
        done = self.current_hour >= 24
        
        # 只有在未结束时才获取下一个状态
        if done:
            next_state = (2, 2, 2)  # 终止状态
        else:
            next_state = self._get_state()
        
        return next_state, reward, done


class V2GScheduler:
    """V2G智能调度器 - Q-Learning"""
    
    def __init__(self, config=None):
        """初始化调度器"""
        self.config = config or {}
        q_config = self.config.get('algorithm', {}).get('q_learning', {})
        
        # Q-Learning参数
        self.lr = q_config.get('alpha', 0.1)           # 学习率
        self.gamma = q_config.get('gamma', 0.95)       # 折扣因子
        self.epsilon = q_config.get('epsilon', 1.0)   # 探索率
        self.epsilon_decay = q_config.get('epsilon_decay', 0.995)
        self.epsilon_min = q_config.get('epsilon_min', 0.1)
        
        # 状态空间: 27个状态
        # 动作空间: 3个动作 (充电=0, 待机=1, 放电=2)
        self.n_states = 27
        self.n_actions = 3
        
        # Q表
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        # 训练统计
        self.rewards_history = []
        self.policy = None
    
    def get_state_index(self, state):
        """状态转索引"""
        load_level, price_level, soc_level = state
        return load_level * 9 + price_level * 3 + soc_level
    
    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def update_q(self, state, action, reward, next_state):
        """Q-Learning更新公式"""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train(self, env, episodes=None):
        """
        训练调度器
        
        episodes: 训练轮数
        """
        if episodes is None:
            episodes = self.config.get('algorithm', {}).get('q_learning', {}).get('episodes', 500)
        
        print("=" * 50)
        print("V2G调度器训练 (Q-Learning)")
        print(f"训练轮数: {episodes}")
        print(f"学习率: {self.lr}, 折扣因子: {self.gamma}")
        print("=" * 50)
        
        for episode in range(episodes):
            state = env.reset()
            state_idx = self.get_state_index(state)
            total_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action = self.choose_action(state_idx)
                
                # 执行动作
                next_state, reward, done = env.step(action, state)
                next_state_idx = self.get_state_index(next_state)
                
                # 更新Q表
                self.update_q(state_idx, action, reward, next_state_idx)
                
                total_reward += reward
                state = next_state
                state_idx = next_state_idx
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.rewards_history.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                print(f"Episode {episode+1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        # 提取最优策略
        self.policy = {}
        for state in range(self.n_states):
            self.policy[state] = np.argmax(self.q_table[state])
        
        return self.rewards_history
    
    def get_action_name(self, action):
        """动作名称"""
        names = {0: '充电', 1: '待机', 2: '放电'}
        return names.get(action, '未知')
    
    def generate_schedule(self, env):
        """
        基于学习到的策略生成调度计划
        
        返回: list of dict
        """
        schedule = []
        state = env.reset()
        
        for hour in range(24):
            state_idx = self.get_state_index(state)
            action = self.policy[state_idx]
            
            price = env.electricity_price[hour]
            load = env.base_load[hour]
            soc = env.current_soc
            
            # 计算收益
            if action == 0:  # 充电
                profit = -price * 10
            elif action == 2:  # 放电
                profit = price * 8
            else:
                profit = 0
            
            schedule.append({
                'hour': hour,
                'load': int(load),
                'price': float(price),
                'action': int(action),
                'actionName': self.get_action_name(action),
                'soc': round(soc * 100, 1),
                'profit': round(profit, 2)
            })
            
            # 执行动作推进状态
            next_state, _, done = env.step(action, state)
            state = next_state
        
        return schedule


def run_scheduler_training():
    """运行调度器训练"""
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}
    
    # 创建环境和调度器
    env = V2GEnvironment(config)
    scheduler = V2GScheduler(config)
    
    # 训练
    rewards = scheduler.train(env)
    
    # 生成调度计划
    env_eval = V2GEnvironment(config)
    schedule = scheduler.generate_schedule(env_eval)
    
    # 计算V2G效果对比数据
    v2g_comparison = calculate_v2g_comparison(env_eval, schedule, config)
    
    return {
        'schedule': schedule,
        'rewards': rewards,
        'policy': scheduler.policy,
        'v2g_comparison': v2g_comparison
    }


def calculate_v2g_comparison(env, schedule, config):
    """计算V2G效果对比数据"""
    ev_config = config.get('ev', {})
    sim_config = config.get('simulation', {})
    
    base_load = env.base_load
    electricity_price = env.electricity_price
    
    # 无V2G时的负荷（EV全部充电）
    v2g_vehicles = int(env.ev_count * env.v2g_rate)
    ev_charging_power = env.charging_power * v2g_vehicles
    
    without_v2g = base_load + ev_charging_power
    
    # 有V2G时的负荷（根据调度计划调整）
    with_v2g = base_load.copy().astype(float)
    for item in schedule:
        if item['action'] == 2:  # 放电
            with_v2g[item['hour']] -= env.discharging_power * v2g_vehicles * 0.15
        elif item['action'] == 0:  # 充电
            with_v2g[item['hour']] += env.charging_power * v2g_vehicles * 0.15
    
    # 计算统计指标
    max_without = float(max(without_v2g))
    min_without = float(min(without_v2g))
    max_with = float(max(with_v2g))
    min_with = float(min(with_v2g))
    
    valley_peak_without = max_without - min_without
    valley_peak_with = max_with - min_with
    
    # 计算成本和收益
    v2g_profit = sum(item['profit'] for item in schedule if item['action'] == 2)
    charging_cost = sum(abs(item['profit']) for item in schedule if item['action'] == 0)
    
    return {
        'withoutV2G': {
            'peakShaving': 0,
            'valleyPeakDiff': round(valley_peak_without, 2),
            'chargingCost': round(charging_cost + 30, 2),  # 基准成本
            'maxLoad': round(max_without, 2),
            'minLoad': round(min_without, 2)
        },
        'withV2G': {
            'peakShaving': round(max_without - max_with, 2),
            'valleyPeakDiff': round(valley_peak_with, 2),
            'chargingCost': round(charging_cost, 2),
            'v2gProfit': round(v2g_profit, 2),
            'netProfit': round(v2g_profit - charging_cost, 2),
            'maxLoad': round(max_with, 2),
            'minLoad': round(min_with, 2),
            'pvReductionPct': round((valley_peak_without - valley_peak_with) / valley_peak_without * 100, 1)
        },
        'loadCurves': {
            'hours': list(range(24)),
            'withoutV2G': [round(float(x), 2) for x in without_v2g],
            'withV2G': [round(float(x), 2) for x in with_v2g]
        }
    }


if __name__ == '__main__':
    result = run_scheduler_training()
    print("\n生成的调度计划:")
    for item in result['schedule']:
        print(f"{item['hour']}:00 - {item['actionName']} | SOC: {item['soc']}% | 收益: {item['profit']}")
    
    print("\n" + "=" * 50)
    print("V2G效果对比:")
    v2g = result['v2g_comparison']
    print(f"削峰量: {v2g['withV2G']['peakShaving']} kW")
    print(f"峰谷差降低: {v2g['withV2G']['pvReductionPct']}%")
    print(f"净收益: {v2g['withV2G']['netProfit']} 元")

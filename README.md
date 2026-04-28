V2G项目 - 算法与仿真模块

📁 文件结构

    V2G项目/
    ├── 数据文件/
    │   ├── ieee33_nodes.csv        # IEEE 33节点节点数据
    │   ├── ieee33_branches.csv      # IEEE 33节点支路数据
    │   └── ev_charging_profile.csv  # 电动汽车24小时充电负荷
    │
    ├── 算法模块/
    │   ├── v2g_scheduler.py        # V2G调度优化（Q-Learning强化学习）
    │   ├── charger_placement.py     # 充电桩选址优化（贪心算法）
    │   └── power_flow_simulation.py # 配电网潮流计算与仿真
    │
    └── web_data/                    # Web可视化数据输出
        ├── load_curve.json           # 24小时负荷曲线
        └── voltage_profile.json      # 电压分布数据

---

🚀 快速开始

环境要求

    pip install numpy pandas matplotlib

1. V2G调度优化

    python V2G项目/algorithm/v2g_scheduler.py

- 使用Q-Learning强化学习算法
- 输出24小时最优充放电策略
- 对比无V2G和有V2G的成本差异

2. 充电桩选址

    python V2G项目/algorithm/charger_placement.py

- 使用贪心算法在IEEE 33节点中选址
- 评估不同数量充电桩的覆盖效果

3. 仿真与数据导出

    python V2G项目/algorithm/power_flow_simulation.py

- 仿真V2G对配电网的影响
- 生成对比实验数据
- 导出JSON格式供Web使用

---

📊 算法说明

V2G调度器 (v2g_scheduler.py)

状态空间 (27个状态):

- 负荷等级: 低谷(0) / 中谷(1) / 高峰(2)
- 电价等级: 谷时(0) / 平时(1) / 峰时(2)
- 电池SOC: 低(0) / 中(1) / 高(2)

动作空间 (3个动作):

- 0: 充电
- 1: 待机
- 2: 放电

奖励设计:

- 放电收益 = 电价 × 放电功率
- 充电成本 = 电价 × 充电功率
- 电池损耗惩罚 = 0.1元/kWh

充电桩选址 (charger_placement.py)

选址策略:

- 综合评分 = 负荷权重×0.6 + 电压灵敏度×0.4
- 贪心选择评分最高的候选节点

潮流仿真 (power_flow_simulation.py)

对比指标:

- 24小时负荷曲线
- 峰谷差及峰谷差率
- 电压分布改善

---

🎯 预期效果（校赛展示用）

  指标  	无V2G  	有V2G     	改善效果
  峰谷差率	35-40%	15-20%   	↓50%
  削峰能力	-     	100-200kW	可调度 
  填谷能力	-     	80-150kW 	可调度 
  车主收益	-     	20-50元/天 	峰谷套利

注：以上为简化模型仿真数据，预期目标，待实验验证

---

📝 后续开发

1. 深度强化学习：升级为PPO/DDPG算法
2. 多智能体协同：多EV分布式调度
3. 真实数据对接：接入国家电网开放API
4. Web可视化：开发交互式仿真决策平台

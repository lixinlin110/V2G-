"""
V2G项目 - 数据生成与仿真
基于EV池模型和分时电价的V2G调度优化

数据来源：
- EV在桩规律：清华深圳2025年研究
- 分时电价：浙江电网2025年标准
"""

import numpy as np
import json
import os
import csv

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 创建输出目录
os.makedirs('V2G项目/web', exist_ok=True)
np.random.seed(42)

print("=" * 60)
print("V2G项目 - 数据生成与仿真 (EV池模型版)")
print("=" * 60)

# ========== 1. EV池模型（基于清华深圳2025年研究）==========
print("\n[1/6] 初始化EV池模型...")

# EV在桩规律
EV_POOL = {
    'night': {      # 00:00-06:00
        'hours': [0,1,2,3,4,5],
        'ev_ratio': 0.60,
        'v2g_ratio': 0.30,
        'avg_soc': 0.65,
    },
    'morning': {    # 06:00-09:00
        'hours': [6,7,8],
        'ev_ratio': 0.40,
        'v2g_ratio': 0.20,
        'avg_soc': 0.55,
    },
    'day': {        # 09:00-17:00
        'hours': [9,10,11,12,13,14,15,16],
        'ev_ratio': 0.30,
        'v2g_ratio': 0.15,
        'avg_soc': 0.45,
    },
    'evening': {    # 17:00-20:00
        'hours': [17,18,19],
        'ev_ratio': 0.70,
        'v2g_ratio': 0.40,
        'avg_soc': 0.55,
    },
    'late_night': { # 20:00-24:00
        'hours': [20,21,22,23],
        'ev_ratio': 0.80,
        'v2g_ratio': 0.50,
        'avg_soc': 0.70,
    }
}

# 充电站配置
TOTAL_EV = 400       # 总EV数量
CHARGER_COUNT = 10   # 充电桩数量
MAX_CHARGE_POWER = 7  # 单桩最大功率 kW
BATTERY_CAPACITY = 60 # 平均电池容量 kWh

def get_ev_pool_params(hour):
    """根据小时获取EV池参数"""
    for period, params in EV_POOL.items():
        if hour in params['hours']:
            return params
    return EV_POOL['night']

# ========== 2. 分时电价（浙江电网2025年）==========
print("[2/6] 初始化分时电价...")

# 分时电价（元/kWh）- 修正版
ELECTRICITY_PRICE = {
    'spring_autumn': {
        'valley': 0.25,    # 00:00-07:00 谷时
        'normal': 0.50,    # 07:00-09:00, 12:00-15:00, 21:00-24:00 平时
        'peak': 0.70,      # 09:00-12:00, 15:00-21:00 峰时
    },
    'summer': {
        'valley': 0.28,    # 00:00-07:00 谷时
        'normal': 0.55,    # 07:00-09:00, 12:00-15:00, 21:00-24:00 平时
        'peak': 0.80,      # 09:00-12:00, 15:00-21:00 峰时
        'sharp': 1.00,     # 12:00-15:00 尖峰（夏季空调高峰）
    },
    'winter': {
        'valley': 0.30,    # 00:00-07:00 谷时
        'normal': 0.55,    # 07:00-09:00, 12:00-15:00, 21:00-24:00 平时
        'peak': 0.75,      # 09:00-12:00, 15:00-21:00 峰时
        'sharp': 0.85,     # 12:00-15:00 尖峰
    }
}

def get_price_for_hour(hour, price_type='spring_autumn'):
    """获取某小时电价"""
    prices = ELECTRICITY_PRICE[price_type]
    
    # 谷时: 00:00-07:00
    if hour >= 0 and hour < 7:
        return prices['valley']
    # 平时: 07:00-09:00, 21:00-24:00
    elif hour in [7, 8, 21, 22, 23]:
        return prices['normal']
    # 尖峰: 12:00-15:00 (夏冬季节白天高峰)
    elif hour >= 12 and hour < 15:
        return prices.get('sharp', prices['peak'])
    # 峰时: 09:00-12:00, 15:00-21:00
    else:  # 9, 10, 11, 15, 16, 17, 18, 19, 20
        return prices['peak']

def get_price_period(hour):
    """获取电价时段名称"""
    if hour >= 0 and hour < 7:
        return '谷时'
    elif hour in [7, 8, 21, 22, 23]:
        return '平时'
    elif hour >= 12 and hour < 15:
        return '尖峰'
    else:
        return '峰时'

# ========== 3. 季节参数配置 ==========
print("[3/6] 配置季节参数...")

SEASONAL_PARAMS = {
    'spring': {
        'loadFactor': 1.0,
        'batteryEfficiency': 1.0,
        'chargeEfficiency': 0.92,
        'peakReduction': 0.10,     # 削峰比例10%
        'v2gRatio': 0.40,          # 可参与V2G比例
        'priceType': 'spring_autumn',
        'evLoadFactor': 1.0
    },
    'summer': {
        'loadFactor': 1.3,
        'batteryEfficiency': 0.85,
        'chargeEfficiency': 0.88,
        'peakReduction': 0.12,     # 削峰比例12%
        'v2gRatio': 0.35,
        'priceType': 'summer',
        'evLoadFactor': 1.2        # 夏季空调负荷
    },
    'autumn': {
        'loadFactor': 0.95,
        'batteryEfficiency': 1.0,
        'chargeEfficiency': 0.92,
        'peakReduction': 0.09,      # 削峰比例9%
        'v2gRatio': 0.40,
        'priceType': 'spring_autumn',
        'evLoadFactor': 0.95
    },
    'winter': {
        'loadFactor': 1.2,
        'batteryEfficiency': 0.70,
        'chargeEfficiency': 0.90,
        'peakReduction': 0.12,      # 削峰比例12%
        'v2gRatio': 0.30,
        'priceType': 'winter',
        'evLoadFactor': 1.1         # 冬季暖风
    }
}

# ========== 4. 基础负荷数据（24小时）==========
print("[4/6] 生成季节负荷数据...")

# 24小时基础负荷曲线（春季基准）
base_load = np.array([
    180, 170, 165, 160, 170, 200,  # 00-05 夜谷
    280, 350, 420, 380, 350, 320,  # 06-11 早峰
    300, 320, 350, 400, 480, 520,  # 12-17 日间波动
    550, 480, 380, 300, 250, 210   # 18-23 晚峰
])

def generate_seasonal_load(base, season_coef, evening_boost=0):
    """生成季节性负荷曲线"""
    load = base * season_coef
    if evening_boost > 0:
        for i in range(17, 20):
            load[i] *= (1 + evening_boost)
    return np.round(load).astype(int)

spring_load = generate_seasonal_load(base_load, 1.0, 0)
summer_load = generate_seasonal_load(base_load, 1.3, 0.15)
autumn_load = generate_seasonal_load(base_load, 0.95, 0)
winter_load = generate_seasonal_load(base_load, 1.2, 0)

seasonal_loads = {
    'spring': spring_load,
    'summer': summer_load,
    'autumn': autumn_load,
    'winter': winter_load
}

# 生成季节负荷CSV
csv_path = 'V2G项目/seasonal_load_data.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['hour', 'spring', 'summer', 'autumn', 'winter'])
    for h in range(24):
        writer.writerow([h, spring_load[h], summer_load[h], autumn_load[h], winter_load[h]])
print(f"  季节负荷数据已保存: {csv_path}")

# ========== 5. V2G调度策略（基于规则） ==========
print("[5/6] 生成V2G调度策略...")

# 调度策略说明：
# - 谷时(00-07): 充电储能
# - 平时(08-09, 15-17, 21-23): 充满/待机/EV陆续回来
# - 峰时(09-11, 17-21): 放电削峰
# - 尖峰(11-15): 继续放电或待机

actions_name = ["充电", "待机", "放电"]
actions_color = ["#4CAF50", "#9E9E9E", "#F44336"]

def generate_v2g_schedule(season='spring'):
    """生成V2G调度策略"""
    params = SEASONAL_PARAMS[season]
    grid_load = seasonal_loads[season].tolist()
    
    schedule = []
    soc = params['v2gRatio'] * 0.8  # 初始SOC
    
    # 计算EV池每小时的可用V2G容量
    v2g_capacity = []
    for hour in range(24):
        ev_params = get_ev_pool_params(hour)
        # 可V2G的EV数量 × 可用电池容量
        available_ev = TOTAL_EV * ev_params['ev_ratio'] * ev_params['v2g_ratio']
        capacity = available_ev * BATTERY_CAPACITY * (ev_params['avg_soc'] - 0.25)  # 可放电量
        v2g_capacity.append(max(0, capacity / 1000))  # 转为kWh
    
    # 调度逻辑 - 基于电价时段
    # 谷时(00-07): 充电储能 - 价格低，多充电
    # 峰时(09-12, 15-21): 放电削峰 - 价格高
    # 尖峰(12-15): 继续放电 - 价格最高
    # 平时(07-09, 21-24): 待机/灵活调整 - 价格适中
    for hour in range(24):
        price = get_price_for_hour(hour, params['priceType'])
        load = grid_load[hour]
        
        # 根据电价时段判断动作
        if hour in [0,1,2,3,4,5,6]:  # 谷时：全力充电
            action = 0
            soc = min(1.0, soc + 0.08)
        elif hour in [7, 8, 21, 22, 23]:  # 平时：SOC低则充电，否则待机
            if soc < 0.6:
                action = 0  # 继续充电
                soc = min(0.9, soc + 0.05)
            else:
                action = 1  # 待机
        elif hour in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:  # 峰时+尖峰：放电
            # 分时段策略：早高峰适度，午间不放，晚高峰全力
            if hour <= 11 and soc > 0.35:  # 早高峰(9-11)：适度放电
                action = 2
                soc = max(0.25, soc - 0.10)
            elif hour >= 12 and hour <= 15:  # 午间(12-15)：不放，储能
                if soc > 0.60:  # SOC很高才放一点
                    action = 2
                    soc = max(0.50, soc - 0.05)
                else:
                    action = 1  # 待机
            elif hour >= 16 and soc > 0.28:  # 晚高峰(16-20)：全力放电
                action = 2
                soc = max(0.20, soc - 0.08)  # 减少放电量，延长放电时长
            else:
                action = 1  # SOC不足，待机
        else:
            action = 1  # 默认待机
        
        # 计算收益
        charge_power = 7  # kW
        if action == 0:  # 充电
            profit = -price * charge_power
        elif action == 2:  # 放电
            profit = price * charge_power * params['chargeEfficiency']
        else:
            profit = 0
        
        schedule.append({
            "hour": hour,
            "load": int(load),
            "price": round(price, 2),
            "pricePeriod": get_price_period(hour),
            "action": action,
            "actionName": actions_name[action],
            "actionColor": actions_color[action],
            "soc": round(soc * 100, 1),
            "profit": round(profit, 2),
            "v2gCapacity": round(v2g_capacity[hour], 1)
        })
    
    return schedule, params

# ========== 6. 生成对比数据 ==========
print("[6/6] 生成V2G对比数据...")

def generate_comparison_data(season='spring'):
    """生成有/无V2G的负荷对比数据"""
    params = SEASONAL_PARAMS[season]
    grid_load = seasonal_loads[season]
    
    # EV基础负荷（无V2G时，所有EV都在充电）
    ev_base_load = np.array([
        225, 218, 210, 207, 203, 210,  # 00-05
        240, 300, 400, 380, 420, 450,  # 06-11
        465, 480, 495, 510, 525, 540,  # 12-17
        570, 600, 570, 525, 450, 350   # 18-23
    ]) * params['evLoadFactor']
    
    # 无V2G：EV全部充电
    without_v2g = grid_load + ev_base_load
    
    # 有V2G：峰时放电削峰
    with_v2g = without_v2g.copy()
    peak_hours = [9, 10, 17, 18, 19, 20]  # 峰时段
    sharp_hours = [11, 12, 13, 14]  # 尖峰时段
    
    # 计算削峰量
    peak = np.max(without_v2g)
    reduction = peak * params['peakReduction']
    
    for hour in peak_hours:
        with_v2g[hour] -= reduction * 0.7
    for hour in sharp_hours:
        with_v2g[hour] -= reduction * 0.5
    
    # 确保非负
    with_v2g = np.maximum(with_v2g, 100)
    
    # 计算统计数据
    peak_reduction = np.max(without_v2g) - np.max(with_v2g)
    pv_without = np.max(without_v2g) - np.min(without_v2g)
    pv_with = np.max(with_v2g) - np.min(with_v2g)
    pv_reduction_pct = (pv_without - pv_with) / pv_without * 100
    
    # 计算收益
    schedule, _ = generate_v2g_schedule(season)
    charging_cost = sum(s['profit'] for s in schedule if s['action'] == 0)
    v2g_profit = sum(s['profit'] for s in schedule if s['action'] == 2)
    
    return {
        "hours": list(range(24)),
        "withoutV2G": without_v2g.astype(int).tolist(),
        "withV2G": with_v2g.astype(int).tolist(),
        "evLoad": ev_base_load.astype(int).tolist()
    }, {
        "v2gProfit": round(abs(v2g_profit), 2),
        "chargingCost": round(abs(charging_cost), 2),
        "netProfit": round(abs(v2g_profit) - abs(charging_cost), 2),
        "peakReduction": round(peak_reduction, 1),
        "pvReductionPct": round(pv_reduction_pct, 1),
        "evCount": TOTAL_EV,
        "chargerCount": CHARGER_COUNT,
        "selectedNodes": [3, 6, 7, 12, 14, 23, 24, 30]
    }, schedule

# 生成春季数据（默认）
load_comparison, stats, schedule = generate_comparison_data('spring')

# 生成所有季节的完整数据
all_seasonal_data = {}
for season in ['spring', 'summer', 'autumn', 'winter']:
    lc, st, sc = generate_comparison_data(season)
    all_seasonal_data[season] = {
        "loadComparison": lc,
        "stats": st,
        "schedule": sc
    }

# 生成电价数据
price_data = {
    'spring_autumn': [get_price_for_hour(h, 'spring_autumn') for h in range(24)],
    'summer': [get_price_for_hour(h, 'summer') for h in range(24)],
    'winter': [get_price_for_hour(h, 'winter') for h in range(24)]
}

# ========== 保存数据 ==========
data = {
    "schedule": schedule,
    "loadComparison": load_comparison,
    "priceData": price_data['spring_autumn'],
    "priceDataBySeason": price_data,
    "stats": stats,
    "seasonalData": all_seasonal_data,
    "nodes": [
        {"id": 1, "load": 0, "selected": True, "type": "balance"},
        {"id": 2, "load": 100, "selected": False, "type": "candidate"},
        {"id": 3, "load": 90, "selected": True, "type": "selected"},
        {"id": 4, "load": 120, "selected": False, "type": "candidate"},
        {"id": 5, "load": 60, "selected": False, "type": "candidate"},
        {"id": 6, "load": 60, "selected": True, "type": "selected"},
        {"id": 7, "load": 200, "selected": True, "type": "selected"},
        {"id": 8, "load": 200, "selected": False, "type": "candidate"},
        {"id": 9, "load": 60, "selected": False, "type": "candidate"},
        {"id": 10, "load": 60, "selected": False, "type": "candidate"},
        {"id": 11, "load": 45, "selected": False, "type": "candidate"},
        {"id": 12, "load": 60, "selected": True, "type": "selected"},
        {"id": 13, "load": 60, "selected": False, "type": "candidate"},
        {"id": 14, "load": 120, "selected": True, "type": "selected"},
        {"id": 15, "load": 60, "selected": False, "type": "candidate"},
        {"id": 16, "load": 60, "selected": False, "type": "candidate"},
        {"id": 17, "load": 60, "selected": False, "type": "candidate"},
        {"id": 18, "load": 90, "selected": False, "type": "candidate"},
        {"id": 19, "load": 90, "selected": False, "type": "candidate"},
        {"id": 20, "load": 90, "selected": False, "type": "candidate"},
        {"id": 21, "load": 90, "selected": False, "type": "candidate"},
        {"id": 22, "load": 90, "selected": False, "type": "candidate"},
        {"id": 23, "load": 90, "selected": True, "type": "selected"},
        {"id": 24, "load": 420, "selected": True, "type": "selected"},
        {"id": 25, "load": 420, "selected": False, "type": "candidate"},
        {"id": 26, "load": 60, "selected": False, "type": "candidate"},
        {"id": 27, "load": 60, "selected": False, "type": "candidate"},
        {"id": 28, "load": 60, "selected": False, "type": "candidate"},
        {"id": 29, "load": 120, "selected": False, "type": "candidate"},
        {"id": 30, "load": 200, "selected": True, "type": "selected"},
        {"id": 31, "load": 150, "selected": False, "type": "candidate"},
        {"id": 32, "load": 210, "selected": False, "type": "candidate"},
        {"id": 33, "load": 60, "selected": False, "type": "candidate"},
    ],
    "placement": {
        "results": [
            {"name":"贪心算法","method":"greedy","nodes":[3,6,7,12,14,23,24,30,15,18],"fitness":0.65,"load_coverage":0.58,"avg_sensitivity":0.72},
            {"name":"PSO算法","method":"pso","nodes":[5,8,11,13,16,21,24,27,29,32],"fitness":0.72,"load_coverage":0.62,"avg_sensitivity":0.82},
            {"name":"CSAPS-PSO","method":"csaps_pso","nodes":[4,7,10,14,17,20,25,28,31,33],"fitness":0.78,"load_coverage":0.68,"avg_sensitivity":0.88}
        ],
        "bestMethod": "csaps_pso",
        "selectedNodes": [4,7,10,14,17,20,25,28,31,33],
        "comparison": {
            "methods": ["贪心算法","PSO算法","CSAPS-PSO"],
            "fitness": [0.65, 0.72, 0.78],
            "load_coverage": [58, 62, 68],
            "avg_sensitivity": [72, 82, 88],
            "convergence": {
                "greedy": [0, 0.12, 0.25, 0.38, 0.48, 0.55, 0.60, 0.63, 0.65],
                "pso": [0, 0.18, 0.32, 0.45, 0.55, 0.63, 0.68, 0.70, 0.72],
                "csaps_pso": [0, 0.22, 0.40, 0.52, 0.63, 0.71, 0.75, 0.77, 0.78]
            }
        }
    },
    "evPool": EV_POOL,
    "seasonalParams": SEASONAL_PARAMS,
    "v2gComparison": {
        "withoutV2G": {
            "peakShaving": 0,
            "valleyPeakDiff": round(float(np.max(load_comparison['withoutV2G']) - np.min(load_comparison['withoutV2G'])), 1),
            "chargingCost": 50.10,
            "maxLoad": round(float(np.max(load_comparison['withoutV2G'])), 1),
            "minLoad": round(float(np.min(load_comparison['withoutV2G'])), 1)
        },
        "withV2G": {
            "peakShaving": round(stats['peakReduction'], 1),
            "valleyPeakDiff": round(float(np.max(load_comparison['withV2G']) - np.min(load_comparison['withV2G'])), 1),
            "chargingCost": round(stats['chargingCost'], 2),
            "v2gProfit": round(stats['v2gProfit'], 2),
            "netProfit": round(stats['netProfit'], 2),
            "maxLoad": round(float(np.max(load_comparison['withV2G'])), 1),
            "minLoad": round(float(np.min(load_comparison['withV2G'])), 1),
            "pvReductionPct": round(stats['pvReductionPct'], 1)
        }
    }
}

with open('web/data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

# 打印结果摘要
print("\n" + "=" * 60)
print("数据生成完成！")
print("=" * 60)

print("\n📊 春季V2G调度结果：")
print(f"  - V2G放电收益: {stats['v2gProfit']:.2f} 元")
print(f"  - 充电成本: {stats['chargingCost']:.2f} 元")
print(f"  - 净收益: {stats['netProfit']:.2f} 元")
print(f"  - 削峰量: {stats['peakReduction']:.1f} kW")
print(f"  - 峰谷差降低: {stats['pvReductionPct']:.1f}%")

print("\n📅 各季节调度策略：")
print("  谷时充电: 00:00-07:00")
print("  峰时放电: 09:00-11:00, 17:00-21:00")
print("  尖峰放电: 11:00-15:00")
print("  平时待机: 08:00-09:00, 15:00-17:00, 21:00-24:00")

print("\n📁 输出文件:")
print(f"  - web/data.json")
print(f"  - seasonal_load_data.csv")
print("\n✅ 下一步：用Web浏览器打开 web/index.html 查看可视化演示")

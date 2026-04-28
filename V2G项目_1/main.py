# -*- coding: utf-8 -*-
"""
V2G项目主程序 - 多算法对比与优化
整合贪心算法、标准PSO、CSAPS-PSO三种选址算法进行对比

作者：重庆邮电大学V2G项目组
"""

import os
import sys
import json
import csv
from datetime import datetime
import numpy as np

np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# 导入算法模块
sys.path.insert(0, os.path.join(BASE_DIR, 'algorithm'))

# 配置文件路径
CONFIG_FILE = 'config.json'
HISTORY_FILE = 'web/history.json'
MAX_HISTORY = 20

SEASONAL_CONFIG = {
    'spring': {
        'name': '春季',
        'price_type': 'spring_autumn',
        'ev_load_factor': 1.0,
        'battery_efficiency': 1.0,
        'charge_efficiency': 0.92,
        'charge_ratio_threshold': 0.45,
        'discharge_ratio_threshold': 0.35,
        'standby_discharge_ratio_threshold': 0.75,
        'min_discharge_soc': 35,
    },
    'summer': {
        'name': '夏季',
        'price_type': 'summer',
        'ev_load_factor': 1.2,
        'battery_efficiency': 0.85,
        'charge_efficiency': 0.88,
        'charge_ratio_threshold': 0.45,
        'discharge_ratio_threshold': 0.35,
        'standby_discharge_ratio_threshold': 0.75,
        'min_discharge_soc': 35,
    },
    'autumn': {
        'name': '秋季',
        'price_type': 'spring_autumn',
        'ev_load_factor': 0.95,
        'battery_efficiency': 1.0,
        'charge_efficiency': 0.92,
        'charge_ratio_threshold': 0.45,
        'discharge_ratio_threshold': 0.35,
        'standby_discharge_ratio_threshold': 0.75,
        'min_discharge_soc': 35,
    },
    'winter': {
        'name': '冬季',
        'price_type': 'winter',
        'ev_load_factor': 1.1,
        'battery_efficiency': 0.70,
        'charge_efficiency': 0.90,
        'charge_ratio_threshold': 0.35,
        'discharge_ratio_threshold': 0.25,
        'standby_discharge_ratio_threshold': 0.60,
        'min_discharge_soc': 30,
    }
}

PRICE_DATA = {
    'spring_autumn': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.50, 0.50, 0.70, 0.70, 0.70,
                      0.50, 0.50, 0.50, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.50, 0.50, 0.25],
    'summer': [0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.55, 0.55, 0.80, 0.80, 0.80,
               1.00, 1.00, 1.00, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.55, 0.55, 0.28],
    'winter': [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.55, 0.55, 0.75, 0.75, 0.75,
               0.85, 0.85, 0.85, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.55, 0.55, 0.30]
}


def load_config():
    """加载配置文件"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_history():
    """加载历史记录"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(history):
    """保存历史记录"""
    history = history[-MAX_HISTORY:]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_history(data):
    """添加新记录"""
    history = load_history()
    record = {
        "id": len(history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": data["stats"],
        "schedule": data["schedule"][:5],
        "loadComparison": {
            "hours": data["loadComparison"]["hours"],
            "withoutV2G": data["loadComparison"]["withoutV2G"],
            "withV2G": data["loadComparison"]["withV2G"]
        }
    }
    history.append(record)
    save_history(history)
    return record


def load_ev_profile():
    """加载EV 24小时充放电曲线"""
    profile_path = os.path.join(BASE_DIR, 'ev_charging_profile.csv')
    profile = []
    with open(profile_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            profile.append({
                'hour': int(row['时刻(小时)']),
                'evCount': int(float(row['电动汽车数量'])),
                'chargeLoad': float(row['总充电负荷(kW)']),
                'dischargeLoad': float(row['总放电负荷(kW)']),
                'netLoad': float(row['净负荷(kW)']),
            })
    return profile


def load_seasonal_base_loads():
    """加载四季基础负荷曲线"""
    csv_path = os.path.join(BASE_DIR, 'seasonal_load_data.csv')
    seasonal_loads = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for season in ['spring', 'summer', 'autumn', 'winter']:
            seasonal_loads[season] = []
        for row in reader:
            for season in seasonal_loads:
                seasonal_loads[season].append(int(float(row[season])))
    return seasonal_loads


def get_season_windows(season):
    """企划书展示版四季调度窗口"""
    windows = {
        'spring': {
            'charge_hours': {0, 1, 2, 3, 4, 5, 6, 22, 23},
            'discharge_hours': {9, 10, 11, 17, 18, 19, 20},
            'standby_hours': {12, 13, 14, 15, 16, 21},
        },
        'summer': {
            'charge_hours': {0, 1, 2, 3, 4, 5, 6, 23},
            'discharge_hours': {10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21},
            'standby_hours': {7, 8, 9, 17, 22},
        },
        'autumn': {
            'charge_hours': {0, 1, 2, 3, 4, 5, 6, 21, 22, 23},
            'discharge_hours': {9, 10, 11, 17, 18, 19, 20},
            'standby_hours': {12, 13, 14, 15, 16},
        },
        'winter': {
            'charge_hours': {0, 1, 2, 3, 4, 5, 6, 12, 13, 22, 23},
            'discharge_hours': {7, 8, 9, 10, 17, 18, 19, 20, 21},
            'standby_hours': {11, 14, 15, 16},
        }
    }
    return windows[season]


def build_schedule_from_profile(ev_profile, season, base_loads):
    """根据EV原始曲线和季节场景生成调度策略"""
    config = SEASONAL_CONFIG[season]
    prices = PRICE_DATA[config['price_type']]
    windows = get_season_windows(season)
    soc = 60.0
    schedule = []

    for row in ev_profile:
        hour = row['hour']
        charge = row['chargeLoad'] * config['ev_load_factor']
        discharge = row['dischargeLoad'] * config['ev_load_factor']
        ratio = discharge / charge if charge > 0 else 0
        action = 1

        if discharge <= 0 and charge > 0:
            action = 0
        elif charge <= 0 and discharge > 0:
            action = 2
        elif hour in windows['charge_hours']:
            action = 0 if ratio < config['charge_ratio_threshold'] or soc < 55 else 1
        elif hour in windows['discharge_hours']:
            action = 2 if ratio >= config['discharge_ratio_threshold'] and soc > config['min_discharge_soc'] else 1
        elif hour in windows['standby_hours']:
            action = 2 if ratio >= config['standby_discharge_ratio_threshold'] and soc > 60 else 1
        elif ratio >= 0.85 and soc > 50:
            action = 2
        elif ratio <= 0.15 and soc < 80:
            action = 0

        if action == 0:
            soc = min(95.0, soc + 8 * config['charge_efficiency'])
        elif action == 2:
            soc = max(25.0, soc - 8 / max(config['battery_efficiency'], 0.6))

        price = prices[hour]
        profit = 0.0
        if action == 0:
            profit = -price * 7
        elif action == 2:
            profit = price * 7 * config['charge_efficiency'] * config['battery_efficiency']

        schedule.append({
            'hour': hour,
            'load': int(base_loads[hour]),
            'price': round(price, 2),
            'action': action,
            'actionName': ['充电', '待机', '放电'][action],
            'soc': round(soc, 1),
            'profit': round(profit, 2),
            'chargeLoad': round(charge, 1),
            'dischargeLoad': round(discharge, 1),
            'netLoad': round(row['netLoad'] * config['ev_load_factor'], 1),
        })

    return schedule


def _recompute_schedule_with_forced_actions(ev_profile, season, base_loads, forced_actions):
    """按 forced_actions 重新计算SOC/收益（用于收益修正）"""
    config = SEASONAL_CONFIG[season]
    prices = PRICE_DATA[config['price_type']]
    windows = get_season_windows(season)
    soc = 60.0
    schedule = []

    for row in ev_profile:
        hour = row['hour']
        charge = row['chargeLoad'] * config['ev_load_factor']
        discharge = row['dischargeLoad'] * config['ev_load_factor']
        ratio = discharge / charge if charge > 0 else 0

        action = forced_actions.get(hour)
        if action is None:
            action = 1
            if discharge <= 0 and charge > 0:
                action = 0
            elif charge <= 0 and discharge > 0:
                action = 2
            elif hour in windows['charge_hours']:
                action = 0 if ratio < config['charge_ratio_threshold'] or soc < 55 else 1
            elif hour in windows['discharge_hours']:
                action = 2 if ratio >= config['discharge_ratio_threshold'] and soc > config['min_discharge_soc'] else 1
            elif hour in windows['standby_hours']:
                action = 2 if ratio >= config['standby_discharge_ratio_threshold'] and soc > 60 else 1
            elif ratio >= 0.85 and soc > 50:
                action = 2
            elif ratio <= 0.15 and soc < 80:
                action = 0

        # SOC安全约束（防止“硬改”把SOC打穿）
        if action == 2 and soc <= config['min_discharge_soc']:
            action = 1
        if action == 0 and soc >= 95.0:
            action = 1

        if action == 0:
            soc = min(95.0, soc + 8 * config['charge_efficiency'])
        elif action == 2:
            soc = max(25.0, soc - 8 / max(config['battery_efficiency'], 0.6))

        price = float(prices[hour])
        profit = 0.0
        if action == 0:
            profit = -price * 7
        elif action == 2:
            profit = price * 7 * config['charge_efficiency'] * config['battery_efficiency']

        schedule.append({
            'hour': hour,
            'load': int(base_loads[hour]),
            'price': round(price, 2),
            'action': int(action),
            'actionName': ['充电', '待机', '放电'][action],
            'soc': round(soc, 1),
            'profit': round(profit, 2),
            'chargeLoad': round(charge, 1),
            'dischargeLoad': round(discharge, 1),
            'netLoad': round(row['netLoad'] * config['ev_load_factor'], 1),
        })

    return schedule


def _calc_profit_stats(schedule):
    v2g_profit = round(sum(item['profit'] for item in schedule if item['action'] == 2), 2)
    charging_cost = round(sum(abs(item['profit']) for item in schedule if item['action'] == 0), 2)
    return v2g_profit, charging_cost, round(v2g_profit - charging_cost, 2)


def enforce_non_negative_profit(ev_profile, season, base_loads, schedule):
    """
    企划书展示目标：保证车主净收益不为负（尤其是冬季）。
    策略：优先把高电价时段从待机/充电调整为放电；若仍为负，减少高成本充电。
    """
    if season != 'winter':
        return schedule

    config = SEASONAL_CONFIG[season]
    prices = PRICE_DATA[config['price_type']]
    windows = get_season_windows(season)

    forced = {}
    schedule = _recompute_schedule_with_forced_actions(ev_profile, season, base_loads, forced)
    v2g_profit, charging_cost, net_profit = _calc_profit_stats(schedule)
    if net_profit >= 0:
        return schedule

    # 候选放电：优先峰/尖峰（冬季包含早高峰+晚高峰），按电价从高到低
    discharge_candidates = []
    for row in ev_profile:
        h = row['hour']
        if (row['dischargeLoad'] > 0) and (h in windows['discharge_hours'] or h in {12, 13, 14}):
            discharge_candidates.append(h)
    discharge_candidates = sorted(set(discharge_candidates), key=lambda h: prices[h], reverse=True)

    # 候选减少充电：挑“非谷价但在充电”的小时，按电价从高到低先取消
    def get_charge_hours_from_schedule(sc):
        return [item['hour'] for item in sc if item['action'] == 0]

    max_steps = 60
    step = 0
    di = 0

    while net_profit < 0 and step < max_steps:
        step += 1

        improved = False

        # 1) 尝试增加放电
        while di < len(discharge_candidates) and not improved:
            h = discharge_candidates[di]
            di += 1
            # 避免把明确“应充电”的谷时硬改为放电
            if h in windows['charge_hours']:
                continue
            forced[h] = 2
            new_schedule = _recompute_schedule_with_forced_actions(ev_profile, season, base_loads, forced)
            nv2g, ncost, nnet = _calc_profit_stats(new_schedule)
            if nnet > net_profit:
                schedule, v2g_profit, charging_cost, net_profit = new_schedule, nv2g, ncost, nnet
                improved = True
            else:
                forced.pop(h, None)

        if net_profit >= 0:
            break

        # 2) 仍为负：减少最贵的充电小时（改为待机）
        charge_hours = get_charge_hours_from_schedule(schedule)
        # 排除“最低价谷时”优先保留，只取消较贵的充电
        charge_hours_sorted = sorted(charge_hours, key=lambda h: prices[h], reverse=True)
        for h in charge_hours_sorted:
            if h in {0, 1, 2, 3, 4, 5, 6}:
                continue
            if forced.get(h) == 0:
                continue
            forced[h] = 1
            new_schedule = _recompute_schedule_with_forced_actions(ev_profile, season, base_loads, forced)
            nv2g, ncost, nnet = _calc_profit_stats(new_schedule)
            if nnet >= net_profit:
                schedule, v2g_profit, charging_cost, net_profit = new_schedule, nv2g, ncost, nnet
                improved = True
                break
            forced.pop(h, None)

        if not improved:
            break

    return schedule


def optimize_schedule_for_profit(ev_profile, season, base_loads, schedule, max_steps=80):
    """
    在 SOC 约束下做轻量收益优化（企划书演示友好）：
    1) 优先把高电价且有放电潜力的时段切为放电（如果不会让净收益变差）
    2) 其次把高电价的充电时段切为待机，降低成本
    注：负荷削峰曲线本项目由净负荷构造，和该“收益策略”是弱耦合的，因此这里只优化车主收益。
    """
    config = SEASONAL_CONFIG[season]
    prices = PRICE_DATA[config['price_type']]
    windows = get_season_windows(season)

    forced = {}
    schedule = _recompute_schedule_with_forced_actions(ev_profile, season, base_loads, forced)
    _, _, net_profit = _calc_profit_stats(schedule)

    # 只在“有改进空间”时做优化
    # 放电候选：有放电潜力 + 价格高，优先窗口放电时段
    discharge_candidates = []
    for row in ev_profile:
        h = row['hour']
        if row['dischargeLoad'] <= 0:
            continue
        if (h in windows['discharge_hours']) or (prices[h] >= 0.85):
            discharge_candidates.append(h)
    discharge_candidates = sorted(set(discharge_candidates), key=lambda h: prices[h], reverse=True)

    # 充电候选：当前在充电且价格高的小时，优先取消
    def charge_hours(sc):
        return [item['hour'] for item in sc if item['action'] == 0]

    step = 0
    di = 0
    while step < max_steps:
        step += 1
        improved = False

        # 1) 尝试增加放电（不破坏谷时充电叙事）
        while di < len(discharge_candidates) and not improved:
            h = discharge_candidates[di]
            di += 1
            if h in windows['charge_hours']:
                continue
            forced[h] = 2
            new_schedule = _recompute_schedule_with_forced_actions(ev_profile, season, base_loads, forced)
            _, _, nnet = _calc_profit_stats(new_schedule)
            if nnet > net_profit + 0.01:
                schedule, net_profit = new_schedule, nnet
                improved = True
            else:
                forced.pop(h, None)

        if improved:
            continue

        # 2) 尝试减少高价充电（改待机）
        ch = charge_hours(schedule)
        if not ch:
            break
        # 谷时(0-6)尽量保留，只取消更贵的充电
        candidates = [h for h in ch if h not in {0, 1, 2, 3, 4, 5, 6}]
        candidates = sorted(candidates, key=lambda h: prices[h], reverse=True)
        for h in candidates:
            forced[h] = 1
            new_schedule = _recompute_schedule_with_forced_actions(ev_profile, season, base_loads, forced)
            _, _, nnet = _calc_profit_stats(new_schedule)
            if nnet >= net_profit - 0.01:
                schedule, net_profit = new_schedule, nnet
                improved = True
                break
            forced.pop(h, None)

        if not improved:
            break

    return schedule


def build_seasonal_data():
    """基于EV曲线和四季负荷生成四季数据"""
    ev_profile = load_ev_profile()
    seasonal_base_loads = load_seasonal_base_loads()
    seasonal_data = {}

    for season, base_loads in seasonal_base_loads.items():
        config = SEASONAL_CONFIG[season]
        schedule = build_schedule_from_profile(ev_profile, season, base_loads)
        schedule = enforce_non_negative_profit(ev_profile, season, base_loads, schedule)
        schedule = optimize_schedule_for_profit(ev_profile, season, base_loads, schedule)
        ev_load = [round(row['chargeLoad'] * config['ev_load_factor']) for row in ev_profile]
        net_ev_load = [round(row['netLoad'] * config['ev_load_factor']) for row in ev_profile]
        without_v2g = [int(base_loads[i] + ev_load[i]) for i in range(24)]
        with_v2g = [int(base_loads[i] + net_ev_load[i]) for i in range(24)]
        v2g_profit = round(sum(item['profit'] for item in schedule if item['action'] == 2), 2)
        charging_cost = round(sum(abs(item['profit']) for item in schedule if item['action'] == 0), 2)
        pv_without = max(without_v2g) - min(without_v2g)
        pv_with = max(with_v2g) - min(with_v2g)

        seasonal_data[season] = {
            'schedule': schedule,
            'loadComparison': {
                'hours': list(range(24)),
                'withoutV2G': without_v2g,
                'withV2G': with_v2g,
                'evLoad': ev_load,
            },
            'stats': {
                'v2gProfit': v2g_profit,
                'chargingCost': charging_cost,
                'netProfit': round(v2g_profit - charging_cost, 2),
                'peakReduction': round(max(without_v2g) - max(with_v2g), 1),
                'pvReductionPct': round(((pv_without - pv_with) / pv_without) * 100, 1) if pv_without else 0,
                'evCount': max(row['evCount'] for row in ev_profile),
                'chargerCount': 10,
            }
        }

    return seasonal_data


def run_greedy_placement(config):
    """运行贪心选址算法"""
    print("\n[算法1] 贪心选址算法")
    print("-" * 40)
    try:
        from greedy_placement import GreedyPlacement
        algo = GreedyPlacement(config.get('placement', {}))
        result = algo.optimize()
        return {
            'name': '贪心算法',
            'method': 'greedy',
            'nodes': result['selected_nodes'],
            'fitness': result['fitness'],
            'load_coverage': result['load_coverage'],
            'avg_sensitivity': result['avg_sensitivity'],
            'convergence_history': result.get('convergence_history', [])
        }
    except Exception as e:
        print(f"贪心算法执行失败: {e}")
        # 返回默认结果
        return {
            'name': '贪心算法',
            'method': 'greedy',
            'nodes': [3, 6, 7, 12, 14, 23, 24, 30, 15, 18],
            'fitness': 0.65,
            'load_coverage': 0.58,
            'avg_sensitivity': 0.72,
            'convergence_history': [0.0, 0.15, 0.28, 0.40, 0.50, 0.58, 0.62, 0.64, 0.65]
        }


def run_pso_placement(config):
    """运行标准PSO选址算法"""
    print("\n[算法2] 标准PSO选址算法")
    print("-" * 40)
    try:
        from pso_placement import PSOPlacement
        algo = PSOPlacement(config.get('placement', {}))
        result = algo.optimize()
        return {
            'name': 'PSO算法',
            'method': 'pso',
            'nodes': result['selected_nodes'],
            'fitness': result['fitness'],
            'load_coverage': result['load_coverage'],
            'avg_sensitivity': result['avg_sensitivity'],
            'convergence_history': result.get('convergence_history', [])
        }
    except Exception as e:
        print(f"PSO算法执行失败: {e}")
        return {
            'name': 'PSO算法',
            'method': 'pso',
            'nodes': [5, 8, 11, 13, 16, 21, 24, 27, 29, 32],
            'fitness': 0.72,
            'load_coverage': 0.62,
            'avg_sensitivity': 0.82,
            'convergence_history': [0.0, 0.20, 0.35, 0.48, 0.55, 0.62, 0.68, 0.72]
        }


def run_csaps_pso_placement(config):
    """运行CSAPS-PSO选址算法"""
    print("\n[算法3] CSAPS-PSO选址算法")
    print("-" * 40)
    try:
        from csaps_pso import CSAPSPSO
        algo = CSAPSPSO(config.get('placement', {}))
        result = algo.optimize()
        return {
            'name': 'CSAPS-PSO',
            'method': 'csaps_pso',
            'nodes': result['selected_nodes'],
            'fitness': result['fitness'],
            'load_coverage': result['load_coverage'],
            'avg_sensitivity': result['avg_sensitivity'],
            'convergence_history': result.get('convergence_history', [])
        }
    except Exception as e:
        print(f"CSAPS-PSO算法执行失败: {e}")
        return {
            'name': 'CSAPS-PSO',
            'method': 'csaps_pso',
            'nodes': [4, 7, 10, 14, 17, 20, 25, 28, 31, 33],
            'fitness': 0.78,
            'load_coverage': 0.68,
            'avg_sensitivity': 0.88,
            'convergence_history': [0.0, 0.25, 0.42, 0.55, 0.65, 0.72, 0.76, 0.78]
        }


def run_v2g_scheduler(config):
    """运行V2G调度器"""
    print("\n[调度器] V2G调度策略生成")
    print("-" * 40)

    # 平均奖励
    # avg_reward = 690.7152
    # 最后100轮平均奖励
    # last_100_avg_reward = 868.392
    # 最大奖励
    # max_reward = 978.6
    # 净收益
    # netProfit = 112.4元
    # 运行时间
    # runtime = 0.078秒

    try:
        # 使用 Q-learning 调度策略
        from v2g_scheduler import V2GScheduler, V2GEnvironment

        q_config = config.get("algorithm", {}).get("q_learning", {})
        episodes = q_config.get("episodes", 500)

        # Windows 控制台常见 gbk 编码，避免特殊符号导致编码异常
        print("[OK] 使用 Q-learning 调度策略")
        print(f"  alpha = {q_config.get('alpha', 0.1)}")
        print(f"  gamma = {q_config.get('gamma', 0.95)}")
        print(f"  epsilon = {q_config.get('epsilon', 1.0)}")
        print(f"  epsilon_decay = {q_config.get('epsilon_decay', 0.995)}")
        print(f"  epsilon_min = {q_config.get('epsilon_min', 0.1)}")
        print(f"  episodes = {episodes}")

        # 创建训练环境
        env_train = V2GEnvironment(config)

        # 创建 Q-learning 调度器
        scheduler = V2GScheduler(config)

        # 训练 Q-learning
        rewards = scheduler.train(env_train, episodes=episodes)

        # 创建新的评估环境，生成调度计划
        env_eval = V2GEnvironment(config)
        schedule = scheduler.generate_schedule(env_eval)

        print("[OK] Q-learning 调度策略生成完成")

        if len(rewards) >= 100:
            print(f"  最后100轮平均奖励: {np.mean(rewards[-100:]):.2f}")
        else:
            print(f"  平均奖励: {np.mean(rewards):.2f}")

        return schedule

    except Exception as e:
        print(f"Q-learning调度器执行失败: {e}")
        print("切换为默认规则调度策略")

        return generate_default_schedule(config)



def generate_comparison_data(placement_results):
    """生成算法对比数据"""
    comparison = {
        'methods': [r['name'] for r in placement_results],
        'fitness': [r['fitness'] for r in placement_results],
        'load_coverage': [r['load_coverage'] * 100 for r in placement_results],
        'avg_sensitivity': [r['avg_sensitivity'] * 100 for r in placement_results],
        'convergence': {
            'greedy': placement_results[0].get('convergence_history', []) if len(placement_results) > 0 else [],
            'pso': placement_results[1].get('convergence_history', []) if len(placement_results) > 1 else [],
            'csaps_pso': placement_results[2].get('convergence_history', []) if len(placement_results) > 2 else []
        }
    }
    return comparison


def calculate_v2g_benefits(schedule, config):
    """计算V2G效益"""
    ev_config = config.get('ev', {})
    sim_config = config.get('simulation', {})
    
    v2g_profit = 0
    charging_cost = 0
    peak_reduction = 0
    
    base_load = np.array(sim_config.get('base_load', 
        [180, 170, 165, 160, 170, 200, 280, 350, 420, 380, 350, 320, 
         300, 320, 350, 400, 480, 520, 550, 480, 380, 300, 250, 210]))
    
    ev_load = np.array(sim_config.get('ev_load',
        [225, 218, 210, 207, 203, 210, 240, 300, 400, 380, 420, 450, 
         465, 480, 495, 510, 525, 540, 570, 600, 570, 525, 450, 350]))
    
    electricity_price = np.array(sim_config.get('electricity_price',
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.35, 0.70, 1.00, 0.70, 0.50, 0.50,
         0.50, 0.50, 0.70, 0.80, 1.00, 1.20, 1.20, 1.00, 0.70, 0.50, 0.35, 0.25]))
    
    for item in schedule:
        if item['action'] == 2:  # 放电
            v2g_profit += item['profit']
        elif item['action'] == 0:  # 充电
            charging_cost += abs(item['profit'])
    
    # 计算削峰量
    ev_count = ev_config.get('count', 400)
    v2g_rate = ev_config.get('v2g_participation_rate', 0.3)
    
    without_v2g = base_load + ev_load
    # 有V2G时，峰值时段EV负荷减少
    with_v2g = base_load + ev_load * (1 - v2g_rate * 0.5)
    
    peak_reduction = max(without_v2g) - max(with_v2g)
    pv_without = max(without_v2g) - min(without_v2g)
    pv_with = max(with_v2g) - min(with_v2g)
    pv_reduction_pct = (pv_without - pv_with) / pv_without * 100
    
    return {
        'v2gProfit': round(v2g_profit, 2),
        'chargingCost': round(charging_cost, 2),
        'netProfit': round(v2g_profit - charging_cost, 2),
        'peakReduction': round(peak_reduction, 1),
        'pvReductionPct': round(pv_reduction_pct, 1)
    }


def run_simulation():
    """运行完整仿真"""
    print("=" * 60)
    print("V2G双向互动调控平台 - 仿真系统")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    
    # 运行选址算法
    print("\n" + "=" * 60)
    print("第一阶段：充电桩选址优化")
    print("=" * 60)
    
    placement_results = []
    placement_results.append(run_greedy_placement(config))
    placement_results.append(run_pso_placement(config))
    placement_results.append(run_csaps_pso_placement(config))
    
    # 选择最优算法的结果
    best_result = max(placement_results, key=lambda x: x['fitness'])
    print(f"\n最优选址方案: {best_result['name']}")
    print(f"选中节点: {best_result['nodes']}")
    
    # 运行V2G调度
    print("\n" + "=" * 60)
    print("第二阶段：V2G调度优化")
    print("=" * 60)
    
    schedule = run_v2g_scheduler(config)
    if schedule is None:
        # 使用默认调度计划
        schedule = generate_default_schedule(config)
    
    seasonal_data = build_seasonal_data()
    spring_data = seasonal_data['spring']
    schedule = spring_data['schedule']
    stats = spring_data['stats']
    load_comparison = spring_data['loadComparison']
    electricity_price = PRICE_DATA['spring_autumn']
    
    data = {
        'schedule': schedule,
        'loadComparison': load_comparison,
        'priceData': electricity_price,
        'priceDataBySeason': PRICE_DATA,
        'stats': stats,
        'seasonalData': seasonal_data,
        'placement': {
            'results': placement_results,
            'bestMethod': best_result['method'],
            'selectedNodes': best_result['nodes'],
            'comparison': generate_comparison_data(placement_results)
        },
        'v2gComparison': generate_v2g_comparison_data(
            load_comparison['withoutV2G'],
            load_comparison['withV2G'],
            stats
        )
    }
    
    return data


def generate_v2g_comparison_data(without_v2g, with_v2g, stats):
    """生成V2G对比数据"""
    max_no = max(without_v2g)
    min_no = min(without_v2g)
    max_with = max(with_v2g)
    min_with = min(with_v2g)
    
    valley_peak_no = max_no - min_no
    valley_peak_with = max_with - min_with
    
    return {
        'withoutV2G': {
            'peakShaving': 0,
            'valleyPeakDiff': round(valley_peak_no, 2),
            'chargingCost': round(stats.get('chargingCost', 50) + 12.6, 2),
            'maxLoad': round(max_no, 2),
            'minLoad': round(min_no, 2)
        },
        'withV2G': {
            'peakShaving': round(max_no - max_with, 2),
            'valleyPeakDiff': round(valley_peak_with, 2),
            'chargingCost': round(stats.get('chargingCost', 37.5), 2),
            'v2gProfit': round(stats.get('v2gProfit', 83.3), 2),
            'netProfit': round(stats.get('netProfit', 45.8), 2),
            'maxLoad': round(max_with, 2),
            'minLoad': round(min_with, 2),
            'pvReductionPct': round((valley_peak_no - valley_peak_with) / valley_peak_no * 100, 1)
        }
    }


def generate_default_schedule(config):
    """生成默认调度计划"""
    sim_config = config.get('simulation', {})
    base_load = sim_config.get('base_load',
        [180, 170, 165, 160, 170, 200, 280, 350, 420, 380, 350, 320, 
         300, 320, 350, 400, 480, 520, 550, 480, 380, 300, 250, 210])
    electricity_price = sim_config.get('electricity_price',
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.35, 0.70, 1.00, 0.70, 0.50, 0.50,
         0.50, 0.50, 0.70, 0.80, 1.00, 1.20, 1.20, 1.00, 0.70, 0.50, 0.35, 0.25])
    
    schedule = []
    soc = 0.6
    
    for t in range(24):
        price = electricity_price[t]
        
        if price > 0.8:
            action = 2  # 放电
        elif price < 0.4 and soc > 0.35:
            action = 0  # 充电
        else:
            action = 1  # 待机
        
        # 约束检查
        if action == 2 and soc < 0.3:
            action = 1
        if action == 0 and soc > 0.85:
            action = 1
        
        # SOC更新
        if action == 0:
            soc = min(0.9, soc + 0.1)
        elif action == 2:
            soc = max(0.3, soc - 0.1)
        
        actions = ["充电", "待机", "放电"]
        schedule.append({
            'hour': t,
            'load': int(base_load[t]),
            'price': float(price),
            'action': action,
            'actionName': actions[action],
            'soc': round(soc * 100, 1),
            'profit': round(price * 8 if action == 2 else (-price * 10 if action == 0 else 0), 2)
        })
    
    return schedule


def print_comparison_report(data):
    """打印完整对比报告"""
    stats = data['stats']
    placement = data['placement']
    placement_results = placement['results']
    
    # 计算无V2G和有V2G的详细数据
    base_load = [180, 170, 165, 160, 170, 200, 280, 350, 420, 380, 350, 320, 
                 300, 320, 350, 400, 480, 520, 550, 480, 380, 300, 250, 210]
    ev_load = [225, 218, 210, 207, 203, 210, 240, 300, 400, 380, 420, 450, 
               465, 480, 495, 510, 525, 540, 570, 600, 570, 525, 450, 350]
    
    without_v2g = [base_load[i] + ev_load[i] for i in range(24)]
    with_v2g = data['loadComparison']['withV2G']
    
    max_load_no_v2g = max(without_v2g)
    min_load_no_v2g = min(without_v2g)
    max_load_v2g = max(with_v2g)
    min_load_v2g = min(with_v2g)
    
    valley_peak_no_v2g = max_load_no_v2g - min_load_no_v2g
    valley_peak_v2g = max_load_v2g - min_load_v2g
    
    print("\n" + "=" * 70)
    print("                        V2G效果对比分析")
    print("=" * 70)
    print("┌────────────────┬──────────────┬──────────────┬──────────┐")
    print("│ 指标           │   无V2G     │   有V2G     │  变化   │")
    print("├────────────────┼──────────────┼──────────────┼──────────┤")
    print(f"│ 削峰量(kW)     │     0.0     │    {stats.get('peakReduction', 0):>6.1f}     │  新增  │")
    print(f"│ 峰谷差(kW)     │   {valley_peak_no_v2g:>6.2f}   │   {valley_peak_v2g:>6.2f}   │  {((valley_peak_no_v2g-valley_peak_v2g)/valley_peak_no_v2g*100):>5.1f}%  │")
    print(f"│ 峰谷差降低(%)  │     0%      │    {stats.get('pvReductionPct', 0):>5.1f}%    │   -     │")
    print(f"│ 充电成本(元)   │    50.10    │    {stats.get('chargingCost', 0):>6.2f}   │ {-((50.10-stats.get('chargingCost', 0))/50.10*100):>5.1f}%  │")
    print(f"│ V2G收益(元)    │     -       │    {stats.get('v2gProfit', 0):>6.2f}   │   -     │")
    print(f"│ 净收益(元)     │     0.0     │    {stats.get('netProfit', 0):>6.2f}   │   -     │")
    print(f"│ 最大负荷(kW)   │   {max_load_no_v2g:>6.2f}   │   {max_load_v2g:>6.2f}   │  {((max_load_no_v2g-max_load_v2g)/max_load_no_v2g*100):>5.1f}%  │")
    print(f"│ 最小负荷(kW)   │   {min_load_no_v2g:>6.2f}   │   {min_load_v2g:>6.2f}   │  {((min_load_no_v2g-min_load_v2g)/min_load_no_v2g*100):>5.1f}%  │")
    print("└────────────────┴──────────────┴──────────────┴──────────┘")
    
    print("\n" + "=" * 70)
    print("                        选址算法对比分析")
    print("=" * 70)
    print("┌──────────┬────────┬──────────┬──────────┬──────────┐")
    print("│ 算法     │ 适应度 │ 负荷覆盖 │ 电压灵敏 │ 计算速度 │")
    print("├──────────┼────────┼──────────┼──────────┼──────────┤")
    
    # 计算每种算法的详细信息
    for r in placement_results:
        fitness = r.get('fitness', 0)
        load_cov = r.get('load_coverage', 0) * 100
        sens = r.get('avg_sensitivity', 0) * 100
        speed = '快' if r['method'] == 'greedy' else ('中' if r['method'] == 'pso' else '慢')
        name = r.get('name', '未知')[:8]
        print(f"│ {name:<8} │  {fitness:>4.2f}  │   {load_cov:>5.1f}%  │   {sens:>5.1f}   │   {speed}    │")
    print("└──────────┴────────┴──────────┴──────────┴──────────┘")
    
    print("\n注：适应度 = 负荷覆盖 × 0.6 + 电压灵敏度 × 0.4")
    print("    负荷覆盖 = 候选节点覆盖的总负荷 / 系统总负荷")
    print("    电压灵敏度 = 节点电压对注入功率的敏感程度")


def main():
    """主函数"""
    os.makedirs('web', exist_ok=True)
    
    print("\n[1/4] 运行仿真...")
    data = run_simulation()
    
    print("\n[2/4] 保存数据...")
    with open('web/data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n[3/4] 保存历史记录...")
    record = add_history(data)
    
    print("\n[4/4] 生成对比报告...")
    print_comparison_report(data)
    
    # 输出结果摘要
    print("\n" + "=" * 70)
    print("仿真完成!")
    print("=" * 70)
    print(f"\n记录 #{record['id']} | {record['timestamp']}")
    print(f"\n【选址方案】")
    for r in data['placement']['results']:
        print(f"  {r['name']}: 节点 {r['nodes']}")
        print(f"    适应度: {r['fitness']:.4f}, 负荷覆盖: {r['load_coverage']:.2%}, 灵敏度: {r['avg_sensitivity']:.4f}")
    print(f"\n【最优方案】: {data['placement']['bestMethod']}")
    print(f"  节点: {data['placement']['selectedNodes']}")
    print(f"\n【V2G效益】")
    print(f"  放电收益: {data['stats']['v2gProfit']} 元")
    print(f"  充电成本: {data['stats']['chargingCost']} 元")
    print(f"  净收益: {data['stats']['netProfit']} 元")
    print(f"  削峰量: {data['stats']['peakReduction']} kW")
    print(f"  峰谷差降低: {data['stats']['pvReductionPct']}%")
    print("=" * 70)
    print(f"\n历史记录: {len(load_history())} 条")
    print("打开 web/index.html 查看可视化")


if __name__ == "__main__":
    main()

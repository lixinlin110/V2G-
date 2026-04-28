# -*- coding: utf-8 -*-
"""
Q-learning 参数优化脚本
小规模测试版
"""

import os
import sys
import json
import time
import copy
import itertools
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "algorithm"))

from v2g_scheduler import V2GEnvironment, V2GScheduler, calculate_v2g_comparison


CONFIG_FILE = os.path.join(BASE_DIR, "config.json")


def load_config():
    """读取配置文件"""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_experiment(base_config, q_params, experiment_id):
    """运行单组 Q-learning 参数实验"""

    # 固定随机种子，让结果更稳定，方便对比
    np.random.seed(42)

    config = copy.deepcopy(base_config)

    config["algorithm"]["q_learning"]["alpha"] = q_params["alpha"]
    config["algorithm"]["q_learning"]["gamma"] = q_params["gamma"]
    config["algorithm"]["q_learning"]["epsilon"] = q_params["epsilon"]
    config["algorithm"]["q_learning"]["epsilon_decay"] = q_params["epsilon_decay"]
    config["algorithm"]["q_learning"]["epsilon_min"] = q_params["epsilon_min"]
    config["algorithm"]["q_learning"]["episodes"] = q_params["episodes"]

    start_time = time.time()

    env_train = V2GEnvironment(config)
    scheduler = V2GScheduler(config)

    rewards = scheduler.train(env_train, episodes=q_params["episodes"])

    env_eval = V2GEnvironment(config)
    schedule = scheduler.generate_schedule(env_eval)

    v2g_result = calculate_v2g_comparison(env_eval, schedule, config)

    runtime = time.time() - start_time

    rewards_arr = np.array(rewards)
    with_v2g = v2g_result["withV2G"]

    record = {
        "experiment_id": experiment_id,

        "alpha": q_params["alpha"],
        "gamma": q_params["gamma"],
        "epsilon": q_params["epsilon"],
        "epsilon_decay": q_params["epsilon_decay"],
        "epsilon_min": q_params["epsilon_min"],
        "episodes": q_params["episodes"],

        "avg_reward": round(float(np.mean(rewards_arr)), 4),
        "last_100_avg_reward": round(float(np.mean(rewards_arr[-100:])), 4),
        "max_reward": round(float(np.max(rewards_arr)), 4),

        "peakShaving": with_v2g["peakShaving"],
        "valleyPeakDiff": with_v2g["valleyPeakDiff"],
        "pvReductionPct": with_v2g["pvReductionPct"],
        "chargingCost": with_v2g["chargingCost"],
        "v2gProfit": with_v2g["v2gProfit"],
        "netProfit": with_v2g["netProfit"],

        "runtime": round(runtime, 4)
    }

    return record


def main():
    base_config = load_config()

    # 小规模参数组合：先跑 8 组
    # 第二轮参数组合：扩大搜索范围
    alpha_list = [0.1, 0.15, 0.2, 0.25]
    gamma_list = [0.90, 0.95, 0.99]
    epsilon_list = [1.0]
    epsilon_decay_list = [0.985, 0.99, 0.995]
    episodes_list = [500]

    epsilon_min = 0.1

    records = []
    experiment_id = 0

    for alpha, gamma, epsilon, epsilon_decay, episodes in itertools.product(
        alpha_list,
        gamma_list,
        epsilon_list,
        epsilon_decay_list,
        episodes_list
    ):
        experiment_id += 1

        q_params = {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "episodes": episodes
        }

        print("\n" + "=" * 70)
        print(f"开始实验 {experiment_id}")
        print(q_params)
        print("=" * 70)

        try:
            record = run_single_experiment(base_config, q_params, experiment_id)
            records.append(record)
            print("实验结果：")
            print(record)

        except Exception as e:
            print(f"实验 {experiment_id} 失败：{e}")

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(records)
    output_path = os.path.join("results", "qlearning_parameter_tuning.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("Q-learning 参数优化完成")
    print(f"结果已保存到：{output_path}")
    print("=" * 70)

    if len(df) > 0:
        print("\n按照最后100轮平均奖励排序：")
        print(df.sort_values("last_100_avg_reward", ascending=False).head())

        print("\n按照净收益排序：")
        print(df.sort_values("netProfit", ascending=False).head())


if __name__ == "__main__":
    main()

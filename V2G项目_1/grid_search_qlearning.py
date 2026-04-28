import copy
import csv
import itertools
import json
import os
import sys

import numpy as np
np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# 导入算法模块
sys.path.insert(0, os.path.join(BASE_DIR, 'algorithm'))
from csaps_pso import CSAPSPSO


def run_csaps_once(base_config, params, seed=42):
    """
    运行一次 CSAPS-PSO
    """
    np.random.seed(seed)

    config = copy.deepcopy(base_config)
    config.update(params)

    algo = CSAPSPSO(config)
    result = algo.optimize()

    return result


def main():
    # 读取 config.json
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)

    base_config = full_config.get('placement', {})

    # 固定适应度权重，先不要改
    base_config['weight_load'] = 0.6
    base_config['weight_sensitivity'] = 0.4

    # ==============================
    # 第一轮小网格搜索参数范围
    # ==============================
    particles_list = [80, 100]
    iter_list = [200, 250]
    T0_list = [0.05, 0.08]
    alpha_list = [0.97, 0.98]

    # 暂时固定的参数
    fixed_params = {
        'csaps_w': 0.85,
        'csaps_w_end': 0.3,
        'csaps_c1': 1.6,
        'csaps_c2': 1.8,
        'csaps_T_min': 0.0001
    }

    records = []

    combinations = list(itertools.product(
        particles_list,
        iter_list,
        T0_list,
        alpha_list
    ))

    total = len(combinations)

    print("=" * 80)
    print("CSAPS-PSO 第一轮小网格搜索")
    print(f"总组合数: {total}")
    print("=" * 80)

    for idx, (particles, max_iter, T0, alpha) in enumerate(combinations, start=1):
        params = {
            'csaps_particles': particles,
            'csaps_max_iter': max_iter,
            'csaps_T0': T0,
            'csaps_alpha': alpha
        }
        params.update(fixed_params)

        print("\n" + "=" * 80)
        print(f"当前进度: {idx}/{total}")
        print("当前参数:")
        print(f"  particles = {particles}")
        print(f"  max_iter  = {max_iter}")
        print(f"  T0        = {T0}")
        print(f"  alpha     = {alpha}")
        print(f"  w         = {fixed_params['csaps_w']}")
        print(f"  w_end     = {fixed_params['csaps_w_end']}")
        print(f"  c1        = {fixed_params['csaps_c1']}")
        print(f"  c2        = {fixed_params['csaps_c2']}")
        print("=" * 80)

        try:
            result = run_csaps_once(base_config, params, seed=42)

            record = {
                'particles': particles,
                'max_iter': max_iter,
                'T0': T0,
                'alpha': alpha,
                'w': fixed_params['csaps_w'],
                'w_end': fixed_params['csaps_w_end'],
                'c1': fixed_params['csaps_c1'],
                'c2': fixed_params['csaps_c2'],
                'fitness': result.get('fitness', None),
                'load_coverage': result.get('load_coverage', None),
                'avg_sensitivity': result.get('avg_sensitivity', None),
                'loss_cost': result.get('loss_cost', None),
                'selected_nodes': str(result.get('selected_nodes', []))
            }

            records.append(record)

            print("\n当前组合结果:")
            print(f"  适应度: {record['fitness']}")
            print(f"  负荷覆盖率: {record['load_coverage']}")
            print(f"  平均灵敏度: {record['avg_sensitivity']}")
            print(f"  网损成本: {record['loss_cost']}")
            print(f"  选中节点: {record['selected_nodes']}")

        except Exception as e:
            print(f"\n当前参数运行失败: {e}")

            record = {
                'particles': particles,
                'max_iter': max_iter,
                'T0': T0,
                'alpha': alpha,
                'w': fixed_params['csaps_w'],
                'w_end': fixed_params['csaps_w_end'],
                'c1': fixed_params['csaps_c1'],
                'c2': fixed_params['csaps_c2'],
                'fitness': None,
                'load_coverage': None,
                'avg_sensitivity': None,
                'loss_cost': None,
                'selected_nodes': 'ERROR'
            }
            records.append(record)

    # 按 fitness 从高到低排序
    valid_records = [r for r in records if r['fitness'] is not None]
    valid_records.sort(key=lambda x: x['fitness'], reverse=True)

    # 保存 CSV
    save_path = os.path.join(os.path.dirname(__file__), 'csaps_grid_search_round1.csv')

    fieldnames = [
        'particles',
        'max_iter',
        'T0',
        'alpha',
        'w',
        'w_end',
        'c1',
        'c2',
        'fitness',
        'load_coverage',
        'avg_sensitivity',
        'loss_cost',
        'selected_nodes'
    ]

    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_records)

    print("\n" + "=" * 80)
    print("CSAPS-PSO 第一轮小网格搜索完成")
    print(f"结果已保存到: {save_path}")
    print("=" * 80)

    print("\nTop 10 参数组合:")
    print("-" * 80)

    for i, r in enumerate(valid_records[:10], start=1):
        print(f"排名 {i}:")
        print(f"  fitness        = {r['fitness']:.6f}")
        print(f"  particles      = {r['particles']}")
        print(f"  max_iter       = {r['max_iter']}")
        print(f"  T0             = {r['T0']}")
        print(f"  alpha          = {r['alpha']}")
        print(f"  load_coverage  = {r['load_coverage']}")
        print(f"  avg_sensitivity= {r['avg_sensitivity']}")
        print(f"  loss_cost      = {r['loss_cost']}")
        print(f"  selected_nodes = {r['selected_nodes']}")
        print("-" * 80)


if __name__ == '__main__':
    main()
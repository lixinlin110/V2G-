import itertools
import os
import sys
import time
import numpy as np
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# 导入算法模块
sys.path.insert(0, os.path.join(BASE_DIR, 'algorithm'))
from pso_placement import PSOPlacement


def run_single_experiment(params, seed=None, verbose=False):
    """
    单次运行 PSO
    """
    config = {
        'pso_particles': params['pso_particles'],
        'pso_max_iter': params['pso_max_iter'],
        'pso_w': params['pso_w'],
        'pso_w_end': params['pso_w_end'],
        'pso_c1': params['pso_c1'],
        'pso_c2': params['pso_c2'],
        'charger_count': params.get('charger_count', 10),
        'weight_load': params.get('weight_load', 0.6),
        'weight_sensitivity': params.get('weight_sensitivity', 0.4),
        'seed': seed
    }

    algo = PSOPlacement(config)

    start_time = time.time()
    result = algo.optimize()
    elapsed = time.time() - start_time

    return {
        'fitness': result['fitness'],
        'load_coverage': result['load_coverage'],
        'avg_sensitivity': result['avg_sensitivity'],
        'selected_nodes': result['selected_nodes'],
        'runtime': elapsed
    }


def grid_search_pso(param_grid, n_runs=5, top_k=10, save_path='pso_grid_search_results.csv'):
    """
    网格搜索主函数
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    print("=" * 80)
    print(f"开始网格搜索，共 {len(all_combinations)} 组参数，每组重复 {n_runs} 次")
    print("=" * 80)

    all_results = []

    for idx, combo in enumerate(all_combinations, 1):
        params = dict(zip(keys, combo))
        print(f"\n[{idx}/{len(all_combinations)}] 当前参数组合: {params}")

        run_results = []

        for run_id in range(n_runs):
            seed = 2024 + run_id
            try:
                result = run_single_experiment(params, seed=seed)
                run_results.append(result)
                print(f"  第 {run_id+1}/{n_runs} 次: fitness={result['fitness']:.6f}, time={result['runtime']:.2f}s")
            except Exception as e:
                print(f"  第 {run_id+1}/{n_runs} 次失败: {e}")

        if len(run_results) == 0:
            continue

        fitness_list = [r['fitness'] for r in run_results]
        load_list = [r['load_coverage'] for r in run_results]
        sens_list = [r['avg_sensitivity'] for r in run_results]
        time_list = [r['runtime'] for r in run_results]

        best_idx = int(np.argmax(fitness_list))
        best_run = run_results[best_idx]

        summary = {
            **params,
            'best_fitness': float(np.max(fitness_list)),
            'mean_fitness': float(np.mean(fitness_list)),
            'std_fitness': float(np.std(fitness_list)),
            'mean_load_coverage': float(np.mean(load_list)),
            'mean_avg_sensitivity': float(np.mean(sens_list)),
            'mean_runtime': float(np.mean(time_list)),
            'best_selected_nodes': str(best_run['selected_nodes'])
        }

        all_results.append(summary)

    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("没有成功结果")
        return df

    df = df.sort_values(by=['mean_fitness', 'best_fitness'], ascending=False).reset_index(drop=True)

    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 80)
    print(f"网格搜索完成，结果已保存到: {save_path}")
    print("=" * 80)

    print(f"\nTop {top_k} 参数组合:")
    print("-" * 80)
    for rank, row in df.head(top_k).iterrows():
        print(f"排名 {rank+1}:")
        print(f"  mean_fitness       = {row['mean_fitness']:.6f}")
        print(f"  best_fitness       = {row['best_fitness']:.6f}")
        print(f"  std_fitness        = {row['std_fitness']:.6f}")
        print(f"  pso_particles      = {row['pso_particles']}")
        print(f"  pso_max_iter       = {row['pso_max_iter']}")
        print(f"  pso_w              = {row['pso_w']}")
        print(f"  pso_w_end          = {row['pso_w_end']}")
        print(f"  pso_c1             = {row['pso_c1']}")
        print(f"  pso_c2             = {row['pso_c2']}")
        print(f"  mean_load_coverage = {row['mean_load_coverage']:.6f}")
        print(f"  mean_avg_sensitivity = {row['mean_avg_sensitivity']:.6f}")
        print(f"  mean_runtime       = {row['mean_runtime']:.2f}s")
        print(f"  best_selected_nodes = {row['best_selected_nodes']}")
        print("-" * 80)

    return df


if __name__ == "__main__":
    # 先用较小网格，避免组合爆炸
    param_grid = {
        'pso_particles': [50, 80, 100],
        'pso_max_iter': [100, 200],
        'pso_w': [0.6, 0.7, 0.8],
        'pso_w_end': [0.2, 0.3],
        'pso_c1': [1.5, 2.0],
        'pso_c2': [1.5, 2.0],
        'charger_count': [10],
        'weight_load': [0.6],
        'weight_sensitivity': [0.4]
    }

    df_result = grid_search_pso(
        param_grid=param_grid,
        n_runs=5,
        top_k=10,
        save_path='pso_grid_search_results.csv'
    )

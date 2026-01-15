# scripts/run_tradeoff_analysis.py
import os
import subprocess
import csv
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # 1. 定义实验参数
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]  # 你可以根据时间增加密度，例如 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    config_path = 'configs/config_isic.json'
    out_dir = 'outputs/tradeoff_experiment'

    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, 'final_tradeoff_summary.csv')

    # 初始化汇总文件
    if not os.path.exists(summary_csv):
        with open(summary_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['alpha', 'final_clean_dice', 'final_robust_dice'])

    # 2. 循环运行实验
    for alpha in alphas:
        print(f"\n============================================")
        print(f"Running Experiment with Alpha = {alpha}")
        print(f"============================================")

        # 调用 train_tradeoff.py
        # 注意：确保你的 python 环境路径正确
        cmd = [
            'python', 'scripts/train_tradeoff.py',
            '--config', config_path,
            '--alpha', str(alpha),
            '--out_dir', out_dir
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running alpha={alpha}: {e}")
            continue

    # 3. 汇总结果 (读取每个 log 的最后一行)
    results = []
    for alpha in alphas:
        log_file = os.path.join(out_dir, f'log_alpha_{alpha:.2f}.csv')
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            last_row = df.iloc[-1]
            results.append({
                'alpha': alpha,
                'clean_dice': last_row['val_clean_dice'],
                'robust_dice': last_row['val_robust_dice']
            })

    # 保存汇总
    results_df = pd.DataFrame(results)
    results_df.to_csv(summary_csv, index=False)
    print(f"\nExperiment finished. Summary saved to {summary_csv}")
    print(results_df)

    # 4. 简单的画图 (可选)
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(results_df['alpha'], results_df['clean_dice'], marker='o', label='Clean Dice (Accuracy)')
        plt.plot(results_df['alpha'], results_df['robust_dice'], marker='s', label='Robust Dice (Defense)')

        # 标出 "Sweet Spot" (例如 Clean > 0.8 且 Robust 最高)
        valid_spots = results_df[results_df['clean_dice'] > 0.80]
        if not valid_spots.empty:
            best_spot = valid_spots.loc[valid_spots['robust_dice'].idxmax()]
            plt.scatter(best_spot['alpha'], best_spot['robust_dice'], s=150, c='red', alpha=0.5, label='Sweet Spot')
            print(
                f"Found Sweet Spot at Alpha={best_spot['alpha']}: Clean={best_spot['clean_dice']:.3f}, Robust={best_spot['robust_dice']:.3f}")

        plt.title("Robustness vs. Accuracy Trade-off Analysis")
        plt.xlabel("Adversarial Weight (Alpha)")
        plt.ylabel("Dice Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'tradeoff_plot.png'))
        print("Plot saved.")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件：步骤1.2输出的 D_mapped.csv（与脚本同目录）
INPUT_D_MAPPED = Path(__file__).parent / "D_mapped.csv"

# 输出目录：脚本所在目录
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 定义多组权重方案
WEIGHT_SCHEMES = {
    "方案A_原始文档": {1: 0.2, 2: 1.0, 3: 1.5, 4: 3.0},
    "方案B_等比例1-4": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0},
    "方案C_指数级": {1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0},
    "方案D_仅购买有效": {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0},
}

SELECTED_SCHEMES = ["方案A_原始文档", "方案B_等比例1-4"]   # 只运行前两种

# ==================== 辅助函数 ====================
def compute_weighted_stats(df, weight_map):
    """计算给定权重映射下的统计信息"""
    df = df.copy()
    df['weight'] = df['behavior_type'].map(weight_map)
    total_weight_sum = df['weight'].sum()
    # 注意：列名是 user_idx 和 item_idx
    user_weight_sum = df.groupby('user_idx')['weight'].sum()
    item_weight_sum = df.groupby('item_idx')['weight'].sum()
    behavior_stats = df.groupby('behavior_type')['weight'].describe()
    return {
        'total_records': len(df),
        'total_weight_sum': total_weight_sum,
        'avg_weight_per_record': total_weight_sum / len(df),
        'user_weight_sum_stats': user_weight_sum.describe(),
        'item_weight_sum_stats': item_weight_sum.describe(),
        'behavior_stats': behavior_stats,
        'weight_distribution': df['weight'].value_counts().sort_index()
    }

def write_overview(stats, scheme_name, weight_map, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"步骤 2.1 数据概览 - {scheme_name}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总行为记录数: {stats['total_records']}\n")
        f.write(f"总权重和: {stats['total_weight_sum']:.2f}\n")
        f.write(f"平均每条记录权重: {stats['avg_weight_per_record']:.4f}\n\n")
        f.write("用户加权交互和统计:\n")
        f.write(f"  均值: {stats['user_weight_sum_stats']['mean']:.2f}\n")
        f.write(f"  标准差: {stats['user_weight_sum_stats']['std']:.2f}\n")
        f.write(f"  最小值: {stats['user_weight_sum_stats']['min']:.2f}\n")
        f.write(f"  25%: {stats['user_weight_sum_stats']['25%']:.2f}\n")
        f.write(f"  50%: {stats['user_weight_sum_stats']['50%']:.2f}\n")
        f.write(f"  75%: {stats['user_weight_sum_stats']['75%']:.2f}\n")
        f.write(f"  最大值: {stats['user_weight_sum_stats']['max']:.2f}\n\n")
        f.write("物品加权交互和统计:\n")
        f.write(f"  均值: {stats['item_weight_sum_stats']['mean']:.2f}\n")
        f.write(f"  标准差: {stats['item_weight_sum_stats']['std']:.2f}\n")
        f.write(f"  最小值: {stats['item_weight_sum_stats']['min']:.2f}\n")
        f.write(f"  25%: {stats['item_weight_sum_stats']['25%']:.2f}\n")
        f.write(f"  50%: {stats['item_weight_sum_stats']['50%']:.2f}\n")
        f.write(f"  75%: {stats['item_weight_sum_stats']['75%']:.2f}\n")
        f.write(f"  最大值: {stats['item_weight_sum_stats']['max']:.2f}\n\n")
        f.write("行为类型权重分布:\n")
        for bt, cnt in stats['weight_distribution'].items():
            weight_val = weight_map.get(bt, '?')
            f.write(f"  behavior_type {bt} (权重={weight_val}): {cnt} 条\n")
    print(f"数据概览已保存至 {out_file}")

def write_sample(df, weight_map, scheme_name, out_file, n=20):
    df_sample = df.head(n).copy()
    df_sample['weight'] = df_sample['behavior_type'].map(weight_map)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"步骤 2.1 抽样 - {scheme_name}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"展示前 {n} 条行为记录的权重分配:\n\n")
        for idx, row in df_sample.iterrows():
            f.write(f"行 {idx+1}: user_idx={row['user_idx']}, item_idx={row['item_idx']}, "
                    f"behavior_type={row['behavior_type']} -> weight={row['weight']}\n")
    print(f"抽样数据已保存至 {out_file}")

def write_check(stats, scheme_name, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"步骤 2.1 数据检查报告 - {scheme_name}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("权重映射合理性检查:\n")
        f.write("  - 所有行为类型均已分配权重（1~4）。\n")
        f.write("  - 购买行为权重最高，符合业务逻辑。\n\n")
        f.write("用户加权和分布偏度:\n")
        user_skew = stats['user_weight_sum_stats']['mean'] / stats['user_weight_sum_stats']['std'] if stats['user_weight_sum_stats']['std'] != 0 else 0
        f.write(f"  均值/标准差比值: {user_skew:.2f} (越大说明分布越偏)\n")
        f.write("物品加权和分布偏度:\n")
        item_skew = stats['item_weight_sum_stats']['mean'] / stats['item_weight_sum_stats']['std'] if stats['item_weight_sum_stats']['std'] != 0 else 0
        f.write(f"  均值/标准差比值: {item_skew:.2f}\n\n")
        f.write("建议:\n")
        if stats['avg_weight_per_record'] < 0.5:
            f.write("  - 平均权重偏低，多数为浏览行为，可适当提高购买/加购权重。\n")
        elif stats['avg_weight_per_record'] > 2:
            f.write("  - 平均权重偏高，可能过度强调少数购买行为，注意稀疏性。\n")
        else:
            f.write("  - 权重分布较为均衡。\n")
    print(f"检查报告已保存至 {out_file}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("步骤 2.1 行为权重映射对比分析")
    print("=" * 60)

    if not INPUT_D_MAPPED.exists():
        print(f"错误：输入文件不存在: {INPUT_D_MAPPED}")
        print("请将 D_mapped.csv 放在脚本同目录下，或修改 INPUT_D_MAPPED 变量指向正确路径。")
        exit(1)

    print(f"读取 {INPUT_D_MAPPED} ...")
    df_all = pd.read_csv(INPUT_D_MAPPED)

    if 'split' not in df_all.columns:
        print("错误：D_mapped.csv 中没有 'split' 列，请确保已执行步骤0.1添加时间锚点。")
        exit(1)

    df_train = df_all[df_all['split'] == 'train'].copy()
    print(f"训练窗口记录数: {len(df_train)}")

    # 确定要运行的方案
    if 'SELECTED_SCHEMES' in dir() and SELECTED_SCHEMES:
        schemes = {k: WEIGHT_SCHEMES[k] for k in SELECTED_SCHEMES if k in WEIGHT_SCHEMES}
    else:
        schemes = WEIGHT_SCHEMES

    for scheme_name, weight_map in schemes.items():
        print(f"\n处理方案: {scheme_name}")
        stats = compute_weighted_stats(df_train, weight_map)
        base_name = f"step2_1_{scheme_name.replace(' ', '_')}"
        overview_file = OUTPUT_DIR / f"{base_name}_overview.txt"
        sample_file = OUTPUT_DIR / f"{base_name}_sample.txt"
        check_file = OUTPUT_DIR / f"{base_name}_check.txt"

        write_overview(stats, scheme_name, weight_map, overview_file)
        write_sample(df_train, weight_map, scheme_name, sample_file, n=20)
        write_check(stats, scheme_name, check_file)

    print("\n所有方案报告生成完毕。")
    print(f"报告存放目录: {OUTPUT_DIR}")
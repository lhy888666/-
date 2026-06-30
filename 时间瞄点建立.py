import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
INPUT_USER_FILE = "cleaned_user_behavior.csv"   # 清洗后的用户行为表 D
INPUT_ITEM_FILE = "cleaned_item_subset.csv"     # 清洗后的商品子集表 P（可选）

# 输出目录
OUT_DIR = Path("./step0_1")
OUT_DIR.mkdir(exist_ok=True)

OVERVIEW_TXT = OUT_DIR / "step0_1_overview.txt"
SAMPLE_TXT = OUT_DIR / "step0_1_sample.txt"
CHECK_TXT = OUT_DIR / "step0_1_check.txt"

# 时间墙定义
TRAIN_END_DATE = pd.Timestamp("2014-12-17")
VAL_END_DATE = pd.Timestamp("2014-12-18")

# ==================== 开始处理 ====================
print("=" * 60)
print("步骤 0.1：时间锚点确立（添加 split 标签）")
print("=" * 60)

# 读取清洗后的数据
print(f"加载用户行为数据: {INPUT_USER_FILE}")
dtype_dict = {
    'user_id': 'int32',
    'item_id': 'int32',
    'behavior_type': 'int8',
    'user_geohash': 'object',
    'item_category': 'int16',
    'time': 'object'
}
df = pd.read_csv(INPUT_USER_FILE, dtype=dtype_dict)
original_count = len(df)
print(f"原始记录数: {original_count}")

# 解析 time 字段
print("解析 time 字段...")
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H', errors='coerce')
before_drop = len(df)
df = df.dropna(subset=['time']).copy()
after_drop = len(df)
if before_drop - after_drop > 0:
    print(f"警告: time 字段解析失败 {before_drop - after_drop} 条，已剔除")
print(f"有效时间记录数: {len(df)}")

# 添加 split 标签
print("添加 split 标签...")
conditions = [
    df['time'] < TRAIN_END_DATE,
    (df['time'] >= TRAIN_END_DATE) & (df['time'] < VAL_END_DATE),
    df['time'] >= VAL_END_DATE
]
choices = ['train', 'val', 'test']
df['split'] = np.select(conditions, choices, default='unknown')
unknown_cnt = (df['split'] == 'unknown').sum()
if unknown_cnt > 0:
    print(f"警告: 存在 {unknown_cnt} 条记录未匹配任何划分")

# ==================== 数据概览 ====================
train_cnt = (df['split'] == 'train').sum()
val_cnt = (df['split'] == 'val').sum()
test_cnt = (df['split'] == 'test').sum()
total_valid = train_cnt + val_cnt + test_cnt

with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 0.1 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_USER_FILE}\n")
    if Path(INPUT_ITEM_FILE).exists():
        f.write(f"参考文件: {INPUT_ITEM_FILE}\n")
    f.write(f"原始记录数: {original_count}\n")
    f.write(f"时间解析后记录数: {total_valid}\n")
    f.write(f"剔除记录数: {original_count - total_valid}\n\n")
    f.write("split 标签分布:\n")
    f.write(f"  训练集 (train): {train_cnt} 条 ({train_cnt/total_valid*100:.2f}%)\n")
    f.write(f"  验证集 (val)  : {val_cnt} 条 ({val_cnt/total_valid*100:.2f}%)\n")
    f.write(f"  测试集 (test) : {test_cnt} 条 ({test_cnt/total_valid*100:.2f}%)\n")
    f.write(f"时间范围: {df['time'].min()} ～ {df['time'].max()}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 抽样 ====================
sample_size = min(10, len(df))
sample_df = df.head(sample_size)  # 取前10行，展示顺序未打乱
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 0.1 抽样（用于人工检查）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 行数据（行顺序与原始输入一致，未随机打乱）:\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"行 {idx+1}: user_id={row['user_id']}, time={row['time']}, split={row['split']}\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 检查报告 ====================
train_max_time = df[df['split'] == 'train']['time'].max() if train_cnt > 0 else None
val_min_time = df[df['split'] == 'val']['time'].min() if val_cnt > 0 else None
val_max_time = df[df['split'] == 'val']['time'].max() if val_cnt > 0 else None
test_min_time = df[df['split'] == 'test']['time'].min() if test_cnt > 0 else None

with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 0.1 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("时间墙检验:\n")
    f.write(f"  训练集最大时间: {train_max_time}\n")
    f.write(f"  验证集最小时间: {val_min_time}\n")
    f.write(f"  验证集最大时间: {val_max_time}\n")
    f.write(f"  测试集最小时间: {test_min_time}\n")
    if train_max_time is not None and train_max_time < TRAIN_END_DATE:
        f.write("  ✓ 训练集时间 < 2014-12-17\n")
    else:
        f.write("  ✗ 训练集时间超出范围\n")
    if val_min_time is not None and val_min_time >= TRAIN_END_DATE and val_max_time < VAL_END_DATE:
        f.write("  ✓ 验证集时间在 [2014-12-17, 2014-12-18)\n")
    else:
        f.write("  ✗ 验证集时间范围错误\n")
    if test_min_time is not None and test_min_time >= VAL_END_DATE:
        f.write("  ✓ 测试集时间 >= 2014-12-18\n")
    else:
        f.write("  ✗ 测试集时间范围错误\n")
    f.write("\n行顺序检验:\n")
    f.write("  数据行顺序与原始输入一致，未进行任何随机打乱或排序。\n")
    f.write("  （抽样中的行顺序即为原始文件顺序）\n")
    f.write("\n数据完整性:\n")
    f.write(f"  总有效记录数: {total_valid}\n")
    f.write(f"  unknown 标签数: {unknown_cnt}\n")
    if unknown_cnt == 0:
        f.write("  ✓ 所有记录均已正确分配 split 标签。\n")
    else:
        f.write("  ⚠ 存在未分配标签的记录，请检查时间范围。\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 0.1 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
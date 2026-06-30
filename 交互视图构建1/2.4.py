import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件（步骤2.1输出）
INPUT_CSV = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\train_actions_weighted.csv"

# 输出目录
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\2 交互视图构造")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUTPUT_CSV = OUT_DIR / "train_actions_cold_mimic.csv"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step2_4_overview.txt"
SAMPLE_TXT = OUT_DIR / "step2_4_sample.txt"
CHECK_TXT = OUT_DIR / "step2_4_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step2_4_checksums.log"

# 冷启动模拟参数：删除的商品比例
REMOVE_RATIO = 0.20   # 20%
RANDOM_SEED = 42      # 固定随机种子保证可复现

# ==================== MD5 函数 ====================
def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def log_checksum(file_path, description=""):
    if not Path(file_path).exists():
        print(f"警告: {file_path} 不存在，跳过MD5计算")
        return None
    md5 = compute_md5(file_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CHECKSUM_LOG, "a") as f:
        f.write(f"[{timestamp}] {description} | {file_path} | MD5: {md5}\n")
    print(f"  MD5({Path(file_path).name}) = {md5}")
    return md5

# ==================== 1. 读取数据 ====================
print("步骤 2.4：冷启动模拟抽样")
print(f"读取 {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
original_count = len(df)
print(f"原始记录数: {original_count}")

# 获取所有商品
all_items = df['item_idx'].unique()
print(f"商品总数: {len(all_items)}")

# ==================== 2. 随机选取20%的商品 ====================
np.random.seed(RANDOM_SEED)
n_remove = int(len(all_items) * REMOVE_RATIO)
removed_items = np.random.choice(all_items, size=n_remove, replace=False)
print(f"拟删除商品数: {len(removed_items)} ({REMOVE_RATIO*100:.0f}%)")

# ==================== 3. 删除这些商品的所有记录 ====================
print("删除被选中商品的所有交互记录...")
df_remaining = df[~df['item_idx'].isin(removed_items)].copy()
remaining_count = len(df_remaining)
print(f"剩余记录数: {remaining_count} (删除 {original_count - remaining_count} 条)")

# ==================== 4. 保存输出文件 ====================
print(f"保存 {OUTPUT_CSV} ...")
df_remaining.to_csv(OUTPUT_CSV, index=False)
log_checksum(OUTPUT_CSV, "输出：冷启动模拟训练集")

# ==================== 5. 生成数据概览txt ====================
removed_item_list = removed_items.tolist()
unique_users_remaining = df_remaining['user_idx'].nunique()
unique_items_remaining = df_remaining['item_idx'].nunique()
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.4 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_CSV}\n")
    f.write(f"原始记录数: {original_count}\n")
    f.write(f"原始商品数: {len(all_items)}\n")
    f.write(f"随机删除商品比例: {REMOVE_RATIO*100:.0f}%\n")
    f.write(f"删除商品数: {len(removed_items)}\n")
    f.write(f"剩余记录数: {remaining_count}\n")
    f.write(f"剩余商品数: {unique_items_remaining}\n")
    f.write(f"剩余用户数: {unique_users_remaining}\n")
    f.write(f"删除率（记录）: {(original_count - remaining_count)/original_count*100:.2f}%\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 6. 生成抽样txt ====================
# 展示被删除的商品（前20个）以及剩余数据的前10条记录
sample_removed = removed_items[:20]
sample_remaining = df_remaining.head(10)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.4 抽样（冷启动模拟）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"被删除的商品（前20个）: {sample_removed.tolist()}\n")
    if len(removed_items) > 20:
        f.write(f"... 剩余 {len(removed_items)-20} 个商品未展示\n\n")
    else:
        f.write("\n")
    f.write("剩余数据前10条记录示例:\n")
    for idx, row in sample_remaining.iterrows():
        f.write(f"  user_idx={row['user_idx']}, item_idx={row['item_idx']}, "
                f"behavior_type={row['behavior_type']}, weight={row['weight']}\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 7. 生成检查报告txt ====================
# 验证删除的商品在剩余数据中确实不存在
still_exist = df_remaining['item_idx'].isin(removed_items).any()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.4 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("删除完整性检查:\n")
    f.write(f"  被删除商品是否在剩余数据中仍出现: {still_exist} (应为 False)\n")
    f.write(f"  原始商品数: {len(all_items)}\n")
    f.write(f"  剩余商品数: {unique_items_remaining}\n")
    f.write(f"  删除商品数: {len(removed_items)}\n")
    f.write(f"  商品数验证: {len(all_items) == unique_items_remaining + len(removed_items)}\n\n")
    f.write("随机性检查:\n")
    f.write(f"  随机种子: {RANDOM_SEED} (固定，保证可复现)\n\n")
    f.write("数据完整性:\n")
    f.write(f"  剩余数据无缺失值: {df_remaining.isnull().sum().sum() == 0}\n")
    f.write(f"  剩余数据记录数: {remaining_count}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 2.4 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
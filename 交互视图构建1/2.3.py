import pandas as pd
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
OUTPUT_CSV = OUT_DIR / "train_pos_pairs.csv"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step2_3_overview.txt"
SAMPLE_TXT = OUT_DIR / "step2_3_sample.txt"
CHECK_TXT = OUT_DIR / "step2_3_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step2_3_checksums.log"

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
print("步骤 2.3：正样本对提取")
print(f"读取 {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"总记录数: {len(df)}")

# ==================== 2. 筛选购买行为 ====================
print("筛选 behavior_type=4 (购买) 的记录...")
df_buy = df[df['behavior_type'] == 4].copy()
print(f"购买记录数: {len(df_buy)}")

# ==================== 3. 提取正样本对并去重 ====================
print("提取 (user_idx, item_idx) 对并去重...")
pos_pairs = df_buy[['user_idx', 'item_idx']].drop_duplicates()
print(f"去重后正样本对数: {len(pos_pairs)}")

# ==================== 4. 保存输出文件 ====================
print(f"保存 {OUTPUT_CSV} ...")
pos_pairs.to_csv(OUTPUT_CSV, index=False)
log_checksum(OUTPUT_CSV, "输出：训练正样本对")

# ==================== 5. 生成数据概览txt ====================
unique_users = pos_pairs['user_idx'].nunique()
unique_items = pos_pairs['item_idx'].nunique()
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.3 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_CSV}\n")
    f.write(f"输入总记录数: {len(df)}\n")
    f.write(f"购买行为记录数: {len(df_buy)}\n")
    f.write(f"去重后正样本对数: {len(pos_pairs)}\n")
    f.write(f"唯一用户数: {unique_users}\n")
    f.write(f"唯一物品数: {unique_items}\n")
    f.write(f"平均每个用户的购买物品数: {len(pos_pairs)/unique_users:.2f}\n" if unique_users > 0 else "平均每个用户的购买物品数: N/A\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 6. 生成抽样txt ====================
sample_size = min(20, len(pos_pairs))
sample_df = pos_pairs.head(sample_size)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.3 抽样（正样本对示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 个正样本对 (user_idx, item_idx):\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"  ({row['user_idx']}, {row['item_idx']})\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 7. 生成检查报告txt ====================
# 检查是否有重复（去重后应无）
dup_count = pos_pairs.duplicated().sum()
# 检查用户索引和物品索引是否合法（非负）
user_min = pos_pairs['user_idx'].min()
item_min = pos_pairs['item_idx'].min()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.3 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("重复检查:\n")
    f.write(f"  重复边数量: {dup_count} (应为0)\n")
    f.write(f"  去重后正样本对数: {len(pos_pairs)}\n\n")
    f.write("索引合法性:\n")
    f.write(f"  user_idx 最小值: {user_min}\n")
    f.write(f"  item_idx 最小值: {item_min}\n")
    f.write(f"  所有索引均非负: {user_min >= 0 and item_min >= 0}\n\n")
    f.write("数据完整性:\n")
    f.write(f"  购买行为记录中无缺失 user_idx 或 item_idx: {df_buy[['user_idx', 'item_idx']].isna().sum().sum() == 0}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 2.3 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
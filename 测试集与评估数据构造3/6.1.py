import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件（步骤1.2输出）
INPUT_CSV = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\D_mapped.csv"

# 输出目录
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\5 元数据生成")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUTPUT_CSV = OUT_DIR / "test_pos_pairs.csv"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step6_1_overview.txt"
SAMPLE_TXT = OUT_DIR / "step6_1_sample.txt"
CHECK_TXT = OUT_DIR / "step6_1_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step6_1_checksums.log"

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

# ==================== 1. 读取数据并筛选测试集 ====================
print("步骤 6.1：测试正样本提取")
print(f"读取 {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"总记录数: {len(df)}")

# 筛选测试集
df_test = df[df['split'] == 'test'].copy()
print(f"测试集记录数: {len(df_test)}")

# 记录输入文件MD5
log_checksum(INPUT_CSV, "输入：映射后的用户行为表")

# ==================== 2. 筛选购买行为 ====================
print("筛选 behavior_type=4 (购买) 的记录...")
df_buy = df_test[df_test['behavior_type'] == 4].copy()
print(f"测试集中购买记录数: {len(df_buy)}")

# ==================== 3. 提取正样本对并去重 ====================
print("提取 (user_idx, item_idx) 对并去重...")
pos_pairs = df_buy[['user_idx', 'item_idx']].drop_duplicates()
print(f"去重后正样本对数: {len(pos_pairs)}")

# ==================== 4. 保存输出文件 ====================
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUTPUT_CSV} ...")
pos_pairs.to_csv(OUTPUT_CSV, index=False)
log_checksum(OUTPUT_CSV, "输出：测试正样本对")

# ==================== 5. 统计信息 ====================
unique_users = pos_pairs['user_idx'].nunique()
unique_items = pos_pairs['item_idx'].nunique()

# ==================== 6. 生成数据概览txt ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.1 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_CSV}\n")
    f.write(f"总记录数: {len(df)}\n")
    f.write(f"测试集记录数: {len(df_test)}\n")
    f.write(f"测试集中购买记录数: {len(df_buy)}\n")
    f.write(f"去重后正样本对数: {len(pos_pairs)}\n")
    f.write(f"唯一用户数: {unique_users}\n")
    f.write(f"唯一物品数: {unique_items}\n")
    if unique_users > 0:
        f.write(f"平均每个测试用户的正样本数: {len(pos_pairs)/unique_users:.2f}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 7. 生成抽样txt ====================
sample_size = min(20, len(pos_pairs))
sample_df = pos_pairs.head(sample_size)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.1 抽样（正样本对示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 个测试正样本对 (user_idx, item_idx):\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"  ({row['user_idx']}, {row['item_idx']})\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 8. 生成检查报告txt ====================
dup_count = pos_pairs.duplicated().sum()
null_count = pos_pairs.isnull().sum().sum()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.1 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("重复检查:\n")
    f.write(f"  重复边数量: {dup_count} (应为0)\n")
    f.write(f"  去重后正样本对数: {len(pos_pairs)}\n\n")
    f.write("数据完整性:\n")
    f.write(f"  缺失值数量: {null_count}\n")
    f.write(f"  无缺失值: {null_count == 0}\n\n")
    f.write("索引合法性:\n")
    f.write(f"  user_idx 最小值: {pos_pairs['user_idx'].min()}\n")
    f.write(f"  item_idx 最小值: {pos_pairs['item_idx'].min()}\n")
    f.write(f"  所有索引均非负: {pos_pairs['user_idx'].min() >= 0 and pos_pairs['item_idx'].min() >= 0}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 6.1 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
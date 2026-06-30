import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件路径（根据您的实际存放位置调整）
INPUT_TEST_POS = Path(r"E:\计算机设计大赛\第一次数据处理\6 测试集与评估数据构造\test_pos_pairs.csv")
INPUT_TRAIN_ACTIONS = Path(r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\train_actions_weighted.csv")

# 输出目录
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\6 测试集与评估数据构造")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUTPUT_CSV = OUT_DIR / "test_pos_labeled.csv"
OVERVIEW_TXT = OUT_DIR / "step6_2_overview.txt"
SAMPLE_TXT = OUT_DIR / "step6_2_sample.txt"
CHECK_TXT = OUT_DIR / "step6_2_check.txt"
CHECKSUM_LOG = OUT_DIR / "step6_2_checksums.log"

COLD_START_THRESHOLD = 5

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

# ==================== 主处理 ====================
print("步骤 6.2：冷启动标签打标")

# 检查输入文件是否存在
if not INPUT_TEST_POS.exists():
    raise FileNotFoundError(f"找不到测试正样本文件: {INPUT_TEST_POS}\n请先运行步骤6.1生成该文件。")
if not INPUT_TRAIN_ACTIONS.exists():
    raise FileNotFoundError(f"找不到训练行为文件: {INPUT_TRAIN_ACTIONS}\n请先运行步骤2.1生成该文件。")

print(f"读取 {INPUT_TEST_POS} ...")
df_pos = pd.read_csv(INPUT_TEST_POS)
print(f"测试正样本数: {len(df_pos)}")

print(f"读取 {INPUT_TRAIN_ACTIONS} ...")
df_train = pd.read_csv(INPUT_TRAIN_ACTIONS)
print(f"训练行为记录数: {len(df_train)}")

log_checksum(INPUT_TEST_POS, "输入：测试正样本对")
log_checksum(INPUT_TRAIN_ACTIONS, "输入：训练窗口加权行为数据")

# 统计训练数据中每个商品的出现次数
print("统计训练数据中商品出现次数...")
item_cnt = df_train.groupby('item_idx').size().reset_index(name='train_cnt')
print(f"训练数据中唯一商品数: {len(item_cnt)}")
print(f"商品出现次数统计: min={item_cnt['train_cnt'].min()}, max={item_cnt['train_cnt'].max()}, mean={item_cnt['train_cnt'].mean():.2f}")

# 为测试正样本添加 train_cnt 和冷启动标签
df_pos = df_pos.merge(item_cnt, on='item_idx', how='left')
df_pos['train_cnt'] = df_pos['train_cnt'].fillna(0).astype(int)
df_pos['is_cold'] = (df_pos['train_cnt'] < COLD_START_THRESHOLD).astype(int)

cold_cnt = df_pos['is_cold'].sum()
warm_cnt = len(df_pos) - cold_cnt
print(f"冷启动样本数: {cold_cnt} ({cold_cnt/len(df_pos)*100:.2f}%), 非冷启动: {warm_cnt}")

# 保存输出
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUTPUT_CSV} ...")
df_pos.to_csv(OUTPUT_CSV, index=False)
log_checksum(OUTPUT_CSV, "输出：带冷启动标签的测试正样本")

# ==================== 生成数据概览 ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.2 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入测试正样本文件: {INPUT_TEST_POS}\n")
    f.write(f"输入训练行为文件: {INPUT_TRAIN_ACTIONS}\n")
    f.write(f"输出文件: {OUTPUT_CSV}\n")
    f.write(f"冷启动阈值: 训练中出现次数 < {COLD_START_THRESHOLD}\n\n")
    f.write(f"测试正样本总数: {len(df_pos)}\n")
    f.write(f"冷启动样本数 (is_cold=1): {cold_cnt} ({cold_cnt/len(df_pos)*100:.2f}%)\n")
    f.write(f"非冷启动样本数 (is_cold=0): {warm_cnt} ({warm_cnt/len(df_pos)*100:.2f}%)\n\n")
    f.write("训练数据中商品出现次数统计（用于打标）:\n")
    f.write(f"  训练商品总数: {len(item_cnt)}\n")
    f.write(f"  出现次数最小值: {item_cnt['train_cnt'].min()}\n")
    f.write(f"  出现次数最大值: {item_cnt['train_cnt'].max()}\n")
    f.write(f"  出现次数均值: {item_cnt['train_cnt'].mean():.2f}\n")
    f.write(f"  出现次数中位数: {item_cnt['train_cnt'].median():.0f}\n")
    f.write(f"  出现次数 ≤ {COLD_START_THRESHOLD} 的商品数: {(item_cnt['train_cnt'] < COLD_START_THRESHOLD).sum()} ({(item_cnt['train_cnt'] < COLD_START_THRESHOLD).mean()*100:.2f}%)\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 生成抽样txt ====================
sample_size = min(20, len(df_pos))
sample_df = df_pos.head(sample_size)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.2 抽样（正样本及冷启动标签）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 条测试正样本的冷启动标签:\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"  user={row['user_idx']}, item={row['item_idx']}, train_cnt={row['train_cnt']}, is_cold={row['is_cold']}\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 生成检查报告 ====================
mismatch = ((df_pos['is_cold'] == 1) & (df_pos['train_cnt'] >= COLD_START_THRESHOLD)) | \
           ((df_pos['is_cold'] == 0) & (df_pos['train_cnt'] < COLD_START_THRESHOLD))
mismatch_cnt = mismatch.sum()
dup_count = df_pos.duplicated(subset=['user_idx', 'item_idx']).sum()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.2 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("标签一致性检查:\n")
    f.write(f"  标签不一致的样本数: {mismatch_cnt} (应为0)\n")
    if mismatch_cnt == 0:
        f.write("  ✓ 所有样本的冷启动标签与训练出现次数一致。\n")
    else:
        f.write("  ✗ 存在标签不一致，请检查逻辑。\n\n")
    f.write("重复检查:\n")
    f.write(f"  重复样本数量: {dup_count}\n")
    if dup_count == 0:
        f.write("  ✓ 无重复样本。\n")
    else:
        f.write("  注意：存在重复样本，可能影响评估。\n\n")
    f.write("train_cnt 统计:\n")
    f.write(f"  最小值: {df_pos['train_cnt'].min()}\n")
    f.write(f"  最大值: {df_pos['train_cnt'].max()}\n")
    f.write(f"  均值: {df_pos['train_cnt'].mean():.2f}\n")
    f.write(f"  中位数: {df_pos['train_cnt'].median():.0f}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 6.2 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
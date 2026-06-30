import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ==================== 配置 ====================
# 输入文件
INPUT_TEST_LABELED = r"E:\计算机设计大赛\第一次数据处理\6 测试集与评估数据构造\test_pos_labeled.csv"
INPUT_TRAIN_MATRIX = r"E:\计算机设计大赛\第一次数据处理\2 交互视图构造\train_ui_matrix.npz"

# 输出目录
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\6 测试集与评估数据构造")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUTPUT_PKL = OUT_DIR / "eval_pairs.pkl"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step6_3_overview.txt"
SAMPLE_TXT = OUT_DIR / "step6_3_sample.txt"
CHECK_TXT = OUT_DIR / "step6_3_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step6_3_checksums.log"

# 负采样参数
NEG_SAMPLE_SIZE = 99
RANDOM_SEED = 42

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

# ==================== 1. 加载数据 ====================
print("步骤 6.3：负采样构造")
print(f"读取 {INPUT_TEST_LABELED} ...")
df_pos = pd.read_csv(INPUT_TEST_LABELED)
print(f"测试正样本数: {len(df_pos)}")

print(f"读取 {INPUT_TRAIN_MATRIX} ...")
train_matrix = sp.load_npz(INPUT_TRAIN_MATRIX).tocsr()
print(f"训练矩阵形状: {train_matrix.shape}, 非零元素数: {train_matrix.nnz}")

# 记录输入文件MD5
log_checksum(INPUT_TEST_LABELED, "输入：带冷启动标签的测试正样本")
log_checksum(INPUT_TRAIN_MATRIX, "输入：用户-物品评分矩阵")

# ==================== 2. 获取训练商品集合 ====================
# 训练中出现过的所有商品（非零列）
train_items = set(np.where(train_matrix.indptr[1:] > train_matrix.indptr[:-1])[0])
print(f"训练商品总数: {len(train_items)}")

# ==================== 3. 获取每个用户在训练中交互过的商品 ====================
print("提取每个用户在训练集中交互过的商品...")
user_interacted = {}
for user in range(train_matrix.shape[0]):
    # 获取该用户的非零列索引
    start = train_matrix.indptr[user]
    end = train_matrix.indptr[user + 1]
    if start < end:
        items = train_matrix.indices[start:end]
        user_interacted[user] = set(items)
    else:
        user_interacted[user] = set()
print(f"训练用户数: {len(user_interacted)}")

# ==================== 4. 负采样 ====================
np.random.seed(RANDOM_SEED)
eval_data = []   # 每个元素为 (user_idx, pos_item, neg_list)
total_neg = 0

# 按用户分组正样本
user_pos_items = df_pos.groupby('user_idx')['item_idx'].apply(list).to_dict()

for user, pos_items in user_pos_items.items():
    interacted = user_interacted.get(user, set())
    # 候选负样本：训练商品中排除该用户已交互的商品
    candidate_neg = list(train_items - interacted)
    if len(candidate_neg) == 0:
        print(f"警告: 用户 {user} 无可用负样本（训练商品全部被交互），跳过该用户")
        continue
    # 对于每个正样本，采样99个负样本（无放回，若不足则采样全部）
    for pos_item in pos_items:
        if len(candidate_neg) >= NEG_SAMPLE_SIZE:
            neg_samples = np.random.choice(candidate_neg, size=NEG_SAMPLE_SIZE, replace=False).tolist()
        else:
            neg_samples = candidate_neg.copy()   # 全部负样本
        eval_data.append((user, pos_item, neg_samples))
        total_neg += len(neg_samples)

print(f"生成评估样本数: {len(eval_data)}, 总负样本数: {total_neg}")

# ==================== 5. 保存输出 ====================
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUTPUT_PKL} ...")
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(eval_data, f)
log_checksum(OUTPUT_PKL, "输出：测试集负采样评估对")

# ==================== 6. 生成数据概览txt ====================
total_users = len(user_pos_items)
total_pos = len(df_pos)
avg_neg_per_user = total_neg / total_users if total_users > 0 else 0
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.3 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入测试正样本文件: {INPUT_TEST_LABELED}\n")
    f.write(f"输入训练矩阵文件: {INPUT_TRAIN_MATRIX}\n")
    f.write(f"输出文件: {OUTPUT_PKL}\n")
    f.write(f"负采样数量: 每个正样本 {NEG_SAMPLE_SIZE} 个负样本\n\n")
    f.write(f"测试用户数（有正样本的用户）: {total_users}\n")
    f.write(f"总正样本数: {total_pos}\n")
    f.write(f"总负样本数: {total_neg}\n")
    f.write(f"平均每个用户的负样本数: {avg_neg_per_user:.2f}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 7. 生成抽样txt ====================
sample_size = min(10, len(eval_data))
sample_data = eval_data[:sample_size]
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.3 抽样（负采样示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 个样本（用户, 正样本, 负样本前20个）:\n\n")
    for idx, (user, pos, negs) in enumerate(sample_data):
        f.write(f"样本 {idx+1}:\n")
        f.write(f"  用户ID: {user}\n")
        f.write(f"  正样本商品: {pos}\n")
        f.write(f"  负样本商品（前20）: {negs[:20]}\n")
        f.write(f"  负样本总数: {len(negs)}\n\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 8. 生成检查报告txt ====================
# 检查负样本合法性：是否在训练商品集合中，且不等于正样本，且不在用户交互集中
invalid_count = 0
neg_counts = []
for user, pos, negs in eval_data:
    neg_counts.append(len(negs))
    interacted = user_interacted.get(user, set())
    for neg in negs:
        if neg not in train_items:
            invalid_count += 1
        if neg in interacted:
            invalid_count += 1
        if neg == pos:
            invalid_count += 1
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.3 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"用户总数（有正样本）: {total_users}\n")
    f.write(f"评估样本总数: {len(eval_data)}\n")
    f.write(f"总负样本数: {total_neg}\n\n")
    f.write("负样本合法性检查:\n")
    f.write(f"  不合规的负样本数量: {invalid_count}\n")
    if invalid_count == 0:
        f.write("  ✓ 所有负样本均合法（在训练商品中、非用户已交互、非正样本）。\n")
    else:
        f.write("  ✗ 存在不合规负样本，请检查采样逻辑。\n\n")
    f.write("每个样本的负样本数量统计:\n")
    if neg_counts:
        f.write(f"  最小值: {min(neg_counts)}\n")
        f.write(f"  最大值: {max(neg_counts)}\n")
        f.write(f"  均值: {np.mean(neg_counts):.2f}\n")
        f.write(f"  中位数: {np.median(neg_counts):.0f}\n")
        insufficient = sum(1 for c in neg_counts if c < NEG_SAMPLE_SIZE)
        if insufficient > 0:
            f.write(f"\n警告: 有 {insufficient} 个样本的负样本数量不足 {NEG_SAMPLE_SIZE}（训练商品不足）。\n")
    else:
        f.write("  无负样本数据。\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 6.3 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_PKL}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
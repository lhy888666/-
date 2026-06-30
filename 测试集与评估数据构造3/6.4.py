import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ==================== 配置 ====================
INPUT_D_MAPPED = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\D_mapped.csv"
INPUT_TRAIN_ACTIONS = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\train_actions_weighted.csv"
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\6 测试集与评估数据构造")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PKL = OUT_DIR / "val_pairs.pkl"
OVERVIEW_TXT = OUT_DIR / "step6_4_overview.txt"
SAMPLE_TXT = OUT_DIR / "step6_4_sample.txt"
CHECK_TXT = OUT_DIR / "step6_4_check.txt"
CHECKSUM_LOG = OUT_DIR / "step6_4_checksums.log"

NEG_SAMPLE_SIZE = 99
RANDOM_SEED = 42
MAX_RETRIES = 1000   # 防止死循环（实际不会触发）

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
print("步骤 6.4：验证集评估数据构造（优化版：拒绝采样）")
print(f"读取 {INPUT_D_MAPPED} ...")
df_all = pd.read_csv(INPUT_D_MAPPED)
df_val = df_all[df_all['split'] == 'val'].copy()
print(f"验证集记录数: {len(df_val)}")

print(f"读取 {INPUT_TRAIN_ACTIONS} ...")
df_train = pd.read_csv(INPUT_TRAIN_ACTIONS)
print(f"训练行为记录数: {len(df_train)}")

log_checksum(INPUT_D_MAPPED, "输入：映射后的用户行为表（含split）")
log_checksum(INPUT_TRAIN_ACTIONS, "输入：训练窗口加权行为数据")

# ==================== 2. 提取验证集正样本 ====================
df_val_pos = df_val[df_val['behavior_type'] == 4].copy()
print(f"验证集购买记录数: {len(df_val_pos)}")

if len(df_val_pos) == 0:
    print("警告：验证集无购买行为，输出空列表")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump([], f)
    with open(OVERVIEW_TXT, 'w') as f:
        f.write("步骤6.4：验证集无正样本，输出为空。")
    print("步骤 6.4 完成（无样本）。")
    exit(0)

# ==================== 3. 获取训练商品集合（用于快速成员判断） ====================
all_items = df_train['item_idx'].unique()
train_items_set = set(all_items)
print(f"训练商品总数: {len(train_items_set)}")

# ==================== 4. 获取每个用户在训练集中交互过的商品 ====================
print("提取每个用户在训练集中交互过的商品...")
user_interacted = defaultdict(set)
for user, group in df_train.groupby('user_idx'):
    user_interacted[user] = set(group['item_idx'].unique())
print(f"训练用户数: {len(user_interacted)}")

# ==================== 5. 按用户分组正样本 ====================
user_pos_items = df_val_pos.groupby('user_idx')['item_idx'].apply(list).to_dict()

# ==================== 6. 负采样（拒绝采样） ====================
np.random.seed(RANDOM_SEED)
eval_data = []
total_neg = 0

for user, pos_items in user_pos_items.items():
    interacted = user_interacted.get(user, set())
    for pos_item in pos_items:
        neg_samples = []
        sampled_set = set()
        attempts = 0
        while len(neg_samples) < NEG_SAMPLE_SIZE and attempts < MAX_RETRIES:
            cand = np.random.choice(all_items)
            if (cand in train_items_set and
                cand not in interacted and
                cand != pos_item and
                cand not in sampled_set):
                sampled_set.add(cand)
                neg_samples.append(cand)
            attempts += 1
        # 如果拒绝采样未能采满，则从补全集里直接取（极少数情况）
        if len(neg_samples) < NEG_SAMPLE_SIZE:
            # 可用补全集 = 训练商品 - 用户已交互 - {pos_item}
            candidate_pool = list(train_items_set - interacted - {pos_item})
            need = NEG_SAMPLE_SIZE - len(neg_samples)
            if need <= len(candidate_pool):
                extra = np.random.choice(candidate_pool, size=need, replace=False).tolist()
                neg_samples.extend(extra)
        eval_data.append((user, pos_item, neg_samples))
        total_neg += len(neg_samples)

print(f"生成验证样本数: {len(eval_data)}, 总负样本数: {total_neg}")

# ==================== 7. 保存输出 ====================
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUTPUT_PKL} ...")
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(eval_data, f)
log_checksum(OUTPUT_PKL, "输出：验证集评估文件")

# ==================== 8. 生成数据概览 ====================
total_users = len(user_pos_items)
total_pos = len(df_val_pos)
avg_neg_per_user = total_neg / total_users if total_users > 0 else 0
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.4 数据概览（验证集）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入验证集文件: {INPUT_D_MAPPED} (split='val')\n")
    f.write(f"输入训练行为文件: {INPUT_TRAIN_ACTIONS}\n")
    f.write(f"输出文件: {OUTPUT_PKL}\n")
    f.write(f"负采样数量: 每个正样本 {NEG_SAMPLE_SIZE} 个负样本\n\n")
    f.write(f"验证用户数（有正样本的用户）: {total_users}\n")
    f.write(f"总正样本数: {total_pos}\n")
    f.write(f"总负样本数: {total_neg}\n")
    f.write(f"平均每个用户的负样本数: {avg_neg_per_user:.2f}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 9. 抽样 ====================
sample_size = min(10, len(eval_data))
sample_data = eval_data[:sample_size]
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.4 抽样（验证集负采样示例）\n")
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

# ==================== 10. 检查报告 ====================
invalid_count = 0
neg_counts = []
for user, pos, negs in eval_data:
    neg_counts.append(len(negs))
    interacted = user_interacted.get(user, set())
    for neg in negs:
        if neg not in train_items_set:
            invalid_count += 1
        if neg in interacted:
            invalid_count += 1
        if neg == pos:
            invalid_count += 1
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 6.4 数据检查报告（验证集）\n")
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

print("\n步骤 6.4 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_PKL}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
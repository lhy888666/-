import pickle
import numpy as np
import random
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入目录（步骤3.1输出）
INPUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\3 序列数据构造")
INPUT_SEQ = INPUT_DIR / "user_sequences.pkl"

# 输出文件
OUT_MASK = INPUT_DIR / "user_seq_mask.pkl"
OUT_SHUFFLE = INPUT_DIR / "user_seq_shuffle.pkl"

# 输出txt报告
OVERVIEW_TXT = INPUT_DIR / "step3_3_overview.txt"
SAMPLE_TXT = INPUT_DIR / "step3_3_sample.txt"
CHECK_TXT = INPUT_DIR / "step3_3_check.txt"

# MD5日志
CHECKSUM_LOG = INPUT_DIR / "step3_3_checksums.log"

# 序列参数（与步骤3.1保持一致）
MAX_SEQ_LEN = 50
PAD_VALUE = -1
MASK_TOKEN = -2

# 增强参数
MASK_RATIO = 0.2  # 掩码比例（针对有效物品）
SHUFFLE_RATIO = 0.05  # 乱序交换比例（针对有效物品位置，至少交换1对）
RANDOM_SEED = 42  # 固定随机种子保证可复现


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


# ==================== 增强函数 ====================
def random_mask(seq, mask_ratio=MASK_RATIO, mask_token=MASK_TOKEN, pad_value=PAD_VALUE):
    """对序列中的有效物品按比例随机替换为 mask_token"""
    seq = seq.copy()
    valid_indices = [i for i, x in enumerate(seq) if x != pad_value]
    n_mask = max(1, int(len(valid_indices) * mask_ratio))
    mask_indices = random.sample(valid_indices, n_mask)
    for idx in mask_indices:
        seq[idx] = mask_token
    return seq, mask_indices


def local_shuffle(seq, shuffle_ratio=SHUFFLE_RATIO, pad_value=PAD_VALUE):
    """对序列中的有效物品进行局部乱序：随机交换若干对有效物品的位置"""
    seq = seq.copy()
    valid_indices = [i for i, x in enumerate(seq) if x != pad_value]
    n_swaps = max(1, int(len(valid_indices) * shuffle_ratio // 2))  # 每对交换计为2个物品
    for _ in range(n_swaps):
        if len(valid_indices) < 2:
            break
        i, j = random.sample(valid_indices, 2)
        seq[i], seq[j] = seq[j], seq[i]
    return seq


# ==================== 主处理 ====================
print("步骤 3.3：序列增强视图生成")
print(f"加载 {INPUT_SEQ} ...")
with open(INPUT_SEQ, 'rb') as f:
    user_sequences = pickle.load(f)

# 记录输入文件MD5
log_checksum(INPUT_SEQ, "输入：用户物品序列")

# 设置随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

total_users = len(user_sequences)
print(f"用户总数: {total_users}")

# 生成增强视图
user_mask_dict = {}
user_shuffle_dict = {}
mask_stats = {'total_masked_items': 0, 'users_with_mask': 0}
shuffle_stats = {'total_swapped_pairs': 0, 'users_with_shuffle': 0}

for uid, seq in user_sequences.items():
    # 随机掩码
    masked_seq, masked_indices = random_mask(seq)
    user_mask_dict[uid] = masked_seq
    if masked_indices:
        mask_stats['total_masked_items'] += len(masked_indices)
        mask_stats['users_with_mask'] += 1

    # 局部乱序
    shuffled_seq = local_shuffle(seq)
    user_shuffle_dict[uid] = shuffled_seq
    # 统计交换对数（与原序列比较，通过位置变化计数）
    changes = sum(1 for i in range(MAX_SEQ_LEN)
                  if seq[i] != shuffled_seq[i] and seq[i] != PAD_VALUE)
    n_swaps = changes // 2
    if n_swaps > 0:
        shuffle_stats['total_swapped_pairs'] += n_swaps
        shuffle_stats['users_with_shuffle'] += 1

mask_stats['coverage'] = mask_stats['users_with_mask'] / total_users * 100
mask_stats['avg_mask_per_user'] = mask_stats['total_masked_items'] / total_users
shuffle_stats['coverage'] = shuffle_stats['users_with_shuffle'] / total_users * 100
shuffle_stats['avg_swaps_per_user'] = shuffle_stats['total_swapped_pairs'] / total_users

print(f"掩码: 总掩码物品数={mask_stats['total_masked_items']}, 覆盖率={mask_stats['coverage']:.1f}%")
print(f"乱序: 总交换对数={shuffle_stats['total_swapped_pairs']}, 覆盖率={shuffle_stats['coverage']:.1f}%")

# 保存输出
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUT_MASK} ...")
with open(OUT_MASK, 'wb') as f:
    pickle.dump(user_mask_dict, f)
log_checksum(OUT_MASK, "输出：随机掩码后的序列")

print(f"保存 {OUT_SHUFFLE} ...")
with open(OUT_SHUFFLE, 'wb') as f:
    pickle.dump(user_shuffle_dict, f)
log_checksum(OUT_SHUFFLE, "输出：局部乱序后的序列")

# ==================== 生成数据概览 ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.3 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_SEQ}\n")
    f.write(f"输出文件: {OUT_MASK}, {OUT_SHUFFLE}\n")
    f.write(f"序列固定长度: {MAX_SEQ_LEN}\n")
    f.write(f"填充值: {PAD_VALUE}\n")
    f.write(f"掩码标记: {MASK_TOKEN}\n")
    f.write(f"掩码比例: {MASK_RATIO * 100}% (针对有效物品)\n")
    f.write(f"乱序交换比例: {SHUFFLE_RATIO * 100}% (有效物品位置对)\n\n")
    f.write(f"用户总数: {total_users}\n\n")
    f.write("随机掩码 (Mask) 统计:\n")
    f.write(f"  总掩码物品数: {mask_stats['total_masked_items']}\n")
    f.write(f"  平均每用户掩码数: {mask_stats['avg_mask_per_user']:.2f}\n")
    f.write(f"  掩码覆盖率: {mask_stats['coverage']:.2f}% (有掩码的用户比例)\n\n")
    f.write("局部乱序 (Shuffle) 统计:\n")
    f.write(f"  总交换对数: {shuffle_stats['total_swapped_pairs']}\n")
    f.write(f"  平均每用户交换对数: {shuffle_stats['avg_swaps_per_user']:.2f}\n")
    f.write(f"  乱序覆盖率: {shuffle_stats['coverage']:.2f}% (有交换的用户比例)\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 生成抽样（至少10个用户） ====================
sample_users = list(user_sequences.keys())[:10]  # 取前10个用户
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.3 抽样（用于人工检查）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"说明：展示前10个用户的原序列、掩码后序列、乱序后序列（前30个位置）\n")
    f.write(f"填充值={PAD_VALUE}，掩码标记={MASK_TOKEN}\n\n")
    for uid in sample_users:
        orig = user_sequences[uid]
        mask_seq = user_mask_dict[uid]
        shuffle_seq = user_shuffle_dict[uid]
        f.write(f"用户 {uid}:\n")
        f.write(f"  原序列 (前30): {orig[:30]}\n")
        f.write(f"  掩码后序列 (前30): {mask_seq[:30]}\n")
        f.write(f"  乱序后序列 (前30): {shuffle_seq[:30]}\n")
        f.write("\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 生成检查报告 ====================
# 检查掩码序列合法性
mask_errors = []
shuffle_errors = []
for uid in user_sequences:
    orig = user_sequences[uid]
    mask = user_mask_dict[uid]
    shuffle = user_shuffle_dict[uid]
    # 长度检查
    if len(mask) != MAX_SEQ_LEN or len(shuffle) != MAX_SEQ_LEN:
        mask_errors.append(f"用户{uid}长度不一致")
        continue
    # 填充位置应保持不变
    for i in range(MAX_SEQ_LEN):
        if orig[i] == PAD_VALUE:
            if mask[i] != PAD_VALUE:
                mask_errors.append(f"用户{uid}填充位置{i}被修改")
            if shuffle[i] != PAD_VALUE:
                shuffle_errors.append(f"用户{uid}填充位置{i}被修改")
        else:
            # 掩码序列：非填充位置只能变为掩码或保持原值
            if mask[i] not in (orig[i], MASK_TOKEN):
                mask_errors.append(f"用户{uid}有效位置{i}变为非法值{mask[i]}")
    # 乱序序列：有效物品的多重集合应保持不变
    orig_valid = sorted([x for x in orig if x != PAD_VALUE])
    shuffle_valid = sorted([x for x in shuffle if x != PAD_VALUE])
    if orig_valid != shuffle_valid:
        shuffle_errors.append(f"用户{uid}乱序后有效物品集合与原序列不一致")

# 统计掩码标记分布
all_mask_counts = [sum(1 for x in seq if x == MASK_TOKEN) for seq in user_mask_dict.values()]

with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.3 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"用户总数: {total_users}\n")
    f.write(f"序列固定长度: {MAX_SEQ_LEN}\n\n")
    f.write("掩码序列检查:\n")
    if not mask_errors:
        f.write("  ✓ 所有掩码序列符合规范（长度正确、填充位置不变、掩码仅替换有效物品）\n")
    else:
        f.write(f"  ✗ 发现 {len(mask_errors)} 个错误，例如: {mask_errors[:3]}\n")
    f.write("\n乱序序列检查:\n")
    if not shuffle_errors:
        f.write("  ✓ 所有乱序序列符合规范（长度正确、填充位置不变、有效物品集合与原序列一致）\n")
    else:
        f.write(f"  ✗ 发现 {len(shuffle_errors)} 个错误，例如: {shuffle_errors[:3]}\n")
    f.write("\n掩码标记分布（每用户掩码数量）:\n")
    if all_mask_counts:
        f.write(f"  最小值: {min(all_mask_counts)}\n")
        f.write(f"  最大值: {max(all_mask_counts)}\n")
        f.write(f"  均值: {np.mean(all_mask_counts):.2f}\n")
        f.write(f"  中位数: {np.median(all_mask_counts):.0f}\n")
    else:
        f.write("  无掩码标记\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 3.3 完成。")
print(f"输出目录: {INPUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUT_MASK}")
print(f"  - {OUT_SHUFFLE}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
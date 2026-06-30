import pickle
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入目录（步骤3.1输出）
INPUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\3 序列数据构造")
INPUT_SEQ = INPUT_DIR / "user_sequences.pkl"
INPUT_TS = INPUT_DIR / "user_timestamps.pkl"

# 输出文件
OUTPUT_PKL = INPUT_DIR / "user_delta_time.pkl"

# 输出txt报告
OVERVIEW_TXT = INPUT_DIR / "step3_2_overview.txt"
SAMPLE_TXT = INPUT_DIR / "step3_2_sample.txt"
CHECK_TXT = INPUT_DIR / "step3_2_check.txt"

# MD5日志
CHECKSUM_LOG = INPUT_DIR / "step3_2_checksums.log"

# 序列参数（与步骤3.1保持一致）
MAX_SEQ_LEN = 50
DELTA_PAD = 0.0          # 时间差填充值（小时）

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

# ==================== 计算时间差 ====================
def compute_delta_time(timestamps):
    """
    输入: 时间戳列表（长度 MAX_SEQ_LEN，填充位置为 None）
    输出: 时间差列表（长度相同，填充位置为 DELTA_PAD，第一个有效时间戳的 delta 为 0）
    时间差单位: 小时
    """
    deltas = []
    last_valid_ts = None
    for ts in timestamps:
        if ts is None:
            deltas.append(DELTA_PAD)
        else:
            if last_valid_ts is None:
                deltas.append(0.0)          # 第一个有效行为，无前驱
            else:
                delta_seconds = (ts - last_valid_ts).total_seconds()
                delta_hours = delta_seconds / 3600.0
                deltas.append(delta_hours)
            last_valid_ts = ts
    return deltas

# ==================== 主处理 ====================
print("步骤 3.2：时间间隔矩阵构造")
print(f"加载 {INPUT_SEQ} ...")
with open(INPUT_SEQ, 'rb') as f:
    user_sequences = pickle.load(f)

print(f"加载 {INPUT_TS} ...")
with open(INPUT_TS, 'rb') as f:
    user_timestamps = pickle.load(f)

# 记录输入文件MD5
log_checksum(INPUT_SEQ, "输入：用户物品序列")
log_checksum(INPUT_TS, "输入：用户时间戳序列")

# 检查用户键是否一致
if user_sequences.keys() != user_timestamps.keys():
    print("警告：user_sequences 和 user_timestamps 的键不一致，将取交集")
    common_keys = set(user_sequences.keys()) & set(user_timestamps.keys())
    user_sequences = {k: user_sequences[k] for k in common_keys}
    user_timestamps = {k: user_timestamps[k] for k in common_keys}

print(f"用户总数: {len(user_sequences)}")

# 计算时间差
user_delta_time = {}
all_deltas = []          # 收集所有非填充时间差用于统计
valid_counts = []        # 每个用户的有效时间差个数（非填充）

for uid, ts_list in user_timestamps.items():
    deltas = compute_delta_time(ts_list)
    user_delta_time[uid] = deltas
    # 统计有效时间差（非填充）
    valid = [d for d in deltas if d != DELTA_PAD]
    valid_counts.append(len(valid))
    all_deltas.extend(valid)

print(f"生成时间差序列数: {len(user_delta_time)}")
print(f"有效时间差条目总数: {len(all_deltas)}")

# 保存输出
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUTPUT_PKL} ...")
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(user_delta_time, f)
log_checksum(OUTPUT_PKL, "输出：用户时间差序列字典")

# ==================== 生成数据概览 ====================
avg_delta = np.mean(all_deltas) if all_deltas else 0.0
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.2 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_SEQ}, {INPUT_TS}\n")
    f.write(f"输出文件: {OUTPUT_PKL}\n")
    f.write(f"序列固定长度: {MAX_SEQ_LEN}\n")
    f.write(f"时间差填充值: {DELTA_PAD} 小时\n")
    f.write(f"用户总数: {len(user_delta_time)}\n")
    f.write(f"有效时间差条目总数（非填充）: {len(all_deltas)}\n")
    f.write(f"有效时间差均值: {avg_delta:.2f} 小时\n")
    if all_deltas:
        f.write(f"有效时间差中位数: {np.median(all_deltas):.2f} 小时\n")
        f.write(f"有效时间差最大值: {max(all_deltas):.2f} 小时\n")
    else:
        f.write("有效时间差中位数: N/A\n有效时间差最大值: N/A\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 生成抽样 ====================
sample_users = list(user_delta_time.keys())[:3]   # 取前3个用户
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.2 抽样（用于人工检查）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("说明：展示部分用户的时间差序列（前20个位置，填充值=0小时）\n\n")
    for uid in sample_users:
        deltas = user_delta_time[uid]
        f.write(f"用户 {uid}:\n")
        f.write(f"  时间差序列 (前20): {deltas[:20]}\n")
        valid = [d for d in deltas if d != DELTA_PAD]
        if valid:
            f.write(f"  非填充时间差统计: 均值={np.mean(valid):.2f}h, 中位数={np.median(valid):.2f}h, 最大值={max(valid):.2f}h\n")
        else:
            f.write(f"  非填充时间差统计: 无有效时间差\n")
        f.write("\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 生成检查报告 ====================
negative_count = sum(1 for d in all_deltas if d < -0.001)   # 允许微小误差
length_errors = [uid for uid, deltas in user_delta_time.items() if len(deltas) != MAX_SEQ_LEN]
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.2 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"用户总数: {len(user_delta_time)}\n")
    f.write(f"序列固定长度: {MAX_SEQ_LEN}\n")
    f.write(f"时间差填充值: {DELTA_PAD} 小时\n\n")
    f.write("每个用户的有效时间差个数（即相邻行为对数）统计:\n")
    if valid_counts:
        f.write(f"  最小值: {min(valid_counts)}\n")
        f.write(f"  最大值: {max(valid_counts)}\n")
        f.write(f"  均值: {np.mean(valid_counts):.2f}\n")
        f.write(f"  中位数: {np.median(valid_counts):.0f}\n")
        f.write(f"  25分位数: {np.percentile(valid_counts, 25):.0f}\n")
        f.write(f"  75分位数: {np.percentile(valid_counts, 75):.0f}\n\n")
    else:
        f.write("  无有效时间差数据\n\n")
    f.write("所有有效时间差（小时）统计:\n")
    if all_deltas:
        f.write(f"  最小值: {min(all_deltas):.2f}\n")
        f.write(f"  最大值: {max(all_deltas):.2f}\n")
        f.write(f"  均值: {np.mean(all_deltas):.2f}\n")
        f.write(f"  标准差: {np.std(all_deltas):.2f}\n")
        f.write(f"  25分位数: {np.percentile(all_deltas, 25):.2f}\n")
        f.write(f"  75分位数: {np.percentile(all_deltas, 75):.2f}\n\n")
    else:
        f.write("  无有效时间差数据\n\n")
    f.write(f"负时间差数量（时间顺序异常）: {negative_count}\n")
    if negative_count == 0:
        f.write("✓ 所有时间差非负，时间顺序正确。\n\n")
    else:
        f.write("⚠ 存在负时间差，请检查时间排序是否正确。\n\n")
    f.write("序列长度一致性检查:\n")
    if length_errors:
        f.write(f"✗ 发现 {len(length_errors)} 个用户的时间差序列长度不等于 {MAX_SEQ_LEN}，示例: {length_errors[:5]}\n")
    else:
        f.write(f"✓ 所有用户的时间差序列长度均为 {MAX_SEQ_LEN}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 3.2 完成。")
print(f"输出目录: {INPUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_PKL}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
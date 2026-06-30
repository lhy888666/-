import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件（步骤2.1输出）
INPUT_CSV = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\train_actions_weighted.csv"

# 输出目录（步骤3.1输出）
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\3 序列数据构造")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUT_SEQ = OUT_DIR / "user_sequences.pkl"
OUT_TIMESTAMP = OUT_DIR / "user_timestamps.pkl"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step3_1_overview.txt"
SAMPLE_TXT = OUT_DIR / "step3_1_sample.txt"
CHECK_TXT = OUT_DIR / "step3_1_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step3_1_checksums.log"

# 序列参数
MAX_SEQ_LEN = 50
PAD_VALUE = -1          # 填充值（与有效物品ID 0~N 不冲突）
PAD_TIME = None         # 时间戳填充值

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
print("步骤 3.1：用户行为序列生成")
print(f"读取 {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"总记录数: {len(df)}")

# 检查必要列
required_cols = ['user_idx', 'item_idx', 'time']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"输入文件缺少必要列: {col}")

# 确保time列为datetime
if not pd.api.types.is_datetime64_any_dtype(df['time']):
    df['time'] = pd.to_datetime(df['time'])

# ==================== 2. 按用户分组，按时间排序，生成序列 ====================
print("按 user_idx 分组并按时间排序...")
# 按用户分组，每组按时间排序后提取 item_idx 和 time
grouped = df.groupby('user_idx').apply(
    lambda g: g.sort_values('time')[['item_idx', 'time']].values.tolist()
).to_dict()

user_sequences = {}
user_timestamps = {}
for user_idx, items_time_list in grouped.items():
    items = [int(pair[0]) for pair in items_time_list]
    times = [pair[1] for pair in items_time_list]

    # 截断：保留最近的 MAX_SEQ_LEN 条（时间升序，最近的在末尾）
    if len(items) > MAX_SEQ_LEN:
        items = items[-MAX_SEQ_LEN:]
        times = times[-MAX_SEQ_LEN:]
    # 填充：在开头填充 PAD_VALUE 和 PAD_TIME
    if len(items) < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - len(items)
        items = [PAD_VALUE] * pad_len + items
        times = [PAD_TIME] * pad_len + times

    user_sequences[user_idx] = items
    user_timestamps[user_idx] = times

print(f"生成用户序列数: {len(user_sequences)}")

# ==================== 3. 保存pkl文件并记录MD5 ====================
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUT_SEQ} ...")
with open(OUT_SEQ, 'wb') as f:
    pickle.dump(user_sequences, f)
log_checksum(OUT_SEQ, "输出：用户物品序列字典")

print(f"保存 {OUT_TIMESTAMP} ...")
with open(OUT_TIMESTAMP, 'wb') as f:
    pickle.dump(user_timestamps, f)
log_checksum(OUT_TIMESTAMP, "输出：用户时间戳序列字典")

# ==================== 4. 生成数据概览txt ====================
unique_users = len(user_sequences)
# 统计有效序列长度（原始行为数）
seq_lengths = [sum(1 for x in seq if x != PAD_VALUE) for seq in user_sequences.values()]

with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.1 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_CSV}\n")
    f.write(f"输入总记录数: {len(df)}\n")
    f.write(f"输入唯一用户数: {df['user_idx'].nunique()}\n")
    f.write(f"输入唯一物品数: {df['item_idx'].nunique()}\n")
    f.write(f"时间范围: {df['time'].min()} ～ {df['time'].max()}\n\n")
    f.write(f"输出文件: {OUT_SEQ}, {OUT_TIMESTAMP}\n")
    f.write(f"输出用户数: {unique_users}\n")
    f.write(f"序列固定长度: {MAX_SEQ_LEN}\n")
    f.write(f"填充值: {PAD_VALUE}\n")
    f.write(f"时间戳填充: {PAD_TIME}\n\n")
    f.write("有效序列长度（原始行为数）统计:\n")
    f.write(f"  最小值: {min(seq_lengths)}\n")
    f.write(f"  最大值: {max(seq_lengths)}\n")
    f.write(f"  均值: {np.mean(seq_lengths):.2f}\n")
    f.write(f"  中位数: {np.median(seq_lengths):.0f}\n")
    f.write(f"  25分位数: {np.percentile(seq_lengths, 25):.0f}\n")
    f.write(f"  75分位数: {np.percentile(seq_lengths, 75):.0f}\n\n")
    f.write("填充比例统计（填充位置占比）:\n")
    pad_ratios = [(MAX_SEQ_LEN - l) / MAX_SEQ_LEN for l in seq_lengths]
    f.write(f"  最小值: {min(pad_ratios):.4f}\n")
    f.write(f"  最大值: {max(pad_ratios):.4f}\n")
    f.write(f"  均值: {np.mean(pad_ratios):.4f}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 5. 生成抽样txt ====================
sample_users = list(user_sequences.keys())[:3]   # 取前3个用户
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.1 抽样（用于人工检查）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("说明：展示部分用户的序列前20个位置（填充值用 -1 表示）\n\n")
    for uid in sample_users:
        seq = user_sequences[uid]
        ts = user_timestamps[uid]
        f.write(f"用户 {uid}:\n")
        f.write(f"  物品序列 (前20): {seq[:20]}\n")
        # 时间戳格式化
        valid_ts = []
        for t in ts[:20]:
            if t is None:
                valid_ts.append('PAD')
            else:
                valid_ts.append(t.strftime('%Y-%m-%d %H:%M:%S'))
        f.write(f"  时间戳 (前20): {valid_ts}\n\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 6. 生成检查报告txt ====================
# 统计全填充用户
all_pad_users = [uid for uid, seq in user_sequences.items() if all(x == PAD_VALUE for x in seq)]
# 时间戳None统计
total_ts_slots = sum(len(ts) for ts in user_timestamps.values())
none_ts_count = sum(1 for ts_list in user_timestamps.values() for ts in ts_list if ts is None)

with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.1 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"用户总数: {unique_users}\n")
    f.write(f"序列固定长度: {MAX_SEQ_LEN}\n")
    f.write(f"填充值: {PAD_VALUE}\n\n")
    f.write("有效序列长度（原始行为数）统计:\n")
    f.write(f"  最小值: {min(seq_lengths)}\n")
    f.write(f"  最大值: {max(seq_lengths)}\n")
    f.write(f"  均值: {np.mean(seq_lengths):.2f}\n")
    f.write(f"  中位数: {np.median(seq_lengths):.0f}\n")
    f.write(f"  25分位数: {np.percentile(seq_lengths, 25):.0f}\n")
    f.write(f"  75分位数: {np.percentile(seq_lengths, 75):.0f}\n\n")
    f.write("填充比例统计:\n")
    pad_ratios = [(MAX_SEQ_LEN - l) / MAX_SEQ_LEN for l in seq_lengths]
    f.write(f"  最小值: {min(pad_ratios):.4f}\n")
    f.write(f"  最大值: {max(pad_ratios):.4f}\n")
    f.write(f"  均值: {np.mean(pad_ratios):.4f}\n\n")
    f.write("时间戳检查:\n")
    f.write(f"  总时间戳槽位数: {total_ts_slots}\n")
    f.write(f"  None（填充位）数量: {none_ts_count}\n")
    f.write(f"  非None数量: {total_ts_slots - none_ts_count}\n")
    f.write("  注意：填充位置的时间戳为None是正常的。\n\n")
    if all_pad_users:
        f.write(f"警告：发现 {len(all_pad_users)} 个用户的序列全为填充值（原始行为数为0），这些用户应被过滤掉。\n")
    else:
        f.write("✓ 所有用户至少有一条有效行为，无全填充序列。\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 3.1 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUT_SEQ}")
print(f"  - {OUT_TIMESTAMP}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
INPUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\3 序列数据构造")
INPUT_SEQ = INPUT_DIR / "user_sequences.pkl"
OUTPUT_OVERVIEW = INPUT_DIR / "step3_4_overview.txt"   # 只生成这个

SESSION_LEN = 10
STEP = 1
PAD_VALUE = -1

# ==================== 生成会话 ====================
def generate_sessions(user_sequences, session_len, step, pad_value):
    sessions = []
    for user_idx, seq in user_sequences.items():
        valid_seq = [x for x in seq if x != pad_value]
        if len(valid_seq) < session_len + 1:
            continue
        for start in range(0, len(valid_seq) - session_len, step):
            session = valid_seq[start:start + session_len]
            target = valid_seq[start + session_len]
            sessions.append([user_idx, session, target])
    return pd.DataFrame(sessions, columns=['user_idx', 'session_seq', 'next_item'])

# ==================== 主处理 ====================
print("步骤 3.4：Session 子序列滑窗构造（仅生成概览）")
print(f"加载 {INPUT_SEQ} ...")
with open(INPUT_SEQ, 'rb') as f:
    user_sequences = pickle.load(f)

print(f"用户总数: {len(user_sequences)}")

df_sessions = generate_sessions(user_sequences, SESSION_LEN, STEP, PAD_VALUE)
print(f"去重前样本数: {len(df_sessions)}")

if len(df_sessions) > 0:
    # 添加字符串列用于去重
    df_sessions['session_seq_str'] = df_sessions['session_seq'].apply(lambda x: ','.join(map(str, x)))
    df_sessions = df_sessions.drop_duplicates(subset=['user_idx', 'session_seq_str', 'next_item'])
    df_sessions = df_sessions.reset_index(drop=True)
    print(f"去重后样本数: {len(df_sessions)}")
else:
    print("无会话样本生成。")

total_sessions = len(df_sessions)
unique_users = df_sessions['user_idx'].nunique() if total_sessions > 0 else 0

# ==================== 只输出数据概览 ====================
with open(OUTPUT_OVERVIEW, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 3.4 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_SEQ}\n")
    f.write(f"会话窗口长度: {SESSION_LEN}\n")
    f.write(f"滑动步长: {STEP}\n")
    f.write(f"填充值: {PAD_VALUE}\n\n")
    f.write(f"生成的会话样本总数: {total_sessions}\n")
    f.write(f"覆盖的用户数: {unique_users}\n")
    if total_sessions > 0:
        f.write(f"平均每个用户的会话数: {total_sessions / unique_users:.2f}\n")
        f.write("\n目标物品（next_item）出现频次（前10）:\n")
        target_counts = df_sessions['next_item'].value_counts().head(10)
        for item, cnt in target_counts.items():
            f.write(f"  item {item}: {cnt} 次\n")
    else:
        f.write("\n警告：未生成任何会话样本，请检查用户序列长度是否均小于会话窗口长度+1。\n")

print(f"数据概览已保存至 {OUTPUT_OVERVIEW}")
print("\n步骤 3.4 完成（仅生成概览）。")

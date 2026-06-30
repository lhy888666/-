import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件（步骤1.2输出）
D_MAPPED_PATH = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\D_mapped.csv"

# 输出目录（与输入同目录）
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUTPUT_CSV = OUT_DIR / "train_actions_weighted.csv"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step2_1_overview.txt"
SAMPLE_TXT = OUT_DIR / "step2_1_sample.txt"
CHECK_TXT = OUT_DIR / "step2_1_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step2_1_checksums.log"

# 行为权重映射规则
WEIGHT_MAP = {
    1: 0.2,   # 浏览
    2: 1.0,   # 收藏
    3: 1.5,   # 加购
    4: 3.0    # 购买
}

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

# ==================== 1. 读取并筛选训练窗口数据 ====================
print("步骤 2.1：行为权重映射")
print(f"读取 {D_MAPPED_PATH} ...")
df = pd.read_csv(D_MAPPED_PATH)
original_count = len(df)

# 只保留训练窗口数据
df_train = df[df['split'] == 'train'].copy()
print(f"训练窗口记录数: {len(df_train)} (总记录数: {original_count})")

# ==================== 2. 添加权重列 ====================
print("添加 weight 列...")
df_train['weight'] = df_train['behavior_type'].map(WEIGHT_MAP)
# 检查是否有未映射的行为类型（理论上behavior_type只有1-4）
if df_train['weight'].isna().any():
    missing = df_train[df_train['weight'].isna()]['behavior_type'].unique()
    print(f"警告: 发现未映射的行为类型: {missing}，将填充为0")
    df_train['weight'] = df_train['weight'].fillna(0.0)

# ==================== 3. 保存输出文件 ====================
print(f"保存 {OUTPUT_CSV} ...")
df_train.to_csv(OUTPUT_CSV, index=False)
log_checksum(OUTPUT_CSV, "输出：训练窗口加权行为数据")

# ==================== 4. 生成数据概览txt ====================
unique_users = df_train['user_idx'].nunique()
unique_items = df_train['item_idx'].nunique()
total_weight_sum = df_train['weight'].sum()
avg_weight = df_train['weight'].mean()

with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.1 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {D_MAPPED_PATH}\n")
    f.write(f"输出文件: {OUTPUT_CSV}\n")
    f.write(f"训练窗口记录数: {len(df_train)}\n")
    f.write(f"唯一用户数: {unique_users}\n")
    f.write(f"唯一物品数: {unique_items}\n")
    f.write(f"总权重和: {total_weight_sum:.2f}\n")
    f.write(f"平均权重: {avg_weight:.4f}\n\n")
    f.write("行为类型权重映射规则:\n")
    for bt, w in WEIGHT_MAP.items():
        behavior_name = {1: '浏览', 2: '收藏', 3: '加购', 4: '购买'}.get(bt, '未知')
        f.write(f"  behavior_type={bt} ({behavior_name}) -> weight={w}\n")
    f.write("\n行为类型分布（训练窗口）:\n")
    behavior_cnt = df_train['behavior_type'].value_counts().sort_index()
    for bt, cnt in behavior_cnt.items():
        f.write(f"  behavior_type={bt}: {cnt} 条\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 5. 生成抽样txt ====================
sample_size = min(20, len(df_train))
sample_df = df_train.head(sample_size)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.1 抽样（行为权重示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 条记录的权重分配:\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"行 {idx+1}: user_idx={row['user_idx']}, item_idx={row['item_idx']}, "
                f"behavior_type={row['behavior_type']} -> weight={row['weight']}\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 6. 生成检查报告txt ====================
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.1 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("权重映射完整性检查:\n")
    f.write(f"  所有行为类型均已分配权重（1-4）: {df_train['weight'].notna().all()}\n")
    f.write(f"  权重最小值: {df_train['weight'].min()}, 最大值: {df_train['weight'].max()}\n\n")
    f.write("用户加权交互和统计（仅训练窗口）:\n")
    user_weight_sum = df_train.groupby('user_idx')['weight'].sum()
    f.write(f"  均值: {user_weight_sum.mean():.2f}\n")
    f.write(f"  标准差: {user_weight_sum.std():.2f}\n")
    f.write(f"  最小值: {user_weight_sum.min():.2f}\n")
    f.write(f"  最大值: {user_weight_sum.max():.2f}\n\n")
    f.write("物品加权交互和统计:\n")
    item_weight_sum = df_train.groupby('item_idx')['weight'].sum()
    f.write(f"  均值: {item_weight_sum.mean():.2f}\n")
    f.write(f"  标准差: {item_weight_sum.std():.2f}\n")
    f.write(f"  最小值: {item_weight_sum.min():.2f}\n")
    f.write(f"  最大值: {item_weight_sum.max():.2f}\n\n")
    f.write("数据完整性:\n")
    f.write(f"  无缺失 weight 值的记录: {df_train['weight'].isna().sum() == 0}\n")
    f.write(f"  输出文件记录数: {len(df_train)}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 2.1 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
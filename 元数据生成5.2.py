import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件（步骤2.1输出）
INPUT_CSV = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\train_actions_weighted.csv"

# 输出目录
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\5 元数据生成")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUT_USER_FEAT = OUT_DIR / "user_stat_features.csv"
OUT_ITEM_FEAT = OUT_DIR / "item_stat_features.csv"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step5_2_overview.txt"
SAMPLE_TXT = OUT_DIR / "step5_2_sample.txt"
CHECK_TXT = OUT_DIR / "step5_2_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step5_2_checksums.log"

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
print("步骤 5.2：统计特征提取")
print(f"读取 {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"总记录数: {len(df)}")

# 记录输入文件MD5
log_checksum(INPUT_CSV, "输入：训练窗口加权行为数据")

# 确保time列为datetime
if not pd.api.types.is_datetime64_any_dtype(df['time']):
    df['time'] = pd.to_datetime(df['time'])

# ==================== 2. 用户级统计特征 ====================
print("计算用户统计特征...")
# 按用户分组
user_group = df.groupby('user_idx')

# 基础统计
user_stats = pd.DataFrame()
user_stats['user_idx'] = user_group['user_idx'].first()
user_stats['total_actions'] = user_group.size()
user_stats['total_buy'] = user_group['behavior_type'].apply(lambda x: (x == 4).sum())
user_stats['buy_rate'] = user_stats['total_buy'] / user_stats['total_actions']
user_stats['total_view'] = user_group['behavior_type'].apply(lambda x: (x == 1).sum())
user_stats['total_fav'] = user_group['behavior_type'].apply(lambda x: (x == 2).sum())
user_stats['total_cart'] = user_group['behavior_type'].apply(lambda x: (x == 3).sum())
user_stats['avg_weight'] = user_group['weight'].mean()
user_stats['total_weight'] = user_group['weight'].sum()
# 活跃天数（唯一日期）
user_stats['active_days'] = user_group['time'].apply(lambda x: x.dt.date.nunique())
# 平均每小时行为数（假设时间跨度）
time_range = (df['time'].max() - df['time'].min()).total_seconds() / 3600
user_stats['actions_per_hour'] = user_stats['total_actions'] / time_range if time_range > 0 else 0
# 购买行为的平均权重（购买权重固定为3.0，此处计算用户购买时的平均权重可能无意义，跳过）
# 用户购买商品种类数（去重）
user_stats['distinct_items_bought'] = user_group.apply(lambda g: g[g['behavior_type'] == 4]['item_idx'].nunique())
user_stats['distinct_items_interacted'] = user_group['item_idx'].nunique()

print(f"用户特征表大小: {len(user_stats)}")

# ==================== 3. 物品级统计特征 ====================
print("计算物品统计特征...")
item_group = df.groupby('item_idx')

item_stats = pd.DataFrame()
item_stats['item_idx'] = item_group['item_idx'].first()
item_stats['total_interactions'] = item_group.size()
item_stats['total_buy'] = item_group['behavior_type'].apply(lambda x: (x == 4).sum())
item_stats['buy_rate'] = item_stats['total_buy'] / item_stats['total_interactions']
item_stats['total_view'] = item_group['behavior_type'].apply(lambda x: (x == 1).sum())
item_stats['total_fav'] = item_group['behavior_type'].apply(lambda x: (x == 2).sum())
item_stats['total_cart'] = item_group['behavior_type'].apply(lambda x: (x == 3).sum())
item_stats['avg_weight'] = item_group['weight'].mean()
item_stats['total_weight'] = item_group['weight'].sum()
item_stats['distinct_users'] = item_group['user_idx'].nunique()

print(f"物品特征表大小: {len(item_stats)}")

# ==================== 4. 保存输出 ====================
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUT_USER_FEAT} ...")
user_stats.to_csv(OUT_USER_FEAT, index=False)
log_checksum(OUT_USER_FEAT, "输出：用户统计特征")

print(f"保存 {OUT_ITEM_FEAT} ...")
item_stats.to_csv(OUT_ITEM_FEAT, index=False)
log_checksum(OUT_ITEM_FEAT, "输出：物品统计特征")

# ==================== 5. 生成数据概览txt ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.2 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_CSV}\n")
    f.write(f"输入总记录数: {len(df)}\n")
    f.write(f"唯一用户数: {df['user_idx'].nunique()}\n")
    f.write(f"唯一物品数: {df['item_idx'].nunique()}\n\n")
    f.write(f"输出用户特征文件: {OUT_USER_FEAT.name}\n")
    f.write(f"输出用户特征数: {len(user_stats)}\n")
    f.write(f"用户特征列: {list(user_stats.columns)}\n\n")
    f.write(f"输出物品特征文件: {OUT_ITEM_FEAT.name}\n")
    f.write(f"输出物品特征数: {len(item_stats)}\n")
    f.write(f"物品特征列: {list(item_stats.columns)}\n\n")
    f.write("用户特征统计摘要:\n")
    f.write(f"  total_actions: min={user_stats['total_actions'].min()}, max={user_stats['total_actions'].max()}, mean={user_stats['total_actions'].mean():.2f}\n")
    f.write(f"  buy_rate: min={user_stats['buy_rate'].min():.4f}, max={user_stats['buy_rate'].max():.4f}, mean={user_stats['buy_rate'].mean():.4f}\n")
    f.write("物品特征统计摘要:\n")
    f.write(f"  total_interactions: min={item_stats['total_interactions'].min()}, max={item_stats['total_interactions'].max()}, mean={item_stats['total_interactions'].mean():.2f}\n")
    f.write(f"  buy_rate: min={item_stats['buy_rate'].min():.4f}, max={item_stats['buy_rate'].max():.4f}, mean={item_stats['buy_rate'].mean():.4f}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 6. 生成抽样txt ====================
user_sample = user_stats.head(10)
item_sample = item_stats.head(10)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.2 抽样（特征示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("用户特征（前10条）:\n")
    f.write(user_sample.to_string(index=False) + "\n\n")
    f.write("物品特征（前10条）:\n")
    f.write(item_sample.to_string(index=False) + "\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 7. 生成检查报告txt ====================
user_null = user_stats.isnull().sum().sum()
item_null = item_stats.isnull().sum().sum()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.2 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("用户特征表:\n")
    f.write(f"  记录数: {len(user_stats)}\n")
    f.write(f"  缺失值总数: {user_null}\n")
    f.write(f"  无缺失值: {user_null == 0}\n")
    f.write("物品特征表:\n")
    f.write(f"  记录数: {len(item_stats)}\n")
    f.write(f"  缺失值总数: {item_null}\n")
    f.write(f"  无缺失值: {item_null == 0}\n")
    f.write("\n业务合理性检查:\n")
    f.write(f"  用户购买率范围: [{user_stats['buy_rate'].min():.4f}, {user_stats['buy_rate'].max():.4f}]\n")
    f.write(f"  物品购买率范围: [{item_stats['buy_rate'].min():.4f}, {item_stats['buy_rate'].max():.4f}]\n")
    f.write("  注意：购买率可能为0或1，属于正常情况。\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 5.2 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUT_USER_FEAT}")
print(f"  - {OUT_ITEM_FEAT}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
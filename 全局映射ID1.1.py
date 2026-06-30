import pandas as pd
import pickle
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件
D_WITH_SPLIT_PATH = "D_with_split.csv"  # 步骤0.1输出（带split标签）
P_CLEANED_PATH = "cleaned_item_subset.csv"  # 清洗后的商品子集表

# 输出目录（存放pkl和txt）
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出pkl文件
USER2IDX_PATH = OUT_DIR / "user2idx.pkl"
ITEM2IDX_PATH = OUT_DIR / "item2idx.pkl"
CATE2IDX_PATH = OUT_DIR / "cate2idx.pkl"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step1_1_overview.txt"
SAMPLE_TXT = OUT_DIR / "step1_1_sample.txt"
CHECK_TXT = OUT_DIR / "step1_1_check.txt"

# MD5日志文件（可选，存放于同一目录）
CHECKSUM_LOG = OUT_DIR / "step1_1_checksums.log"

# 特殊ID常量
UNK_USER_ID = "<UNK_USER>"
UNK_ITEM_ID = "<UNK_ITEM>"
UNK_CATE_ID = "<UNK_CATE>"


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
print("步骤 1.1：建立全局 ID 映射字典（基于训练窗口）")
print("加载数据...")
df_d = pd.read_csv(D_WITH_SPLIT_PATH)
df_p = pd.read_csv(P_CLEANED_PATH)

print("输入文件 MD5:")
log_checksum(D_WITH_SPLIT_PATH, "输入：带split标签的用户行为表")
log_checksum(P_CLEANED_PATH, "输入：清洗后的商品子集表")

# 只使用训练窗口数据
df_train = df_d[df_d['split'] == 'train'].copy()
print(f"训练窗口记录数: {len(df_train)}")

# ==================== 2. 提取训练窗口中出现的实体 ====================
train_users = set(df_train['user_id'].unique())
train_items = set(df_train['item_id'].unique())
train_cates = set(df_train['item_category'].unique())
train_cates.update(df_p['item_category'].unique())  # 确保P中的类别也有映射

print(f"训练窗口用户数: {len(train_users)}")
print(f"训练窗口商品数: {len(train_items)}")
print(f"训练窗口+商品子集类别数: {len(train_cates)}")

# ==================== 3. 构建映射字典 ====================
user2idx = {user: idx for idx, user in enumerate(sorted(train_users))}
user2idx[UNK_USER_ID] = len(user2idx)

item2idx = {item: idx for idx, item in enumerate(sorted(train_items))}
item2idx[UNK_ITEM_ID] = len(item2idx)

cate2idx = {cate: idx for idx, cate in enumerate(sorted(train_cates))}
cate2idx[UNK_CATE_ID] = len(cate2idx)

print(f"用户映射表大小: {len(user2idx)} (含UNK)")
print(f"商品映射表大小: {len(item2idx)} (含UNK)")
print(f"类别映射表大小: {len(cate2idx)} (含UNK)")

# ==================== 4. 保存pkl文件并记录MD5 ====================
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()  # 清空旧日志

with open(USER2IDX_PATH, 'wb') as f:
    pickle.dump(user2idx, f)
print(f"保存 {USER2IDX_PATH}")
log_checksum(USER2IDX_PATH, "输出：用户ID映射字典")

with open(ITEM2IDX_PATH, 'wb') as f:
    pickle.dump(item2idx, f)
print(f"保存 {ITEM2IDX_PATH}")
log_checksum(ITEM2IDX_PATH, "输出：商品ID映射字典")

with open(CATE2IDX_PATH, 'wb') as f:
    pickle.dump(cate2idx, f)
print(f"保存 {CATE2IDX_PATH}")
log_checksum(CATE2IDX_PATH, "输出：商品类别ID映射字典")

# ==================== 5. 生成数据概览txt ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 1.1 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {D_WITH_SPLIT_PATH}\n")
    f.write(f"输入文件: {P_CLEANED_PATH}\n")
    f.write(f"训练窗口记录数: {len(df_train)}\n")
    f.write(f"训练窗口唯一用户数: {len(train_users)}\n")
    f.write(f"训练窗口唯一商品数: {len(train_items)}\n")
    f.write(f"训练窗口唯一类别数（含P表补充）: {len(train_cates)}\n\n")
    f.write("映射字典大小（含UNK）:\n")
    f.write(f"  user2idx: {len(user2idx)}\n")
    f.write(f"  item2idx: {len(item2idx)}\n")
    f.write(f"  cate2idx: {len(cate2idx)}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 6. 生成抽样txt ====================
sample_size = min(10, len(user2idx))
sample_user_items = list(user2idx.items())[:sample_size]
sample_item_items = list(item2idx.items())[:sample_size]
sample_cate_items = list(cate2idx.items())[:sample_size]

with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 1.1 抽样（映射示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"用户映射示例（前{sample_size}个）:\n")
    for k, v in sample_user_items:
        f.write(f"  {k} -> {v}\n")
    f.write(f"\n商品映射示例（前{sample_size}个）:\n")
    for k, v in sample_item_items:
        f.write(f"  {k} -> {v}\n")
    f.write(f"\n类别映射示例（前{sample_size}个）:\n")
    for k, v in sample_cate_items:
        f.write(f"  {k} -> {v}\n")
    f.write(f"\nUNK 映射:\n")
    f.write(f"  {UNK_USER_ID} -> {user2idx[UNK_USER_ID]}\n")
    f.write(f"  {UNK_ITEM_ID} -> {item2idx[UNK_ITEM_ID]}\n")
    f.write(f"  {UNK_CATE_ID} -> {cate2idx[UNK_CATE_ID]}\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 7. 生成检查报告txt ====================
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 1.1 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("实体提取完整性检查:\n")
    if len(train_users) > 0:
        f.write(f"  ✓ 训练窗口用户数: {len(train_users)}\n")
    else:
        f.write("  ✗ 训练窗口用户数为0\n")
    if len(train_items) > 0:
        f.write(f"  ✓ 训练窗口商品数: {len(train_items)}\n")
    else:
        f.write("  ✗ 训练窗口商品数为0\n")
    if len(train_cates) > 0:
        f.write(f"  ✓ 训练窗口+商品子集类别数: {len(train_cates)}\n")
    else:
        f.write("  ✗ 类别数为0\n\n")

    f.write("映射字典一致性检查:\n")
    f.write(f"  user2idx 包含所有训练用户: {len(train_users) <= len(user2idx) - 1}\n")
    f.write(f"  item2idx 包含所有训练商品: {len(train_items) <= len(item2idx) - 1}\n")
    f.write(f"  cate2idx 包含所有训练类别及P表类别: {len(train_cates) <= len(cate2idx) - 1}\n\n")

    f.write("UNK预留检查:\n")
    f.write(f"  UNK_USER 索引: {user2idx[UNK_USER_ID]}\n")
    f.write(f"  UNK_ITEM 索引: {item2idx[UNK_ITEM_ID]}\n")
    f.write(f"  UNK_CATE 索引: {cate2idx[UNK_CATE_ID]}\n")
    f.write("  注：UNK索引为映射表最后一个位置，用于处理未登录实体。\n\n")

    f.write("输出文件清单:\n")
    f.write(f"  - {USER2IDX_PATH}\n")
    f.write(f"  - {ITEM2IDX_PATH}\n")
    f.write(f"  - {CATE2IDX_PATH}\n")
    f.write(f"  - {OVERVIEW_TXT}\n")
    f.write(f"  - {SAMPLE_TXT}\n")
    f.write(f"  - {CHECK_TXT}\n")
    f.write(f"  - {CHECKSUM_LOG}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 1.1 完成。")
print(f"输出目录: {OUT_DIR}")
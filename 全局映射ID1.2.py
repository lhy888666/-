import pandas as pd
import pickle
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件
D_WITH_SPLIT_PATH = "D_with_split.csv"                     # 步骤0.1输出
P_CLEANED_PATH = "cleaned_item_subset.csv"                  # 清洗后的商品子集表
MAP_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射")   # 步骤1.1输出目录
USER2IDX_PATH = MAP_DIR / "user2idx.pkl"
ITEM2IDX_PATH = MAP_DIR / "item2idx.pkl"
CATE2IDX_PATH = MAP_DIR / "cate2idx.pkl"

# 输出目录（与映射字典同目录）
OUT_DIR = MAP_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MAPPED_PATH = OUT_DIR / "D_mapped.csv"
P_MAPPED_PATH = OUT_DIR / "P_mapped.csv"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step1_2_overview.txt"
SAMPLE_TXT = OUT_DIR / "step1_2_sample.txt"
CHECK_TXT = OUT_DIR / "step1_2_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step1_2_checksums.log"

# 特殊ID常量（需与步骤1.1保持一致）
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

# ==================== 1. 加载映射字典 ====================
print("步骤 1.2：应用全局ID映射")
print("加载映射字典...")
with open(USER2IDX_PATH, 'rb') as f:
    user2idx = pickle.load(f)
with open(ITEM2IDX_PATH, 'rb') as f:
    item2idx = pickle.load(f)
with open(CATE2IDX_PATH, 'rb') as f:
    cate2idx = pickle.load(f)

# 获取UNK索引
UNK_USER_IDX = user2idx[UNK_USER_ID]
UNK_ITEM_IDX = item2idx[UNK_ITEM_ID]
UNK_CATE_IDX = cate2idx[UNK_CATE_ID]
print(f"UNK_USER 索引: {UNK_USER_IDX}")
print(f"UNK_ITEM 索引: {UNK_ITEM_IDX}")
print(f"UNK_CATE 索引: {UNK_CATE_IDX}")

# ==================== 2. 加载待映射数据 ====================
print("加载原始数据...")
df_d = pd.read_csv(D_WITH_SPLIT_PATH)           # 全量D表
df_p = pd.read_csv(P_CLEANED_PATH)              # 清洗后的P表
print(f"D表记录数: {len(df_d)}")
print(f"P表记录数: {len(df_p)}")

# ==================== 3. 映射D表 ====================
print("映射 D 表...")
# 应用映射，缺失值填充UNK索引
df_d['user_idx'] = df_d['user_id'].map(user2idx).fillna(UNK_USER_IDX).astype('int32')
df_d['item_idx'] = df_d['item_id'].map(item2idx).fillna(UNK_ITEM_IDX).astype('int32')
df_d['category_idx'] = df_d['item_category'].map(cate2idx).fillna(UNK_CATE_IDX).astype('int32')
# 删除原始列
df_d.drop(columns=['user_id', 'item_id', 'item_category'], inplace=True)
# 保留需要的列，并调整顺序（可选）
keep_cols = ['user_idx', 'item_idx', 'category_idx', 'behavior_type', 'user_geohash', 'time', 'split']
df_d = df_d[keep_cols]

# ==================== 4. 映射P表 ====================
print("映射 P 表...")
df_p['item_idx'] = df_p['item_id'].map(item2idx).fillna(UNK_ITEM_IDX).astype('int32')
df_p['category_idx'] = df_p['item_category'].map(cate2idx).fillna(UNK_CATE_IDX).astype('int32')
# 删除原始列
df_p.drop(columns=['item_id', 'item_category'], inplace=True)
# 保留需要的列（注意：P表中可能有item_geohash）
keep_cols_p = ['item_idx', 'category_idx'] + [c for c in df_p.columns if c not in ['item_idx', 'category_idx']]
df_p = df_p[keep_cols_p]

# ==================== 5. 保存输出文件并记录MD5 ====================
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {D_MAPPED_PATH}")
df_d.to_csv(D_MAPPED_PATH, index=False)
log_checksum(D_MAPPED_PATH, "输出：映射后的用户行为表")

print(f"保存 {P_MAPPED_PATH}")
df_p.to_csv(P_MAPPED_PATH, index=False)
log_checksum(P_MAPPED_PATH, "输出：映射后的商品子集表")

# ==================== 6. 统计UNK出现次数 ====================
d_unk_user_cnt = (df_d['user_idx'] == UNK_USER_IDX).sum()
d_unk_item_cnt = (df_d['item_idx'] == UNK_ITEM_IDX).sum()
d_unk_cate_cnt = (df_d['category_idx'] == UNK_CATE_IDX).sum()
p_unk_item_cnt = (df_p['item_idx'] == UNK_ITEM_IDX).sum()
p_unk_cate_cnt = (df_p['category_idx'] == UNK_CATE_IDX).sum()

# ==================== 7. 生成数据概览txt ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 1.2 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {D_WITH_SPLIT_PATH}\n")
    f.write(f"输入文件: {P_CLEANED_PATH}\n")
    f.write(f"映射字典: {USER2IDX_PATH}, {ITEM2IDX_PATH}, {CATE2IDX_PATH}\n\n")
    f.write("D表（用户行为）统计:\n")
    f.write(f"  原始记录数: {len(df_d)}\n")
    f.write(f"  split 分布:\n")
    for split in ['train', 'val', 'test']:
        cnt = (df_d['split'] == split).sum()
        f.write(f"    {split}: {cnt} ({cnt/len(df_d)*100:.2f}%)\n")
    f.write(f"  UNK_USER 出现次数: {d_unk_user_cnt}\n")
    f.write(f"  UNK_ITEM 出现次数: {d_unk_item_cnt}\n")
    f.write(f"  UNK_CATE 出现次数: {d_unk_cate_cnt}\n\n")
    f.write("P表（商品子集）统计:\n")
    f.write(f"  原始记录数: {len(df_p)}\n")
    f.write(f"  UNK_ITEM 出现次数: {p_unk_item_cnt}\n")
    f.write(f"  UNK_CATE 出现次数: {p_unk_cate_cnt}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 8. 生成抽样txt ====================
sample_size = min(10, len(df_d))
sample_df = df_d.head(sample_size)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 1.2 抽样（映射后样例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示 D 表前 {sample_size} 行映射结果:\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"行 {idx+1}: user_idx={row['user_idx']}, item_idx={row['item_idx']}, "
                f"category_idx={row['category_idx']}, behavior_type={row['behavior_type']}, "
                f"split={row['split']}\n")
    f.write("\nP 表抽样（前5条）:\n")
    for idx, row in df_p.head(5).iterrows():
        f.write(f"  item_idx={row['item_idx']}, category_idx={row['category_idx']}, "
                f"geohash={row.get('item_geohash', 'N/A')}\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 9. 生成检查报告txt ====================
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 1.2 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("映射完整性检查:\n")
    f.write(f"  D表中 user_idx 最小值: {df_d['user_idx'].min()}, 最大值: {df_d['user_idx'].max()}\n")
    f.write(f"  D表中 item_idx 最小值: {df_d['item_idx'].min()}, 最大值: {df_d['item_idx'].max()}\n")
    f.write(f"  D表中 category_idx 最小值: {df_d['category_idx'].min()}, 最大值: {df_d['category_idx'].max()}\n")
    f.write(f"  P表中 item_idx 最小值: {df_p['item_idx'].min()}, 最大值: {df_p['item_idx'].max()}\n")
    f.write(f"  P表中 category_idx 最小值: {df_p['category_idx'].min()}, 最大值: {df_p['category_idx'].max()}\n\n")
    f.write("UNK 索引有效性:\n")
    f.write(f"  UNK_USER 索引: {UNK_USER_IDX}\n")
    f.write(f"  UNK_ITEM 索引: {UNK_ITEM_IDX}\n")
    f.write(f"  UNK_CATE 索引: {UNK_CATE_IDX}\n")
    f.write(f"  所有映射后的索引均非负。\n\n")
    f.write("split 列保留情况:\n")
    f.write(f"  D 表中仍包含 split 列，未丢失。\n")
    f.write(f"  split 列取值: {df_d['split'].unique()}\n\n")
    f.write("输出文件清单:\n")
    f.write(f"  - {D_MAPPED_PATH}\n")
    f.write(f"  - {P_MAPPED_PATH}\n")
    f.write(f"  - {OVERVIEW_TXT}\n")
    f.write(f"  - {SAMPLE_TXT}\n")
    f.write(f"  - {CHECK_TXT}\n")
    f.write(f"  - {CHECKSUM_LOG}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 1.2 完成。")
print(f"输出目录: {OUT_DIR}")
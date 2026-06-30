import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件（步骤1.2输出）
INPUT_CSV = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\P_mapped.csv"

# 输出目录
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\4 图数据构造")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUTPUT_CSV = OUT_DIR / "edge_item_category.csv"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step4_2_overview.txt"
SAMPLE_TXT = OUT_DIR / "step4_2_sample.txt"
CHECK_TXT = OUT_DIR / "step4_2_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step4_2_checksums.log"

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
print("步骤 4.2：物品-类目异构图边列表构造")
print(f"读取 {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"原始商品记录数: {len(df)}")

# 记录输入文件MD5
log_checksum(INPUT_CSV, "输入：映射后的商品子集表")

# ==================== 2. 提取边并去重 ====================
print("提取 (item_idx, category_idx) 并去重...")
# 确保列存在
required_cols = ['item_idx', 'category_idx']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"输入文件缺少必要列: {col}")
edges = df[['item_idx', 'category_idx']].drop_duplicates()
print(f"去重后边数: {len(edges)}")

# ==================== 3. 统计信息 ====================
unique_items = edges['item_idx'].nunique()
unique_categories = edges['category_idx'].nunique()
print(f"唯一物品数: {unique_items}, 唯一类目数: {unique_categories}")

# ==================== 4. 保存输出 ====================
print(f"保存 {OUTPUT_CSV} ...")
edges.to_csv(OUTPUT_CSV, index=False)
log_checksum(OUTPUT_CSV, "输出：物品-类目边列表")

# ==================== 5. 生成数据概览txt ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 4.2 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_CSV}\n")
    f.write(f"原始商品记录数: {len(df)}\n")
    f.write(f"去重后边数: {len(edges)}\n")
    f.write(f"唯一物品数: {unique_items}\n")
    f.write(f"唯一类目数: {unique_categories}\n")
    f.write(f"平均每个物品对应的类目数: {len(edges)/unique_items:.2f}\n" if unique_items > 0 else "平均每个物品对应的类目数: N/A\n")
    f.write(f"平均每个类目对应的物品数: {len(edges)/unique_categories:.2f}\n" if unique_categories > 0 else "平均每个类目对应的物品数: N/A\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 6. 生成抽样txt ====================
sample_size = min(20, len(edges))
sample_df = edges.head(sample_size)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 4.2 抽样（边示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 条边 (item_idx, category_idx):\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"  ({row['item_idx']}, {row['category_idx']})\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 7. 生成检查报告txt ====================
# 检查重复（去重后应无）
dup_count = edges.duplicated().sum()
# 检查索引合法性
item_min = edges['item_idx'].min()
cate_min = edges['category_idx'].min()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 4.2 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("重复检查:\n")
    f.write(f"  重复边数量: {dup_count} (应为0)\n")
    f.write(f"  去重后边数: {len(edges)}\n\n")
    f.write("索引合法性:\n")
    f.write(f"  item_idx 最小值: {item_min}\n")
    f.write(f"  category_idx 最小值: {cate_min}\n")
    f.write(f"  所有索引均非负: {item_min >= 0 and cate_min >= 0}\n\n")
    f.write("数据完整性:\n")
    f.write(f"  无缺失值: {edges.isnull().sum().sum() == 0}\n")
    f.write(f"  输出文件记录数: {len(edges)}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 4.2 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
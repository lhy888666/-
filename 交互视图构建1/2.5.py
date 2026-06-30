import pandas as pd
import numpy as np
import scipy.sparse as sp
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
# 输入文件（步骤2.1输出）
INPUT_CSV = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\train_actions_weighted.csv"

# 输出目录
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\2 交互视图构造")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
OUTPUT_NPZ = OUT_DIR / "user_category_matrix.npz"

# 输出txt报告
OVERVIEW_TXT = OUT_DIR / "step2_5_overview.txt"
SAMPLE_TXT = OUT_DIR / "step2_5_sample.txt"
CHECK_TXT = OUT_DIR / "step2_5_check.txt"

# MD5日志
CHECKSUM_LOG = OUT_DIR / "step2_5_checksums.log"

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
print("步骤 2.5：用户-类目交互统计")
print(f"读取 {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"总记录数: {len(df)}")

# 检查必要列
required_cols = ['user_idx', 'category_idx', 'weight']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"输入文件缺少必要列: {col}")

# ==================== 2. 聚合 ====================
print("按 (user_idx, category_idx) 聚合 weight 的和...")
df_agg = df.groupby(['user_idx', 'category_idx'], as_index=False)['weight'].sum()
print(f"聚合后非零交互对数: {len(df_agg)}")

# ==================== 3. 获取矩阵维度 ====================
num_users = df_agg['user_idx'].max() + 1
num_categories = df_agg['category_idx'].max() + 1
print(f"用户数: {num_users}, 类目数: {num_categories}")

# ==================== 4. 构建稀疏矩阵（CSR格式） ====================
print("构建稀疏矩阵...")
rows = df_agg['user_idx'].values
cols = df_agg['category_idx'].values
data = df_agg['weight'].values.astype(np.float32)

coo = sp.coo_matrix((data, (rows, cols)), shape=(num_users, num_categories))
csr = coo.tocsr()
print(f"矩阵非零元素数: {csr.nnz}")
print(f"矩阵密度: {csr.nnz / (num_users * num_categories):.6f}")

# ==================== 5. 保存为.npz文件 ====================
print(f"保存 {OUTPUT_NPZ} ...")
sp.save_npz(OUTPUT_NPZ, csr)
log_checksum(OUTPUT_NPZ, "输出：用户-类目评分矩阵（CSR格式）")

# ==================== 6. 生成数据概览txt ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.5 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_CSV}\n")
    f.write(f"输入记录数: {len(df)}\n")
    f.write(f"聚合后非零交互对数: {len(df_agg)}\n\n")
    f.write(f"输出文件: {OUTPUT_NPZ}\n")
    f.write(f"矩阵形状: {num_users} 用户 × {num_categories} 类目\n")
    f.write(f"非零元素数: {csr.nnz}\n")
    f.write(f"矩阵密度: {csr.nnz / (num_users * num_categories):.8f}\n")
    f.write(f"数据类型: {csr.data.dtype}\n\n")
    f.write("weight 统计:\n")
    f.write(f"  最小值: {df_agg['weight'].min():.4f}\n")
    f.write(f"  最大值: {df_agg['weight'].max():.4f}\n")
    f.write(f"  均值: {df_agg['weight'].mean():.4f}\n")
    f.write(f"  标准差: {df_agg['weight'].std():.4f}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 7. 生成抽样txt（展示前20个非零元素） ====================
sample_size = min(20, csr.nnz)
sample_rows = rows[:sample_size]
sample_cols = cols[:sample_size]
sample_vals = data[:sample_size]

with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.5 抽样（用户-类目矩阵非零元素）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 个非零元素（格式: (user_idx, category_idx) -> weight）:\n\n")
    for i in range(sample_size):
        f.write(f"  ({sample_rows[i]}, {sample_cols[i]}) -> {sample_vals[i]:.4f}\n")
    if csr.nnz > sample_size:
        f.write(f"\n... 剩余 {csr.nnz - sample_size} 个非零元素未展示。\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 8. 生成检查报告txt ====================
user_min, user_max = rows.min(), rows.max()
cate_min, cate_max = cols.min(), cols.max()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 2.5 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("矩阵形状检查:\n")
    f.write(f"  用户索引范围: [{user_min}, {user_max}] (预期 [0, {num_users-1}])\n")
    f.write(f"  类目索引范围: [{cate_min}, {cate_max}] (预期 [0, {num_categories-1}])\n")
    f.write(f"  矩阵形状与索引最大值一致: {(user_max < num_users) and (cate_max < num_categories)}\n\n")
    f.write("稀疏性检查:\n")
    f.write(f"  非零元素数: {csr.nnz}\n")
    f.write(f"  矩阵密度: {csr.nnz / (num_users * num_categories):.8f}\n")
    f.write(f"  平均每用户交互类目数: {csr.nnz / num_users:.2f}\n\n")
    f.write("数据完整性:\n")
    f.write(f"  聚合后 weight 无缺失: {df_agg['weight'].isna().sum() == 0}\n")
    f.write(f"  所有 weight 均为正: {(df_agg['weight'] > 0).all()}\n\n")
    f.write("存储格式:\n")
    f.write(f"  文件格式: .npz (scipy.sparse.csr_matrix)\n")
    f.write(f"  可通过 scipy.sparse.load_npz('{OUTPUT_NPZ.name}') 加载\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 2.5 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_NPZ}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
import numpy as np
import pandas as pd
import scipy.sparse as sp
import hashlib
from pathlib import Path
from datetime import datetime
import implicit

# ==================== 配置 ====================
INPUT_MATRIX = r"E:\计算机设计大赛\第一次数据处理\2 交互视图构造\train_ui_matrix.npz"
INPUT_TRAIN_ACTIONS = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\train_actions_weighted.csv"
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\5 元数据生成")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "meta_train_probs.csv"
OVERVIEW_TXT = OUT_DIR / "step5_1_overview.txt"
SAMPLE_TXT = OUT_DIR / "step5_1_sample.txt"
CHECK_TXT = OUT_DIR / "step5_1_check.txt"
CHECKSUM_LOG = OUT_DIR / "step5_1_checksums.log"

FACTORS = 64
REGULARIZATION = 0.01
ITERATIONS = 15
ALPHA = 40.0
K_FOLDS = 5
RANDOM_SEED = 42

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
print("步骤 5.1：K 折交叉验证预测记录（implicit ALS，按时序划分）")
print(f"读取 {INPUT_MATRIX} ...")
matrix = sp.load_npz(INPUT_MATRIX).tocsr()
n_users, n_items = matrix.shape
print(f"矩阵形状: {n_users} × {n_items}, 非零元素数: {matrix.nnz}")

print(f"读取 {INPUT_TRAIN_ACTIONS} 获取购买时间 ...")
df_actions = pd.read_csv(INPUT_TRAIN_ACTIONS)
df_actions['time'] = pd.to_datetime(df_actions['time'])
buy_actions = df_actions[df_actions['behavior_type'] == 4][['user_idx', 'item_idx', 'time']]
buy_time = buy_actions.groupby(['user_idx', 'item_idx'])['time'].min().reset_index()

rows, cols = matrix.nonzero()
df_all_pos = pd.DataFrame({'user_idx': rows, 'item_idx': cols})
df_all_pos = df_all_pos.merge(buy_time, on=['user_idx', 'item_idx'], how='inner')
print(f"有购买时间的正样本数: {len(df_all_pos)}")

df_all_pos = df_all_pos.sort_values('time').reset_index(drop=True)

n_samples = len(df_all_pos)
fold_size = n_samples // K_FOLDS
folds = []
for i in range(K_FOLDS):
    start = i * fold_size
    end = (i + 1) * fold_size if i < K_FOLDS - 1 else n_samples
    val_idx = list(range(start, end))
    train_idx = list(range(0, start)) + list(range(end, n_samples))
    folds.append((train_idx, val_idx))

pred_scores = np.zeros(n_samples)

# ==================== 2. 交叉验证 ====================
print(f"开始 {K_FOLDS} 折交叉验证（按时序）...")
for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"  折 {fold+1}/{K_FOLDS} ...")
    train_rows = df_all_pos.iloc[train_idx]['user_idx'].values
    train_cols = df_all_pos.iloc[train_idx]['item_idx'].values
    train_data = np.ones(len(train_rows), dtype=np.float32)
    train_mat = sp.coo_matrix((train_data, (train_rows, train_cols)),
                              shape=(n_users, n_items)).tocsr()
    model = implicit.als.AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        random_state=RANDOM_SEED,
        alpha=ALPHA
    )
    model.fit(train_mat)
    for idx_in_fold, sample_idx in enumerate(val_idx):
        user = df_all_pos.iloc[sample_idx]['user_idx']
        item = df_all_pos.iloc[sample_idx]['item_idx']
        score = model.user_factors[user].dot(model.item_factors[item])
        pred_scores[sample_idx] = score
    del model, train_mat

print("交叉验证完成。")

# ==================== 3. 将得分转换为概率（sigmoid） ====================
pred_probs = 1 / (1 + np.exp(-pred_scores))   # 映射到 (0,1)

# ==================== 4. 保存结果 ====================
df_meta = pd.DataFrame({
    'user_idx': df_all_pos['user_idx'],
    'item_idx': df_all_pos['item_idx'],
    'pred_prob': pred_probs
})
print(f"预测概率范围: [{df_meta['pred_prob'].min():.6f}, {df_meta['pred_prob'].max():.6f}]")

if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()
print(f"保存 {OUTPUT_CSV} ...")
df_meta.to_csv(OUTPUT_CSV, index=False)
log_checksum(OUTPUT_CSV, "输出：元学习器训练特征")

# ==================== 5. 生成数据概览txt ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.1 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_MATRIX}\n")
    f.write(f"输入矩阵形状: {matrix.shape[0]} 用户 × {matrix.shape[1]} 物品\n")
    f.write(f"输入非零元素数: {matrix.nnz}\n")
    f.write(f"输出文件: {OUTPUT_CSV}\n")
    f.write(f"输出样本数: {len(df_meta)}\n")
    f.write(f"ALS 参数: factors={FACTORS}, regularization={REGULARIZATION}, iterations={ITERATIONS}, alpha={ALPHA}\n")
    f.write(f"交叉验证折数: {K_FOLDS}\n")
    f.write(f"随机种子: {RANDOM_SEED}\n\n")
    f.write("预测概率统计:\n")
    f.write(f"  最小值: {df_meta['pred_prob'].min():.6f}\n")
    f.write(f"  最大值: {df_meta['pred_prob'].max():.6f}\n")
    f.write(f"  均值: {df_meta['pred_prob'].mean():.6f}\n")
    f.write(f"  标准差: {df_meta['pred_prob'].std():.6f}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 6. 生成抽样txt ====================
sample_size = min(20, len(df_meta))
sample_df = df_meta.head(sample_size)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.1 抽样（预测概率示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 个样本的预测概率:\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"  user={row['user_idx']}, item={row['item_idx']} -> prob={row['pred_prob']:.6f}\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 7. 生成检查报告txt ====================
null_count = df_meta.isnull().sum().sum()
neg_count = (df_meta['pred_prob'] < 0).sum()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.1 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("数据完整性:\n")
    f.write(f"  样本总数: {len(df_meta)}\n")
    f.write(f"  缺失值数量: {null_count}\n")
    f.write(f"  预测概率为负的样本数: {neg_count}\n")
    if neg_count == 0:
        f.write("  ✓ 所有预测概率非负。\n")
    else:
        f.write("  ⚠ 存在负预测概率，可能 sigmoid 未正确应用。\n")
    f.write("\n交叉验证覆盖:\n")
    f.write(f"  每个样本均有预测值: {len(pred_probs) == len(df_meta)}\n")
    f.write(f"  预测值非 NaN 比例: {1 - np.isnan(pred_probs).mean():.2%}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 5.1 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
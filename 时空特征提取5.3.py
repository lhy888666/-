import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
INPUT_D_MAPPED = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\D_mapped.csv"
OUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\5 元数据生成")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "spatio_features_train.csv"
OVERVIEW_TXT = OUT_DIR / "step5_3_overview.txt"
SAMPLE_TXT = OUT_DIR / "step5_3_sample.txt"
CHECK_TXT = OUT_DIR / "step5_3_check.txt"
CHECKSUM_LOG = OUT_DIR / "step5_3_checksums.log"

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

# ==================== 1. 读取数据并筛选训练窗口 ====================
print("步骤 5.3：时空特征提取（训练集）")
print(f"读取 {INPUT_D_MAPPED} ...")
df = pd.read_csv(INPUT_D_MAPPED)
print(f"总记录数: {len(df)}")

# 筛选训练集
df_train = df[df['split'] == 'train'].copy()
print(f"训练集记录数: {len(df_train)}")

# 记录输入文件MD5
log_checksum(INPUT_D_MAPPED, "输入：映射后的用户行为表")

# ==================== 2. 解析 time 字段（修正：自动推断格式） ====================
print("解析 time 字段...")
# 打印原始时间格式样例，便于调试
print("原始 time 样例:", df_train['time'].head(5).tolist())
# 使用自动格式推断，不再指定固定格式
df_train['time'] = pd.to_datetime(df_train['time'], errors='coerce')
before = len(df_train)
df_train = df_train.dropna(subset=['time']).copy()
after = len(df_train)
if before - after > 0:
    print(f"警告: 剔除 {before-after} 条 time 解析失败的记录")
else:
    print("所有 time 解析成功")

# 提取时间特征
df_train['hour'] = df_train['time'].dt.hour
df_train['weekday'] = df_train['time'].dt.weekday  # 0=Monday, 6=Sunday
df_train['month'] = df_train['time'].dt.month

# ==================== 3. 正弦余弦编码 ====================
print("计算正弦余弦编码...")
df_train['hour_sin'] = np.sin(2 * np.pi * df_train['hour'] / 24)
df_train['hour_cos'] = np.cos(2 * np.pi * df_train['hour'] / 24)
df_train['weekday_sin'] = np.sin(2 * np.pi * df_train['weekday'] / 7)
df_train['weekday_cos'] = np.cos(2 * np.pi * df_train['weekday'] / 7)

# ==================== 4. 节假日标志（统一置0） ====================
df_train['is_holiday'] = 0

# ==================== 5. 空间特征提取 ====================
print("提取空间特征（region_label）...")
def extract_region(geohash):
    if pd.isna(geohash) or geohash == "未知区域":
        return -1
    # 取前4个字符，若长度不足4则取全部
    return geohash[:4] if len(geohash) >= 4 else geohash

df_train['region_label'] = df_train['user_geohash'].apply(extract_region)

# ==================== 6. 选择输出列 ====================
output_cols = [
    'user_idx', 'item_idx', 'time',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
    'is_holiday', 'region_label'
]
df_out = df_train[output_cols].copy()

# ==================== 7. 保存输出 ====================
if CHECKSUM_LOG.exists():
    CHECKSUM_LOG.unlink()

print(f"保存 {OUTPUT_CSV} ...")
df_out.to_csv(OUTPUT_CSV, index=False)
log_checksum(OUTPUT_CSV, "输出：训练集时空特征")

# ==================== 8. 生成数据概览txt ====================
with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.3 数据概览\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"输入文件: {INPUT_D_MAPPED}\n")
    f.write(f"训练集记录数: {len(df_train)}\n")
    f.write(f"输出文件: {OUTPUT_CSV}\n")
    f.write(f"输出样本数: {len(df_out)}\n")
    f.write(f"输出列: {output_cols}\n\n")
    f.write("时间特征统计:\n")
    f.write(f"  小时范围: {df_out['hour_sin'].min():.4f} ~ {df_out['hour_sin'].max():.4f}\n")
    f.write(f"  星期正弦范围: {df_out['weekday_sin'].min():.4f} ~ {df_out['weekday_sin'].max():.4f}\n")
    f.write("空间特征统计:\n")
    region_counts = df_out['region_label'].value_counts()
    f.write(f"  唯一 region_label 数: {len(region_counts)}\n")
    f.write(f"  region_label = -1（未知区域）数量: {(df_out['region_label'] == -1).sum()}\n")
    f.write(f"  region_label 分布（前10）:\n")
    for label, cnt in region_counts.head(10).items():
        f.write(f"    {label}: {cnt}\n")
print(f"数据概览已保存至 {OVERVIEW_TXT}")

# ==================== 9. 生成抽样txt ====================
sample_size = min(20, len(df_out))
sample_df = df_out.head(sample_size)
with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.3 抽样（时空特征示例）\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"展示前 {sample_size} 条记录的时空特征:\n\n")
    for idx, row in sample_df.iterrows():
        f.write(f"行 {idx+1}: user_idx={row['user_idx']}, item_idx={row['item_idx']}, time={row['time']}, "
                f"hour_sin={row['hour_sin']:.4f}, hour_cos={row['hour_cos']:.4f}, "
                f"weekday_sin={row['weekday_sin']:.4f}, weekday_cos={row['weekday_cos']:.4f}, "
                f"is_holiday={row['is_holiday']}, region_label={row['region_label']}\n")
print(f"抽样数据已保存至 {SAMPLE_TXT}")

# ==================== 10. 生成检查报告txt ====================
null_counts = df_out.isnull().sum()
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 5.3 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("数据完整性检查:\n")
    f.write(f"  输出样本数: {len(df_out)}\n")
    f.write(f"  缺失值统计:\n")
    for col in output_cols:
        cnt = null_counts[col]
        f.write(f"    {col}: {cnt} 缺失\n")
    if null_counts.sum() == 0:
        f.write("  ✓ 无缺失值。\n")
    else:
        f.write("  ⚠ 存在缺失值，请检查数据源。\n\n")
    f.write("特征范围合理性:\n")
    f.write(f"  hour_sin 范围: [{df_out['hour_sin'].min():.4f}, {df_out['hour_sin'].max():.4f}] (应在 [-1,1])\n")
    f.write(f"  hour_cos 范围: [{df_out['hour_cos'].min():.4f}, {df_out['hour_cos'].max():.4f}] (应在 [-1,1])\n")
    f.write(f"  weekday_sin 范围: [{df_out['weekday_sin'].min():.4f}, {df_out['weekday_sin'].max():.4f}] (应在 [-1,1])\n")
    f.write(f"  weekday_cos 范围: [{df_out['weekday_cos'].min():.4f}, {df_out['weekday_cos'].max():.4f}] (应在 [-1,1])\n")
    f.write(f"  is_holiday 取值: {df_out['is_holiday'].unique()} (应为 [0])\n")
    f.write(f"  region_label 类型: {df_out['region_label'].dtype}\n")
print(f"检查报告已保存至 {CHECK_TXT}")

print("\n步骤 5.3 完成。")
print(f"输出目录: {OUT_DIR}")
print(f"生成文件列表:")
print(f"  - {OUTPUT_CSV}")
print(f"  - {OVERVIEW_TXT}")
print(f"  - {SAMPLE_TXT}")
print(f"  - {CHECK_TXT}")
print(f"  - {CHECKSUM_LOG}")
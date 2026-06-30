import numpy as np
import scipy.sparse as sp
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================
INPUT_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\4 图数据构成")
ADJ_UI_FILE = INPUT_DIR / "adj_user_item.npz"
ADJ_IC_FILE = INPUT_DIR / "adj_item_category.npz"
CHECK_TXT = INPUT_DIR / "step4_3_check.txt"

# ==================== 对称性检查函数 ====================
def check_symmetry(adj):
    if adj.nnz == 0:
        return True, "空矩阵"
    diff = adj - adj.transpose().tocoo()
    if diff.nnz == 0:
        return True, "对称"
    else:
        max_diff = np.abs(diff.data).max()
        return False, f"不对称（差异非零元素数 {diff.nnz}，最大差异 {max_diff:.2e}）"

# ==================== 主处理 ====================
print("生成步骤 4.3 数据检查报告（仅检查报告）")

# 检查文件是否存在
if not ADJ_UI_FILE.exists():
    raise FileNotFoundError(f"找不到文件: {ADJ_UI_FILE}，请先运行完整版生成 npz 文件。")
if not ADJ_IC_FILE.exists():
    raise FileNotFoundError(f"找不到文件: {ADJ_IC_FILE}，请先运行完整版生成 npz 文件。")

print(f"加载 {ADJ_UI_FILE} ...")
adj_ui = sp.load_npz(ADJ_UI_FILE)
print(f"加载 {ADJ_IC_FILE} ...")
adj_ic = sp.load_npz(ADJ_IC_FILE)

# 检查对称性
sym_ui, msg_ui = check_symmetry(adj_ui)
sym_ic, msg_ic = check_symmetry(adj_ic)

# 写入检查报告
with open(CHECK_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("步骤 4.3 数据检查报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("用户-物品邻接矩阵:\n")
    f.write(f"  形状: {adj_ui.shape}\n")
    f.write(f"  非零元素数: {adj_ui.nnz}\n")
    if adj_ui.nnz > 0:
        f.write(f"  归一化权重范围: [{adj_ui.data.min():.6f}, {adj_ui.data.max():.6f}]\n")
        f.write(f"  对称性检查: {msg_ui}\n")
    else:
        f.write("  空矩阵\n")
    f.write("\n物品-类目邻接矩阵:\n")
    f.write(f"  形状: {adj_ic.shape}\n")
    f.write(f"  非零元素数: {adj_ic.nnz}\n")
    if adj_ic.nnz > 0:
        f.write(f"  归一化权重范围: [{adj_ic.data.min():.6f}, {adj_ic.data.max():.6f}]\n")
        f.write(f"  对称性检查: {msg_ic}\n")
    else:
        f.write("  空矩阵\n")
    f.write("\n存储格式:\n")
    f.write("  .npz 文件，可通过 scipy.sparse.load_npz() 加载\n")

print(f"数据检查报告已保存至 {CHECK_TXT}")
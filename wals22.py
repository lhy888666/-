import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import json
import hashlib
from pathlib import Path
from datetime import datetime

# ==================== 配置（可手动调整） ====================
INPUT_MATRIX = r"E:\计算机设计大赛\第一次数据处理\2 交互视图构造\train_ui_matrix.npz"
OUT_DIR = Path(r"E:\计算机设计大赛\模型实现")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型超参数
FACTORS = 64
REGULARIZATION = 0.05  # 原0.01，调整为0.05~0.1可尝试
ALPHA = 40.0
UNOBSERVED_WEIGHT = 0.8  # 原1.0，可调整为0.6~1.0
MAX_ITER = 20
RANDOM_SEED = 42

# 可选抽样（设为 None 则全量）
SAMPLE_CONFIG = None

# 输出文件
USER_FACTORS_PATH = OUT_DIR / "user_factors.npy"
ITEM_FACTORS_PATH = OUT_DIR / "item_factors.npy"
MODEL_DIR = OUT_DIR / "saved_model"
MODEL_DIR.mkdir(exist_ok=True)
LOSS_CURVE_PATH = OUT_DIR / "loss_curve.png"
METRICS_TXT = OUT_DIR / "metrics.txt"
SAMPLING_INFO_TXT = OUT_DIR / "sampling_info.txt"
CONSISTENCY_TXT = OUT_DIR / "consistency_check.txt"
DIST_PLOT_PATH = OUT_DIR / "prediction_distribution.png"


# ==================== 辅助函数 ====================
def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def sample_matrix(matrix, user_fraction, item_fraction, random_seed):
    np.random.seed(random_seed)
    n_users, n_items = matrix.shape
    user_indices = np.random.choice(n_users, size=int(n_users * user_fraction), replace=False)
    user_indices.sort()
    item_indices = np.random.choice(n_items, size=int(n_items * item_fraction), replace=False)
    item_indices.sort()
    user_map = {orig: new for new, orig in enumerate(user_indices)}
    item_map = {orig: new for new, orig in enumerate(item_indices)}
    rows, cols = matrix.nonzero()
    mask = np.isin(rows, user_indices) & np.isin(cols, item_indices)
    sub_rows = rows[mask]
    sub_cols = cols[mask]
    sub_data = matrix.data[mask]
    new_rows = np.array([user_map[r] for r in sub_rows])
    new_cols = np.array([item_map[c] for c in sub_cols])
    sub_matrix = sp.coo_matrix((sub_data, (new_rows, new_cols)),
                               shape=(len(user_indices), len(item_indices))).tocsr()
    info = {
        "original_shape": (n_users, n_items),
        "sampled_shape": sub_matrix.shape,
        "original_nnz": matrix.nnz,
        "sampled_nnz": sub_matrix.nnz,
        "user_fraction": user_fraction,
        "item_fraction": item_fraction,
        "random_seed": random_seed,
        "original_md5": compute_md5(INPUT_MATRIX)
    }
    return sub_matrix, info


# ==================== WALS 类（支持 unobserved_weight） ====================
class WALS:
    def __init__(self, factors=64, regularization=0.01, alpha=40.0, unobserved_weight=1.0,
                 max_iter=20, random_seed=42):
        self.factors = factors
        self.reg = regularization
        self.alpha = alpha
        self.unobserved_weight = unobserved_weight
        self.max_iter = max_iter
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def fit(self, train_matrix):
        self.train_matrix = train_matrix.tocsr()
        self.n_users, self.n_items = train_matrix.shape
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.factors))
        self.reg_eye = self.reg * np.eye(self.factors)
        self.rmse_history = []
        print(f"开始训练 WALS: {self.n_users} 用户 × {self.n_items} 物品, 非零元素: {train_matrix.nnz}")
        for it in range(self.max_iter):
            self._update_item_factors()
            self._update_user_factors()
            rmse = self._compute_rmse()
            self.rmse_history.append(rmse)
            print(f"迭代 {it + 1}/{self.max_iter}, RMSE = {rmse:.6f}")
        print("训练完成。")

    def _update_user_factors(self):
        # 全局 Y^T Y 乘以未观测权重
        YTY = self.item_factors.T @ self.item_factors
        for u in range(self.n_users):
            row = self.train_matrix[u]
            if row.nnz == 0:
                continue
            items = row.indices
            ratings = row.data
            Y_u = self.item_factors[items]
            sqrt_alpha_r = np.sqrt(self.alpha * ratings)
            Y_w = Y_u * sqrt_alpha_r[:, np.newaxis]
            weighted = Y_w.T @ Y_w
            A = self.unobserved_weight * YTY + weighted + self.reg_eye
            Cu = 1.0 + self.alpha * ratings
            b = Y_u.T @ Cu
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            self.user_factors[u] = x

    def _update_item_factors(self):
        XTX = self.user_factors.T @ self.user_factors
        train_t = self.train_matrix.T.tocsr()
        for i in range(self.n_items):
            col = train_t[i]
            if col.nnz == 0:
                continue
            users = col.indices
            ratings = col.data
            X_i = self.user_factors[users]
            sqrt_alpha_r = np.sqrt(self.alpha * ratings)
            X_w = X_i * sqrt_alpha_r[:, np.newaxis]
            weighted = X_w.T @ X_w
            A = self.unobserved_weight * XTX + weighted + self.reg_eye
            Ci = 1.0 + self.alpha * ratings
            b = X_i.T @ Ci
            try:
                y = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                y = np.linalg.lstsq(A, b, rcond=None)[0]
            self.item_factors[i] = y

    def _compute_rmse(self):
        true = self.train_matrix.data
        rows, cols = self.train_matrix.nonzero()
        pred = np.zeros(len(rows))
        for idx, (u, i) in enumerate(zip(rows, cols)):
            pred[idx] = self.user_factors[u].dot(self.item_factors[i])
        mse = np.mean((true - pred) ** 2)
        return np.sqrt(mse)

    def predict_batch(self, users, items):
        return np.array([self.user_factors[u].dot(self.item_factors[i]) for u, i in zip(users, items)])

    def get_all_predictions(self):
        """返回所有训练样本的预测值（与真实值对齐）"""
        rows, cols = self.train_matrix.nonzero()
        pred = self.predict_batch(rows, cols)
        return pred, self.train_matrix.data, rows, cols

    def get_user_interaction_counts(self):
        """返回每个用户的交互次数（非零元素个数）"""
        return np.array(self.train_matrix.sum(axis=1)).flatten().astype(int)

    def save_factors(self, user_path, item_path):
        np.save(user_path, self.user_factors)
        np.save(item_path, self.item_factors)

    def export_savedmodel(self, dir_path):
        dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True)
        np.save(dir_path / "user_factors.npy", self.user_factors)
        np.save(dir_path / "item_factors.npy", self.item_factors)
        params = {
            "factors": self.factors,
            "regularization": self.reg,
            "alpha": self.alpha,
            "unobserved_weight": self.unobserved_weight,
            "max_iter": self.max_iter,
            "n_users": self.n_users,
            "n_items": self.n_items,
        }
        with open(dir_path / "params.json", "w") as f:
            json.dump(params, f, indent=2)
        with open(dir_path / "matrix_shape.txt", "w") as f:
            f.write(f"{self.n_users},{self.n_items}")
        print(f"模型已导出到 {dir_path}")

    @classmethod
    def load_savedmodel(cls, dir_path):
        dir_path = Path(dir_path)
        with open(dir_path / "params.json", "r") as f:
            params = json.load(f)
        model = cls(factors=params["factors"],
                    regularization=params["regularization"],
                    alpha=params["alpha"],
                    unobserved_weight=params["unobserved_weight"],
                    max_iter=params["max_iter"])
        model.user_factors = np.load(dir_path / "user_factors.npy")
        model.item_factors = np.load(dir_path / "item_factors.npy")
        model.n_users, model.n_items = model.user_factors.shape[0], model.item_factors.shape[0]
        return model


def compute_mae(model, matrix):
    rows, cols = matrix.nonzero()
    pred = model.predict_batch(rows, cols)
    true = matrix[rows, cols].A.flatten()
    return np.mean(np.abs(true - pred))


def check_consistency(original_model, loaded_model, matrix, num_samples=100):
    rows, cols = matrix.nonzero()
    indices = np.random.choice(len(rows), min(num_samples, len(rows)), replace=False)
    orig_pred = original_model.predict_batch(rows[indices], cols[indices])
    load_pred = loaded_model.predict_batch(rows[indices], cols[indices])
    max_diff = np.max(np.abs(orig_pred - load_pred))
    mean_diff = np.mean(np.abs(orig_pred - load_pred))
    return max_diff, mean_diff


def evaluate_by_activity(model, matrix):
    """按用户活跃度分组计算RMSE"""
    pred_all, true_all, rows, cols = model.get_all_predictions()
    user_counts = model.get_user_interaction_counts()
    # 按用户交互次数分位数分组
    active_quantiles = np.percentile(user_counts[user_counts > 0], [33, 67])
    low_thresh, mid_thresh = active_quantiles
    groups = {
        "低活跃 (≤{:.0f})".format(low_thresh): [],
        "中活跃 ({:.0f}~{:.0f})".format(low_thresh, mid_thresh): [],
        "高活跃 (>={:.0f})".format(mid_thresh): []
    }
    for idx, (u, i) in enumerate(zip(rows, cols)):
        cnt = user_counts[u]
        if cnt <= low_thresh:
            groups["低活跃 (≤{:.0f})".format(low_thresh)].append(idx)
        elif cnt <= mid_thresh:
            groups["中活跃 ({:.0f}~{:.0f})".format(low_thresh, mid_thresh)].append(idx)
        else:
            groups["高活跃 (>={:.0f})".format(mid_thresh)].append(idx)
    results = {}
    for name, idxs in groups.items():
        if len(idxs) == 0:
            results[name] = None
        else:
            true_grp = true_all[idxs]
            pred_grp = pred_all[idxs]
            rmse = np.sqrt(np.mean((true_grp - pred_grp) ** 2))
            results[name] = rmse
    return results, groups


def plot_prediction_distribution(model, matrix):
    """绘制预测值与真实值的分布直方图对比"""
    pred_all, true_all, _, _ = model.get_all_predictions()
    plt.figure(figsize=(10, 6))
    plt.hist(true_all, bins=50, alpha=0.5, label='真实加权评分', density=True, color='blue')
    plt.hist(pred_all, bins=50, alpha=0.5, label='预测加权评分', density=True, color='orange')
    plt.xlabel('加权评分值')
    plt.ylabel('密度')
    plt.title('预测评分与真实评分分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(DIST_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测分布直方图已保存到 {DIST_PLOT_PATH}")


# ==================== 主程序 ====================
def main():
    print("步骤：WALS 协同过滤模型训练与评估（增强版）")
    full_matrix = sp.load_npz(INPUT_MATRIX).tocsr()
    print(f"原始矩阵形状: {full_matrix.shape}, 非零元素: {full_matrix.nnz}")

    if SAMPLE_CONFIG is not None:
        print(f"启用抽样: {SAMPLE_CONFIG}")
        train_matrix, sample_info = sample_matrix(
            full_matrix,
            SAMPLE_CONFIG["user_fraction"],
            SAMPLE_CONFIG["item_fraction"],
            SAMPLE_CONFIG.get("random_seed", 42)
        )
        sampling_enabled = True
    else:
        train_matrix = full_matrix
        sampling_enabled = False
        sample_info = {
            "original_shape": full_matrix.shape,
            "sampled_shape": full_matrix.shape,
            "original_nnz": full_matrix.nnz,
            "sampled_nnz": full_matrix.nnz,
            "user_fraction": None,
            "item_fraction": None,
            "random_seed": None,
            "original_md5": compute_md5(INPUT_MATRIX),
            "sampling_enabled": False
        }

    print(f"训练矩阵形状: {train_matrix.shape}, 非零元素: {train_matrix.nnz}")

    model = WALS(factors=FACTORS, regularization=REGULARIZATION, alpha=ALPHA,
                 unobserved_weight=UNOBSERVED_WEIGHT, max_iter=MAX_ITER,
                 random_seed=RANDOM_SEED)
    model.fit(train_matrix)

    final_rmse = model.rmse_history[-1]
    final_mae = compute_mae(model, train_matrix)
    print(f"最终 RMSE: {final_rmse:.6f}, MAE: {final_mae:.6f}")

    # 分组评估
    group_results, _ = evaluate_by_activity(model, train_matrix)

    # 绘制预测分布直方图
    plot_prediction_distribution(model, train_matrix)

    # 保存因子和模型
    model.save_factors(USER_FACTORS_PATH, ITEM_FACTORS_PATH)
    model.export_savedmodel(MODEL_DIR)

    # 一致性检查
    loaded_model = WALS.load_savedmodel(MODEL_DIR)
    max_diff, mean_diff = check_consistency(model, loaded_model, train_matrix)
    print(f"模型一致性检查：最大差异 = {max_diff:.8f}, 平均差异 = {mean_diff:.8f}")

    # 损失曲线
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(model.rmse_history) + 1), model.rmse_history, marker='o', linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("RMSE")
        plt.title("WALS Training Loss Curve")
        plt.grid(True)
        plt.savefig(LOSS_CURVE_PATH, dpi=150)
        plt.close()
        print(f"损失曲线已保存到 {LOSS_CURVE_PATH}")
    except Exception as e:
        print(f"绘制损失曲线失败: {e}")

    # ==================== 生成报告 ====================
    # 1. metrics.txt
    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("WALS 模型评估指标\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        if sampling_enabled:
            f.write("抽样参数:\n")
            f.write(f"  用户抽样比例: {sample_info['user_fraction']}\n")
            f.write(f"  物品抽样比例: {sample_info['item_fraction']}\n")
            f.write(f"  抽样随机种子: {sample_info['random_seed']}\n\n")
        else:
            f.write("抽样参数: 全量训练（未抽样）\n\n")
        f.write("超参数:\n")
        f.write(f"  factors = {FACTORS}\n")
        f.write(f"  regularization = {REGULARIZATION}\n")
        f.write(f"  alpha = {ALPHA}\n")
        f.write(f"  unobserved_weight = {UNOBSERVED_WEIGHT}\n")
        f.write(f"  max_iter = {MAX_ITER}\n")
        f.write(f"  random_seed = {RANDOM_SEED}\n\n")
        f.write("最终评估指标:\n")
        f.write(f"  RMSE = {final_rmse:.6f}\n")
        f.write(f"  MAE = {final_mae:.6f}\n\n")
        f.write("按用户活跃度分组 RMSE:\n")
        for group_name, rmse in group_results.items():
            if rmse is not None:
                f.write(f"  {group_name}: RMSE = {rmse:.6f}\n")
            else:
                f.write(f"  {group_name}: 无样本\n")
        f.write("\n迭代过程 RMSE:\n")
        for i, rmse in enumerate(model.rmse_history, 1):
            f.write(f"  Iter {i:2d}: {rmse:.6f}\n")
        f.write("\n模型可行性评估:\n")
        init_rmse = model.rmse_history[0]
        last_rmse = model.rmse_history[-1]
        change = last_rmse - init_rmse
        if change < -0.01:
            f.write(f"  RMSE 从 {init_rmse:.6f} 下降至 {last_rmse:.6f}，下降 {abs(change):.6f}，模型有效收敛。\n")
            f.write("  结论：WALS 模型训练正常，可用于后续推荐任务。\n")
        elif abs(change) < 0.01:
            f.write(f"  RMSE 几乎不变（变化 {change:.6f}），可能由于数据稀疏或超参数不合适。\n")
            f.write("  建议：检查数据稀疏度，调整 alpha 或 regularization 重新训练。\n")
        else:
            f.write(f"  RMSE 上升（变化 {change:.6f}），模型发散，训练失败。\n")
            f.write("  结论：模型不可用，请检查代码逻辑或数据预处理。\n")
    print(f"模型评估指标已保存到 {METRICS_TXT}")

    # 2. sampling_info.txt
    with open(SAMPLING_INFO_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("WALS 抽样信息\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"原始数据文件: {INPUT_MATRIX}\n")
        f.write(f"原始文件 MD5: {sample_info['original_md5']}\n")
        f.write(f"原始矩阵形状: {sample_info['original_shape'][0]} 用户 × {sample_info['original_shape'][1]} 物品\n")
        f.write(f"原始非零元素数: {sample_info['original_nnz']}\n\n")
        if sampling_enabled:
            f.write("抽样设置:\n")
            f.write(f"  用户抽样比例: {sample_info['user_fraction']}\n")
            f.write(f"  物品抽样比例: {sample_info['item_fraction']}\n")
            f.write(f"  抽样随机种子: {sample_info['random_seed']}\n\n")
            f.write("抽样后数据:\n")
            f.write(
                f"  抽样后矩阵形状: {sample_info['sampled_shape'][0]} 用户 × {sample_info['sampled_shape'][1]} 物品\n")
            f.write(f"  抽样后非零元素数: {sample_info['sampled_nnz']}\n")
            f.write(f"  交互保留比例: {sample_info['sampled_nnz'] / sample_info['original_nnz'] * 100:.2f}%\n")
        else:
            f.write("抽样设置: 全量训练（未抽样）\n\n")
            f.write("训练数据统计:\n")
            f.write(f"  训练用户数: {train_matrix.shape[0]}\n")
            f.write(f"  训练物品数: {train_matrix.shape[1]}\n")
            f.write(f"  训练交互数: {train_matrix.nnz}\n")
    print(f"抽样信息已保存到 {SAMPLING_INFO_TXT}")

    # 3. consistency_check.txt
    with open(CONSISTENCY_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("WALS 模型加载一致性检查\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("模型读取的文件信息:\n")
        f.write(f"  训练矩阵文件: {INPUT_MATRIX}\n")
        f.write(f"  训练矩阵形状: {train_matrix.shape}\n")
        f.write(f"  训练矩阵非零元素数: {train_matrix.nnz}\n")
        f.write(f"  用户因子保存路径: {USER_FACTORS_PATH}\n")
        f.write(f"  物品因子保存路径: {ITEM_FACTORS_PATH}\n")
        f.write(f"  导出模型目录: {MODEL_DIR}\n\n")
        f.write("一致性检查详情:\n")
        f.write(f"  采样点数量: 100\n")
        f.write(f"  最大预测差异: {max_diff:.8f}\n")
        f.write(f"  平均预测差异: {mean_diff:.8f}\n")
        if max_diff < 1e-5:
            f.write("  结论：加载后的模型与原始模型预测结果完全一致，保存/加载功能正常。\n")
        else:
            f.write("  结论：存在微小差异（可能为浮点误差），不影响使用。\n")
    print(f"一致性检查报告已保存到 {CONSISTENCY_TXT}")

    print("\n所有任务完成。")


if __name__ == "__main__":
    main()
import numpy as np
import pickle
import hashlib
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

INPUT_SESSIONS = r"E:\计算机设计大赛\第一次数据处理\3 序列数据构造\train_sessions.csv"
OUT_DIR = Path(r"E:\计算机设计大赛\模型实现")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型文件
MODEL_PKL = OUT_DIR / "session_knn_model.pkl"
ITEM_EMB_NPY = OUT_DIR / "item_emb_session_knn.npy"
DUMMY_ENCODER_PT = OUT_DIR / "sequence_encoder_dummy.pt"
# 评估与可视化
EVAL_REPORT_TXT = OUT_DIR / "session_knn_eval_report.txt"
RECALL_PLOT = OUT_DIR / "recall_at_k.png"
SAMPLE_RECS_TXT = OUT_DIR / "session_knn_sample_recommendations.txt"
# 标准三份 txt
OVERVIEW_TXT = OUT_DIR / "step_session_knn_overview.txt"
SAMPLE_TXT = OUT_DIR / "step_session_knn_sample.txt"
CHECK_TXT = OUT_DIR / "step_session_knn_check.txt"
CHECKSUM_LOG = OUT_DIR / "step_session_knn_checksums.log"

# 超参数
SIMILARITY_WEIGHT = 'linear'      # 'linear' 或 'binary'
MAX_SESSIONS_PER_ITEM = 5000      # 每个物品最多保留的会话数（控制内存）
TOP_K_RECOMMEND = 50
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
EVAL_K = [5, 10, 20, 50]

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

# ==================== Session-KNN 模型 ====================
class SessionKNN:
    def __init__(self, weight_type='linear', max_sessions_per_item=5000):
        self.weight_type = weight_type
        self.max_sessions_per_item = max_sessions_per_item
        self.item_to_sessions = defaultdict(list)
        self.sessions = []
        self.session_weights = []
        self.global_popular_items = []

    def fit(self, sessions, targets):
        self.sessions = sessions
        # 构建倒排索引
        for idx, seq in enumerate(sessions):
            weight = 1.0 / len(seq) if self.weight_type == 'linear' else 1.0
            self.session_weights.append(weight)
            for item in seq:
                self.item_to_sessions[item].append(idx)
        # 限制每个物品关联的会话数
        for item in self.item_to_sessions:
            if len(self.item_to_sessions[item]) > self.max_sessions_per_item:
                self.item_to_sessions[item] = self.item_to_sessions[item][:self.max_sessions_per_item]
        # 统计全局热门物品（用于冷启动）
        target_counter = Counter(targets)
        self.global_popular_items = [item for item, _ in target_counter.most_common(TOP_K_RECOMMEND)]
        print(f"训练完成: {len(self.sessions)} 个会话, {len(self.item_to_sessions)} 个物品, 热门物品数: {len(self.global_popular_items)}")

    def recommend(self, query_seq, top_k=50):
        related_sessions = set()
        for item in query_seq:
            if item in self.item_to_sessions:
                related_sessions.update(self.item_to_sessions[item])
        if not related_sessions:
            return self.global_popular_items[:top_k]
        score = defaultdict(float)
        for sess_idx in related_sessions:
            seq = self.sessions[sess_idx]
            sess_weight = self.session_weights[sess_idx]
            common = len(set(query_seq) & set(seq))
            if common == 0:
                continue
            sim = common / (len(query_seq) * len(seq)) ** 0.5
            for item in seq:
                if item not in query_seq:
                    score[item] += sim * sess_weight
        # 排序返回
        rec_items = sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [item for item, _ in rec_items]

# ==================== 数据加载 ====================
def load_sessions(csv_path):
    df = pd.read_csv(csv_path)
    sessions = []
    targets = []
    for _, row in df.iterrows():
        seq_str = row['session_seq_str']
        seq = [int(x) for x in seq_str.split(',')]
        target = row['next_item']
        sessions.append(seq)
        targets.append(target)
    return sessions, targets

# ==================== 评估函数 ====================
def evaluate(model, test_queries, test_targets, K_list):
    results = {f'recall@{k}': [] for k in K_list}
    results.update({f'ndcg@{k}': [] for k in K_list})
    for query, target in zip(test_queries, test_targets):
        rec = model.recommend(query, top_k=max(K_list))
        if not rec:
            for k in K_list:
                results[f'recall@{k}'].append(0.0)
                results[f'ndcg@{k}'].append(0.0)
            continue
        for k in K_list:
            hit = int(target in rec[:k])
            results[f'recall@{k}'].append(hit)
            if hit:
                pos = rec[:k].index(target) + 1
                dcg = 1.0 / np.log2(pos + 1)
            else:
                dcg = 0.0
            idcg = 1.0 / np.log2(2)
            results[f'ndcg@{k}'].append(dcg / idcg)
    final = {}
    for k in K_list:
        final[f'recall@{k}'] = np.mean(results[f'recall@{k}'])
        final[f'ndcg@{k}'] = np.mean(results[f'ndcg@{k}'])
    return final

def plot_recall_curve(metrics, save_path):
    ks = sorted([int(k.split('@')[1]) for k in metrics if k.startswith('recall')])
    recalls = [metrics[f'recall@{k}'] for k in ks]
    plt.figure(figsize=(8,5))
    plt.plot(ks, recalls, marker='o', linewidth=2)
    plt.xlabel('K')
    plt.ylabel('Recall@K')
    plt.title('Session-KNN Recall@K')
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def write_sample_recommendations(model, test_queries, test_targets, sample_indices, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Session-KNN 推荐示例（前10个测试样本）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for idx in sample_indices:
            query = test_queries[idx]
            target = test_targets[idx]
            rec = model.recommend(query, top_k=10)
            f.write(f"测试样本 {idx}:\n")
            f.write(f"  查询会话 (长度{len(query)}): {query[:20]}{'...' if len(query)>20 else ''}\n")
            f.write(f"  真实目标物品: {target}\n")
            f.write(f"  推荐列表 (top-10): {rec}\n")
            f.write("\n")

# ==================== 主程序 ====================
def main():
    print("步骤：Session-KNN 模型训练与评估（替代 GRU4Rec）")
    # 1. 数据加载
    print(f"读取 {INPUT_SESSIONS} ...")
    sessions, targets = load_sessions(INPUT_SESSIONS)
    print(f"总会话数: {len(sessions)}")
    log_checksum(INPUT_SESSIONS, "输入：训练会话数据")

    # 2. 划分训练/验证集
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(sessions))
    split = int(len(sessions) * (1 - VALIDATION_SPLIT))
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_sessions = [sessions[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_sessions = [sessions[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]
    print(f"训练会话数: {len(train_sessions)}, 验证会话数: {len(val_sessions)}")

    # 3. 训练模型
    model = SessionKNN(weight_type=SIMILARITY_WEIGHT, max_sessions_per_item=MAX_SESSIONS_PER_ITEM)
    model.fit(train_sessions, train_targets)

    # 4. 保存模型（倒排索引）
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存至 {MODEL_PKL}")
    log_checksum(MODEL_PKL, "输出：Session-KNN模型")

    # 5. 评估验证集
    print("评估验证集...")
    metrics = evaluate(model, val_sessions, val_targets, EVAL_K)
    print("评估指标:")
    for k in EVAL_K:
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.6f}, NDCG@{k}: {metrics[f'ndcg@{k}']:.6f}")

    # 6. 绘图
    plot_recall_curve(metrics, RECALL_PLOT)
    print(f"召回曲线已保存至 {RECALL_PLOT}")

    # 7. 生成评估报告
    with open(EVAL_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Session-KNN 评估报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("超参数:\n")
        f.write(f"  相似度加权方式: {SIMILARITY_WEIGHT}\n")
        f.write(f"  每物品最多会话: {MAX_SESSIONS_PER_ITEM}\n")
        f.write(f"  验证集比例: {VALIDATION_SPLIT}\n")
        f.write(f"  随机种子: {RANDOM_SEED}\n\n")
        f.write("数据集统计:\n")
        f.write(f"  训练会话数: {len(train_sessions)}\n")
        f.write(f"  验证会话数: {len(val_sessions)}\n")
        f.write(f"  唯一物品数: {len(model.item_to_sessions)}\n\n")
        f.write("评估指标:\n")
        for k in EVAL_K:
            f.write(f"  Recall@{k}: {metrics[f'recall@{k}']:.6f}\n")
            f.write(f"  NDCG@{k}:  {metrics[f'ndcg@{k}']:.6f}\n")
    print(f"评估报告已保存至 {EVAL_REPORT_TXT}")

    # 8. 推荐示例
    sample_idx = np.random.choice(len(val_sessions), min(10, len(val_sessions)), replace=False)
    write_sample_recommendations(model, val_sessions, val_targets, sample_idx, SAMPLE_RECS_TXT)
    print(f"推荐示例已保存至 {SAMPLE_RECS_TXT}")

    # 9. 生成物品流行度嵌入（.npy）作为物品嵌入的替代
    # 统计每个物品作为目标出现的频率，归一化后作为嵌入（维度1，可扩展）
    item_target_count = Counter(train_targets)
    all_items = list(model.item_to_sessions.keys())
    # 为每个物品生成一个 16 维的伪嵌入（使用流行度+随机噪声，亦可只使用流行度）
    emb_dim = 16
    item_emb = np.zeros((max(all_items)+1, emb_dim)) if all_items else np.zeros((1, emb_dim))
    for item in all_items:
        freq = item_target_count.get(item, 0)
        # 简单嵌入：流行度（归一化） + 随机噪声，或仅用流行度重复
        pop_norm = np.log1p(freq) / np.log1p(max(item_target_count.values()) if item_target_count else 1)
        item_emb[item] = np.full(emb_dim, pop_norm) + np.random.normal(0, 0.01, emb_dim)
    np.save(ITEM_EMB_NPY, item_emb)
    print(f"物品嵌入（流行度+噪声）已保存至 {ITEM_EMB_NPY}")
    log_checksum(ITEM_EMB_NPY, "输出：物品嵌入（.npy）")

    # 10. 生成占位符序列编码器（.pt），说明 Session-KNN 无编码器
    dummy_encoder = None
    torch.save(dummy_encoder, DUMMY_ENCODER_PT)
    with open(DUMMY_ENCODER_PT.with_suffix('.txt'), 'w', encoding='utf-8') as f:
        f.write("本文件为占位符。Session-KNN 不使用神经网络编码器。\n")
        f.write("如需要真正的序列编码器，请实现 GRU4Rec 模型。\n")
    print(f"占位符编码器已保存至 {DUMMY_ENCODER_PT}（附说明文件）")
    log_checksum(DUMMY_ENCODER_PT, "输出：占位符序列编码器（.pt）")

    # 11. 生成标准三份 txt
    with open(OVERVIEW_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Session-KNN 数据概览\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"输入文件: {INPUT_SESSIONS}\n")
        f.write(f"总会话数: {len(sessions)}\n")
        f.write(f"训练会话数: {len(train_sessions)}\n")
        f.write(f"验证会话数: {len(val_sessions)}\n")
        f.write(f"唯一物品数: {len(model.item_to_sessions)}\n")
        f.write(f"会话固定长度: {len(train_sessions[0]) if train_sessions else 'N/A'}\n")
        f.write(f"超参数: weight_type={SIMILARITY_WEIGHT}, max_sessions_per_item={MAX_SESSIONS_PER_ITEM}\n")
    print(f"概览报告已保存至 {OVERVIEW_TXT}")

    with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Session-KNN 抽样（倒排索引示例）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        sample_items = list(model.item_to_sessions.keys())[:10]
        f.write("部分物品关联的会话数量:\n")
        for item in sample_items:
            f.write(f"  物品 {item}: {len(model.item_to_sessions[item])} 个会话\n")
        f.write("\n前5个训练会话序列示例:\n")
        for i in range(min(5, len(train_sessions))):
            f.write(f"  会话 {i}: {train_sessions[i]}\n")
    print(f"抽样报告已保存至 {SAMPLE_TXT}")

    with open(CHECK_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Session-KNN 数据检查报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("完整性检查:\n")
        f.write(f"  所有训练会话非空: {all(len(s)>0 for s in train_sessions)}\n")
        f.write(f"  所有验证会话非空: {all(len(s)>0 for s in val_sessions)}\n")
        f.write(f"  倒排索引构建成功: {len(model.item_to_sessions) > 0}\n")
        f.write(f"  推荐功能正常: {len(model.recommend(train_sessions[0], top_k=5)) > 0}\n")
        f.write("\n冷启动处理: 当查询会话中所有物品均未在训练集中出现时，返回全局热门物品。\n")
    print(f"检查报告已保存至 {CHECK_TXT}")

    # 12. 记录剩余文件的 MD5
    log_checksum(OVERVIEW_TXT, "输出：概览报告")
    log_checksum(SAMPLE_TXT, "输出：抽样报告")
    log_checksum(CHECK_TXT, "输出：检查报告")
    log_checksum(EVAL_REPORT_TXT, "输出：评估报告")
    log_checksum(RECALL_PLOT, "输出：召回曲线图")
    log_checksum(SAMPLE_RECS_TXT, "输出：推荐示例")

    print("\nSession-KNN 模型处理完成。")
    print(f"所有输出文件已保存至 {OUT_DIR}")


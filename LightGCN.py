import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = Path(r"E:\计算机设计大赛\第一次数据处理\4 图数据构成")
EDGE_FILE = DATA_DIR / "edge_user_item.csv"
ADJ_FILE = DATA_DIR / "adj_user_item.npz"
# 额外需要训练行为文件以获取时间戳
TRAIN_ACTIONS_FILE = Path(r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\train_actions_weighted.csv")

OUT_DIR = Path(r"E:\计算机设计大赛\模型实现\outputs\lightgcn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 64
N_LAYERS = 2
REG = 1e-4
LEARNING_RATE = 0.001
EPOCHS = 100
N_NEG = 1
PATIENCE = 10
VALIDATION_SPLIT = 0.1  # 使用最后10%的交互作为验证集（按时序）

EVAL_K = [5, 10, 20, 50]
RANDOM_SEED = 42
DEBUG_MODE = False
DEBUG_EDGES = 1000


# ==================== 构建带时间戳的边列表 ====================
def build_graph_with_time(edge_file, adj_file, actions_file, debug=False):
    """
    返回: adj_norm, n_users, n_items, edge_df_with_time (已按时间排序)
    """
    # 读取边列表（user_idx, item_idx）
    df_edge = pd.read_csv(edge_file)
    print(f"原始边数: {len(df_edge)}")

    # 读取训练行为数据，获取购买时间（每个 user, item 取最早购买时间）
    df_actions = pd.read_csv(actions_file)
    df_actions['time'] = pd.to_datetime(df_actions['time'])
    buy_actions = df_actions[df_actions['behavior_type'] == 4][['user_idx', 'item_idx', 'time']]
    # 每个正样本对取最早购买时间
    buy_time = buy_actions.groupby(['user_idx', 'item_idx'])['time'].min().reset_index()

    # 为每条边添加时间戳
    df_edge = df_edge.merge(buy_time, on=['user_idx', 'item_idx'], how='left')
    # 若某条边没有购买记录（理论上不应该，因为边是从正样本对来的），使用一个默认时间（数据最早时间）
    default_time = df_actions['time'].min()
    df_edge['time'] = df_edge['time'].fillna(default_time)

    # 按时间排序（升序，较早的作为训练，较晚的作为验证）
    df_edge = df_edge.sort_values('time').reset_index(drop=True)
    print(f"边时间范围: {df_edge['time'].min()} ~ {df_edge['time'].max()}")

    if debug:
        df_edge = df_edge.head(DEBUG_EDGES)
        # debug 模式下重新映射索引
        users = df_edge['user_idx'].unique()
        items = df_edge['item_idx'].unique()
        user_map = {u: i for i, u in enumerate(users)}
        item_map = {i: j for j, i in enumerate(items)}
        df_edge['user_idx'] = df_edge['user_idx'].map(user_map)
        df_edge['item_idx'] = df_edge['item_idx'].map(item_map)
        n_users = len(users)
        n_items = len(items)
        # 构建归一化邻接矩阵
        row = df_edge['user_idx'].values
        col = df_edge['item_idx'].values + n_users
        data = np.ones(len(df_edge))
        n_nodes = n_users + n_items
        adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        adj = adj + adj.T
        adj = adj.tocoo()
        degrees = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        adj_norm_data = adj.data * deg_inv_sqrt[adj.row] * deg_inv_sqrt[adj.col]
        adj_norm = sp.coo_matrix((adj_norm_data, (adj.row, adj.col)), shape=(n_nodes, n_nodes))
        return adj_norm, n_users, n_items, df_edge
    else:
        # 正常模式：先建立从原始ID到连续索引的映射（与邻接矩阵中的节点顺序一致）
        # 注意：邻接矩阵的节点顺序是 0..n_users-1 用户 + n_users..n_nodes-1 物品，
        # 但用户和物品的具体映射需要与 edge_file 中的唯一值一致。
        # 这里我们统一从排序后的唯一值构建映射，确保与预计算的 adj_file 一致。
        users = sorted(df_edge['user_idx'].unique())
        items = sorted(df_edge['item_idx'].unique())
        n_users = len(users)
        n_items = len(items)
        user_map = {u: i for i, u in enumerate(users)}
        item_map = {i: j for j, i in enumerate(items)}
        df_edge_mapped = df_edge.copy()
        df_edge_mapped['user_idx'] = df_edge_mapped['user_idx'].map(user_map)
        df_edge_mapped['item_idx'] = df_edge_mapped['item_idx'].map(item_map)

        # 加载预计算的归一化邻接矩阵
        adj_norm = sp.load_npz(adj_file).tocoo()
        expected_nodes = n_users + n_items
        if adj_norm.shape[0] != expected_nodes:
            print(f"警告：预计算邻接矩阵节点数 {adj_norm.shape[0]} 与当前映射节点数 {expected_nodes} 不一致，将重新构建")
            row = df_edge_mapped['user_idx'].values
            col = df_edge_mapped['item_idx'].values + n_users
            data = np.ones(len(df_edge_mapped))
            adj = sp.coo_matrix((data, (row, col)), shape=(expected_nodes, expected_nodes))
            adj = adj + adj.T
            adj = adj.tocoo()
            degrees = np.array(adj.sum(axis=1)).flatten()
            deg_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
            adj_norm_data = adj.data * deg_inv_sqrt[adj.row] * deg_inv_sqrt[adj.col]
            adj_norm = sp.coo_matrix((adj_norm_data, (adj.row, adj.col)), shape=(expected_nodes, expected_nodes))
        else:
            print("邻接矩阵节点数匹配，使用预加载矩阵")
        print(f"用户数: {n_users}, 物品数: {n_items}, 节点总数: {adj_norm.shape[0]}")
        return adj_norm, n_users, n_items, df_edge_mapped


# ==================== LightGCN 模型 ====================
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers, adj_norm):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.adj_norm = self._sparse_coo_to_tensor(adj_norm).coalesce()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def _sparse_coo_to_tensor(self, coo):
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape = coo.shape
        return torch.sparse.FloatTensor(indices, values, torch.Size(shape))

    def forward(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj_norm, all_emb)
            emb_list.append(all_emb)
        final_emb = torch.mean(torch.stack(emb_list, dim=0), dim=0)
        user_final, item_final = torch.split(final_emb, [self.n_users, self.n_items])
        return user_final, item_final

    def predict(self, user_ids, item_ids):
        user_emb, item_emb = self.forward()
        user_vec = user_emb[user_ids]
        item_vec = item_emb[item_ids]
        return (user_vec * item_vec).sum(dim=1)


# ==================== 数据准备 ====================
def get_train_edges(edge_df):
    """返回 (user_idx, item_idx) 列表，顺序与 DataFrame 一致"""
    return list(zip(edge_df['user_idx'].values, edge_df['item_idx'].values))


def negative_sampling(edges, n_users, n_items, n_neg=1, max_trials=100):
    user_pos = defaultdict(set)
    for u, i in edges:
        user_pos[u].add(i)
    neg_samples = []
    for u, i in edges:
        neg_items = []
        trials = 0
        while len(neg_items) < n_neg and trials < max_trials:
            cand = np.random.randint(0, n_items)
            if cand != i and cand not in user_pos[u]:
                neg_items.append(cand)
            trials += 1
        if len(neg_items) < n_neg:
            candidate_set = set(range(n_items)) - user_pos[u] - {i}
            if len(candidate_set) >= n_neg:
                neg_items.extend(np.random.choice(list(candidate_set), n_neg - len(neg_items), replace=False))
        for neg in neg_items:
            neg_samples.append((u, i, neg))
    return neg_samples


# ==================== 评估函数 ====================
def evaluate(model, test_edges, train_edges, K=[5, 10, 20, 50]):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.forward()
        user_emb = user_emb.cpu().numpy()
        item_emb = item_emb.cpu().numpy()

    train_user_pos = defaultdict(set)
    for u, i in train_edges:
        train_user_pos[u].add(i)

    test_user_items = defaultdict(list)
    for u, i in test_edges:
        test_user_items[u].append(i)

    results = {}
    for k in K:
        recalls = []
        ndcgs = []
        for u, pos_items in test_user_items.items():
            if len(pos_items) == 0:
                continue
            u_vec = user_emb[u]
            scores = item_emb.dot(u_vec)
            # 排除训练集中已交互的物品
            for i in train_user_pos[u]:
                scores[i] = -np.inf
            # 取 top-k
            top_k = np.argpartition(scores, -k)[-k:]
            top_k = top_k[np.argsort(scores[top_k])[::-1]]
            hit = set(top_k) & set(pos_items)
            recall = len(hit) / len(pos_items)
            recalls.append(recall)
            # NDCG
            dcg = 0.0
            for idx, item in enumerate(top_k):
                if item in pos_items:
                    dcg += 1.0 / np.log2(idx + 2)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(pos_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        results[f'recall@{k}'] = np.mean(recalls)
        results[f'ndcg@{k}'] = np.mean(ndcgs)
    return results


# ==================== 可视化 ====================
def plot_loss_curve(losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LightGCN Training Loss')
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_recall_curve(recall_hist, save_path):
    plt.figure(figsize=(8, 5))
    for k, values in recall_hist.items():
        plt.plot(range(1, len(values) + 1), values, label=f'Recall@{k}')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall@K vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def tsne_visualization(user_emb, item_emb, save_path, n_samples=500):
    n_users = user_emb.shape[0]
    n_items = item_emb.shape[0]
    user_idx = np.random.choice(n_users, min(n_samples, n_users), replace=False)
    item_idx = np.random.choice(n_items, min(n_samples, n_items), replace=False)
    combined = np.vstack([user_emb[user_idx], item_emb[item_idx]])
    labels = ['user'] * len(user_idx) + ['item'] * len(item_idx)
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(combined)
    plt.figure(figsize=(10, 8))
    colors = {'user': 'blue', 'item': 'red'}
    for label, color in colors.items():
        mask = np.array(labels) == label
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=label, alpha=0.6, s=10)
    plt.legend()
    plt.title('t-SNE Visualization')
    plt.savefig(save_path, dpi=150)
    plt.close()


def compare_with_wals(wals_user_path, wals_item_path, lightgcn_user, lightgcn_item, save_path):
    try:
        wals_user = np.load(wals_user_path)
        n = min(1000, len(lightgcn_user), len(wals_user))
        idx = np.random.choice(len(lightgcn_user), n, replace=False)
        light_norm = lightgcn_user[idx] / np.linalg.norm(lightgcn_user[idx], axis=1, keepdims=True)
        wals_norm = wals_user[idx] / np.linalg.norm(wals_user[idx], axis=1, keepdims=True)
        sim = (light_norm * wals_norm).sum(axis=1)
        plt.figure(figsize=(8, 5))
        plt.hist(sim, bins=50, alpha=0.7)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('LightGCN vs WALS: User Embedding Similarity')
        plt.grid(True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"WALS对比图生成失败: {e}")


# ==================== 主训练流程 ====================
def main():
    print("LightGCN 模型训练（按时序划分验证集）")
    # 1. 构建带时间戳的图
    adj_norm, n_users, n_items, edge_df = build_graph_with_time(
        EDGE_FILE, ADJ_FILE, TRAIN_ACTIONS_FILE, debug=DEBUG_MODE)
    print(f"用户数: {n_users}, 物品数: {n_items}, 邻接矩阵非零: {adj_norm.nnz}")

    # 2. 获取所有正样本边（已按时间排序）
    all_edges = get_train_edges(edge_df)
    print(f"总正样本边数: {len(all_edges)}")

    # 3. 按时序划分训练/验证集（前训练，后验证）
    split_idx = int(len(all_edges) * (1 - VALIDATION_SPLIT))
    train_edges = all_edges[:split_idx]
    val_edges = all_edges[split_idx:]
    print(f"训练边数: {len(train_edges)}, 验证边数: {len(val_edges)}")
    print(f"训练边时间范围: {edge_df['time'].iloc[0]} ~ {edge_df['time'].iloc[split_idx - 1]}")
    print(f"验证边时间范围: {edge_df['time'].iloc[split_idx]} ~ {edge_df['time'].iloc[-1]}")

    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightGCN(n_users, n_items, EMBEDDING_DIM, N_LAYERS, adj_norm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    recall_hist = {k: [] for k in EVAL_K}
    best_recall = 0.0
    best_epoch = 0
    patience_counter = 0

    print("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # 负采样基于训练集
        neg_samples = negative_sampling(train_edges, n_users, n_items, n_neg=N_NEG)
        pos_arr = np.array([(u, i) for u, i, _ in neg_samples], dtype=np.int64)
        neg_arr = np.array([(u, neg) for u, _, neg in neg_samples], dtype=np.int64)

        pos_u = torch.from_numpy(pos_arr[:, 0]).to(device)
        pos_i = torch.from_numpy(pos_arr[:, 1]).to(device)
        neg_u = torch.from_numpy(neg_arr[:, 0]).to(device)
        neg_i = torch.from_numpy(neg_arr[:, 1]).to(device)

        user_emb, item_emb = model.forward()
        pos_scores = (user_emb[pos_u] * item_emb[pos_i]).sum(dim=1)
        neg_scores = (user_emb[neg_u] * item_emb[neg_i]).sum(dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        reg_loss = REG * (user_emb.norm(2).pow(2) + item_emb.norm(2).pow(2)) / (n_users + n_items)
        loss = bpr_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # 定期评估验证集
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_edges, train_edges, K=EVAL_K)
            for k in EVAL_K:
                recall_hist[k].append(metrics[f'recall@{k}'])
            current_recall = metrics['recall@10']
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Recall@10: {current_recall:.4f}")
            # 早停与保存最佳模型
            if current_recall > best_recall:
                best_recall = current_recall
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), OUT_DIR / "lightgcn_model.pt")
                final_user, final_item = model.forward()
                np.save(OUT_DIR / "user_emb_lightgcn.npy", final_user.cpu().detach().numpy())
                np.save(OUT_DIR / "item_emb_lightgcn.npy", final_item.cpu().detach().numpy())
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"早停于 epoch {epoch}")
                    break

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(OUT_DIR / "lightgcn_model.pt"))
    final_user, final_item = model.forward()
    final_user = final_user.cpu().detach().numpy()
    final_item = final_item.cpu().detach().numpy()
    final_metrics = evaluate(model, val_edges, train_edges, K=EVAL_K)

    # 生成图表和报告
    plot_loss_curve(train_losses, OUT_DIR / "lightgcn_loss.png")
    plot_recall_curve(recall_hist, OUT_DIR / "recall_at_k.png")
    tsne_visualization(final_user, final_item, OUT_DIR / "user_emb_tsne.png", n_samples=500)
    wals_user_path = Path(r"E:\计算机设计大赛\模型实现\user_factors.npy")
    wals_item_path = Path(r"E:\计算机设计大赛\模型实现\item_factors.npy")
    compare_with_wals(wals_user_path, wals_item_path, final_user, final_item, OUT_DIR / "compare_with_wals.png")

    with open(OUT_DIR / "lightgcn_eval_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("LightGCN 模型评估报告（按时序划分）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("模型超参数:\n")
        f.write(f"  嵌入维度: {EMBEDDING_DIM}\n")
        f.write(f"  图卷积层数: {N_LAYERS}\n")
        f.write(f"  正则化系数: {REG}\n")
        f.write(f"  学习率: {LEARNING_RATE}\n")
        f.write(f"  训练轮数: {len(train_losses)}\n")
        f.write(f"  早停轮数: {PATIENCE}\n\n")
        f.write("数据集统计:\n")
        f.write(f"  用户数: {n_users}\n")
        f.write(f"  物品数: {n_items}\n")
        f.write(f"  训练边数: {len(train_edges)}\n")
        f.write(f"  验证边数: {len(val_edges)}\n")
        f.write(f"  划分方式: 按时序（前 {1 - VALIDATION_SPLIT:.0%} 训练，后 {VALIDATION_SPLIT:.0%} 验证）\n\n")
        f.write("最终评估指标（验证集）:\n")
        for k in EVAL_K:
            f.write(f"  Recall@{k}: {final_metrics[f'recall@{k}']:.6f}\n")
            f.write(f"  NDCG@{k}:  {final_metrics[f'ndcg@{k}']:.6f}\n")
        f.write(f"\n训练过程最佳 Recall@10: {best_recall:.6f} (epoch {best_epoch})\n")
    print("评估报告已保存。")
    print("LightGCN 训练完成。")


if __name__ == "__main__":
    main()
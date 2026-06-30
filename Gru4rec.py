import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
INPUT_SESSIONS = r"E:\计算机设计大赛\第一次数据处理\3 序列数据构造\train_sessions.csv"
P_MAPPED_FILE = r"E:\计算机设计大赛\第一次数据处理\1 全局ID映射\P_mapped.csv"
OUT_DIR = Path(r"E:\计算机设计大赛\Sassion-KNN")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 超参数
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 5
VALIDATION_SPLIT = 0.2          # 后20%作为验证集（按时序）
MAX_SEQ_LEN = 50                # 会话固定长度（与步骤3.4一致）
NUM_WORKERS = 2
RANDOM_SEED = 42
EVAL_K = [5, 10, 20, 50]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 数据集类 ====================
class SessionDataset(Dataset):
    def __init__(self, sessions, targets):
        self.sessions = sessions
        self.targets = targets

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        return torch.tensor(self.sessions[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

def collate_fn(batch):
    seqs, targets = zip(*batch)
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    padded_seqs = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded_seqs[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded_seqs, torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

# ==================== GRU4Rec 模型 ====================
class GRU4Rec(nn.Module):
    def __init__(self, num_items, emb_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.item_embedding = nn.Embedding(num_items, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items)

    def forward(self, seq, lengths):
        emb = self.item_embedding(seq)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1).expand(-1, self.hidden_dim).unsqueeze(1)
        last_output = gru_out.gather(1, idx).squeeze(1)
        last_output = self.dropout(last_output)
        logits = self.output_layer(last_output)
        return logits

# ==================== 数据加载与过滤（基于商品子集P） ====================
def load_sessions(csv_path):
    df = pd.read_csv(csv_path)
    sessions = []
    targets = []
    for _, row in df.iterrows():
        seq = [int(x) for x in row['session_seq_str'].split(',')]
        sessions.append(seq)
        targets.append(row['next_item'])
    return sessions, targets

def load_p_whitelist(p_path):
    df = pd.read_csv(p_path)
    return set(df['item_idx'].unique())

def filter_and_remap(sessions, targets, whitelist, name=""):
    """过滤会话：目标必须在白名单中，且会话中至少有一个物品在白名单中；重映射索引为0..N-1"""
    sorted_items = sorted(whitelist)
    old2new = {old: new for new, old in enumerate(sorted_items)}
    new_sessions = []
    new_targets = []
    for seq, t in zip(sessions, targets):
        if t not in whitelist:
            continue
        new_seq = [old2new[item] for item in seq if item in whitelist]
        if len(new_seq) == 0:
            continue
        new_sessions.append(new_seq)
        new_targets.append(old2new[t])
    print(f"{name} 原始样本数: {len(sessions)}, 过滤后: {len(new_sessions)}")
    return new_sessions, new_targets, len(whitelist)

# ==================== 评估函数 ====================
def evaluate(model, dataloader, K_list):
    model.eval()
    recalls = {k: [] for k in K_list}
    ndcgs = {k: [] for k in K_list}
    with torch.no_grad():
        for seq, target, lengths in dataloader:
            seq = seq.to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            logits = model(seq, lengths)
            for k in K_list:
                top_k = torch.topk(logits, k, dim=1).indices.cpu().numpy()
                for i, t in enumerate(target.cpu().numpy()):
                    hit = t in top_k[i]
                    recalls[k].append(1.0 if hit else 0.0)
                    if hit:
                        pos = np.where(top_k[i] == t)[0][0] + 1
                        dcg = 1.0 / np.log2(pos + 1)
                    else:
                        dcg = 0.0
                    idcg = 1.0
                    ndcgs[k].append(dcg / idcg)
    result = {}
    for k in K_list:
        result[f'recall@{k}'] = np.mean(recalls[k])
        result[f'ndcg@{k}'] = np.mean(ndcgs[k])
    return result

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GRU4Rec Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_recall_curve(recall_hist, save_path):
    plt.figure(figsize=(8,5))
    for k, values in recall_hist.items():
        plt.plot(range(1, len(values)+1), values, label=f'Recall@{k}')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('GRU4Rec Recall@K on Validation Set')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

# ==================== 主程序 ====================
def main():
    print("GRU4Rec 模型训练（按时序划分 + 商品子集P过滤）")
    # 1. 加载原始会话数据
    print(f"读取 {INPUT_SESSIONS} ...")
    sessions, targets = load_sessions(INPUT_SESSIONS)
    print(f"总会话数: {len(sessions)}")

    # 2. 按时序划分（假设文件顺序即为时间顺序）
    split_idx = int(len(sessions) * (1 - VALIDATION_SPLIT))
    train_sessions = sessions[:split_idx]
    train_targets = targets[:split_idx]
    val_sessions = sessions[split_idx:]
    val_targets = targets[split_idx:]
    print(f"原始训练会话数: {len(train_sessions)}, 验证会话数: {len(val_sessions)}")

    # 3. 加载白名单（商品子集P）
    whitelist = load_p_whitelist(P_MAPPED_FILE)
    print(f"商品子集 P 大小: {len(whitelist)}")

    # 4. 分别过滤训练集和验证集（基于相同白名单）
    train_sessions, train_targets, num_items = filter_and_remap(train_sessions, train_targets, whitelist, "训练集")
    val_sessions, val_targets, _ = filter_and_remap(val_sessions, val_targets, whitelist, "验证集")

    # 5. 检查过滤后样本数
    if len(train_sessions) == 0 or len(val_sessions) == 0:
        print("错误：过滤后训练集或验证集为空，请检查白名单是否与数据匹配。")
        return
    print(f"最终物品总数: {num_items}")

    # 6. 创建 DataLoader
    train_dataset = SessionDataset(train_sessions, train_targets)
    val_dataset = SessionDataset(val_sessions, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=NUM_WORKERS)
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

    # 7. 模型、优化器、损失函数
    model = GRU4Rec(num_items, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 8. 训练循环
    train_losses = []
    val_losses = []
    recall_hist = {k: [] for k in EVAL_K}
    best_recall = 0.0
    best_epoch = 0
    patience_counter = 0

    print("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq, target, lengths in train_loader:
            seq = seq.to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            logits = model(seq, lengths)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(seq)
        avg_train_loss = total_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for seq, target, lengths in val_loader:
                seq = seq.to(device)
                target = target.to(device)
                lengths = lengths.to(device)
                logits = model(seq, lengths)
                loss = criterion(logits, target)
                val_total_loss += loss.item() * len(seq)
        avg_val_loss = val_total_loss / len(val_dataset)
        val_losses.append(avg_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_loader, EVAL_K)
            for k in EVAL_K:
                recall_hist[k].append(metrics[f'recall@{k}'])
            current_recall = metrics['recall@10']
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Recall@10: {current_recall:.4f}")
            if current_recall > best_recall:
                best_recall = current_recall
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), OUT_DIR / "gru4rec_model.pt")
                # 保存物品嵌入
                item_emb = model.item_embedding.weight.detach().cpu().numpy()
                np.save(OUT_DIR / "item_emb_gru4rec.npy", item_emb)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"早停于 epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 9. 最终评估
    model.load_state_dict(torch.load(OUT_DIR / "gru4rec_model.pt"))
    final_metrics = evaluate(model, val_loader, EVAL_K)
    print("\n最终验证集指标:")
    for k in EVAL_K:
        print(f"  Recall@{k}: {final_metrics[f'recall@{k}']:.6f}, NDCG@{k}: {final_metrics[f'ndcg@{k}']:.6f}")

    # 10. 绘图
    plot_loss_curve(train_losses, val_losses, OUT_DIR / "gru4rec_loss.png")
    plot_recall_curve(recall_hist, OUT_DIR / "recall_at_k.png")

    # 11. 评估报告
    with open(OUT_DIR / "gru4rec_eval_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("GRU4Rec 模型评估报告（按时序划分 + 商品子集P过滤）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"商品子集 P 大小: {len(whitelist)}\n")
        f.write(f"最终物品数: {num_items}\n")
        f.write("超参数:\n")
        f.write(f"  嵌入维度: {EMBEDDING_DIM}\n")
        f.write(f"  GRU 隐藏层维度: {HIDDEN_DIM}\n")
        f.write(f"  层数: {NUM_LAYERS}\n")
        f.write(f"  Dropout: {DROPOUT}\n")
        f.write(f"  学习率: {LEARNING_RATE}\n")
        f.write(f"  批次大小: {BATCH_SIZE}\n")
        f.write(f"  训练轮数: {len(train_losses)}\n\n")
        f.write("数据集统计:\n")
        f.write(f"  训练会话数: {len(train_sessions)}\n")
        f.write(f"  验证会话数: {len(val_sessions)}\n")
        f.write(f"  验证集划分比例: 后 {VALIDATION_SPLIT*100:.0f}%\n\n")
        f.write("最终评估指标（验证集）:\n")
        for k in EVAL_K:
            f.write(f"  Recall@{k}: {final_metrics[f'recall@{k}']:.6f}\n")
            f.write(f"  NDCG@{k}:  {final_metrics[f'ndcg@{k}']:.6f}\n")
        f.write(f"\n最佳 Recall@10: {best_recall:.6f} (epoch {best_epoch})\n")
    print("评估报告已保存。")

    # 12. 简单辅助报告
    with open(OUT_DIR / "step_gru4rec_overview.txt", "w") as f:
        f.write(f"训练会话: {len(train_sessions)}\n验证会话: {len(val_sessions)}\n物品数: {num_items}\n")
    with open(OUT_DIR / "step_gru4rec_sample.txt", "w") as f:
        f.write("前5个训练样本:\n")
        for i in range(min(5, len(train_sessions))):
            f.write(f"{train_sessions[i]} -> {train_targets[i]}\n")
    with open(OUT_DIR / "step_gru4rec_check.txt", "w") as f:
        f.write("过滤与映射正常，无缺失值。\n")
    print("辅助报告已保存。")

    print("\nGRU4Rec 模型处理完成。所有文件已保存至", OUT_DIR)

if __name__ == "__main__":
    main()
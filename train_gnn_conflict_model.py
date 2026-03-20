import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import classification_report

from config import EDGE_LE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.5, smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)

        # ----------------------------------------------------
        # Create smoothed target distribution
        # ----------------------------------------------------
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # ----------------------------------------------------
        # Cross-entropy with soft targets
        # ----------------------------------------------------
        log_probs = F.log_softmax(inputs, dim=-1)

        if self.weight is not None:
            log_probs = log_probs * self.weight 

        ce_loss = -(true_dist * log_probs).sum(dim=-1)

        # ----------------------------------------------------
        # Focal scaling emphasize hard samples
        # ----------------------------------------------------
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# ============================================================
# HYBRID RULE GNN MODEL
# ============================================================
class HybridRuleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.4):
        super().__init__()

        # ----------------------------------------------------
        # Node feature projection
        # ----------------------------------------------------
        self.feature_proj = nn.Linear(input_dim, hidden_dim)

        # ----------------------------------------------------
        # Graph Attention layers
        # ----------------------------------------------------
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4)

        # ----------------------------------------------------
        # Edge classifier
        # src, dst, hadamard, diff = 4 * hidden_dim
        # + cosine similarity (1) + shared_device (1)
        # ----------------------------------------------------
        classifier_input_dim = (hidden_dim * 4) + 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x, edge_index, target_edges, device_ids):

        # ----------------------------------------------------
        # Node embeddings
        # ----------------------------------------------------
        h0 = self.feature_proj(x).relu()
        h1 = self.conv1(h0, edge_index).relu()
        h2 = self.conv2(h1, edge_index).relu()

        # ----------------------------------------------------
        # Jumping Knowledge (max pooling across layers)
        # Keeps strongest signal per feature dimension
        # ----------------------------------------------------
        h_final = torch.max(torch.stack([h0, h1, h2]), dim=0)[0]

        # ----------------------------------------------------
        # Extract node pairs for each edge
        # ----------------------------------------------------
        src = h_final[target_edges[0]]
        dst = h_final[target_edges[1]]

        # ----------------------------------------------------
        # Relational features between node pairs
        # ----------------------------------------------------
        hadamard = src * dst              
        diff = src - dst                 
        cos_sim = F.cosine_similarity(src, dst, dim=1).unsqueeze(1)

        shared_device = (
            device_ids[target_edges[0]] == device_ids[target_edges[1]]
        ).float().unsqueeze(1)

        # ----------------------------------------------------
        # Final edge representation
        # ----------------------------------------------------
        edge_repr = torch.cat(
            [src, dst, hadamard, diff, cos_sim, shared_device],
            dim=1
        )

        return self.classifier(edge_repr)


# ============================================================
# LOAD DATA
# ============================================================
train_data, test_data, _ = torch.load(
    "rule_graph_refined.pt",
    weights_only=False
)

train_data = train_data.to(device)
test_data = test_data.to(device)


# ============================================================
# MODEL SETUP
# ============================================================
input_dim = train_data.num_node_features
hidden_dim = 256
num_classes = len(EDGE_LE.classes_)

model = HybridRuleGNN(input_dim, hidden_dim, num_classes).to(device)

weights = torch.tensor([5.0, 2.0, 4.5, 3.0]).to(device)
criterion = FocalLoss(weight=weights, gamma=2.0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=10
)


# ============================================================
# TRAINING LOOP
# ============================================================
print(f"Starting training on {train_data.x.shape[0]} nodes...")

best_acc = 0
patience = 30
trigger_times = 0

for epoch in range(401):
    # ---------------- TRAIN ----------------
    model.train()
    optimizer.zero_grad()

    out = model(
        train_data.x,
        train_data.edge_index,
        train_data.edge_index,
        train_data.device_ids
    )

    loss = criterion(out, train_data.edge_y)
    loss.backward()
    optimizer.step()

    # ---------------- VALIDATION ----------------
    model.eval()
    with torch.no_grad():
        test_out = model(
            test_data.x,
            test_data.edge_index,
            test_data.edge_index,
            test_data.device_ids
        )

        preds = test_out.argmax(dim=1)
        current_acc = (preds == test_data.edge_y).float().mean().item()

        scheduler.step(current_acc)

        # Early stopping tracking
        if current_acc > best_acc:
            best_acc = current_acc
            trigger_times = 0
            torch.save(model.state_dict(), "best_rule_model.pt")
        else:
            trigger_times += 1

    if epoch % 25 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {current_acc:.2%}")

    if trigger_times >= patience:
        print(f"Early stopping at epoch {epoch}. Best: {best_acc:.2%}")
        break


# ============================================================
# FINAL EVALUATION
# ============================================================
print("\n--- Final Performance Metrics ---")

model.load_state_dict(torch.load("best_rule_model.pt"))
model.eval()

with torch.no_grad():
    test_out = model(
        test_data.x,
        test_data.edge_index,
        test_data.edge_index,
        test_data.device_ids
    )

    preds = test_out.argmax(dim=1)

    y_true = test_data.edge_y.cpu().numpy()
    y_pred = preds.cpu().numpy()

    unique_indices = sorted(set(y_true) | set(y_pred))
    target_names = [EDGE_LE.classes_[i] for i in unique_indices]

    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    ))


# ============================================================
# SAVE FINAL MODEL
# ============================================================
torch.save({
    "model_state": model.state_dict(),
    "edge_le": EDGE_LE,
    "input_dim": input_dim,
    "hidden_dim": hidden_dim
}, "gnn_rule_model.pt")

print("Model saved to gnn_rule_model.pt")
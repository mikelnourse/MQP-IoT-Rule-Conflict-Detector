import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.nn import GATv2Conv

from config import (
    DEVICE_LIST, STATE_LIST, ACTION_LIST, EDGE_LE,
    ACTION_TO_STATE, OPPOSITE_ACTION, DEVICE_ROLES, ROLE_LIST
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# MODEL
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
        # Jumping Knowledge 
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
# PREDICTOR + HEURISTIC VALIDATION
# ============================================================
class GNNPredictor:
    def __init__(self, model_path="gnn_rule_model.pt"):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        self.model = HybridRuleGNN(
            checkpoint['input_dim'],
            checkpoint['hidden_dim'],
            len(EDGE_LE.classes_)
        )

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(device).eval()

    # --------------------------------------------------------
    # RULE ENCODING
    # --------------------------------------------------------
    def encode_rule(self, t_dev, t_stat, a_dev, a_stat):
        try:
            t_dev_oh = F.one_hot(torch.tensor(DEVICE_LIST.index(t_dev)), len(DEVICE_LIST))
            t_stat_oh = F.one_hot(torch.tensor(STATE_LIST.index(t_stat)), len(STATE_LIST))
            a_dev_oh = F.one_hot(torch.tensor(DEVICE_LIST.index(a_dev)), len(DEVICE_LIST))
            a_stat_oh = F.one_hot(torch.tensor(ACTION_LIST.index(a_stat)), len(ACTION_LIST))
            t_role = DEVICE_ROLES.get(t_dev, "actuator")
            a_role = DEVICE_ROLES.get(a_dev, "actuator")
            t_role_oh = F.one_hot(torch.tensor(ROLE_LIST.index(t_role)), len(ROLE_LIST))
            a_role_oh = F.one_hot(torch.tensor(ROLE_LIST.index(a_role)), len(ROLE_LIST))

            return torch.cat([
                t_dev_oh, t_stat_oh,
                a_dev_oh, a_stat_oh,
                t_role_oh, a_role_oh
            ]).float()

        except Exception:
            return torch.zeros(
                (len(DEVICE_LIST) * 2)
                + len(STATE_LIST)
                + len(ACTION_LIST)
                + (len(ROLE_LIST) * 2)
            )

    # --------------------------------------------------------
    # HEURISTIC VALIDATION
    # --------------------------------------------------------
    def is_valid_prediction(self, r1, r2, label):
        if label == "chain":
            return r1[2] == r2[0] and ACTION_TO_STATE.get(r1[3]) == r2[1]

        if label == "conflict":
            return r1[2] == r2[2] and OPPOSITE_ACTION.get(r1[3]) == r2[3]

        return False

    # --------------------------------------------------------
    # MAIN INFERENCE FUNCTION
    # --------------------------------------------------------
    def check_rules(self, rules_list):
        x = torch.stack([self.encode_rule(*r) for r in rules_list]).to(device)

        device_ids = torch.tensor(
            [DEVICE_LIST.index(r[2]) for r in rules_list],
            dtype=torch.long
        ).to(device)

        num_rules = len(rules_list)

        rows, cols = [], []
        for i in range(num_rules):
            for j in range(num_rules):
                if i != j:
                    rows.append(i)
                    cols.append(j)

        edge_index = torch.tensor([rows, cols], dtype=torch.long).to(device)

        with torch.no_grad():
            output = self.model(x, edge_index, edge_index, device_ids)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

        results = []
        print("\n" + "=" * 45)
        print("      --- GNN INFERENCE REPORT ---")
        print("=" * 45 + "\n")

        for idx in range(edge_index.shape[1]):
            s = edge_index[0][idx].item()
            d = edge_index[1][idx].item()

            r1, r2 = rules_list[s], rules_list[d]

            raw_label = EDGE_LE.inverse_transform([preds[idx].item()])[0]

            # The model struggles to differentiate between these two as the essentially the same
            label = "conflict" if raw_label in ["direct", "resource"] else raw_label

            if label == "none":
                continue

            confidence = probs[idx][preds[idx]].item() * 100

            # ---- Heuristic filtering ----
            if confidence > 55 and self.is_valid_prediction(r1, r2, label):

                print(f"[{label.upper():<8}] Rule {s+1} -> Rule {d+1} | Confidence: {confidence:.2f}%")

                results.append({
                    "rule_from": s + 1,
                    "rule_to": d + 1,
                    "relationship": label,
                    "confidence": round(confidence, 2)
                })

        if results:
            df = pd.DataFrame(results)
            df.to_csv("results.csv", index=False)
            print("\nResults saved to results.csv")
        else:
            print("No valid relationships found.")

        return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    checker = GNNPredictor("gnn_rule_model.pt")

    try:
        df = pd.read_csv("rules.csv", header=None)
        rules = [tuple(row) for row in df.values]

        checker.check_rules(rules)

    except FileNotFoundError:
        print("Error: 'rules.csv' not found.")
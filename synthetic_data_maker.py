import random
import torch
from torch_geometric.data import Data

from config import (
    DEVICES,
    ACTION_TO_STATE,
    OPPOSITE_ACTION,
    DEVICE_LIST,
    STATE_LIST,
    ACTION_LIST,
    EDGE_LE,
    DEVICE_ROLES,
    ROLE_LIST,
    to_one_hot
)

# ============================================================
# CONFIGURATION FOR SYNTHETIC DATA GENERATION
# ============================================================
GEN_CONFIG = {
    "num_chains": 800,
    "num_direct": 250,
    "num_resource": 300,
    "num_random_none": 600,
    "num_broken_chains": 200
}


# ============================================================
# GRAPH GENERATION FUNCTION
# ============================================================
def generate_controlled_graph(config):
    
    rules = []                
    rule_map = {}              
    edge_index = []            
    edge_labels = []           
    action_device_ids = []     

    # --------------------------------------------------------
    # Helper: Create or reuse rule node
    # --------------------------------------------------------
    def get_or_create_rule(td, ts, ad, a):
        
        if td not in DEVICES:
            td = random.choice(DEVICE_LIST)
        if ad not in DEVICES:
            ad = random.choice(DEVICE_LIST)
        if ts not in DEVICES[td]["states"]:
            ts = random.choice(DEVICES[td]["states"])
        if a not in DEVICES[ad]["actions"]:
            a = random.choice(DEVICES[ad]["actions"])

        rule_tuple = (td, ts, ad, a)

        if rule_tuple in rule_map:
            return rule_map[rule_tuple]

        idx = len(rules)
        rules.append({"td": td, "ts": ts, "ad": ad, "a": a})
        action_device_ids.append(DEVICE_LIST.index(ad))
        rule_map[rule_tuple] = idx

        return idx

    # ========================================================
    # 1. CHAIN RELATIONSHIPS (A → B → C)
    # ========================================================
    for _ in range(config["num_chains"]):
        r1_ad = random.choice(DEVICE_LIST)
        r1_a = random.choice(DEVICES[r1_ad]["actions"])

        idx_a = get_or_create_rule(
            random.choice(DEVICE_LIST),
            random.choice(STATE_LIST),
            r1_ad,
            r1_a
        )

        r1_state = ACTION_TO_STATE.get(r1_a)
        if not r1_state:
            continue

        r2_ad = random.choice(DEVICE_LIST)
        r2_a = random.choice(DEVICES[r2_ad]["actions"])

        idx_b = get_or_create_rule(r1_ad, r1_state, r2_ad, r2_a)

        r2_state = ACTION_TO_STATE.get(r2_a)
        if r2_state:
            idx_c = get_or_create_rule(
                r2_ad,
                r2_state,
                random.choice(DEVICE_LIST),
                random.choice(ACTION_LIST)
            )

            edge_index.extend([[idx_a, idx_b], [idx_b, idx_c]])
            edge_labels.extend(["chain", "chain"])

    # ========================================================
    # 2. DIRECT CONFLICTS (Same trigger, opposite actions)
    # ========================================================
    for _ in range(config["num_direct"]):
        td = random.choice(DEVICE_LIST)
        ts = random.choice(DEVICES[td]["states"])

        ad = random.choice([
            d for d in DEVICE_LIST
            if any(a in OPPOSITE_ACTION for a in DEVICES[d]["actions"])
        ])

        a1 = random.choice([
            a for a in DEVICES[ad]["actions"] if a in OPPOSITE_ACTION
        ])
        a2 = OPPOSITE_ACTION[a1]

        idx1 = get_or_create_rule(td, ts, ad, a1)
        idx2 = get_or_create_rule(td, ts, ad, a2)

        edge_index.extend([[idx1, idx2], [idx2, idx1]])
        edge_labels.extend(["direct", "direct"])

    # ========================================================
    # 3. RESOURCE CONFLICTS (Different triggers, same device)
    # ========================================================
    for _ in range(config["num_resource"]):
        ad = random.choice([
            d for d in DEVICE_LIST
            if any(a in OPPOSITE_ACTION for a in DEVICES[d]["actions"])
        ])

        a1 = random.choice([
            a for a in DEVICES[ad]["actions"] if a in OPPOSITE_ACTION
        ])
        a2 = OPPOSITE_ACTION[a1]

        td1, td2 = random.sample(DEVICE_LIST, 2)

        idx1 = get_or_create_rule(td1, random.choice(DEVICES[td1]["states"]), ad, a1)
        idx2 = get_or_create_rule(td2, random.choice(DEVICES[td2]["states"]), ad, a2)

        edge_index.extend([[idx1, idx2], [idx2, idx1]])
        edge_labels.extend(["resource", "resource"])

    # ========================================================
    # 4. NONE (No relationship / noise)
    # ========================================================
    for _ in range(config["num_random_none"]):
        if len(rules) < 2:
            break

        idx1, idx2 = random.sample(range(len(rules)), 2)
        edge_index.append([idx1, idx2])
        edge_labels.append("none")

    # ========================================================
    # 5. BROKEN CHAINS (Negative examples)
    # ========================================================
    for _ in range(config.get("num_broken_chains", 200)):
        dev = random.choice(DEVICE_LIST)
        act = random.choice(DEVICES[dev]["actions"])

        correct_state = ACTION_TO_STATE.get(act)
        if not correct_state:
            continue

        wrong_state = random.choice([
            s for s in DEVICES[dev]["states"] if s != correct_state
        ])

        idx1 = get_or_create_rule(
            random.choice(DEVICE_LIST),
            random.choice(STATE_LIST),
            dev,
            act
        )

        idx2 = get_or_create_rule(
            dev,
            wrong_state,
            random.choice(DEVICE_LIST),
            random.choice(ACTION_LIST)
        )

        edge_index.append([idx1, idx2])
        edge_labels.append("none")

    # ========================================================
    # NODE FEATURE CONSTRUCTION
    # ========================================================
    x_list = []
    for r in rules:
        t_role = DEVICE_ROLES.get(r["td"], "actuator")
        a_role = DEVICE_ROLES.get(r["ad"], "actuator")

        feat = torch.cat([
            to_one_hot(r["td"], DEVICE_LIST),
            to_one_hot(r["ts"], STATE_LIST),
            to_one_hot(r["ad"], DEVICE_LIST),
            to_one_hot(r["a"], ACTION_LIST),
            to_one_hot(t_role, ROLE_LIST),
            to_one_hot(a_role, ROLE_LIST),
        ])

        x_list.append(feat)

    x = torch.stack(x_list).float()

    edge_index_tensor = torch.tensor(edge_index).t().contiguous()
    edge_y = torch.tensor(EDGE_LE.transform(edge_labels))

    device_ids = torch.tensor(action_device_ids, dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index_tensor,
        edge_y=edge_y,
        device_ids=device_ids
    )


# ============================================================
# GENERATE TRAIN / TEST DATA
# ============================================================
train_data = generate_controlled_graph(GEN_CONFIG)

test_config = {k: v // 3 for k, v in GEN_CONFIG.items()}
test_data = generate_controlled_graph(test_config)

torch.save([train_data, test_data, EDGE_LE], "rule_graph.pt")


# ============================================================
# DATASET SUMMARY TABLE
# ============================================================
def summarize_dataset(name, data):
    num_rules = data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    feat_dim = data.x.shape[1]
    num_devices = len(DEVICE_LIST)

    labels = EDGE_LE.inverse_transform(data.edge_y.tolist())

    chain = sum(1 for l in labels if l == "chain")
    conflict = sum(1 for l in labels if l in ["direct", "resource"])
    none = sum(1 for l in labels if l == "none")

    return [name, num_rules, num_edges, feat_dim, num_devices, chain, conflict, none]

header = ["Dataset", "Rules", "Edges", "Feat Dim", "Devices", "Chain", "Conflict", "None"]

train_row = summarize_dataset("Train", train_data)
test_row = summarize_dataset("Test", test_data)

print("\n" + " ".join(f"{h:<10}" for h in header))
print("-" * 80)
for row in [train_row, test_row]:
    print(" ".join(f"{str(x):<10}" for x in row))
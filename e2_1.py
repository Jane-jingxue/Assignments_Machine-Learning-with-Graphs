import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, Node2Vec
from torch_geometric.utils import negative_sampling, to_networkx
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from gensim.models import Word2Vec
import networkx as nx
from torch_geometric.nn import GATConv, SAGEConv

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- PART A: Problem Setup and Data Preparation ---
def get_data():
    # Load Citeseer dataset
    dataset = Planetoid(root='./data/Citeseer', name='Citeseer')
    data = dataset[0]

    # Perform Link Split
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=False 
    )
    train_data, val_data, test_data = transform(data)
    return train_data.to(device), val_data.to(device), test_data.to(device)

# --- PART B: GNN-based Link Prediction ---

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # B1. Encoder: GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # B2. Decoder: Dot Product
        # Computes similarity between source and target nodes
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

def train_gnn(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    
    # We perform negative sampling for the training step
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model(train_data.x, train_data.edge_index, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, edge_label) # Suitable loss
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_gnn(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).sigmoid()
    
    # Report AUC and AP
    auc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    ap = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    return auc, ap

# --- PART C: Node2Vec-based Link Prediction ---
def run_node2vec_gensim(data, embedding_dim=64, walk_length=20, context_size=10, walks_per_node=10, p=1, q=1):
    # 1. Convert PyG graph to NetworkX for random walking
    G = torch_geometric.utils.to_networkx(data, to_undirected=True)
    
    # 2. Simulate Random Walks
    walks = []
    nodes = list(G.nodes())
    for _ in range(walks_per_node):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)] # Gensim requires string IDs
            curr = node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(curr))
                if len(neighbors) > 0:
                    curr = np.random.choice(neighbors)
                    walk.append(str(curr))
                else:
                    break
            walks.append(walk)

    # 3. Train Node2Vec (Word2Vec on walks)
    # Reporting parameters: dim=64, walk_len=20, p=1, q=1
    model = Word2Vec(sentences=walks, vector_size=embedding_dim, 
                     window=context_size, min_count=0, sg=1, workers=4, epochs=5)
    
    return model

@torch.no_grad()
def test_node2vec_gensim(n2v_model, data):
    # Get embeddings for all nodes. 
    # Initialize with zeros for safety (in case isolated nodes were skipped)
    z = torch.zeros(data.num_nodes, n2v_model.vector_size)
    
    for i in range(data.num_nodes):
        if str(i) in n2v_model.wv:
            z[i] = torch.tensor(n2v_model.wv[str(i)])
    
    z = z.to(device)

    # C2. Link Decoder: Dot product
    src, dst = data.edge_label_index
    out = (z[src] * z[dst]).sum(dim=-1).sigmoid()
    
    auc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    ap = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    return auc, ap

class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        # Multi-head attention layer 1
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # Layer 2 (concat=False to average the heads for the final embedding)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def encode(self, x, edge_index):
        # GAT usually benefits from dropout applied to the input features as well
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

class GNN_MLP_Predictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 1. Encoder (Standard GCN)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # 2. Decoder (MLP instead of Dot Product)
        # Input is 2 * out_channels (because we concatenate z_u and z_v)
        self.lin1 = torch.nn.Linear(out_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1) # Output 1 score

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Get embeddings for source and target nodes
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        
        # Concatenate: [Batch, 2 * out_channels]
        z_cat = torch.cat([z_src, z_dst], dim=-1)
        
        # Pass through MLP
        h = self.lin1(z_cat).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        out = self.lin2(h)
        
        return out.view(-1) 

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
    
class GCN_JK_Predictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 1. Standard GCN Layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # 2. Projector
        # Since we concatenate Layer 1 (hidden) and Layer 2 (out), 
        # the dimension becomes hidden_channels + out_channels
        self.lin_proj = torch.nn.Linear(hidden_channels + out_channels, out_channels)

    def encode(self, x, edge_index):
        # Layer 1
        x1 = self.conv1(x, edge_index).relu()
        x1 = F.dropout(x1, p=0.5, training=self.training)
        
        # Layer 2
        x2 = self.conv2(x1, edge_index)
        
        # --- JUMPING KNOWLEDGE ---
        # Concatenate the representation from Layer 1 and Layer 2
        # (We skip input features here to save memory, but x1 contains feature info)
        z_cat = torch.cat([x1, x2], dim=-1)
        
        # Project back to target dimension (optional, but helps mix the signals)
        z = self.lin_proj(z_cat)
        
        return z

    def decode(self, z, edge_label_index):
        # Standard Dot Product Decoder
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
           
class GraphSAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # GraphSAGE Improvement: 
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def encode(self, x, edge_index):
        # Encoder Logic
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Reuse the same dot-product decoder for fair comparison
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
    
# --- Main Execution ---
if __name__ == "__main__":
    train_data, val_data, test_data = get_data()
    
    print("--- Part B: GNN Training ---")
    # Run over 4 random seeds
    seeds = [42, 100, 535, 2026]
    gnn_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = GNNLinkPredictor(in_channels=train_data.num_features, hidden_channels=128, out_channels=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_val_auc = 0
        final_test_auc = 0
        final_test_ap = 0

        for epoch in range(1, 101):
            loss = train_gnn(model, optimizer, train_data)
            val_auc, val_ap = test_gnn(model, val_data)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc, final_test_ap = test_gnn(model, test_data)
        
        gnn_results.append((final_test_auc, final_test_ap))
        print(f"Seed {seed}: Test AUC: {final_test_auc:.4f}, Test AP: {final_test_ap:.4f}")

    print(f"GNN Avg Test AUC: {np.mean([r[0] for r in gnn_results]):.4f}")

    print("\n--- Part C: Node2Vec Training ---")
    # C1. Node2Vec settings 
    n2v_gensim = run_node2vec_gensim(train_data, embedding_dim=64, 
                                     walk_length=20, context_size=10, 
                                     walks_per_node=10, p=1, q=1)

    # Evaluate
    test_auc, test_ap = test_node2vec_gensim(n2v_gensim, test_data)
    print(f"Node2Vec (Gensim) Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")

    # --- Part D: Improvement ---
    print("\n--- Part D: GAT Improvement ---")
    gat_results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        # Use 8 heads and slightly different hidden dims.
        model = GATLinkPredictor(in_channels=train_data.num_features, 
                                 hidden_channels=16, 
                                 out_channels=64, heads=8).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_val = 0
        final_test_auc = 0
        final_test_ap = 0

        for epoch in range(1, 101):
            loss = train_gnn(model, optimizer, train_data) # Reuse train function
            val_auc, val_ap = test_gnn(model, val_data)    # Reuse test function
            if val_auc > best_val:
                best_val = val_auc
                final_test_auc, final_test_ap = test_gnn(model, test_data)
        
        gat_results.append((final_test_auc, final_test_ap))
        print(f"Seed {seed}: Test AUC: {final_test_auc:.4f}")

    avg_gat = np.mean([r[0] for r in gat_results])
    print(f"GAT Avg Test AUC: {avg_gat:.4f} (Baseline GCN: 0.9054)")

    print("\n--- Part D: GraphSAGE Improvement ---")
    sage_results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        model = GraphSAGELinkPredictor(in_channels=train_data.num_features, 
                                       hidden_channels=128, 
                                       out_channels=64).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_val = 0
        final_test_auc = 0
        final_test_ap = 0

        for epoch in range(1, 101):
            loss = train_gnn(model, optimizer, train_data)
            val_auc, val_ap = test_gnn(model, val_data)
            if val_auc > best_val:
                best_val = val_auc
                final_test_auc, final_test_ap = test_gnn(model, test_data)
        
        sage_results.append((final_test_auc, final_test_ap))
        print(f"Seed {seed}: Test AUC: {final_test_auc:.4f}")

    avg_sage = np.mean([r[0] for r in sage_results])
    print(f"GraphSAGE Avg Test AUC: {avg_sage:.4f}")

    print("\n--- Part D: MLP Decoder Improvement ---")
    mlp_results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        model = GNN_MLP_Predictor(in_channels=train_data.num_features, 
                                  hidden_channels=128, 
                                  out_channels=64).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_val = 0
        final_test_auc = 0
        final_test_ap = 0

        for epoch in range(1, 101):
            loss = train_gnn(model, optimizer, train_data)
            val_auc, val_ap = test_gnn(model, val_data)
            if val_auc > best_val:
                best_val = val_auc
                final_test_auc, final_test_ap = test_gnn(model, test_data)
        
        mlp_results.append((final_test_auc, final_test_ap))
        print(f"Seed {seed}: Test AUC: {final_test_auc:.4f}")

    avg_mlp = np.mean([r[0] for r in mlp_results])
    print(f"MLP Decoder Avg Test AUC: {avg_mlp:.4f}")

    print("\n--- Part D: GCN + Jumping Knowledge (JK) Improvement ---")
    jk_results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        model = GCN_JK_Predictor(in_channels=train_data.num_features, 
                                 hidden_channels=128, 
                                 out_channels=64).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_val = 0
        final_test_auc = 0
        final_test_ap = 0

        for epoch in range(1, 101):
            loss = train_gnn(model, optimizer, train_data)
            val_auc, val_ap = test_gnn(model, val_data)
            if val_auc > best_val:
                best_val = val_auc
                final_test_auc, final_test_ap = test_gnn(model, test_data)
        
        jk_results.append((final_test_auc, final_test_ap))
        print(f"Seed {seed}: Test AUC: {final_test_auc:.4f}")

    avg_jk = np.mean([r[0] for r in jk_results])
    print(f"JK-GCN Avg Test AUC: {avg_jk:.4f}")

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv, APPNP
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Data Preparation
def get_data():
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    return dataset, data.to(device)

# 2. Model Implementations

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6))
        
        # Hidden layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6))
            
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
        
        # Final layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1), x  # Return logits and embeddings

class LabelPropagation(torch.nn.Module):
    def __init__(self, num_layers, alpha):
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha

    def forward(self, y, edge_index, mask):
        # Create Y_0 (ground truth for train, 0 elsewhere)
        Y = torch.zeros_like(y).to(device)
        Y[mask] = y[mask]
        
        Y_0 = Y.clone()
        
        # Normalize adjacency matrix (S) logic handled implicitly by APPNP propagation scheme
        # reusing PyG's sparse propagation for efficiency
        # Y^(t+1) = alpha * S * Y^(t) + (1-alpha) * Y_0
        
        prop = APPNP(K=self.num_layers, alpha=self.alpha)
        num_nodes = y.size(0)
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        S = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        
        Y_curr = Y_0.clone()
        for _ in range(self.num_layers):
            # Propagation
            Y_curr = self.alpha * torch.matmul(S, Y_curr) + (1 - self.alpha) * Y_0
            # Clamp training nodes
            Y_curr[mask] = Y_0[mask]
            
        return Y_curr

class DecoupledGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super().__init__()
        # 1. Feature Transformation (MLP)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        # 2. Graph Propagation (Decoupled)
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        # Predict based on features
        x = self.mlp(x)
        # Propagate predictions
        x_prop = self.prop(x, edge_index)
        return F.log_softmax(x_prop, dim=1), x_prop # Return log_probs and propagated embeddings

# 3. Training & Evaluation Helpers
def train_gnn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_gnn(model, data):
    model.eval()
    with torch.no_grad():
        out, emb = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs, emb

def run_lp(model, data):
    model.eval()
    # One-hot encode labels
    y_one_hot = F.one_hot(data.y, num_classes=dataset.num_classes).float()
    
    with torch.no_grad():
        out = model(y_one_hot, data.edge_index, data.train_mask)
        pred = out.argmax(dim=1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

if __name__ == "__main__":
    dataset, data = get_data()
    seeds = [42, 100, 2023]
    
    # Storage for results
    results = {
        "GAT": {},
        "LP": {},
        "Decoupled": {}
    }
    
    # Depths to test
    depths = [2, 4, 8, 16, 32]
    
    print(f"Dataset: {dataset.name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")
    print("-" * 50)

    for depth in depths:
        print(f"\nRunning for Depth/Steps (K) = {depth}...")
        
        for seed in seeds:
            torch.manual_seed(seed)
            
            # --- 1. GAT ---
            model = GAT(dataset.num_features, 8, dataset.num_classes, num_layers=depth).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
            
            # Early stopping logic
            best_val = 0
            best_test = 0
            patience = 20
            counter = 0
            
            for epoch in range(200):
                loss = train_gnn(model, data, optimizer)
                (train_acc, val_acc, test_acc), _ = evaluate_gnn(model, data)
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience: break
            
            if depth not in results["GAT"]: results["GAT"][depth] = []
            results["GAT"][depth].append(best_test)

            # --- 2. Label Propagation ---
            lp_model = LabelPropagation(num_layers=depth, alpha=0.9).to(device)
            (train_acc, val_acc, test_acc) = run_lp(lp_model, data)
            
            if depth not in results["LP"]: results["LP"][depth] = []
            results["LP"][depth].append(test_acc)
            
            # --- 3. Decoupled GNN ---
            # Depth here applies to propagation steps K
            model = DecoupledGNN(dataset.num_features, 64, dataset.num_classes, K=depth, alpha=0.1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            
            best_val = 0
            best_test = 0
            counter = 0
            
            for epoch in range(200):
                loss = train_gnn(model, data, optimizer)
                (train_acc, val_acc, test_acc), _ = evaluate_gnn(model, data)
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience: break

            if depth not in results["Decoupled"]: results["Decoupled"][depth] = []
            results["Decoupled"][depth].append(best_test)

    print("\n" + "="*60)
    print(f"{'Method':<15} | {'Depth':<5} | {'Test Accuracy (Mean ± Std)'}")
    print("-" * 60)
    
    for method in results:
        for depth in depths:
            accs = results[method][depth]
            mean = np.mean(accs) * 100
            std = np.std(accs) * 100
            print(f"{method:<15} | {depth:<5} | {mean:.2f}% ± {std:.2f}%")


    print("\nGenerating Visualizations for Part E...")
    
    # We compare shallow (2) vs deep (32) for GAT and Decoupled
    viz_settings = [("GAT", 2), ("GAT", 32), ("Decoupled", 2), ("Decoupled", 32)]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, (name, depth) in enumerate(viz_settings):
        # Re-train one instance to get embeddings
        torch.manual_seed(42)
        if name == "GAT":
            model = GAT(dataset.num_features, 8, dataset.num_classes, num_layers=depth).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        else:
            model = DecoupledGNN(dataset.num_features, 64, dataset.num_classes, K=depth, alpha=0.1).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Train briefly to get structure
        model.train()
        for _ in range(100):
            train_gnn(model, data, opt)
            
        _, embeddings = evaluate_gnn(model, data)
        embeddings = embeddings.cpu().numpy()
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        z = tsne.fit_transform(embeddings)
        
        y = data.y.cpu().numpy()
        axs[i].scatter(z[:, 0], z[:, 1], s=10, c=y, cmap="tab10")
        axs[i].set_title(f"{name} (Depth/K={depth})")
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig("exercise2_embeddings.png")
    plt.show()
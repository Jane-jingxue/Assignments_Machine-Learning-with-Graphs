import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge
import random

dataset = GeometricShapes(root='data/GeometricShapes', transform=T.NormalizeFeatures())
loader = DataLoader(dataset,batch_size=20)
print(f"Number of graphs: {len(dataset)}")
print(f"Feature shape: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")
# Pick a specific graph by index (e.g., index 0)
graph_index = 20
data = dataset[graph_index]  # This returns a torch_geometric.data.Data object

# Inspect the graph
print("\n=== Graph Info ===")
print(data)
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Label: {data.y}")

dataface = Data(face=data.face)
transform = FaceToEdge(remove_faces=False)  # Keep faces if needed
dataface = transform(dataface)
data_edge=Data(x=data.pos[:,0:2],edge_index=dataface.edge_index,y=data.y)
data_netx = to_networkx(data_edge, to_undirected=True)
plt.figure(figsize=(6, 4))
nx.draw(data_netx, with_labels=True, node_color='lightblue', font_weight='bold')
ind=data_edge.edge_index[0,:]==0

# Node to vec random walk
# start: Probabilities 1/q everywhere
# save previous point neighbors and compare to new node neighbors
# common neighbors have prob=1
# non-common have prob 1/q, and back-step has 1/p

def node2vec(G,node,p,q,L,iter):
    all_walked_nodes = np.array([], dtype=np.int32)
    for i in range(iter):
        walked_nodes=np.array([], dtype=np.int32)
        walked_nodes = np.append(walked_nodes,node)
        # Finding the starting nodes indices (False/True):
        # data_edge.edge_index[0,:]==walked_nodes[-1]
        # Getting the neighbors through row nr. 1
        # data_edge.edge_index[1,data_edge.edge_index[0,:]==walked_nodes[-1]]
        neighbor = G.edge_index[1,G.edge_index[0,:]==walked_nodes[-1]] 
        weights = np.ones(len(neighbor))*(1/q)
        for j in range(L):
            walked_nodes = np.append(walked_nodes,random.choices(neighbor,weights=weights)) #append next node
            neighbor_prev = G.edge_index[1,G.edge_index[0,:]==walked_nodes[-2]]
            neighbor = G.edge_index[1,G.edge_index[0,:]==walked_nodes[-1]] #New neighbors list
            weights = np.zeros(len(neighbor)) #New weights vector
            for k in range(len(neighbor)):
                if neighbor[k]==walked_nodes[-2]: #If previous node
                    weights[k]=1/p
                elif any(neighbor[k]==neighbor_prev): # If neighbors are common
                    weights[k]=1
                else: #all others
                    weights[k]=1/q
        all_walked_nodes = np.append(all_walked_nodes,walked_nodes)
    return all_walked_nodes
p=10
q=1 #q=0.1 for small and q=1 for large
L=20 #walk length
iter=20
X = torch.zeros((data.num_nodes,(L+1)*iter),dtype=torch.int32)
for i in range(data.num_nodes):
    X[i,:] = torch.tensor(node2vec(data_edge,i,p,q,L,iter))
d=2
# take the d most common:
X_d = torch.zeros(X.shape[0],d)
for i in range(X.shape[0]):
    maxes = torch.zeros(X.shape[0],dtype=torch.int32)
    for j in X[i,:]:
        if j!=i:
            maxes[j] += 1
    sorted_tensor, indices = torch.sort(maxes,descending=True)
    for k in range(d):
        X_d[i,k] = indices[k]

X_edge=torch.zeros(2,data_edge.num_nodes*2)
for i in range(data_edge.num_nodes):
    X_edge[0,2*i] = i
    X_edge[0,2*i+1] = i
    X_edge[1,2*i] = X_d[i,0]
    X_edge[1,2*i+1] = X_d[i,1]

# B3.2
data_node2vec_1=Data(x=data_edge.x,edge_index=X_edge,y=data.y)
data_netx_node2vec_1 = to_networkx(data_node2vec_1, to_undirected=True)
plt.figure(figsize=(6, 4))
nx.draw(data_netx_node2vec_1, with_labels=True, node_color='lightsalmon', font_weight='bold')

# B3.3
data_node2vec_2=Data(x=X_d,edge_index=X_edge,y=data.y)
data_netx_node2vec_2 = to_networkx(data_node2vec_2, to_undirected=True)
plt.figure(figsize=(6, 4))
nx.draw(data_netx_node2vec_2, with_labels=True, node_color='limegreen', font_weight='bold')
plt.show()



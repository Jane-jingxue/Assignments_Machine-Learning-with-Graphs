import torch
import networkx as nx
import matplotlib.pyplot as plt
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
data = Data(pos=dataset[graph_index].pos,face=dataset[graph_index].face,y=dataset[graph_index].y,num_nodes=dataset[graph_index].num_nodes)  # This returns a torch_geometric.data.Data object

# Inspect the graph
print("\n=== Graph Info ===")
print(data)
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Label: {data.y}")

dataface = Data(face=data.face)

# Apply FaceToEdge transform
transform = FaceToEdge(remove_faces=False)  # Keep faces if needed
dataface = transform(dataface)

data_edge=Data(x=data.pos[:,0:2],edge_index=dataface.edge_index,y=data.y,num_nodes=data.num_nodes)

data_netx = to_networkx(data_edge, to_undirected=True)
plt.figure(figsize=(6, 4))
nx.draw(data_netx, with_labels=True, node_color='lightblue', font_weight='bold')

# struct2vec
# find w=e^f_k(u,v)
# fk(u,v) = fk−1(u,v) + g(s(Rk(u)),s(Rk(v)))
# f-1(u,v)=0
# find the degrees of all nodes, and put them in some tuple together with respective node index
# R0(u) is u and s(R0(u)) is the degree of u
# R1(u) are the direct neighbors of u, and s(R1(u)) are their nodes
# R2(u) should not include R1(u)
# weights w(u,v) should give a directional matrix
def degree(G):
    #Only for undirected graphs
    degree_array=torch.zeros(G.num_nodes,dtype=torch.int32)
    for i in range(G.num_nodes):
        degree_array[i] = len(G.edge_index[1, G.edge_index[0,:]==i])
    return degree_array

def dgr_seq(G,u,k):
    # Return a list of k numpy arrays
    # use .append() to append to list
    # s(Rk(u)) gives the k-hop nodes
    all_nodes = torch.tensor([u],dtype=torch.int32) # Keeping a list of all 0 to k hop neighbor nodes
    R = [all_nodes] # list of k-hop nodes, first list element is 0-hop, second 1-hop etc...
    s = [] # s is same as R, but instead of node indices we have their degrees
    for i in range(k+1):
        k_neighbors=torch.tensor([],dtype=torch.int32)
        for j in range(len(R[i])):
            j_neighbors = G.edge_index[1,torch.tensor(G.edge_index[0,:]==R[i][j])]
            k_neighbors = torch.cat((k_neighbors,j_neighbors))
        k_neighbors = torch.unique(k_neighbors)
        for j in range(len(all_nodes)):
            k_neighbors = k_neighbors[k_neighbors!=all_nodes[j]]
        all_nodes = torch.cat((all_nodes,k_neighbors))
        R.append(k_neighbors)
    # the returned s list needs to contain the degrees
    dgr = degree(G) # vector of all of the node degrees
    for i in range(len(R)):
        len_Ri = len(R[i])
        dgr_vec = torch.zeros(len_Ri,dtype=torch.int32)
        for j in range(len_Ri):
            dgr_vec[j] = dgr[R[i][j]]
        s.append(dgr_vec)
    return s

def dist(su,sv):
    if len(su)!=0 and len(sv)!=0:
        dist_max = torch.max(torch.tensor([torch.max(su),torch.max(sv)]))
        dist_min = torch.min(torch.tensor([torch.min(su),torch.min(sv)]))
    elif len(su)==0:
        su = torch.tensor([0])
        dist_max = torch.max(torch.tensor([torch.max(su),torch.max(sv)]))
        dist_min = torch.min(torch.tensor([torch.min(su),torch.min(sv)]))
    elif len(sv)==0:
        sv = torch.tensor([0])
        dist_max = torch.max(torch.tensor([torch.max(su),torch.max(sv)]))
        dist_min = torch.min(torch.tensor([torch.min(su),torch.min(sv)]))
    distance = dist_max/dist_min - 1
    return distance

def struct_dist(G,k):
    dgr = degree(G)
    #k=3
    fk_matrix = torch.zeros((G.num_nodes,G.num_nodes)) # zeros because f-1=0
    for a in range(G.num_nodes):
        su = dgr_seq(G,a,k) # will get the degrees of 0 to k hop neighbors of u 
        for b in range(G.num_nodes):
            if b!=a:
                sv = dgr_seq(G,b,k) # will get the degrees of 0 to k hop neighbors of v
                for c in range(k):
                    # here we find f_k(u,v)
                    fk_matrix[a,b] = fk_matrix[a,b] + dist(su[c],sv[c])
                    #g(s(Rk(u)),s(Rk(v)))
                    #s(Rk(u)) k-neighbor nodes
                    # dgr_seq(G,u,k) will get 0 to k hop neighbors of u 
    return fk_matrix
k=3
f = struct_dist(data_edge,k)
weights = torch.exp(-f)
weights_forWalk = weights
for i in range(data_edge.num_nodes):
    weights_forWalk[i,i] = 0
    norm = torch.sqrt(torch.sum(weights_forWalk[i,:]**2))
    weights_forWalk[i,:] = weights_forWalk[i,:]/norm

# when performing random walks the weights should be the probabilities
# start at a node
# list the neighbors
# list their weights (probabilities) and normalize
# perform the random walk
# repeat

Lstruct = 20 # do longer and pick the 2 most common at the end
Zstruct = torch.zeros(data_edge.num_nodes,Lstruct,dtype=torch.int32)
for i in range(data_edge.num_nodes):
    for j in range(Lstruct):
        random_choice = random.choices(torch.tensor(range(data_edge.num_nodes)),weights=weights_forWalk[Zstruct[i,j-1],:])[0]
        Zstruct[i,j] = random_choice

def most_occuring(Z,d):
    z_struct = torch.zeros(Z.shape[0],d)
    for i in range(Z.shape[0]):
        maxes = torch.zeros(Z.shape[0],dtype=torch.int32)
        for j in Z[i,:]:
            if j!=i:
                maxes[j] += 1
        sorted_tensor, indices = torch.sort(maxes,descending=True)
        for k in range(d):
            z_struct[i,k] = indices[k]
    '''
    z_struct = torch.zeros(data_edge.num_nodes,d,dtype=torch.int32)
    for i in range(data_edge.num_nodes):
        index_count=torch.zeros(data_edge.num_nodes,dtype=torch.int32)
        sorted, indices = torch.sort(Z[i,:])
        for j in range(len(sorted)):
            if j!=i:
                index_count[sorted[j]] += 1
        # use sort again and take d first values
        sorted, indices = torch.sort(index_count, descending=True)
        z_struct[i,:] = indices[0:2]
    return z_struct
    '''
    return z_struct

d=2
Zstruct_d = most_occuring(Zstruct,d)

struct_edge=torch.zeros(d,data_edge.num_nodes*2)
for i in range(data_edge.num_nodes):
    struct_edge[0,2*i] = i
    struct_edge[0,2*i+1] = i
    struct_edge[1,2*i] = Zstruct_d[i,0]
    struct_edge[1,2*i+1] = Zstruct_d[i,1]

data_struct2vec=Data(x=Zstruct_d,edge_index=struct_edge,y=data.y)
data_struct2vec_netx = to_networkx(data_struct2vec, to_undirected=True)
plt.figure(figsize=(6, 4))
nx.draw(data_struct2vec_netx, with_labels=True, node_color='gray', font_weight='bold')
plt.show()
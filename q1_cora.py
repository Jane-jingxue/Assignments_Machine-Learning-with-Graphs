import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
loader = DataLoader(dataset,batch_size=20)
for row in loader:
    print(row)

data_netx = to_networkx(dataset[0], node_attrs=["x"], to_undirected=True)
# Degree
dgr = data_netx.degree()
# Betweenness centrality
btwn_cntrlty = nx.betweenness_centrality(data_netx, normalized=True)
# Closeness centrality
clsnss_cntrlty = nx.closeness_centrality(data_netx)
# PageRank
pr = nx.pagerank(data_netx, alpha=0.95)
# Eigenvector centrality
eig_cntrlty = nx.eigenvector_centrality(data_netx)
# Clustering coefficient
clstr_coeff=nx.clustering(data_netx)
xrows, xcols = dataset.x.shape
Xstruct=torch.zeros((xrows, 6))

for i in range(dataset.x.shape[0]):
    Xstruct[i,0] = dgr[i]
    Xstruct[i,1] = btwn_cntrlty[i]
    Xstruct[i,2] = clsnss_cntrlty[i]
    Xstruct[i,3] = pr[i]
    Xstruct[i,4] = eig_cntrlty[i]
    Xstruct[i,5] = clstr_coeff[i]
for i in range(6):
    Xstruct[:,i] = Xstruct[:,i]/torch.sqrt(torch.sum(Xstruct[:,i]**2)) # normalizing the rows

Xattr=dataset.x
Xcomb = torch.cat((torch.tensor(Xstruct),Xattr),dim=1)

#For Xstruct
model = LogisticRegression(max_iter=100)
multi_model = OneVsRestClassifier(model)
multi_model.fit(Xstruct, dataset.train_mask)
print("For struct:")
print("Validation Multi-class Accuracy:", accuracy_score(dataset.val_mask, multi_model.predict(Xstruct)))
print("Test Multi-class Accuracy:", accuracy_score(dataset.test_mask, multi_model.predict(Xstruct)))

#For Xattr
model = LogisticRegression(max_iter=100)
multi_model = OneVsRestClassifier(model)
multi_model.fit(Xattr, dataset.train_mask)
print("For attr:")
print("Validation Multi-class Accuracy:", accuracy_score(dataset.val_mask, multi_model.predict(Xattr)))
print("Test Multi-class Accuracy:", accuracy_score(dataset.test_mask, multi_model.predict(Xattr)))

#For Xcomb
model = LogisticRegression(max_iter=100)
multi_model = OneVsRestClassifier(model)
multi_model.fit(Xcomb, dataset.train_mask)
print("For comb:")
print("Validation Multi-class Accuracy:", accuracy_score(dataset.val_mask, multi_model.predict(Xcomb)))
print("Test Multi-class Accuracy:", accuracy_score(dataset.test_mask, multi_model.predict(Xcomb)))


import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Part A

dataset=KarateClub()
loader = DataLoader(dataset,batch_size=20)
for row in loader:
    print(row)
#Printed:
#DataBatch(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34], batch=[34], ptr=[2])

#We have:
# 34 nodes
# 156 edges
# x is a identity matrix of size 34
# edge is a 2-by-156 matrix
# y contains values 0-3, designating the groups

#Creating the group colors for plot
#dataset.y contains the group values
# change depending on range of y
color_labels=['red','blue','green','purple']
node_colors=[]
for i in range(dataset.x.shape[0]):
    node_colors.append(color_labels[dataset.y[i]])

#Part B
#Converting to networkx and plotting
data_netx = to_networkx(dataset[0], node_attrs=["x"], to_undirected=True)
plt.figure(figsize=(6, 4))
nx.draw(data_netx, with_labels=True, node_color=node_colors, font_weight='bold')
#plt.show()

def degree(G):
    #Only for undirected graphs
    degree_array=torch.zeros(G.x.shape[0],dtype=torch.int32)
    for i in range(G.x.shape[0]):
        degree_array[i] = len(G.edge_index[1, G.edge_index[0,:]==i])
    norm = torch.sqrt(torch.sum(degree_array**2))
    return degree_array/norm
def clustering(G):
    # works for directional and undirectionial edges
    # Number of edges between u and its neighbors divided by the maximum possible edges
    # maximum possible should be 2*(1+2+3+4...) both directions
    # find the neighbors and count the edge_index vector
    clustering_array=torch.zeros(G.x.shape[0],dtype=torch.float32)
    for i in range(G.x.shape[0]):
        num_nghbr_u,ind_nghbr_u = neighbors(i,G.edge_index) # outgoing edges
        for j in ind_nghbr_u:
            num_nghbr_v = 0
            _,ind_nghbr_v = neighbors(j,G.edge_index) # outgoing edges
            for k in ind_nghbr_v:
                num_nghbr_v += torch.sum(ind_nghbr_u==k)
            #check commonality between ind_nghbr_u and ind_nghbr_v
        # 2: 1
        # 3: 3
        # 4: 6
        # sum(i)from 2 to n
        den = sum(range(1,num_nghbr_u))
        # case den = 0 and clustering array becomes infinite
        if den>0:
            clustering_array[i] = (num_nghbr_u + num_nghbr_v)/sum(range(1,num_nghbr_u))
        else:
            clustering_array[i] = (num_nghbr_u + num_nghbr_v)
        #neighbors
    #NORMALIZE
    norm = torch.sqrt(torch.sum(clustering_array**2))
    return clustering_array/norm
def neighbors(u,edge):
    ind = edge[0,:]==u
    num_neighbors = torch.sum(ind)
    ind_neighbors = edge[1,ind]
    return num_neighbors, ind_neighbors
def pagerank(G,beta):
    # Reference http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf
    #R(u)=c*sum(R(v)/Nv+c*(1-beta))
    # R: ranks
    # Nv: outgoing edges from node v
    # loop over all nodes for some time until a condition holds
    tol = 0.01
    iter = 1
    iter_max = 100
    tot_diff_rank = 1
    R = torch.zeros(G.x.shape[0],dtype=torch.float32)
    while tot_diff_rank>tol:
        checkR = R.clone().detach()
        for i in range(G.x.shape[0]):
            #count neighbors of u
            _,ind_nghbr_i = neighbors(i,G.edge_index)
            for j in ind_nghbr_i:
                num_nghbr_j,_ = neighbors(j,G.edge_index)
                R[i] = R[j]/num_nghbr_j+(1-beta)
        tot_diff_rank = torch.sum(torch.abs(checkR-R))
        if iter>iter_max:
            print("PageRank: No convergence, max iteration reached!")
            break
        # some tot_diff_rank calculation
    #NORMALIZE
    norm = torch.sqrt(torch.sum(R**2))
    return R/norm
def eigenvector_centrality(G):
    # Ax=lambda x
    # A is the adjcency matrix
    # lambda is the eigenvalue
    # x is the score vector, eig_cntrlty here
    # x = 1/lambda * sum(a*x), a=1 if neighbors, otherwise =0
    # use: eigenvalues, eigenvectors = torch.linalg.eig(A)
    # Take the largest eigenvalue and its eigenvector, which is our eig_cntrlty 
    A = torch.zeros(G.x.shape[0],G.x.shape[0])
    for i in range(G.x.shape[0]):
        A[i,i] = 1
        _,ind_nghbr_i = neighbors(i,G.edge_index)
        for j in ind_nghbr_i:
            A[i,j] = 1
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    max_ind = torch.max(torch.abs(eigenvalues))==torch.abs(eigenvalues)
    eig_cntrlty = torch.abs(eigenvectors[max_ind])[0]
    #NORMALIZE
    norm = torch.sqrt(torch.sum(eig_cntrlty**2))
    return eig_cntrlty/norm
def hop_neighbors(G,u):
    hop_neighbors_list = [torch.tensor([u])]
    doublettes_neighbors_list = [torch.tensor([u])]
    all_neighbors = torch.tensor([u])
    hops=0
    remaining_nodes = True 
    max_iter = 100
    while remaining_nodes:
        k_neighbors = torch.tensor([],dtype=torch.int32)
        for i in range(len(hop_neighbors_list[hops])):
            new_index = G.edge_index[1,G.edge_index[0,:]==hop_neighbors_list[hops][i]]
            for j in new_index:
                if torch.sum(all_neighbors==j)==0:
                    k_neighbors = torch.cat((k_neighbors,torch.tensor([j])))
                    #all_neighbors = torch.cat((all_neighbors,torch.tensor([j])))
        all_neighbors = torch.cat((all_neighbors,k_neighbors))
        doublettes_neighbors_list.append(k_neighbors)    
        hop_neighbors_list.append(torch.unique(k_neighbors))
        hops += 1
        if len(torch.unique(all_neighbors))==(G.x.shape[0]):
            remaining_nodes = False
        if hops>max_iter:
            print("hop-neighbors: Fail, Max iteration reached!")
            break
    return hop_neighbors_list, doublettes_neighbors_list         
def betweenness(G):
    # g(v)=sigma_st(v)/sigma_st
    # sigma_st: total number of shortest paths between s and t
    # sigma_st(v): shortest number of paths that pass through v
    # a k-hop algorithm for node v until all nodes are covered for
    # For 1-hop no nodes are passed
    # For 2-hop the direct neighbors indices should be added by one
    # For 2-hop: If two 1-step neighbors share a 2-step neighbor t sigma_st is =2
    # For 3-hop: 1-hop and 2-hop neighbor indices added by 1
    # For 3-hop: If 2-step neighbors share common 3-hop neighbors ...
    # Check if all nodes are covered
    g = torch.zeros(G.x.shape[0],dtype=torch.float32)
    sigma_st_v = torch.zeros(G.x.shape[0],dtype=torch.int32)
    sigma_st = torch.zeros(G.x.shape[0],dtype=torch.int32)
    g = torch.zeros(G.x.shape[0],dtype=torch.float32)
    for i in range(G.x.shape[0]):
        # find all k-hop neighbors
        unique_neighbors, doublette_neighbors = hop_neighbors(dataset,i)
        hops = len(unique_neighbors)
        if hops>=3: # should not add first and the last of hops
            for j in range(1,hops-1):
                for k in unique_neighbors[j]: # produce the sigma_st(v) vector
                    sigma_st_v[k] += 1
                for k in doublette_neighbors[j]: # produce the sigma_vt
                    sigma_st[k] += 1
    for i in range(G.x.shape[0]):
        g[i] = sigma_st_v[i]/sigma_st[i]
    #NORMALIZE
    norm = torch.sqrt(torch.sum(g**2))
    return g/norm
def closeness(G):
    # same as betweeness, but without the nominator
    g = torch.zeros(G.x.shape[0],dtype=torch.float32)
    #sigma_st_v = torch.zeros(G.x.shape[0],dtype=torch.int32)
    sigma_st = torch.zeros(G.x.shape[0],dtype=torch.int32)
    g = torch.zeros(G.x.shape[0],dtype=torch.float32)
    for i in range(G.x.shape[0]):
        # find all k-hop neighbors
        unique_neighbors, doublette_neighbors = hop_neighbors(dataset,i)
        hops = len(unique_neighbors)
        if hops>=2:
            for j in range(1,hops):
                #for k in unique_neighbors[j]: # produce the sigma_st(v) vector
                    #sigma_st_v[k] += 1
                for k in doublette_neighbors[j]: # produce the sigma_vt
                    sigma_st[k] += 1
    for i in range(G.x.shape[0]):
        g[i] = 1/sigma_st[i]
    #NORMALIZE
    norm = torch.sqrt(torch.sum(g**2))
    return g/norm

dgr = degree(dataset)
btwn_cntrlty = betweenness(dataset)
clsnss_cntrlty = closeness(dataset)
pr = pagerank(dataset,0.85)
eig_cntrlty = eigenvector_centrality(dataset)
clstr_coeff=clustering(dataset)

#Part C
Xstruct=torch.zeros((dataset.x.shape[0], 6))

for i in range(dataset.x.shape[0]):
    Xstruct[i,0] = dgr[i]
    Xstruct[i,1] = btwn_cntrlty[i]
    Xstruct[i,2] = clsnss_cntrlty[i]
    Xstruct[i,3] = pr[i]
    Xstruct[i,4] = eig_cntrlty[i]
    Xstruct[i,5] = clstr_coeff[i]

def plotPCA(X,y):
    color_labels=['red','blue','green','purple']
    node_colors=[]
    for i in range(X.shape[0]):
        node_colors.append(color_labels[y[i]])
    # Step 2: Apply PCA (reduce to 2 components for visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", np.sum(pca.explained_variance_ratio_))
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca['target'] = y    
    target_names=["Group 1","Group 2","Group 3","Group 4"]
    plt.figure(figsize=(8, 6))
    for target, color in zip(range(len(target_names)), color_labels):
        subset = df_pca[df_pca['target'] == target]
        plt.scatter(subset['PC1'], subset['PC2'], label=target_names[target], alpha=0.7, c=color_labels[target])
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    plt.title("PCA of KarateClub")
    plt.legend()
    plt.grid(True)

plotPCA(Xstruct,dataset.y,)


#0.43 accuracy for Xstruct
#0.29 for X
#0.43 for Xcomb
#Use SGD
#Multinomial logistic regression

# Multi-class classification
Xcomb = torch.cat((dataset.x,Xstruct),dim=1)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(Xstruct, dataset.y, test_size=0.2, random_state=12)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(dataset.x, dataset.y, test_size=0.2, random_state=12)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(Xcomb, dataset.y, test_size=0.2, random_state=12)


#plotPCA(X_test,multi_model.predict(X_test))

#solver='liblinear' best
slvr='liblinear'
model_1 = LogisticRegression(max_iter=100000,solver=slvr,class_weight='balanced')
multi_model_1 = OneVsRestClassifier(model_1)
multi_model_1.fit(X_train_1, y_train_1)
print("Multi-class Accuracy:", accuracy_score(y_test_1, multi_model_1.predict(X_test_1)))
print(multi_model_1.predict(X_test_1))
plotPCA(X_test_1,multi_model_1.predict(X_test_1))

model_2 = LogisticRegression(max_iter=100000,solver=slvr,class_weight='balanced')
multi_model_2 = OneVsRestClassifier(model_2)
multi_model_2.fit(X_train_2, y_train_2)
print("Multi-class Accuracy:", accuracy_score(y_test_2, multi_model_2.predict(X_test_2)))
print(multi_model_2.predict(X_test_2))
plotPCA(X_test_2,multi_model_2.predict(X_test_2))

model_3 = LogisticRegression(max_iter=100000,solver=slvr,class_weight='balanced')
multi_model_3 = OneVsRestClassifier(model_3)
multi_model_3.fit(X_train_3, y_train_3)
print("Multi-class Accuracy:", accuracy_score(y_test_3, multi_model_3.predict(X_test_3)))
print(multi_model_3.predict(X_test_3))
plotPCA(X_test_3,multi_model_3.predict(X_test_3))
'''
model_2 = LogisticRegression(max_iter=100000,solver=slvr,class_weight='balanced')
multi_model_2 = OneVsRestClassifier(model_2)
multi_model_2.fit(dataset.x, dataset.y)
print("Multi-class Accuracy:", accuracy_score(dataset.y, multi_model_2.predict(dataset.x)))
print(multi_model_2.predict(dataset.x))
plotPCA(Xstruct,multi_model_2.predict(dataset.x))

Xcomb = torch.cat((dataset.x,Xstruct),dim=1)
model_3 = LogisticRegression(max_iter=1000000,solver=slvr,class_weight='balanced')
multi_model_3 = OneVsRestClassifier(model_3)
multi_model_3.fit(Xcomb, dataset.y)
print("Multi-class Accuracy:", accuracy_score(dataset.y, multi_model_3.predict(Xcomb)))
print(multi_model_3.predict(Xcomb))
plotPCA(Xstruct,multi_model_3.predict(Xcomb))
'''
plt.show()

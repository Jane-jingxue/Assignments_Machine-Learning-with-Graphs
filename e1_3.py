import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import re
import random
from collections import Counter
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import kagglehub
import os

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

WINDOW_SIZE_COOCCURRENCE = 2  
VOCAB_SIZE = 3000            
EMBEDDING_DIM = 64   
WALK_LENGTH = 10         
NUM_WALKS = 20                
P = 1.0                       
Q = 0.5                      
WINDOW_SIZE_W2V = 5           # Word2Vec window size



def preprocess_text(text):
    """
    Cleans text: lowercase, remove punctuation, tokenize, remove stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation/special chars
    tokens = text.split()
    
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    return tokens

def build_cooccurrence_graph(corpus, window_size=2, vocab_limit=1000):
    """
    Constructs an undirected word co-occurrence graph.
 
    """
    print("Building vocabulary...")
    # 1. Build Vocabulary
    all_words = [word for doc in corpus for word in doc]
    word_counts = Counter(all_words)
    common_words = {word for word, count in word_counts.most_common(vocab_limit)}
    
    print(f"Vocabulary limited to top {vocab_limit} words.")
    
    # 2. Initialize Graph
    G = nx.Graph()
    G.add_nodes_from(common_words)
    
    print("Adding edges based on co-occurrence...")
    # 3. Add Edges 
    for doc in corpus:
        # Filter doc to only include vocab words to speed up and keep graph valid
        doc_filtered = [w for w in doc if w in common_words]
        
        for i, target_word in enumerate(doc_filtered):
            # Look at words in window (i+1 to i+window_size)
            # We only look forward because it's undirected
            end = min(i + 1 + window_size, len(doc_filtered))
            for j in range(i + 1, end):
                context_word = doc_filtered[j]
                
                if target_word == context_word:
                    continue
                
                if G.has_edge(target_word, context_word):
                    G[target_word][context_word]['weight'] += 1
                else:
                    G.add_edge(target_word, context_word, weight=1)
                    
    return G


def get_neighbors(G, node):
    return list(G.neighbors(node))

def node2vec_walk(G, start_node, walk_length, p, q):
    """
    Generates a single biased random walk.
    Adapted from user provided e1_2.py
    """
    walk = [start_node]
    
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = get_neighbors(G, cur)
        
        if len(cur_nbrs) == 0:
            break
            
        if len(walk) == 1:
            # First step: standard weighted random walk
            # Note: For weighted graphs, we use edge weights
            weights = [G[cur][nbr]['weight'] for nbr in cur_nbrs]
            total_w = sum(weights)
            probs = [w/total_w for w in weights]
            walk.append(np.random.choice(cur_nbrs, p=probs))
            continue
            
        prev = walk[-2]
        
        # Calculate unnormalized transition probs alpha based on weights
        weights = []
        for neighbor in cur_nbrs:
            edge_weight = G[cur][neighbor]['weight']
            
            if neighbor == prev:
                weights.append(edge_weight * 1/p)
            elif G.has_edge(neighbor, prev):
                weights.append(edge_weight * 1)
            else:
                weights.append(edge_weight * 1/q)
        
        # Normalize
        total_w = sum(weights)
        if total_w == 0:
            break
        probs = [w/total_w for w in weights]
        
        # Choose next node
        next_node = np.random.choice(cur_nbrs, p=probs)
        walk.append(next_node)
        
    return walk

def train_node2vec(G, dim, walk_length, num_walks, p, q):
    """
    Generates walks and trains embeddings using Skip-Gram.
    
    """
    print(f"Generating Node2Vec walks (p={p}, q={q})...")
    walks = []
    nodes = list(G.nodes())
    
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(node2vec_walk(G, node, walk_length, p, q))
            
    print("Training Word2Vec model...")
    # Train Skip-Gram (sg=1)
    model = Word2Vec(sentences=walks, vector_size=dim, 
                     window=WINDOW_SIZE_W2V, min_count=1, sg=1, workers=4, epochs=5)
    
    return model


def get_weighted_document_embedding(doc, model, word2weight):
    embeddings = []
    weights = []
    
    for word in doc:
        if word in model.wv and word in word2weight:
            embeddings.append(model.wv[word])
            weights.append(word2weight[word])
    
    if len(embeddings) == 0:
        return np.zeros(model.vector_size)
    
    # Compute weighted average
    return np.average(embeddings, axis=0, weights=weights)

def main():
    print("Loading data...")
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    csv_path = os.path.join(path, "IMDB Dataset.csv")
    
    # Use 8000 samples for better results
    df = pd.read_csv(csv_path).sample(8000, random_state=42)

    # Preprocessing
    print("Preprocessing text...")
    df['processed_tokens'] = df['review'].apply(preprocess_text)
    
    # Build Graph
    print("Building graph...")
    G = build_cooccurrence_graph(df['processed_tokens'].tolist(), 
                                 window_size=WINDOW_SIZE_COOCCURRENCE,
                                 vocab_limit=VOCAB_SIZE)
    print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # B. Train Node2Vec 

    model = train_node2vec(G, EMBEDDING_DIM, WALK_LENGTH, NUM_WALKS, P, Q)
    
   
    # Visualize Embeddings 
    print("Visualizing Node2Vec embeddings...")  
    # Define symbolic words to check for clustering
    positive_seeds = [
        'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 
        'best', 'brilliant', 'loved', 'perfect', 'beautiful', 'enjoyed'
    ]
    negative_seeds = [
        'terrible', 'awful', 'worst', 'horrible', 'bad', 
        'waste', 'boring', 'poor', 'disappointing', 'stupid', 'hate'
    ]
    
    # Filter words that actually exist in our trained model
    selected_words = []
    selected_colors = []
    
    for word in positive_seeds:
        if word in model.wv:
            selected_words.append(word)
            selected_colors.append('green') # Positive = Green
            
    for word in negative_seeds:
        if word in model.wv:
            selected_words.append(word)
            selected_colors.append('red')   # Negative = Red

    # If we don't have enough words, fallback to generic top words
    if len(selected_words) < 5:
        print("Warning: Specific sentiment words not found in vocab. Plotting top frequent words instead.")
        selected_words = list(G.nodes())[:50]
        selected_colors = ['blue'] * len(selected_words)

    # Prepare vectors
    vectors = np.array([model.wv[w] for w in selected_words])
    
    if len(vectors) > 0:
        # Perplexity must be less than number of samples
        perp = min(5, len(vectors) - 1)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        vectors_2d = tsne.fit_transform(vectors)
        
        plt.figure(figsize=(10, 8))
        
        # Plot with specific colors
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                    c=selected_colors, edgecolors='k', s=150, alpha=0.75)
        
        for i, label in enumerate(selected_words):
            plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                         fontsize=11, weight='bold', 
                         xytext=(5, 5), textcoords='offset points')
                         
        plt.title("Node2Vec Sentiment Clustering (Green=Pos, Red=Neg)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()
  
    print("Training classifier...")
    # Create X (document embeddings) and y (labels)
    corpus_strings = [" ".join(doc) for doc in df['processed_tokens']]
    vectorizer = TfidfVectorizer(vocabulary=list(G.nodes())) 
    tfidf_matrix = vectorizer.fit_transform(corpus_strings)
    word2weight = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    X = np.array([get_weighted_document_embedding(doc, model, word2weight) 
              for doc in df['processed_tokens']])
    y = df['sentiment'].map({'positive': 1, 'negative': 0}).values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nSentiment Classification Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
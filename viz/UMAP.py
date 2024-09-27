import json
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os

# Function to load embeddings from JSON file
def load_embeddings(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(list(data.values()))

# Function to extract dataset name from filename
def get_dataset_name(filename):
    return filename.split('-trimmed')[0]

# List of JSON files containing embeddings
json_files = [
    "acpas-unique-trimmed-s30-t30-embeddings.json",
    "CMR-trimmed-s30-t30-embeddings.json"
]

# Load embeddings and prepare labels
all_embeddings = []
all_labels = []
dataset_names = []

for i, file in enumerate(json_files):
    embeddings = load_embeddings(file)
    all_embeddings.append(embeddings)
    all_labels.extend([i] * len(embeddings))
    dataset_names.append(get_dataset_name(file))

all_embeddings = np.vstack(all_embeddings)
all_labels = np.array(all_labels)

# Perform UMAP dimensionality reduction
umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(all_embeddings)

# Create the plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                      c=all_labels, cmap='viridis', alpha=0.7)

# Add legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                   label=name, markerfacecolor=plt.cm.viridis(i/(len(dataset_names)-1)), markersize=10)
                   for i, name in enumerate(dataset_names)]
plt.legend(handles=legend_elements, title="Datasets")

plt.title("UMAP Visualization of Embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

# Save the plot
plt.savefig("umap_visualization.png", dpi=300, bbox_inches='tight')
plt.close()

print("UMAP visualization has been saved as 'umap_visualization.png'")
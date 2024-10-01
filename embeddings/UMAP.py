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
# TODO update
def get_dataset_name(filename):
    return filename.split('-trimmed')[0].split("embeddings/")[-1]

def get_filenames(sources):
    files = []
    for data in sources:
        if data['segment'] and data['stride']:
            sampling = f"s{data['segment']}-t{data['stride']}"
        else:
            sampling = "raw"
        path = os.path.join(data['dataset'], sampling)
        if os.path.exists(path):
            filename = os.path.join(path, f"{data['method']}_embeddings.json")
            files.append(filename)       

    if files == []:
        raise ValueError

    return files

def main():
    config = [
        # {
        #     "dataset": "acpas",
        #     "method": "last",
        #     "segment": "30",
        #     "stride": "30"
        # },
        {
            "dataset": "CBF",
            "method": "mean",
            "segment": "15",
            "stride": "15",
        },
        # {
        #     "dataset": "birdsong",
        #     "method": "last",
        #     "segment": None,
        #     "stride": None,            
        # },
        {
            "dataset": "toy",
            "method": "mean",
            "segment": None,
            "stride": None,            
        },
    ]

    # List of JSON files containing embeddings
    json_files = get_filenames(config)

    print(json_files)

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
    umap_model = UMAP(n_neighbors=3, min_dist=0.1)
    umap_embeddings = umap_model.fit_transform(all_embeddings)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                        c=all_labels, cmap='viridis', alpha=0.7)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                    label=name, markerfacecolor=plt.cm.viridis(i/(len(dataset_names)-1)), markersize=10)
                    for i, name in enumerate(dataset_names)]
    plt.legend(handles=legend_elements, title="Datasets")

    plt.title("UMAP Visualization of Embeddings")

    # Save the plot
    plt.savefig("../viz/umap_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("UMAP visualization has been saved as 'umap_visualization.png'")

if __name__ == "__main__":
    main()
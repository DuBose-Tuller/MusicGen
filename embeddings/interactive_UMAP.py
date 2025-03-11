# interactive_umap.py
import plotly.express as px
import numpy as np
import h5py
import argparse
from pathlib import Path
import os

def load_umap_data(h5_file):
    """Load UMAP embeddings from an H5 file created by UMAP.py"""
    with h5py.File(h5_file, 'r') as f:
        embeddings = f['embeddings'][()]
        labels = f['labels'][()]
        # Convert byte strings to regular strings
        class_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                      for name in f['class_names'][()]]
    return embeddings, labels, class_names

def create_interactive_plot(embeddings, labels, class_names, output_path):
    """Create an interactive scatter plot with Plotly"""
    # Create a DataFrame for Plotly Express
    import pandas as pd
    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'dataset': [class_names[label] for label in labels]
    })
    
    # Create interactive plot
    fig = px.scatter(
        df, x='x', y='y', color='dataset',
        title="Interactive UMAP Visualization",
        opacity=0.7,
        category_orders={"dataset": class_names}
    )
    
    # Enhance the plot
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        legend_title_text="Datasets",
        legend=dict(itemsizing='constant'),
        width=1000, height=800
    )
    
    # Save as HTML
    fig.write_html(f"{output_path}.html")
    print(f"Interactive visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create interactive UMAP visualization")
    parser.add_argument('--input', required=True, help='Path to H5 file with UMAP embeddings')
    args = parser.parse_args()

    output_dir = "../results/interactive-UMAP"
    
    input_path = Path(args.input)
    output_path = os.path.join(output_dir, str(input_path).split("/")[-1].split(".")[0])
    
    # Load data and create interactive plot
    embeddings, labels, class_names = load_umap_data(input_path)
    create_interactive_plot(embeddings, labels, class_names, output_path)

if __name__ == "__main__":
    main()

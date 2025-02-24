import numpy as np
import pandas as pd
from embeddings.h5_processor import H5DataProcessor, DatasetConfig, ProcessedDataset, AudioSegmentSplitter
from pathlib import Path
from embeddings.models import RatingsClassifier
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import os

# 1. Load ratings data
print("Loading ratings data...")
df = pd.read_csv('universal_music/FFfull.csv', low_memory=False)
web_df = df[df['study'] == 'web'].copy()

# Get ratings and normalize
web_ratings = web_df[["song", "danc", "heal", "baby", "love"]]
print(f"Loaded ratings for {len(web_ratings)} songs")

# Process datasets using H5DataProcessor
processor = H5DataProcessor(verbose=True)
all_datasets = []
with open("universal_music/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# 2. Process embeddings
print("Loading embeddings...")
datasets = []
for dataset_config in config['datasets']:
    print(f"Processing {dataset_config}")
    
    dataset = processor.process_h5_file(
        processor.get_embedding_path(DatasetConfig(**dataset_config)),
        DatasetConfig(**dataset_config)
    )
    datasets.append(dataset)

# 3. Link song IDs to embeddings
print("Linking song IDs to embeddings...")
# Function to extract song ID from filename
def extract_song_id(filename):
    match = re.search(r'NHSDiscography-(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

# Create a dictionary mapping song IDs to their embeddings
song_embeddings = {}
for dataset in datasets:
    for i, filename in enumerate(dataset.filenames):
        song_id = extract_song_id(filename)
        if song_id not in song_embeddings:
            song_embeddings[song_id] = []
        song_embeddings[song_id].append(dataset.embeddings[i])

print(f"Found embeddings for {len(song_embeddings)} unique songs")

# 4. Create training data with one example per rating
print("Creating training data...")
X_all = []
y_all = []
categories = ["danc", "heal", "baby", "love"]
category_embeddings = {cat: [] for cat in categories}
category_ratings = {cat: [] for cat in categories}

for _, row in web_ratings.iterrows():
    song_id = int(row['song'])
    
    if song_id in song_embeddings and len(song_embeddings[song_id]) > 0:
        # Use the average embedding for this song
        song_embedding = np.mean(song_embeddings[song_id], axis=0)
        
        # For each category, add this embedding with its rating
        for cat in categories:
            if not pd.isna(row[cat]):
                category_embeddings[cat].append(song_embedding)
                category_ratings[cat].append(row[cat])

# Convert to numpy arrays and print stats
for cat in categories:
    print(f"Category {cat}: {len(category_embeddings[cat])} ratings")
    category_embeddings[cat] = np.array(category_embeddings[cat])
    category_ratings[cat] = np.array(category_ratings[cat])

# 5. Train a separate model for each category
print("Training models for each category...")
models = {}
results = {}

for category in categories:
    if len(category_embeddings[category]) > 0:
        print(f"\nTraining model for {category}...")
        X = category_embeddings[category]
        y = category_ratings[category]
        
        # Normalize ratings to be between 0 and 3 (for the 4 rating levels in RatingsClassifier)
        y_normalized = (y * 3 / 7).round().astype(int)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Configure and train model
        # Adjust parameters based on the nature of ratings
        model = RatingsClassifier(
            n_categories=1,  # One model per category
            n_ratings=4,     # 0-3 rating scale
            learning_rate=0.01,
            max_iter=1000,
            l1_penalty=0.1
        )
        
        print(f"Training on {len(X_train)} examples with {len(np.unique(y_train))} unique ratings")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        pred_ratings = model.predict_ratings(X_test_scaled)
        accuracy = np.mean(pred_ratings[:, 0] == y_test)
        
        # Store model and results
        models[category] = model
        results[category] = {
            'accuracy': accuracy,
            'predictions': pred_ratings,
            'true_values': y_test
        }
        
        print(f"Accuracy for {category}: {accuracy:.4f}")
        
        # Distribution of predicted ratings
        unique, counts = np.unique(pred_ratings, return_counts=True)
        print("Distribution of predicted ratings:")
        for val, count in zip(unique, counts):
            print(f"  Rating {val}: {count} instances")

# 6. Save models and results
print("\nSaving models and results...")
os.makedirs('universal_music/models', exist_ok=True)

# Save scaler and models using pickle
import pickle
for category in categories:
    if category in models:
        with open(f'universal_music/models/{category}_model.pkl', 'wb') as f:
            pickle.dump(models[category], f)
        
        # Also save results
        np.savez(
            f'universal_music/models/{category}_results.npz',
            predictions=results[category]['predictions'],
            true_values=results[category]['true_values'],
            accuracy=results[category]['accuracy']
        )

print("Done!")

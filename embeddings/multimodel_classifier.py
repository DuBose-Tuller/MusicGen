"""
Model comparison utility for analyzing music embeddings.
This script compares multiple classification models and performs ablation studies.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from datetime import datetime
import hashlib
import argparse
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from h5_processor import H5DataProcessor, DatasetConfig, ProcessedDataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare multiple models on embedding data")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Ratio of data to use for testing')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', default='../results/model_comparison', help='Output directory')
    parser.add_argument('--feature-count', type=int, default=20, help='Number of top features to display')
    parser.add_argument('--ablation', action='store_true', help='Perform ablation studies')
    parser.add_argument('--verbose', '-v', action="store_true", help="Extra print statements for debugging")
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def compute_metrics(y_test, y_pred, y_prob, is_binary=False):
    """Compute classification metrics."""
    metrics = {
        "f1": f1_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "per_class": {
            "precision": precision_score(y_test, y_pred, average=None).tolist(),
            "recall": recall_score(y_test, y_pred, average=None).tolist(),
            "f1": f1_score(y_test, y_pred, average=None).tolist()
        }
    }

    if is_binary:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
        except Exception:
            metrics["roc_auc"] = None
    
    metrics["confidence"] = {
        "mean": float(np.mean(np.max(y_prob, axis=1))),
        "std": float(np.std(np.max(y_prob, axis=1)))
    }
    
    return metrics

def get_model_config(model):
    """Extract model configuration and hyperparameters."""
    def make_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (int, float, str, bool, type(None))):
            return val
        else:
            return str(val)

    # Handle both direct model objects and pipelines
    if hasattr(model, 'steps'):
        # It's a pipeline
        attributes = {'pipeline_steps': [s[0] for s in model.steps]}
        # Add attributes of the final estimator
        final_estimator = model.steps[-1][1]
        for key, value in vars(final_estimator).items():
            if not key.startswith('_'):
                attributes[key] = make_serializable(value)
    else:
        # Regular model
        attributes = {
            key: make_serializable(value) 
            for key, value in vars(model).items() 
            if not key.startswith('_')
        }

    return {
        "type": model.__class__.__name__,
        "hyperparameters": attributes
    }

def get_models():
    """Initialize classification models to compare."""
    models = {
        "LogisticRegression_L1": LogisticRegression(
            solver='saga', 
            penalty='l1', 
            C=1.0, 
            max_iter=1000, 
            multi_class='multinomial',
            random_state=42
        ),
        
        "LogisticRegression_L2": LogisticRegression(
            solver='saga', 
            penalty='l2', 
            C=1.0, 
            max_iter=1000, 
            multi_class='multinomial',
            random_state=42
        ),
        
        "LogisticRegression_None": LogisticRegression(
            solver='saga', 
            penalty=None, 
            C=1.0, 
            max_iter=1000, 
            multi_class='multinomial',
            random_state=42
        ),
        
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        ),
        
        "LinearSVM": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(penalty='l2', dual=False, max_iter=2000, random_state=42))
        ])
    }
    
    return models

def create_feature_subsets(X, n_features, method='random', n_subsets=5, random_state=42):
    """Create feature subsets for ablation studies.
    
    Parameters:
    -----------
    X : array-like
        Full feature matrix
    n_features : int
        Number of features to select for each subset
    method : str
        Method for creating subsets ('random', 'consecutive', 'pca')
    n_subsets : int
        Number of different subsets to create
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    List of (subset_matrix, mask, description) tuples
    """
    np.random.seed(random_state)
    feature_count = X.shape[1]
    results = []
    
    if method == 'random':
        for i in range(n_subsets):
            mask = np.zeros(feature_count, dtype=bool)
            selected_indices = np.random.choice(feature_count, size=n_features, replace=False)
            mask[selected_indices] = True
            results.append((X[:, mask], mask, f"Random subset {i+1} ({n_features} features)"))
    
    elif method == 'consecutive':
        for i in range(n_subsets):
            start_idx = (i * feature_count // n_subsets) % (feature_count - n_features)
            end_idx = start_idx + n_features
            mask = np.zeros(feature_count, dtype=bool)
            mask[start_idx:end_idx] = True
            results.append((X[:, mask], mask, f"Consecutive subset {i+1} (features {start_idx}-{end_idx-1})"))
    
    elif method == 'pca':
        # First reduce to n_features using PCA, then back to original space
        pca = PCA(n_components=n_features, random_state=random_state)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
        # Reconstruct and select features with highest explained variance
        X_reconstructed = pca.inverse_transform(X_pca)
        explained_variance = np.var(X_reconstructed, axis=0)
        top_features = np.argsort(-explained_variance)[:n_features]
        mask = np.zeros(feature_count, dtype=bool)
        mask[top_features] = True
        results.append((X[:, mask], mask, f"PCA-based features (top {n_features} by variance)"))
        
    return results

def train_evaluate_model(train_data, test_data, models, verbose=False, feature_count=20):
    """Train and evaluate multiple classification models."""
    # Create label mapping for string class labels
    unique_labels = sorted(set(train_data.labels + test_data.labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    X_train = train_data.embeddings
    y_train = np.array([label_to_idx[label] for label in train_data.labels])
    X_test = test_data.embeddings
    y_test = np.array([label_to_idx[label] for label in test_data.labels])
    
    if verbose:
        print("\nTraining set class distribution:")
        for i, label in enumerate(unique_labels):
            print(f"Class {i} ({label}): {np.sum(y_train == i)} samples")
        print("\nTest set class distribution:")
        for i, label in enumerate(unique_labels):
            print(f"Class {i} ({label}): {np.sum(y_test == i)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    for name, model in models.items():
        if verbose:
            print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
        else:
            # For models without predict_proba, like LinearSVC
            y_pred = model.predict(X_test_scaled)
            # Create a simple probability array based on predictions (dummy)
            y_prob = np.zeros((len(y_test), len(unique_labels)))
            for i, pred in enumerate(y_pred):
                y_prob[i, pred] = 1
        
        # Compute metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics = compute_metrics(y_test, y_pred, y_prob, is_binary=len(unique_labels) == 2)
        
        # Extract feature importance if available
        feature_importance = None
        importance_indices = None
        if hasattr(model, 'coef_'):
            # For LogisticRegression
            feature_importance = np.abs(model.coef_).mean(axis=0)
            importance_indices = np.argsort(feature_importance)[::-1][:feature_count]
        elif hasattr(model, 'feature_importances_'):
            # For RandomForest
            feature_importance = model.feature_importances_
            importance_indices = np.argsort(feature_importance)[::-1][:feature_count]
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'coef_'):
            # For Pipeline with LinearSVC at the end
            feature_importance = np.abs(model.steps[-1][1].coef_).mean(axis=0)
            importance_indices = np.argsort(feature_importance)[::-1][:feature_count]
        
        # Add results
        results[name] = {
            'conf_matrix': conf_matrix,
            'metrics': metrics,
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
            'importance_indices': importance_indices.tolist() if importance_indices is not None else None,
            'model_config': get_model_config(model)
        }
        
        if verbose:
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
    
    # Add split information to results
    results['split_info'] = {
        'train_size': len(y_train),
        'test_size': len(y_test),
        'train_class_distribution': [int(np.sum(y_train == i)) for i in range(len(unique_labels))],
        'test_class_distribution': [int(np.sum(y_test == i)) for i in range(len(unique_labels))],
        'label_mapping': {str(label): int(idx) for label, idx in label_to_idx.items()}
    }
    
    return results, unique_labels

def ablation_study(train_data, test_data, feature_percentages=None, methods=None, random_seed=42, verbose=False):
    """Perform ablation studies by training on subsets of features."""
    if feature_percentages is None:
        feature_percentages = [10, 25, 50, 75]
    
    if methods is None:
        methods = ['random', 'consecutive', 'pca']
    
    X_train = train_data.embeddings
    X_test = test_data.embeddings
    
    total_features = X_train.shape[1]
    ablation_results = {}
    
    # Create label mapping
    unique_labels = sorted(set(train_data.labels + test_data.labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    y_train = np.array([label_to_idx[label] for label in train_data.labels])
    y_test = np.array([label_to_idx[label] for label in test_data.labels])
    
    # Base model for ablation
    base_model = LogisticRegression(
        solver='saga', 
        penalty=None,
        max_iter=1000,
        random_state=random_seed
    )
    
    for method in methods:
        method_results = []
        
        for pct in feature_percentages:
            n_features = int(total_features * pct / 100)
            
            if verbose:
                print(f"\nAblation: {method} method, {pct}% features ({n_features} features)")
            
            subsets = create_feature_subsets(
                X_train, 
                n_features=n_features, 
                method=method,
                n_subsets=1,  # Only one subset per percentage
                random_state=random_seed
            )
            
            # We only created one subset
            X_subset_train, mask, description = subsets[0]
            X_subset_test = X_test[:, mask]
            
            # Scale features
            scaler = StandardScaler()
            X_subset_train_scaled = scaler.fit_transform(X_subset_train)
            X_subset_test_scaled = scaler.transform(X_subset_test)
            
            # Train and evaluate
            model = base_model.fit(X_subset_train_scaled, y_train)
            y_pred = model.predict(X_subset_test_scaled)
            y_prob = model.predict_proba(X_subset_test_scaled)
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_prob, is_binary=len(unique_labels) == 2)
            
            # Store results
            method_results.append({
                'percentage': pct,
                'n_features': n_features,
                'description': description,
                'metrics': metrics,
                'feature_mask': mask.tolist()
            })
            
            if verbose:
                print(f"  F1 Score: {metrics['f1']:.4f}")
        
        ablation_results[method] = method_results
    
    return ablation_results, unique_labels

def plot_feature_importance(importances, indices, feature_count=20, title="Feature Importance", 
                           output_dir=None, filename=None):
    """Plot feature importance."""
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(range(feature_count), importances[indices[:feature_count]])
    plt.xticks(range(feature_count), indices[:feature_count], rotation=90)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    
    if output_dir and filename:
        plt.savefig(os.path.join(output_dir, filename))
    
    plt.close()

def plot_model_comparison(results, output_dir=None, filename=None):
    """Plot model comparison."""
    model_names = list(results.keys())
    model_names = [name for name in model_names if name != 'split_info']
    
    metrics = ['f1', 'precision', 'recall']
    metric_values = {metric: [results[model]['metrics'][metric] for model in model_names] for metric in metrics}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width - width, metric_values[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_dir and filename:
        plt.savefig(os.path.join(output_dir, filename))
    
    plt.close()

def plot_ablation_results(ablation_results, output_dir=None, filename=None):
    """Plot ablation study results."""
    methods = list(ablation_results.keys())
    
    plt.figure(figsize=(12, 8))
    
    for method in methods:
        percentages = [r['percentage'] for r in ablation_results[method]]
        f1_scores = [r['metrics']['f1'] for r in ablation_results[method]]
        plt.plot(percentages, f1_scores, 'o-', label=method)
    
    plt.xlabel('Feature Percentage (%)')
    plt.ylabel('F1 Score')
    plt.title('Ablation Study: Effect of Feature Selection Method and Percentage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if output_dir and filename:
        plt.savefig(os.path.join(output_dir, filename))
    
    plt.close()

def save_results(config, results, ablation_results, class_names, output_dir):
    """Save analysis results to file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
        "ablation_results": ablation_results,
        "class_names": class_names
    }
    
    # Generate hash from results
    results_str = json.dumps(output, sort_keys=True)
    results_hash = hashlib.md5(results_str.encode()).hexdigest()[:8]
    
    # Create filename with timestamp and hash
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_comparison_{timestamp}_{results_hash}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    return filepath

def main():
    # Parse arguments and load config
    args = parse_arguments()
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process datasets using H5DataProcessor
    processor = H5DataProcessor(verbose=args.verbose)
    all_train_data = []
    all_test_data = []
    class_names = set()

    # Process each dataset and split
    for dataset_config in config['datasets']:
        if args.verbose:
            print(f"\nProcessing dataset: {dataset_config['dataset']}")
        
        dataset = processor.process_h5_file(
            processor.get_embedding_path(DatasetConfig(**dataset_config)),
            DatasetConfig(**dataset_config)
        )
        
        # Split the dataset
        train_data, test_data = processor.get_train_test_split(
            dataset, 
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
        
        all_train_data.append(train_data)
        all_test_data.append(test_data)
        class_names.update(dataset.labels)
    
    # Combine datasets
    combined_train = ProcessedDataset(
        embeddings=np.vstack([d.embeddings for d in all_train_data]),
        labels=[l for d in all_train_data for l in d.labels],
        filenames=[f for d in all_train_data for f in d.filenames],
        name="combined",
        num_samples=sum(d.num_samples for d in all_train_data)
    )
    
    combined_test = ProcessedDataset(
        embeddings=np.vstack([d.embeddings for d in all_test_data]),
        labels=[l for d in all_test_data for l in d.labels],
        filenames=[f for d in all_test_data for f in d.filenames],
        name="combined",
        num_samples=sum(d.num_samples for d in all_test_data)
    )
    
    # Get models to compare
    models = get_models()
    
    # Train and evaluate models
    results, unique_labels = train_evaluate_model(
        combined_train, combined_test, models, 
        verbose=args.verbose, feature_count=args.feature_count
    )
    
    # Generate model comparison plot
    plot_model_comparison(
        results, 
        output_dir=output_dir, 
        filename=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    # Generate feature importance plots for each model
    for model_name, model_results in results.items():
        if model_name == 'split_info':
            continue
            
        if model_results['feature_importance'] is not None:
            plot_feature_importance(
                np.array(model_results['feature_importance']),
                np.array(model_results['importance_indices']),
                feature_count=args.feature_count,
                title=f"Feature Importance - {model_name}",
                output_dir=output_dir,
                filename=f"feature_importance_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
    
    # Perform ablation studies if requested
    ablation_results = None
    if args.ablation:
        if args.verbose:
            print("\nPerforming ablation studies...")
            
        ablation_results, _ = ablation_study(
            combined_train, combined_test,
            feature_percentages=[5, 10, 25, 50, 75],
            methods=['random', 'consecutive', 'pca'],
            random_seed=args.random_seed,
            verbose=args.verbose
        )
        
        # Generate ablation plot
        plot_ablation_results(
            ablation_results,
            output_dir=output_dir,
            filename=f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    
    # Save results
    output_path = save_results(
        config, results, ablation_results, sorted(class_names), output_dir
    )
    
    print(f"\nResults have been saved to '{output_path}'")
    print(f"Check {output_dir} for visualization plots")

if __name__ == "__main__":
    main()
"""Module for handling H5 file processing and data loading."""

import h5py
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    dataset: str
    method: str
    segment: Optional[int]
    stride: Optional[int]
    merge_subfolders: bool = True

@dataclass
class ProcessedDataset:
    """Container for processed dataset information."""
    embeddings: np.ndarray
    labels: List[str]
    name: str
    num_samples: int

class H5DataProcessor:
    """Handles loading and processing of H5 files containing embeddings."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def get_embedding_path(self, config: DatasetConfig) -> str:
        """Constructs the path to the H5 file based on config."""
        if config.segment and config.stride:
            sampling = f"s{config.segment}-t{config.stride}"
        else:
            sampling = "raw"
            
        path = os.path.join(config.dataset, sampling)
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")
            
        filename = os.path.join(path, f"{config.method}_embeddings.h5")
        return filename

    def process_h5_file(self, filename: str, config: DatasetConfig) -> ProcessedDataset:
        """Process a single H5 file and return embeddings and labels."""
        embeddings = []
        labels = []
        
        with h5py.File(filename, 'r') as f:
            if 'embeddings' not in f:
                raise ValueError(f"No embeddings group in {filename}")
                
            embeddings_group = f['embeddings']
            if len(embeddings_group.keys()) == 0:
                return ProcessedDataset(
                    embeddings=np.array([]),
                    labels=[],
                    name=config.dataset,
                    num_samples=0
                )
            
            # Check if we have a hierarchical structure
            has_subfolders = any(isinstance(embeddings_group[key], h5py.Group) 
                               for key in embeddings_group.keys())
            
            if has_subfolders and not config.merge_subfolders:
                # Handle hierarchical structure with separate classes for subfolders
                for group_name in embeddings_group.keys():
                    group = embeddings_group[group_name]
                    if isinstance(group, h5py.Group):
                        for name in group.keys():
                            embeddings.append(group[name][()])
                            labels.append(f"{config.dataset}/{group_name}")
                    else:
                        # Handle any files in root level
                        embeddings.append(group[()])
                        labels.append(config.dataset)
            else:
                # Either no subfolders or merging subfolders
                def process_group(group: h5py.Group) -> None:
                    for name in group.keys():
                        item = group[name]
                        if isinstance(item, h5py.Group):
                            process_group(item)
                        else:
                            embeddings.append(item[()])
                            labels.append(config.dataset)
                
                process_group(embeddings_group)
        
        embeddings_array = np.array(embeddings)
        
        if self.verbose:
            print(f"\nProcessed {filename}:")
            print(f"  Total samples: {len(labels)}")
            for label in sorted(set(labels)):
                count = labels.count(label)
                print(f"  {label}: {count} samples")
        
        return ProcessedDataset(
            embeddings=embeddings_array,
            labels=labels,
            name=config.dataset,
            num_samples=len(labels)
        )

    def process_configs(self, configs: List[Dict[str, Any]]) -> List[ProcessedDataset]:
        """Process multiple dataset configs and return their data."""
        processed_datasets = []
        
        for config_dict in configs:
            config = DatasetConfig(**config_dict)
            filename = self.get_embedding_path(config)
            
            if self.verbose:
                print(f"Processing {filename}")
            
            processed = self.process_h5_file(filename, config)
            if processed.num_samples > 0:
                processed_datasets.append(processed)
        
        return processed_datasets

    def combine_datasets(self, datasets: List[ProcessedDataset]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Combine multiple processed datasets into arrays for modeling."""
        all_embeddings = []
        all_labels = []
        unique_labels = []
        
        for dataset in datasets:
            if dataset.num_samples > 0:
                all_embeddings.append(dataset.embeddings)
                all_labels.extend(dataset.labels)
                unique_labels.extend(list(set(dataset.labels)))
        
        # Convert to numpy arrays
        X = np.vstack(all_embeddings)
        
        # Create label mapping
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(unique_labels)))}
        y = np.array([label_to_idx[label] for label in all_labels])
        
        if self.verbose:
            print("\nCombined dataset summary:")
            for label, idx in label_to_idx.items():
                count = sum(y == idx)
                print(f"Class {idx} ({label}): {count} samples")
        
        return X, y, sorted(set(unique_labels))

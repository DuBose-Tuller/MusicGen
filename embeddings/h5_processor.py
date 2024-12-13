"""Module for handling H5 file processing and data loading."""

import h5py
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Set
from datetime import datetime

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    dataset: str
    subfolder: str = "raw"  # Default to raw if no attributes specified
    merge_subfolders: bool = True
    attributes: dict = None  # Optional attributes that affect the subfolder name

    def get_subfolder_path(self) -> str:
        """Generate subfolder path based on attributes."""
        if not self.attributes:
            return self.subfolder
        
        # Build subfolder name from attributes
        parts = []
        if 'segment' in self.attributes:
            parts.append(f"s{self.attributes['segment']}")
        if 'stride' in self.attributes:
            parts.append(f"t{self.attributes['stride']}")
        if 'reversed' in self.attributes and self.attributes['reversed']:
            parts.append("reversed")
        if 'noise' in self.attributes:
            parts.append(f"noise{self.attributes['noise']}")
        
        return "_".join(parts) if parts else "raw"

@dataclass
class ProcessedDataset:
    """Container for processed dataset information."""
    embeddings: np.ndarray
    labels: List[str]
    name: str
    num_samples: int

class H5Manager:
    """Manager for H5 file operations with improved metadata handling."""
    
    def __init__(self, h5_file: str, data_path: str):
        self.h5_file = h5_file
        self.data_path = Path(data_path)
        self._ensure_file_structure()
    
    def _ensure_file_structure(self):
        """Ensures the H5 file has the correct structure."""
        with h5py.File(self.h5_file, 'a') as f:
            # Create main groups if they don't exist
            if 'embeddings' not in f:
                f.create_group('embeddings')
            if 'metadata' not in f:
                metadata = f.create_group('metadata')
                # Create processed files tracking dataset
                if 'processed_files' not in metadata:
                    metadata.create_dataset('processed_files', (0,), dtype=h5py.special_dtype(vlen=str),
                                         maxshape=(None,), chunks=True)
                # Add creation timestamp
                metadata.attrs['created_at'] = datetime.now().isoformat()
    
    def get_processed_files(self) -> Set[str]:
        """Get the set of processed files from the H5 file."""
        with h5py.File(self.h5_file, 'r') as f:
            if 'metadata' in f and 'processed_files' in f['metadata']:
                # Decode bytes to strings when reading from HDF5
                return {p.decode('utf-8') if isinstance(p, bytes) else p 
                    for p in f['metadata']['processed_files'][()]}
        return set()
    
    def _add_processed_file(self, filepath: str):
        """Add a filepath to the processed files list."""
        with h5py.File(self.h5_file, 'a') as f:
            processed_files = f['metadata']['processed_files']
            current_size = len(processed_files)
            processed_files.resize((current_size + 1,))
            processed_files[current_size] = str(filepath)
    
    def _get_or_create_group(self, h5file: h5py.File, group_path: Tuple[str, ...]) -> h5py.Group:
        """Creates nested groups if they don't exist and returns the deepest group."""
        current_group = h5file
        for group_name in group_path:
            if group_name not in current_group:
                current_group = current_group.create_group(group_name)
            else:
                current_group = current_group[group_name]
        return current_group
    
    def append_embedding(self, filepath: str, embedding: np.ndarray):
        """Append a single embedding to the HDF5 file."""
        filepath = Path(filepath)
        try:
            relative_parts = filepath.relative_to(self.data_path).parts
        except ValueError:
            relative_parts = (filepath.name,)
        
        # Handle both cases: with subfolders and without
        if len(relative_parts) > 1:
            group_path = relative_parts[:-1]
            filename = relative_parts[-1]
        else:
            group_path = ()
            filename = relative_parts[0]
        
        # Convert filename to a safe HDF5 dataset name
        safe_name = filename.replace('/', '_').replace('\\', '_')
        
        try:
            with h5py.File(self.h5_file, 'a') as f:
                # Get the proper group
                if group_path:
                    current_group = self._get_or_create_group(f['embeddings'], group_path)
                else:
                    current_group = f['embeddings']
                
                # Save the embedding
                if safe_name in current_group:
                    del current_group[safe_name]
                current_group.create_dataset(safe_name, data=np.array(embedding, dtype=np.float32))
                
                # Update processed files list
                self._add_processed_file(str(filepath))
                
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            raise

class H5DataProcessor:
    """Handles loading and processing of H5 files containing embeddings."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def get_embedding_path(self, config: DatasetConfig) -> str:
        """Constructs the path to the H5 file based on config."""
        subfolder = config.get_subfolder_path()
        path = os.path.join(config.dataset, subfolder)
        
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")
        
        # Default to "last" method if not specified in attributes
        method = config.attributes.get('method', 'last') if config.attributes else 'last'
        filename = os.path.join(path, f"{method}_embeddings.h5")
        return filename
    
    def parse_subfolder(self, subfolder: str) -> dict:
        """Parse subfolder name to extract attributes."""
        attributes = {}
        parts = subfolder.replace('-', '_').split('_')
        
        for part in parts:
            if part.startswith('s') and part[1:].isdigit():
                attributes['segment'] = int(part[1:])
            elif part.startswith('t') and part[1:].isdigit():
                attributes['stride'] = int(part[1:])
            elif part == 'reversed':
                attributes['reversed'] = True
            elif part.startswith('noise'):
                try:
                    attributes['noise'] = float(part[5:])
                except ValueError:
                    pass
        
        return attributes
    
    def get_class_name(self, config: DatasetConfig) -> str:
        """Generate a class name based on dataset config and subfolder."""
        base_name = config.dataset
        
        if config.merge_subfolders:
            return base_name
            
        # Get attributes from both explicit config and subfolder name
        subfolder_attributes = self.parse_subfolder(config.subfolder)
        if self.verbose:
            print(f"\nDebug get_class_name:")
            print(f"  base_name: {base_name}")
            print(f"  subfolder: {config.subfolder}")
            print(f"  parsed attributes: {subfolder_attributes}")
        
        # Build distinguishing features
        distinguishing_features = []
        if subfolder_attributes.get('reversed', False):
            distinguishing_features.append('reversed')
        if 'noise' in subfolder_attributes:
            distinguishing_features.append(f"noise{subfolder_attributes['noise']}")
        
        if distinguishing_features:
            return f"{base_name}_{'_'.join(distinguishing_features)}"
        return base_name

    def process_h5_file(self, filename: str, config: DatasetConfig) -> ProcessedDataset:
        """Process a single H5 file and return embeddings and labels."""
        if self.verbose:
            print(f"\nProcessing file: {filename}")
        
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
            
            class_name = self.get_class_name(config)
            
            def process_group(group: h5py.Group, parent_path: str = "") -> None:
                for name in group.keys():
                    item = group[name]
                    if isinstance(item, h5py.Group):
                        if not config.merge_subfolders:
                            new_path = f"{parent_path}/{name}" if parent_path else name
                            process_group(item, new_path)
                        else:
                            process_group(item)
                    else:
                        embeddings.append(item[()])
                        labels.append(class_name)
            
            process_group(embeddings_group)
        
        embeddings_array = np.array(embeddings)
        
        if self.verbose:
            print(f"Processed {filename}:")
            print(f"  Total samples: {len(labels)}")
            print(f"  {class_name}: {len(labels)} samples")
        
        return ProcessedDataset(
            embeddings=embeddings_array,
            labels=labels,
            name=class_name,
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
        if not datasets:
            raise ValueError("No datasets provided to combine")
        
        all_embeddings = []
        all_labels = []
        unique_labels = []
        
        for dataset in datasets:
            if dataset.num_samples > 0:
                all_embeddings.append(dataset.embeddings)
                all_labels.extend(dataset.labels)
                unique_labels.extend(list(set(dataset.labels)))
        
        if not all_embeddings:
            raise ValueError("No valid embeddings found in datasets")
        
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
"""Module for handling H5 file processing and data loading."""

import h5py
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Set
from datetime import datetime
import re

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    dataset: str
    subfolder: str = "raw"  # Default to raw if no attributes specified
    merge_subfolders: bool = False
    label: Optional[str] = None # override class label
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
    """Container for processed dataset information.
    
    Args:
        embeddings: numpy array of embeddings
        labels: list of class labels
        filenames: list of original audio filenames
        name: name of the dataset
        num_samples: number of samples in the dataset
        metadata: optional dictionary of additional metadata
    """
    embeddings: np.ndarray
    labels: List[str]
    filenames: List[str]
    name: str
    num_samples: int
    metadata: Optional[Dict] = None

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
                    metadata.create_dataset('processed_files', (0,), 
                                         dtype=h5py.special_dtype(vlen=str),
                                         maxshape=(None,), chunks=True)
                if 'filenames' not in metadata:  # New dataset for filenames
                    metadata.create_dataset('filenames', (0,),
                                         dtype=h5py.special_dtype(vlen=str),
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
                # Save embedding
                if group_path:
                    current_group = self._get_or_create_group(f['embeddings'], group_path)
                else:
                    current_group = f['embeddings']
                
                # Save the embedding
                if safe_name in current_group:
                    del current_group[safe_name]
                dset = current_group.create_dataset(safe_name, 
                                                  data=np.array(embedding, dtype=np.float32))
                
                # Store filename in dataset attributes
                dset.attrs['original_filename'] = str(filepath)
                
                # Update metadata
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
    
    def parse_subfolder(self, subfolder: str) -> tuple[str, dict]:
        """Parse subfolder name to extract class name and attributes.
        
        Args:
            subfolder: Full subfolder path (e.g., "Dance/s15-t15" or "s15-t15")
            
        Returns:
            tuple of (class_name, attributes_dict)
        """
        # Split path into parts
        parts = subfolder.split('/')
        
        # Last part contains the technical attributes (segment, stride, etc)
        tech_part = parts[-1]
        # Any earlier parts are considered class information
        class_parts = parts[:-1]
        
        # Parse technical attributes
        attributes = {}
        tech_pieces = tech_part.replace('-', '_').split('_')
        
        for piece in tech_pieces:
            if piece.startswith('s') and piece[1:].isdigit():
                attributes['segment'] = int(piece[1:])
            elif piece.startswith('t') and piece[1:].isdigit():
                attributes['stride'] = int(piece[1:])
            elif piece == 'reversed':
                attributes['reversed'] = True
            elif piece.startswith('noise'):
                try:
                    attributes['noise'] = float(piece[5:])
                except ValueError:
                    pass
        
        # Join any class parts to form class name
        class_name = '_'.join(class_parts) if class_parts else None
        
        return class_name, attributes
    
    def get_class_name(self, config: DatasetConfig) -> str:
        """Generate a class name based on dataset config and subfolder."""
        base_name = config.dataset
        
        # Get class name and attributes from subfolder
        subclass_name, subfolder_attributes = self.parse_subfolder(config.subfolder)
        
        if self.verbose:
            print(f"\nDebug get_class_name:")
            print(f"  base_name: {base_name}")
            print(f"  subfolder: {config.subfolder}")
            print(f"  subclass_name: {subclass_name}")
            print(f"  parsed attributes: {subfolder_attributes}")
        
        # Build final class name
        parts = [base_name]
        
        # Add subclass if present (regardless of merge_subfolders setting)
        if subclass_name:
            parts.append(subclass_name)
        
        # Add distinguishing features
        if subfolder_attributes.get('reversed', False):
            parts.append('reversed')
        if 'noise' in subfolder_attributes:
            parts.append(f"noise{subfolder_attributes['noise']}")
        
        return '_'.join(parts)

    def process_h5_file(self, filename: str, config: DatasetConfig) -> ProcessedDataset:
        """Process a single H5 file and return embeddings and labels."""
        if self.verbose:
            print(f"\nProcessing file: {filename}")
        
        embeddings = []
        labels = []
        filenames = []
        
        # Extract the subfolder from the config to use as part of the label
        subfolder_label = config.subfolder.split('/')[0] if '/' in config.subfolder else config.subfolder
        
        with h5py.File(filename, 'r') as f:
            if 'embeddings' not in f:
                raise ValueError(f"No embeddings group in {filename}")
            
            embeddings_group = f['embeddings']
            if len(embeddings_group.keys()) == 0:
                return ProcessedDataset(
                    embeddings=np.array([]),
                    labels=[],
                    filenames=[],
                    name=config.dataset,
                    num_samples=0
                )
            
            def process_group(group, parent_path: str = "") -> None:
                for name in group.keys():
                    item = group[name]
                    if isinstance(item, h5py.Group):
                        new_path = f"{parent_path}/{name}" if parent_path else name
                        process_group(item, new_path)
                    else:
                        embeddings.append(item[()])
                        # Create label from dataset and subfolder
                        label = f"{config.dataset}/{subfolder_label}"
                        labels.append(label)
                        orig_filename = item.attrs.get('original_filename', name)
                        filenames.append(orig_filename)
            
            process_group(embeddings_group)
            
            if self.verbose:
                print(f"\nProcessed {filename}:")
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                for label, count in label_counts.items():
                    print(f"  {label}: {count} samples")
        
        return ProcessedDataset(
            embeddings=np.array(embeddings),
            labels=labels,
            filenames=filenames,
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
    
    def get_train_test_split(self, dataset: ProcessedDataset, 
                            test_ratio: float = 0.2,
                            random_seed: Optional[int] = None) -> Tuple[ProcessedDataset, ProcessedDataset]:
        """Split dataset into train and test sets, keeping segments together."""
        train_idx, test_idx = AudioSegmentSplitter.get_train_test_split(
            dataset.filenames, test_ratio, random_seed
        )

        AudioSegmentSplitter.validate_split(dataset.filenames, train_idx, test_idx, self.verbose)
        
        if self.verbose:
            print(f"\nSplit summary for {dataset.name}:")
            print(f"Train set: {len(train_idx)} samples")
            print(f"Test set: {len(test_idx)} samples")
        
        train_data = ProcessedDataset(
            embeddings=dataset.embeddings[train_idx],
            labels=[dataset.labels[i] for i in train_idx],
            filenames=[dataset.filenames[i] for i in train_idx],
            name=dataset.name,
            num_samples=len(train_idx)
        )
        
        test_data = ProcessedDataset(
            embeddings=dataset.embeddings[test_idx],
            labels=[dataset.labels[i] for i in test_idx],
            filenames=[dataset.filenames[i] for i in test_idx],
            name=dataset.name,
            num_samples=len(test_idx)
        )
        
        return train_data, test_data
    

class AudioSegmentSplitter:
    """Handles splitting of audio segments ensuring segments from the same source
    file stay in the same split."""
    
    @staticmethod
    def extract_base_id(filename: str) -> Optional[str]:
        """Extract the base identifier from a segmented audio filename.
        
        Args:
            filename: The filename to parse
            
        Returns:
            The extracted base ID or None if no pattern is found
        """
        # Remove file extension and path
        base_name = os.path.basename(filename)
        name_without_ext = base_name.rsplit('.', 1)[0]
        
        # Find the last underscore followed by numbers
        pattern = r'^(.+)_\d+$'
        match = re.match(pattern, name_without_ext)

        return match.group(1) if match else None

    @staticmethod
    def get_train_test_split(filenames: List[str], test_ratio: float = 0.2, 
                            random_seed: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """Split indices ensuring segments from same source stay together."""
        # Extract base IDs
        base_ids = [AudioSegmentSplitter.extract_base_id(f) for f in filenames]
        
        # Get unique base IDs (excluding None)
        unique_ids = sorted(set(id_ for id_ in base_ids if id_ is not None))
        
        if not unique_ids:
            raise ValueError("No valid base IDs found in filenames")
        
        # Random split of unique IDs
        if random_seed is not None:
            np.random.seed(random_seed)
        n_test = max(1, int(len(unique_ids) * test_ratio))  # Ensure at least 1 test sample
        test_ids = set(np.random.choice(unique_ids, n_test, replace=False))
        
        # Create train/test indices based on base IDs
        train_indices = []
        test_indices = []
        
        for idx, base_id in enumerate(base_ids):
            if base_id is None:  # Handle files that don't match pattern
                print(f"Warning: no base id found at index {idx}")
                if random_seed is not None:
                    np.random.seed(random_seed + idx)  # Ensure deterministic but different for each file
                if np.random.random() < test_ratio:
                    test_indices.append(idx)
                else:
                    train_indices.append(idx)
            elif base_id in test_ids:
                test_indices.append(idx)
            else:
                train_indices.append(idx)
        
        if len(test_indices) == 0:
            raise ValueError("No test samples found! Check filename patterns.")
        
        return train_indices, test_indices

    @staticmethod
    def validate_split(filenames: List[str], train_idx: List[int], test_idx: List[int], 
                      verbose: bool = False) -> bool:
        """Validate that the split maintains segment grouping integrity."""
        train_ids = {AudioSegmentSplitter.extract_base_id(filenames[i]) for i in train_idx}
        test_ids = {AudioSegmentSplitter.extract_base_id(filenames[i]) for i in test_idx}
        
        # Remove None values
        train_ids.discard(None)
        test_ids.discard(None)
        
        # Check for overlap
        overlap = train_ids & test_ids
        
        if verbose:
            print(f"\nSplit validation:")
            print(f"  Train IDs: {sorted(train_ids)[:5]}...")
            print(f"  Test IDs: {sorted(test_ids)[:5]}...")
            print(f"  Number of unique train IDs: {len(train_ids)}")
            print(f"  Number of unique test IDs: {len(test_ids)}")
            if overlap:
                print(f"  Warning: Found overlapping IDs: {overlap}")
        
        return len(overlap) == 0
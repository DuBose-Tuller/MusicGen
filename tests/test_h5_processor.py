import pytest
import numpy as np
import h5py
import tempfile
import os
from embeddings.h5_processor import H5DataProcessor, DatasetConfig

class TestH5Processor:
    @pytest.fixture
    def sample_h5_file(self):
        """Create a temporary H5 file with test data."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            with h5py.File(f.name, 'w') as h5f:
                embeddings = h5f.create_group('embeddings')
                # Add some test data
                embeddings.create_dataset('sample1', data=np.random.rand(10))
                embeddings.create_dataset('sample2', data=np.random.rand(10))
            return f.name

    def test_process_h5_file(self, sample_h5_file):
        """Test processing of H5 file."""
        processor = H5DataProcessor()
        config = DatasetConfig(
            dataset="test_dataset",
            method="last",
            segment=None,
            stride=None
        )
        
        result = processor.process_h5_file(sample_h5_file, config)
        
        assert result.num_samples == 2
        assert result.name == "test_dataset"
        assert len(result.labels) == 2
        assert all(label == "test_dataset" for label in result.labels)

    def teardown_method(self, method):
        """Cleanup temporary files after tests."""
        if hasattr(self, 'sample_h5_file'):
            os.unlink(self.sample_h5_file)
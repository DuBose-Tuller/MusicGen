import pytest
import os
import h5py
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from embeddings.get_embeddings import H5Manager, process_file, process_directory

class TestH5Manager:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    @pytest.fixture
    def sample_h5_file(self, temp_dir):
        """Create a sample H5 file with some test data."""
        h5_path = os.path.join(temp_dir, 'test.h5')
        with h5py.File(h5_path, 'w') as f:
            f.attrs['processed_files'] = ['file1.wav', 'file2.wav']
            embeddings = f.create_group('embeddings')
            embeddings.create_dataset('file1', data=np.random.rand(128))
        return h5_path

    @pytest.fixture
    def h5_manager(self, temp_dir, sample_h5_file):
        """Create an H5Manager instance for testing."""
        return H5Manager(sample_h5_file, temp_dir)

    def test_load_processed_files(self, h5_manager):
        """Test loading of processed files from H5 file."""
        processed_files = h5_manager._load_processed_files()
        assert isinstance(processed_files, set)
        assert 'file1.wav' in processed_files
        assert 'file2.wav' in processed_files

    def test_load_processed_files_nonexistent(self, temp_dir):
        """Test loading processed files when H5 file doesn't exist."""
        nonexistent_file = os.path.join(temp_dir, 'nonexistent.h5')
        manager = H5Manager(nonexistent_file, temp_dir)
        processed_files = manager._load_processed_files()
        assert isinstance(processed_files, set)
        assert len(processed_files) == 0

    def test_append_embedding_new_file(self, h5_manager, temp_dir):
        """Test appending a new embedding."""
        test_embedding = np.random.rand(128)
        test_filepath = os.path.join(temp_dir, 'new_file.wav')
        
        h5_manager.append_embedding(test_filepath, test_embedding)
        
        with h5py.File(h5_manager.h5_file, 'r') as f:
            assert 'embeddings/new_file' in f
            np.testing.assert_array_almost_equal(
                f['embeddings/new_file'][()],
                test_embedding
            )
            assert str(test_filepath) in f.attrs['processed_files']

    def test_append_embedding_with_subfolders(self, h5_manager, temp_dir):
        """Test appending embedding with subfolder structure."""
        test_embedding = np.random.rand(128)
        subfolder = os.path.join(temp_dir, 'subfolder')
        os.makedirs(subfolder, exist_ok=True)
        test_filepath = os.path.join(subfolder, 'test.wav')
        
        h5_manager.append_embedding(test_filepath, test_embedding)
        
        with h5py.File(h5_manager.h5_file, 'r') as f:
            assert 'embeddings/subfolder/test' in f
            np.testing.assert_array_almost_equal(
                f['embeddings/subfolder/test'][()],
                test_embedding
            )

    def test_append_embedding_replace_existing(self, h5_manager, temp_dir):
        """Test replacing an existing embedding."""
        test_filepath = os.path.join(temp_dir, 'file1.wav')
        new_embedding = np.random.rand(128)
        
        h5_manager.append_embedding(test_filepath, new_embedding)
        
        with h5py.File(h5_manager.h5_file, 'r') as f:
            np.testing.assert_array_almost_equal(
                f['embeddings/file1'][()],
                new_embedding
            )

class TestProcessing:
    @pytest.fixture
    def mock_model(self):
        """Create a mock MusicGen model."""
        mock = Mock()
        mock.set_generation_params = Mock()
        mock.generate_continuation = Mock(return_value=torch.randn(1, 1, 16000))
        return mock

    @pytest.fixture
    def mock_torchaudio(self):
        """Create mock for torchaudio functions."""
        with patch('torchaudio.load') as mock_load:
            mock_load.return_value = (torch.randn(1, 16000), 16000)
            yield mock_load

    def test_process_file(self, mock_model, mock_torchaudio, temp_dir):
        """Test processing a single audio file."""
        test_file = os.path.join(temp_dir, 'test.wav')
        
        # Create a mock state manager
        with patch('get_embeddings.state_manager') as mock_state_manager:
            mock_state_manager.get_embedding.return_value = torch.randn(128)
            
            # Process the file
            embedding = process_file(test_file, mock_model)
            
            # Verify model interactions
            mock_model.set_generation_params.assert_called_once()
            mock_model.generate_continuation.assert_called_once()
            
            # Verify state manager interactions
            mock_state_manager.clear_embedding.assert_called_once()
            mock_state_manager.set_method.assert_called_once()
            
            # Check output
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape == (128,)  # Assuming 128-dim embeddings

    def test_process_directory(self, mock_model, temp_dir):
        """Test processing a directory of audio files."""
        # Create test directory structure with wav files
        os.makedirs(os.path.join(temp_dir, 'subdir'), exist_ok=True)
        test_files = [
            os.path.join(temp_dir, 'test1.wav'),
            os.path.join(temp_dir, 'test2.wav'),
            os.path.join(temp_dir, 'subdir', 'test3.wav')
        ]
        for file in test_files:
            Path(file).touch()

        # Create H5Manager
        h5_file = os.path.join(temp_dir, 'embeddings.h5')
        manager = H5Manager(h5_file, temp_dir)

        # Mock process_file function
        with patch('get_embeddings.process_file') as mock_process:
            mock_process.return_value = np.random.rand(128)
            
            # Process directory
            process_directory(temp_dir, manager, mock_model, 'last', verbose=True)
            
            # Check number of calls to process_file
            assert mock_process.call_count == len(test_files)

    def test_process_directory_with_errors(self, mock_model, temp_dir):
        """Test directory processing with file processing errors."""
        # Create test files
        test_file = os.path.join(temp_dir, 'test.wav')
        Path(test_file).touch()

        # Create H5Manager
        h5_file = os.path.join(temp_dir, 'embeddings.h5')
        manager = H5Manager(h5_file, temp_dir)

        # Mock process_file to raise an exception
        with patch('get_embeddings.process_file') as mock_process:
            mock_process.side_effect = Exception("Test error")
            
            # Process directory - should not raise exception
            process_directory(temp_dir, manager, mock_model, 'last', verbose=True)
            
            # Verify process_file was called
            mock_process.assert_called_once()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

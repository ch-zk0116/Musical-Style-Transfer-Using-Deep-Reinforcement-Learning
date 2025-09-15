# DataLoading.py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import OrderedDict
import torch.multiprocessing as mp

# Paste your entire Dataset class definition here
class MultiChannelLazyDataset(Dataset):
    """
    A LAZY-LOADING dataset that is efficient with multiple workers.
    It does NOT cache entire files, relying on the DataLoader's pre-fetching.
    """
    def __init__(self, matrix_filepaths, labels, segment_length=128, dataset_name="Unknown"):
        self.matrix_filepaths = matrix_filepaths
        self.labels = labels
        self.segment_length = segment_length
        self.pointers = []

        print(f"Scanning files to create segment pointers for '{dataset_name}'...")
        for i, filepath in enumerate(matrix_filepaths):
            try:
                with np.load(filepath, mmap_mode='r') as data:
                    num_timesteps = data['onset'].shape[0]
                
                for start_step in range(0, num_timesteps - self.segment_length + 1, self.segment_length):
                    self.pointers.append({'file_idx': i, 'start_step': start_step})
            except Exception as e:
                print(f"Cannot process file: {filepath}, Error: {e}")
                pass # Suppress warnings for brevity
        
        print(f"Scan complete. Found {len(self.pointers)} total segments.")

    def __len__(self):
        return len(self.pointers)

    def __getitem__(self, idx):
        pointer = self.pointers[idx]
        file_idx, start_step = pointer['file_idx'], pointer['start_step']
        
        filepath = self.matrix_filepaths[file_idx]
        label = self.labels[file_idx]

        with np.load(filepath) as data:
            segment_channels = [
                data[key][start_step : start_step + self.segment_length, :].T
                for key in ['onset', 'sustain', 'velocity', 'offset']
            ]
        
        multi_channel_segment = np.stack(segment_channels, axis=0).astype(np.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return torch.from_numpy(multi_channel_segment), label_tensor
    

class MultiChannelEagerDataset(Dataset):
    """
    An EAGER-LOADING dataset. It loads the ENTIRE dataset into RAM upon
    initialization. This is fast for training if you have enough memory.
    """
    def __init__(self, matrix_filepaths, labels, segment_length=128, dataset_name="Unknown", print_statement=True):
        self.segment_length = segment_length
        temp_segments_list = []
        self.all_labels = []

        if print_statement:
            print(f"EAGERLY loading all multi-channel segments for '{dataset_name}' into RAM...")
        
        # <<< MODIFICATION START >>>
        # Choose the iterator: use tqdm only if print_statement is True
        if print_statement:
            iterator = tqdm(enumerate(matrix_filepaths), total=len(matrix_filepaths), desc="Loading files")
        else:
            iterator = enumerate(matrix_filepaths)
        # <<< MODIFICATION END >>>

        # Use the chosen iterator (either with or without a progress bar)
        for file_idx, filepath in iterator:
            try:
                with np.load(filepath) as data:
                    onset, sustain, velocity, offset = (data[k] for k in ['onset', 'sustain', 'velocity', 'offset'])
                    file_label = labels[file_idx]

                num_timesteps = onset.shape[0]
                matrices = [onset, sustain, velocity, offset]

                for start_step in range(0, num_timesteps - self.segment_length + 1, self.segment_length):
                    segment_channels = [m[start_step : start_step + self.segment_length, :].T for m in matrices]
                    multi_channel_segment = np.stack(segment_channels, axis=0)
                    
                    temp_segments_list.append(multi_channel_segment)
                    self.all_labels.append(file_label)
            except Exception as e:
                # It's still good to know if a specific file fails
                if print_statement:
                    print(f"Cannot process file: {filepath}, Error: {e}")
                pass

        if not temp_segments_list:
            self.all_segments = np.array([])
        else:
            self.all_segments = np.stack(temp_segments_list, axis=0).astype(np.float32)
            
        self.all_labels = np.array(self.all_labels, dtype=np.float32)

        if print_statement:
            print(f"Eager loading complete. Total segments: {len(self)}. Shape: {self.all_segments.shape}")

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        segment = self.all_segments[idx]
        label = self.all_labels[idx]
        return torch.from_numpy(segment), torch.tensor(label).unsqueeze(0)

class MultiChannelChunkedDataset(Dataset):
    """
    A CHUNKED-LOADING dataset. It loads a large 'chunk' of files into a cache
    to speed up training, then loads the next chunk when needed. This balances
    memory usage and I/O overhead.
    """
    def __init__(self, matrix_filepaths, labels, cache_size=500, segment_length=128, dataset_name="Unknown"):
        self.matrix_filepaths = matrix_filepaths
        self.labels = labels
        self.segment_length = segment_length
        self.cache_size = cache_size
        self.pointers = []
        self._cache = OrderedDict() # An ordered dictionary to act as our file cache

        print(f"Scanning files to create segment pointers for '{dataset_name}'...")
        for i, filepath in enumerate(matrix_filepaths):
            try:
                with np.load(filepath, mmap_mode='r') as data:
                    num_timesteps = data['onset'].shape[0]
                for start_step in range(0, num_timesteps - segment_length + 1, self.segment_length):
                    self.pointers.append({'file_idx': i, 'start_step': start_step})
            except Exception:
                pass
        
        print(f"Scan complete. Found {len(self.pointers)} total segments.")

    def __len__(self):
        return len(self.pointers)

    def _load_file_into_cache(self, file_idx):
        # If the cache is full, remove the oldest item
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False) # Remove the first item inserted
        
        filepath = self.matrix_filepaths[file_idx]
        try:
            with np.load(filepath) as data:
                # Load all 4 matrices for this file into the cache
                self._cache[file_idx] = {
                    key: data[key].astype(np.float32)
                    for key in ['onset', 'sustain', 'velocity', 'offset']
                }
        except Exception as e:
            # If a file is corrupted, put a placeholder to avoid re-loading
            self._cache[file_idx] = None
            print(f"Warning: Could not load file for cache: {filepath}, {e}")

    def __getitem__(self, idx):
        pointer = self.pointers[idx]
        file_idx, start_step = pointer['file_idx'], pointer['start_step']
        label = self.labels[file_idx]

        # --- The Caching Logic ---
        # Check if the required file is in our RAM cache
        if file_idx not in self._cache:
            # If not, load it (and potentially others) into the cache
            self._load_file_into_cache(file_idx)
        
        # If the file was corrupted or couldn't be loaded, return a dummy tensor
        if self._cache[file_idx] is None:
            return torch.zeros((4, self.segment_length, 88)), torch.tensor(-1.0).unsqueeze(0)
            
        # Get the data from the cache (fast RAM access)
        cached_data = self._cache[file_idx]
        
        segment_channels = [
            cached_data[key][start_step : start_step + self.segment_length, :].T
            for key in ['onset', 'sustain', 'velocity', 'offset']
        ]
        
        multi_channel_segment = np.stack(segment_channels, axis=0)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return torch.from_numpy(multi_channel_segment), label_tensor

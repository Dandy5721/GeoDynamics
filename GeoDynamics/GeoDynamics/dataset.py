import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import re
import scipy.io as sio
from scipy.io import loadmat
from utils import (
    sorted_aphanumeric,
    fc2vector,
    sliding_window_corrcoef,
)

class Imdb_dataset:
    def __init__(self, imdb, tokenizer):
        self.imdb = imdb
        self.tokenizer = tokenizer
    
    def get_tokenized_dataset(self, split='Train', eval_ratio=0.1):
        if split in self.imdb:
            return self.imdb[split].map(
                self.preprocess_function,
                batched=True,
                remove_columns=['text']
            )

        elif split in ['eval'] and 'test' in self.imdb:
            test_dataset = self.imdb['test']
            total_samples = len(test_dataset)
            eval_samples = int(eval_ratio*total_samples)
            eval_indices = random.sample(range(total_samples), eval_samples)
            eval_dataset = test_dataset.select(eval_indices)
            return eval_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=['text']
            )
        else:
            print('Can not get data')
            return None

    def preprocess_function(self, examples):
        samples = self.tokenizer(
            examples['text'],
            max_length=128,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        samples.pop('attention_mask')

        return samples

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)  


class Dataset_Skeleton_seqFC(Dataset):
    def __init__(self, mat_file, window_size):
        super(Dataset_Skeleton_seqFC, self).__init__()
        self.mat_file = mat_file
        self.window = window_size
        self.data = []
        self.labels = []
        self.load_data()
        self.pad_data()  # Ensure data is padded after loading

    def load_data(self):
        # Load the .mat file
        mat_data = sio.loadmat(self.mat_file)
        
        # Check and extract data
        if 'data' in mat_data:
            data_array = mat_data['data']
            if isinstance(data_array, np.ndarray) and data_array.ndim == 2:
                # Extract each sample from the cell array
                self.data = [data_array[i, 0] for i in range(data_array.shape[0])]
            else:
                raise ValueError("Expected a 2D array for 'data' but got shape", data_array.shape)
        else:
            raise ValueError("Key 'data' not found in .mat file")

        # Check and extract labels
        if 'action_label' in mat_data:
            self.labels = mat_data['action_label'].flatten()  # Ensure labels are 1D
            self.labels = self.labels - 1  # Convert from 1-9 to 0-8
        else:
            raise ValueError("Key 'action_label' not found in .mat file")

    def pad_data(self):
        # Find the maximum length across all samples
        max_length = max(sample.shape[2] for sample in self.data)
        # print(max_length)
        # Pad each sample to the maximum length
        for i in range(len(self.data)):
            sample = self.data[i]
            if sample.shape[2] < max_length:
                # Compute padding sizes
                pad_size = max_length - sample.shape[2]
                # Print padding size for debugging
                # print(f"Padding size for sample {i}: {pad_size}")
                # Pad each dimension of X, Y, Z
                padding = ((0, 0), (0, 0), (0, pad_size))
                self.data[i] = np.pad(sample, padding, mode='constant', constant_values=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]  # Extract each sample
        
        if isinstance(sample_data, np.ndarray):
            if sample_data.ndim != 3:
                raise ValueError(f"Expected 3D array, but got shape {sample_data.shape}")
            
            data_tensor = self.dataCovSlidingWindow(sample_data, self.window)
        else:
            raise TypeError("Sample data is not in the expected format")
        
        label = self.labels[idx]
        return {'labels': label, 'input_ids': torch.tensor(data_tensor).float()}

    def dataCovSlidingWindow(self, sample_data, window_size):
        # Ensure sample_data is a numpy array
        sample_data = np.array(sample_data)
        
        # Extract X, Y, Z from the sample_data
        X = sample_data[0, :, :]
        Y = sample_data[1, :, :]
        Z = sample_data[2, :, :]

        # Get the number of time points (third dimension)
        num_time_points = X.shape[1]
        
        # Determine the number of sliding windows
        num_windows = num_time_points - window_size + 1
        # print(num_windows)
        # Preallocate a matrix to store all covariance matrices for this sample
        N = X.shape[0] * 3  # Number of joints times 3 (X, Y, Z)
        # print(N)
        cov_matrices = np.zeros((num_windows, N, N))
        
        # Slide the window over the time points
        for t in range(num_windows):
            # Extract the current window for X, Y, Z
            X_window = X[:, t:t+window_size]
            Y_window = Y[:, t:t+window_size]
            Z_window = Z[:, t:t+window_size]
            
            # Compute covariance matrix for this window
            concatenated = np.concatenate([X_window, Y_window, Z_window], axis=0)
            C = np.cov(concatenated.T, rowvar=False)
            
            # Store the covariance matrix in the output matrix
            cov_matrices[t, :, :] = C
        
        return cov_matrices


def pick_labels(tensor):
    arr = np.unique(tensor)
    # print(arr)
    mask = np.ones(len(arr), dtype=bool)
    mask[0] = False  # resting
    result = arr[mask]
    # print(result)
    assert len(result) == 8
    return result

class Dataset_WM(Dataset):

    def __init__(self, data_dir, label_dir, slice=None):

        super(Dataset_WM, self).__init__()
        # tau data
        self.base_path = data_dir
        self.data_path = [os.path.join(self.base_path, name) for name in
                          sorted_aphanumeric(os.listdir(data_dir)) if name.endswith('.txt') or name.endswith('.csv')]
        if slice:
            self.data_path = self.data_path[slice]
        self.data = [np.loadtxt(path, delimiter='\t', dtype=np.float32) for path in self.data_path]

        # label data
        self.label_path = label_dir
        self.labels_path = [os.path.join(self.label_path, name) for name in
                          sorted_aphanumeric(os.listdir(label_dir)) if name.endswith('.txt') or name.endswith('.csv')]
        if slice:
            self.labels_path = self.labels_path[slice]
        self.labels = [np.loadtxt(path, delimiter=',', dtype=np.float32) for path in self.labels_path]
        self.labels = self.labels * len(self.data)
        

        self.temp_label = []
        self.temp_clip = []

        for sample, label in zip(self.data, self.labels):
            label_pos = self.pick_labels(label)
            batch_data = sample.squeeze()
            for ll in label_pos:
                self.temp_label.append(int(ll-1))
                self.temp_clip.append(batch_data[label == ll])

        self.data = self.temp_clip
        self.labels = self.temp_label

        self.n_data = len(self.data)
        print(self.n_data)
        print(data_dir)
        print(f'{self.data_path[0]},...,{self.data_path[-1]}')

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):

        input_ids = self.data[idx]
        labels = self.labels[idx]
        output_dict = {'labels': labels, 'input_ids': input_ids}
        return output_dict

    def pick_labels(self, tensor):
        arr = np.unique(tensor)
        # print(arr)
        mask = np.ones(len(arr), dtype=bool)
        mask[0] = False  # resting
        result = arr[mask]
        # print(result)
        assert len(result) == 8
        return result

class SeqFCDataset_WM(Dataset):

    def __init__(self, data_dir, label_dir, slice=None, transpose=False):

        super(SeqFCDataset_WM, self).__init__()
        # tau data
        self.transpose = transpose
        self.base_path = data_dir
        self.data_path = [os.path.join(self.base_path, name) for name in
                          sorted_aphanumeric(os.listdir(data_dir)) if name.endswith('.txt') or name.endswith('.csv')]
        if slice:
            self.data_path = self.data_path[slice]
        self.data = [np.loadtxt(path, delimiter='\t', dtype=np.float32) for path in self.data_path]

        # label data
        self.label_path = label_dir
        self.labels_path = [os.path.join(self.label_path, name) for name in
                          sorted_aphanumeric(os.listdir(label_dir)) if name.endswith('.txt') or name.endswith('.csv')]
        if slice:
            self.labels_path = self.labels_path[slice]
        self.labels = [np.loadtxt(path, delimiter=',', dtype=np.float32) for path in self.labels_path]
        self.labels = self.labels * len(self.data)
        

        self.temp_label = []
        self.temp_clip = []

        for sample, label in zip(self.data, self.labels):
            label_pos = self.pick_labels(label)
            batch_data = sample.squeeze()
            for ll in label_pos:
                self.temp_label.append(int(ll-1))
                self.temp_clip.append(batch_data[label == ll])

        self.data = self.temp_clip
        self.labels = self.temp_label

        self.n_data = len(self.data)
        self.window = [15] * self.n_data
        print(self.n_data)
        print(data_dir)
        print(f'{self.data_path[0]},...,{self.data_path[-1]}')

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.window = np.concatenate((self.window, other.window), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):

        input_ids = self.data[idx]
        # data = torch.corrcoef(torch.tensor(input_ids).squeeze().T)  #fc
        data = sliding_window_corrcoef(
            torch.tensor(input_ids).squeeze().T,
            self.window[idx],
        )  #dfc
        data = torch.from_numpy(data)
        # print(data.shape)
        labels = self.labels[idx]
        # print(labels)
        # output_dict = {'labels': labels, 'input_ids': input_ids} #vec
        output_dict = {'labels': labels, 'input_ids': data} #manifold
        return output_dict

    def pick_labels(self, tensor):
        arr = np.unique(tensor)
        # print(arr)
        mask = np.ones(len(arr), dtype=bool)
        mask[0] = False  # resting
        result = arr[mask]
        # print(result)
        assert len(result) == 8
        return result

class Dataset_ADNI_seqFC(Dataset):
    def __init__(self, data_dir, label_file):
        super(Dataset_ADNI_seqFC, self).__init__()
        self.data_dir = data_dir
        self.label_file = label_file
        self.data_files = []
        self.labels = []
        self.load_data()

    def load_data(self):

        labels_df = pd.read_csv(self.label_file, header=0)

        for filename in os.listdir(self.data_dir):
            if filename.startswith('sub_'):
                id = filename.split('_')[1]

                label_row = labels_df[labels_df['subject_id'] == id]
                if not label_row.empty:
                    label = label_row.iloc[0]['DX']
                    if label in ['CN', 'SMC', 'EMCI']:
                        self.labels.append(0)
                    elif label in ['LMCI', 'AD']:
                         self.labels.append(1)
                    # elif label in ['LMCI']:
                    #     self.labels.append(2)
                    # elif label in ['AD']:
                    #     self.labels.append(3)
                    else:
                        print('Label Error')
                        self.labels.append(-1)
                    self.data_files.append(os.path.join(self.data_dir, filename))
                    self.n_data = len(self.data_files)
                    self.window = [15]*self.n_data

    def __len__(self):
        return len(self.data_files)
    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):
        filename = self.data_files[idx]
        data = np.loadtxt(filename)
        data = torch.tensor(data)[:140].float()
        # print(data.shape)
        # data = torch.corrcoef(torch.tensor(data).squeeze().T)  #fc
        data = sliding_window_corrcoef(
            torch.tensor(data).squeeze().T,
            self.window[idx],
        )  #dfc
        # print(data.shape)
        data = torch.from_numpy(data)
        data = torch.nan_to_num(data)
        label = self.labels[idx]
        output_dict = {'labels': label, 'input_ids': data.float()}
        return output_dict
    

class SeqFCDataset_ADNI_seq(Dataset):

    def __init__(self, data_dir, label_dir):
        super(SeqFCDataset_ADNI_seq, self).__init__()
        self.base_path = data_dir
        self.data_path = [os.path.join(self.base_path, name) for name in
                          (os.listdir(data_dir)) if name.endswith('.txt') or name.endswith('.csv')]

        self.data = [np.loadtxt(path, delimiter=' ', dtype=np.float32) for path in self.data_path]

        # label data
        # self.label_path = label_dir
        self.labels = np.loadtxt(label_dir, delimiter=',', dtype=np.float32)
        self.labels[self.labels > 0] -= 1

        self.data, self.labels = self.filter_data(self.data, self.labels)

        self.n_data = len(self.data)
        print(self.n_data)
        print(data_dir)
        print(f'{self.data_path[0]},...,{self.data_path[-1]}')

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):
        input_ids = self.data[idx][:140]
        labels = self.labels[idx]
        output_dict = {'labels': labels, 'input_ids': input_ids}
        return output_dict
    
    def filter_data(self, data, labels):
        filtered_data = []
        filtered_labels = []
        for d, label in zip(data, labels):
            if d.shape != (976, 116):
                filtered_data.append(d)
                filtered_labels.append(torch.tensor(label).long())
        return filtered_data, filtered_labels
    

from scipy.io import loadmat  

class Dataset_PPMI_seq(Dataset):
    def __init__(self, root_dir):
        super(Dataset_PPMI_seq, self).__init__()
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()
        # self.pad_sentences()

    def load_data(self):
        sentence_sizes = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if 'AAL116_features_timeseries' in file:
                    file_path = os.path.join(subdir, file)
                    data = loadmat(file_path)
                    features = data['data']  
                    sentence_sizes.append(features.shape[0]) 
                    label = self.get_label(subdir)
                    if features.shape[0] > 137:
                        self.data.append(torch.tensor(features[:137]))
                    else:
                        self.data.append(torch.cat((torch.tensor(features), torch.zeros(137 - features.shape[0], features.shape[1])), dim=0))
                    self.labels.append(label)
                     
        self.max_sentence_size = max(sentence_sizes)

    def get_label(self, subdir):
        if 'control' in subdir:
            return 0
        elif 'patient' in subdir:
            return 1
        elif 'prodromal' in subdir:
            return 1
        elif 'swedd' in subdir:
            return 0
        else:
            print("Label error")
            return -1 
        
    def pad_sentences(self):
        self.data = [torch.cat((torch.tensor(sentence), torch.zeros(self.max_sentence_size - sentence.shape[0], sentence.shape[1])), dim=0) for sentence in self.data]        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx].float()
        label = self.labels[idx]
        # if features.shape[0] != self.max_sentence_size:
        #     print("Shape error")

        label = self.labels[idx]
        # data = (features - torch.mean(features, axis=0, keepdims=True)) / torch.std(features, axis=0, keepdims=True)
        output_dict = {'labels': label, 'input_ids': features}

        return output_dict

class Dataset_PPMI_seqFC(Dataset):
    def __init__(self, root_dir):
        super(Dataset_PPMI_seqFC, self).__init__()
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()
        # self.pad_sentences()

    def load_data(self):
        sentence_sizes = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if 'AAL116_features_timeseries' in file:
                    file_path = os.path.join(subdir, file)
                    data = loadmat(file_path)
                    features = data['data']  
                    sentence_sizes.append(features.shape[0]) 
                    label = self.get_label(subdir)
                    if features.shape[0] > 137:
                        self.data.append(torch.tensor(features[:137]))
                    else:
                        self.data.append(torch.cat((torch.tensor(features), torch.zeros(137 - features.shape[0], features.shape[1])), dim=0))
                    self.labels.append(label)
                    self.n_data = len(self.data)
                    self.window = [15]*self.n_data
        self.max_sentence_size = max(sentence_sizes)

    def get_label(self, subdir):
        if 'control' in subdir:
            return 0
        elif 'patient' in subdir:
            return 1
        elif 'prodromal' in subdir:
            return 2
        elif 'swedd' in subdir:
            return 3
        else:
            print("Label error")
            return -1 
        
    def pad_sentences(self):
        self.data = [torch.cat((torch.tensor(sentence), torch.zeros(self.max_sentence_size - sentence.shape[0], sentence.shape[1])), dim=0) for sentence in self.data]        
        
    def __len__(self):
        return len(self.data)
    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):
        features = self.data[idx].float()
        features = sliding_window_corrcoef(
            torch.tensor(features).squeeze().T,
            self.window[idx],
        )  #dfc
        # print(features.shape)
        features = torch.from_numpy(features)
        features = torch.nan_to_num(features)
        label = self.labels[idx]
        # if features.shape[0] != self.max_sentence_size:
        #     print("Shape error")

        label = self.labels[idx]
        # data = (features - torch.mean(features, axis=0, keepdims=True)) / torch.std(features, axis=0, keepdims=True)
        output_dict = {'labels': label, 'input_ids': features}

        return output_dict


class Dataset_OASIS_seqFC(Dataset):
    def __init__(self, root_dir):
        super(Dataset_OASIS_seqFC, self).__init__()
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        sentence_sizes = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.root_dir, filename)
                data = self.load_txt(filepath)
                # if (data.size(0) >= 100) and (data.size(0) <=1000): 
                label = self.get_label(filename)
                if data.size(0) < 328:
                    data = torch.cat((torch.tensor(data), torch.zeros(328 - data.shape[0], data.shape[1])), dim=0)
                else:
                    data = self.resize_matrices(data)
                self.data.append(data)
                self.n_data = len(self.data)
                self.window = [15]*self.n_data
                sentence_sizes.append(data.shape[0]) 
                self.labels.append(label)
        self.max_sentence_size = max(sentence_sizes)

    def load_txt(self, filepath):
        with open(filepath, 'r') as file:
            data = [[float(num) for num in line.split()] for line in file.readlines()]
        return torch.tensor(data)

    def get_label(self, filename):
        if 'CN' in filename:
            return 0
        elif 'AD' in filename:
            return 1
        else:
            print("Label error")
            return -1
    
    def resize_matrices(self, matrix):
        n, dim = matrix.shape
        start_idx = (n - 328) // 2
        end_idx = start_idx + 328
        resized_matrix = matrix[start_idx:end_idx, :]
        return resized_matrix       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = sliding_window_corrcoef(
            torch.tensor(data).squeeze().T,
            self.window[idx],
        )  #dfc
        # print(data.shape)
        data = torch.from_numpy(data)
        data = torch.nan_to_num(data)
        label = self.labels[idx]
        output_dict = {'labels': label, 'input_ids': data}

        return output_dict

class Dataset_OASIS_seq(Dataset):
    def __init__(self, root_dir):
        super(Dataset_OASIS_seq, self).__init__()
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        sentence_sizes = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.root_dir, filename)
                data = self.load_txt(filepath)
                # if (data.size(0) >= 100) and (data.size(0) <=1000): 
                label = self.get_label(filename)
                if data.size(0) < 328:
                    data = torch.cat((torch.tensor(data), torch.zeros(328 - data.shape[0], data.shape[1])), dim=0)
                else:
                    data = self.resize_matrices(data)
                self.data.append(data)
                sentence_sizes.append(data.shape[0]) 
                self.labels.append(label)
        self.max_sentence_size = max(sentence_sizes)

    def load_txt(self, filepath):
        with open(filepath, 'r') as file:
            data = [[float(num) for num in line.split()] for line in file.readlines()]
        return torch.tensor(data)

    def get_label(self, filename):
        if 'CN' in filename:
            return 0
        elif 'AD' in filename:
            return 1
        else:
            print("Label error")
            return -1
    
    def resize_matrices(self, matrix):
        n, dim = matrix.shape
        start_idx = (n - 328) // 2
        end_idx = start_idx + 328
        resized_matrix = matrix[start_idx:end_idx, :]
        return resized_matrix       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.nan_to_num(data)
        label = self.labels[idx]
        output_dict = {'labels': label, 'input_ids': data}

        return output_dict
    
import pandas as pd

# class Dataset_ADNI_seq(Dataset):
#     def __init__(self, data_dir, label_file):
#         super(Dataset_ADNI_seq, self).__init__()
#         self.data_dir = data_dir
#         self.label_file = label_file
#         self.data = []
#         self.labels = []
#         self.load_data()
#         self.pad_sentences()

#     def load_data(self):
#         sentence_sizes = []

#         labels_df = pd.read_csv(self.label_file, header=0)

#         for filename in os.listdir(self.data_dir):
#             if filename.startswith('sub_'):
#                 id = filename.split('_')[1]

#                 label_row = labels_df[labels_df['subject_id'] == id]
#                 if not label_row.empty:
#                     label = label_row.iloc[0]['DX']
#                     if label in ['CN', 'SMC']:
#                         self.labels.append(0)
#                     elif label in ['EMCI']:
#                         self.labels.append(1)
#                     elif label in ['LMCI']:
#                         self.labels.append(2)
#                     elif label in ['AD']:
#                         self.labels.append(3)
#                     else:
#                         print('Label Error')
#                         self.labels.append(-1)
#                     features = np.loadtxt(os.path.join(self.data_dir, filename))
#                     self.data.append(features)
#                     sentence_sizes.append(features.shape[0])
#         self.max_sentence_size = max(sentence_sizes)

#     def pad_sentences(self):
#         self.data = [torch.cat((torch.tensor(sentence), torch.zeros(self.max_sentence_size - sentence.shape[0], sentence.shape[1])), dim=0) for sentence in self.data]        

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data = self.data[idx] 
#         label = self.labels[idx]
#         output_dict = {'labels': label, 'input_ids': data.float()}
#         return output_dict

class Dataset_ADNI_seq(Dataset):
    def __init__(self, data_dir, label_file):
        super(Dataset_ADNI_seq, self).__init__()
        self.data_dir = data_dir
        self.label_file = label_file
        self.data_files = []
        self.labels = []
        self.load_data()

    def load_data(self):

        labels_df = pd.read_csv(self.label_file, header=0)

        for filename in os.listdir(self.data_dir):
            if filename.startswith('sub_'):
                id = filename.split('_')[1]

                label_row = labels_df[labels_df['subject_id'] == id]
                if not label_row.empty:
                    label = label_row.iloc[0]['DX']
                    if label in ['CN', 'SMC', 'EMCI']:
                        self.labels.append(0)
                    elif label in ['LMCI', 'AD']:
                         self.labels.append(1)
                    # elif label in ['LMCI']:
                    #     self.labels.append(2)
                    # elif label in ['AD']:
                    #     self.labels.append(3)
                    else:
                        print('Label Error')
                        self.labels.append(-1)
                    self.data_files.append(os.path.join(self.data_dir, filename))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        filename = self.data_files[idx]
        data = np.loadtxt(filename)
        data = torch.tensor(data)[:140].float()
        # print(data.shape)
        data = torch.corrcoef(torch.tensor(data).squeeze().T)  #fc
        # print(data.shape)
        data = torch.nan_to_num(data)
        label = self.labels[idx]
        output_dict = {'labels': label, 'input_ids': data.float()}
        return output_dict
    

    

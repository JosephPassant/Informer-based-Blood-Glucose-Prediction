import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import json


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)


    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)



class AverageMeter(object):

    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class BloodGlucoseDataset(Dataset):
    """
    Dataset for loading dynamically named blood glucose prediction slices.
    """

    def __init__(self, encoder_dir, decoder_dir, target_dir, shuffle=True):
        """
        Args:
            encoder_dir (str): Path to the directory containing encoder input files.
            decoder_dir (str): Path to the directory containing decoder input files.
            target_dir (str): Path to the directory containing target files.
        """
        self.encoder_dir = encoder_dir
        self.decoder_dir = decoder_dir
        self.target_dir = target_dir
        self.shuffle = shuffle

        # Validate directories
        self._validate_directory(self.encoder_dir)
        self._validate_directory(self.decoder_dir)
        self._validate_directory(self.target_dir)

        # Create a sorted list of unique sample IDs from encoder filenames
        self.sample_ids = list(self._extract_sample_ids(self.encoder_dir))

        if self.shuffle:
            random.shuffle(self.sample_ids)

    @staticmethod
    def _validate_directory(directory):
        """Ensure directory exists and is readable."""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"'{directory}' is not a directory.")
        if not os.access(directory, os.R_OK):
            raise PermissionError(f"Directory '{directory}' is not readable.")

    @staticmethod
    def _extract_sample_ids(directory):
        sample_ids = set()
        for filename in os.listdir(directory):
            if filename.endswith(".pt"):
                parts = filename.split('.')
                sample_id = f"{parts[0]}"  # Extract only PatientID_SliceNumber
                sample_ids.add(sample_id)
        return sorted(sample_ids)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (encoder_input, decoder_input, target)
        """
        sample_id = self.sample_ids[idx]

        encoder_path = os.path.join(self.encoder_dir, f"{sample_id}.pt")
        decoder_path = os.path.join(self.decoder_dir, f"{sample_id}.pt")
        target_path = os.path.join(self.target_dir, f"{sample_id}.pt")

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file missing: {encoder_path}")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"Decoder file missing: {decoder_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file missing: {target_path}")

        # Load tensors
        encoder_input = torch.load(encoder_path).float()
        decoder_input = torch.load(decoder_path).float()
        target = torch.load(target_path).float().squeeze(-1)

        # No need to pass time features separately, already part of encoder_input & decoder_input
        return encoder_input, decoder_input, target
    
       
def compute_batch_derivatives(batch_values, interval=5):
    """
    Compute derivatives for a batch of time series data.
    
    Args:
        batch_values: numpy array of shape (batch_size, seq_len, features)
        interval: time interval between consecutive measurements in minutes (default: 5)
        
    Returns:
        numpy array of derivatives with same shape as input
    """
    batch_size, seq_len = batch_values.shape[0], batch_values.shape[1]
    
    # Initialize output array with same shape as input
    derivatives = np.zeros_like(batch_values)
    
    # Calculate derivatives for each sequence in the batch
    for b in range(batch_size):
        # First derivative is zero (no previous value)
        # For remaining points, calculate rate of change
        derivatives[b, 1:] = np.diff(batch_values[b], axis=0) / interval
    
    return derivatives

def load_config(config_path=None):
    # print(f"Attempting to load config from: {config_path}")  # Debugging print statement
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as file:
        return json.load(file)

class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def create_dir(directory):
    """
    Safely creates a directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print(f"Directory {directory} already exists.")
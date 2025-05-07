import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET


def separate_ptid_data(df):
    """
    Separates the data into individual patient time series.
    
    Args:
    - df: pandas DataFrame containing all patient data
    
    Returns:
    - data_dict: Dictionary containing DataFrames for each patient, with PtID as the key
    """
    data_dict = {} # Initialize dictionary to store DataFrames

    for group in df.groupby('PtID'): # Iterate through each patient group
        PtID = group[0] # Extract the PtID
        data_dict[PtID] = group[1].reset_index(drop=True) # Store the DataFrame in the dictionary with PtID as the key

    return data_dict

def align_start_date(data_dict, base_date="2000-01-01"):
    """
    Aligns the start date of the data for each patient
    
    Args:
    data_dict: dictionary containing dataframes for each patient
    base_date: date to align the data to
    
    Returns:
    dictionary containing dataframes for each patient with aligned start date
    """
    
    updated_data_dict = {}
    
    for ptid, df in data_dict.items():
        # Calculate the difference in days between the base date and the start date
        start_date = df['DateTime'].min()
        days_diff = (start_date - pd.to_datetime(base_date)).days
        
        # Update the DateTime column by subtracting the difference in days
        df['DateTime'] = df['DateTime'] - pd.to_timedelta(days_diff, unit='D')
        
        # Store the updated DataFrame in the dictionary
        updated_data_dict[ptid] = df
    
    return updated_data_dict

def undersample_dict(original_dict, sample_size):
    """Returns a new dictionary with `sample_size` random samples from `original_dict`."""
    if len(original_dict) <= sample_size:  # No need to sample if already within limits
        return original_dict  
    sampled_keys = np.random.choice(list(original_dict.keys()), sample_size, replace=False)
    return {k: original_dict[k] for k in sampled_keys}


def get_first_file(directory, file_extension=".pt"):
    files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
    if files:
        return os.path.join(directory, files[0])
    return None


OHIO_DATA_DIR = os.path.join('..', 'source_data', 'SourceData', 'Ohio')

def get_ohio_data(dataset_type, data_dir_type, Ohio_data_dir = OHIO_DATA_DIR):

    """
    Function to parse the OhioT1DM dataset and return a dictionary of DataFrames for each patient.

    Parameters:
    data_type (str): Type of data to parse (e.g. 'training', 'validation', 'test')
    data_dir (str): Directory containing the XML files, to be appended to the RAW_DATA_DIR

    Returns:
    data_dict (dict): Dictionary containing DataFrames for each patient, with PtID as the key
    """

    # Initialize dictionary to store DataFrames
    data_dict = {}

    # Directory containing the XML files
    dir_path = os.path.join(OHIO_DATA_DIR, data_dir_type)

    # Iterate through each file in the directory
    for filename in os.listdir(dir_path):
        if not filename.endswith('.xml'):
            continue
        
        # Full path to the file
        full_path = os.path.join(dir_path, filename)
        
        # Parse the XML file
        tree = ET.parse(full_path)
        root = tree.getroot()
        data = []

        # Extract Patient ID (as an integer)
        PtID = int(root.attrib['id'])

        # Extract glucose level events including timestamp and value
        for event in root.find('glucose_level').findall('event'):
            row = {'timestamp': event.attrib['ts'], 'value': event.attrib['value']}
            data.append(row)

        # Create a DataFrame for the patient
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
        df = df.sort_values(by='timestamp', ascending=True)

        # Store the DataFrame in the dictionary with PtID as the key
        data_dict[PtID] = df

    # Label the dictionary based on the data type
    dynamic_label = f"ohio_{dataset_type}_data"
    globals()[dynamic_label] = data_dict  # Assign to global variable dynamically

    return  data_dict  # Assign to global variable dynamically
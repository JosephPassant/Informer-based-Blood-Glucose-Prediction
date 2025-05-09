
# set up command line arguments
import argparse
import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import json
import xml.etree.ElementTree as ET  # Added missing import
from pathlib import Path
from IPython.display import display  # For the display() function



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, "models", "jpformer"))

from jpformer import JPFormer

sys.path.append(os.path.join(PROJECT_ROOT, "shared_utilities"))
from masking import *

# ptid command line arguments
parser = argparse.ArgumentParser(description='JPFormer Inference')
parser.add_argument('--ptid', type=str, default='JPFormer', help='patient id number')
parser.add_argument('--optimised_for', type=str, default='hypo', help="optimised for 'hypo' EP% or 'overall' EP%")

args = parser.parse_args()



def get_full_ohio_ptid_data(ptid):

    """
    Read and Process Ohio Patient Data
    Args:
        ptid (str): Patient ID
    Returns:
        pd.DataFrame: DataFrame containing glucose level events with timestamps and values (sorted by timestamp)
    
    """
    
    ptid_file = os.path.join(PROJECT_ROOT, 'data', 'source_data', 'SourceData', 'Ohio', 'Test', f"{ptid}-ws-testing.xml")
        
    # Parse the XML file
    tree = ET.parse(ptid_file)
    root = tree.getroot()
    data = []

    # Extract Patient ID (as an integer)
    file_ptid = int(root.attrib['id'])

    assert file_ptid == ptid, f"Patient ID in file {file_ptid} does not match provided PtID {ptid}"

    # Extract glucose level events including timestamp and value
    for event in root.find('glucose_level').findall('event'):
        row = {'timestamp': event.attrib['ts'], 'glucose_value': event.attrib['value']}
        data.append(row)

    # Create a DataFrame for the patient
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.sort_values(by='timestamp', ascending=True)

    df['real_value_flag'] = 1
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

    mask = (df['time_diff'] > 595) & (df['time_diff'] < 605)
    insert_rows = df[mask].copy()  # Added missing definition

    if not insert_rows.empty:
        # Modify new rows: set `real_value_flag = 0`, shift `DateTime`, and set `GlucoseValue = NaN`
        insert_rows['real_value_flag'] = 0
        insert_rows['timestamp'] -= pd.to_timedelta(5, unit='m')
        insert_rows['glucose_value'] = np.nan

    df = pd.concat([df, insert_rows], ignore_index=True).reset_index(drop=True)


    df['glucose_value'] = df['glucose_value'].astype(float)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    df['time_diff_flag'] = df['time_diff'].apply(lambda x: 0 if x < 295 or x > 305 else 1)
    df['RollingTimeDiffFlag'] = df['time_diff_flag'].rolling(window=72).sum()

    df = df.drop(columns=['time_diff', 'time_diff_flag', 'real_value_flag'])


    #bg_value z-score normalisation

    df['glucose_value'] = (df['glucose_value'] - 152.91051040286524) / 70.27050122812615

    return  df  # Assign to global variable dynamically


def ohio_data_slicing(df, start_index):
    """
    Create sliding windows of data for glucose prediction.

    extracts 6 hour window of glucose data, verifies its quality and prepares encoder and decoder inputs for the model.
    Args:

    """

    input_slice = df[start_index:start_index + 72].reset_index(drop=True)
    

    if input_slice['RollingTimeDiffFlag'].iloc[-1] != 72:
        print("Unable to predict accurate BG values as input data contains consecutive missing values")
        return None, None, None
    

    input_slice = input_slice.drop(columns=['RollingTimeDiffFlag'])

    encoder_input = input_slice.iloc[:72]
    if 'timestamp' in encoder_input.columns:
        encoder_input = encoder_input.drop(columns=['timestamp'])

    # print("DEBUGGING:")

    # print(encoder_input.tail())
    # print("\n"*2)


    start_token = input_slice.iloc[-12:]
    last_timestamp = start_token['timestamp'].iloc[-1]
    start_token = start_token.drop(columns=['timestamp'])

    decoder_time_sequence = pd.DataFrame({
        'glucose_value': [0] * 24,
        # increment timestamps by 5 minutes starting from last_timestamp + 5 minutes
        'timestamp': pd.date_range(start=last_timestamp + pd.Timedelta(minutes=5), periods=24, freq='5min'),
        })
    
    decoder_time_sequence['hour'] = decoder_time_sequence['timestamp'].dt.hour
    decoder_time_sequence['minute'] = decoder_time_sequence['timestamp'].dt.minute
    decoder_time_sequence = decoder_time_sequence.drop(columns=['timestamp'])

    decoder_input = pd.concat([start_token, decoder_time_sequence], ignore_index=True)

    # convert to tensors
    encoder_input = torch.tensor(encoder_input.values, dtype=torch.float32)
    decoder_input = torch.tensor(decoder_input.values, dtype=torch.float32)

    return encoder_input, decoder_input, last_timestamp

"""
=================================================
CONFIGURATION LOADING FUNCTIONS
=================================================
"""

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as file:
        return json.load(file)

class ConfigObject:
    """
    Convert a dictionary to an object with attributes.
    """
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


"""
=================================================
MODEL AND ENVIRONMENT SETUP
=================================================
"""

def setup_device(config):
    """
    Set up and return the appropriate computation device based on availability.
    
    Args:
        config: Configuration object with device preferences
        
    Returns:
        torch.device: The selected computation device
    """
    if torch.cuda.is_available() and config.use_gpu:
        device = torch.device(f"cuda:{config.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (M1/M2)
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def load_model(config, device, pretrained_weights_path=None):
    """
    Initialize and load a model with optional pretrained weights.
    
    Args:
        config: Configuration object with model parameters
        device: Computation device
        pretrained_weights_path: Optional path to pretrained model weights
        
    Returns:
        model: Initialized model
    """
    model = JPFormer(
        enc_in=config.enc_in,
        dec_in=config.dec_in,
        c_out=config.c_out,
        seq_len=config.seq_len,
        label_len=config.label_len,
        out_len=config.pred_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_layers=config.d_layers,
        d_ff=config.d_ff,
        factor=config.factor,
        dropout=config.dropout,
        embed=config.embed,
        activation=config.activation,
        output_attention=config.output_attention,
        mix=config.mix,
        device=device
    ).float().to(device)

    # Debugging: Check model name
    model_name = model.__class__.__name__
    print(f"Initialized model: {model_name}")

    # Load pre-trained weights if provided
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        try:
            model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
            print(f"Successfully loaded pretrained weights from: {pretrained_weights_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            print("Continuing with random initialization...")
    else:
        print(f"Pretrained weights file not found or not provided. Using random initialization.")

    return model, model_name



def inference_loop(encoder_input, decoder_input, last_timestamp, model, device):

    # Perform inference
    with torch.no_grad():
        encoder_input = encoder_input.unsqueeze(0).to(device)
        decoder_input = decoder_input.unsqueeze(0).to(device)
        output = model(encoder_input, decoder_input)

    # Process the output (e.g., denormalization, warnings)
    # convert output to df
    output_df = pd.DataFrame(output.cpu().numpy().squeeze(), columns=['glucose_value'])
    # add timestamp to output_df based on last_timestamp + 5 minutes and 24 increments of 5 minutes
    output_df['timestamp'] = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=5), periods=24, freq='5min')


    # denormalised output

    output_df['glucose_value'] = (output_df['glucose_value'] * 70.27050122812615) + 152.91051040286524

    #  fine first index below 70mg/dl
    hypo_index = output_df[output_df['glucose_value'] < 70].index

    if hypo_index.empty:
        hypo_time = None
    else:
        hypo_time = output_df['timestamp'].iloc[hypo_index[0]]

    hyper_index = output_df[output_df['glucose_value'] > 180].index
    if hyper_index.empty:
        hyper_time = None
    else:
        hyper_time = output_df['timestamp'].iloc[hyper_index[0]]

    # warnings
    return hypo_time, hyper_time, output_df





def main ():

    ptid = int(args.ptid)

    population_model_dir = os.path.join(PROJECT_ROOT, 'models/jpformer/population_jpformer_final_model/population_jpformer_ohio_ptid_results')
    personalised_model_dir = os.path.join(PROJECT_ROOT, 'models/jpformer/fine_tuning_development_files/loss_function_weights_lowest')

    population_performance_df = pd.read_csv(os.path.join(population_model_dir, f"patient_{args.ptid}/base_model_eval/patient_{args.ptid}_base_model_overall_cg_ega.csv"))
    personalised_performance_df = pd.read_csv(os.path.join(personalised_model_dir, f"patient_{args.ptid}/fine_tuning_eval/patient_{args.ptid}_overall_cg_ega.csv"))
                                            

    population_performance_df['EP%'] = (population_performance_df['EP'] / population_performance_df['Count']) * 100
    personalised_performance_df['EP%'] = (personalised_performance_df['EP'] / personalised_performance_df['Count']) * 100



    if parser.parse_args().optimised_for == 'hypo':
        population_hypo_percent = population_performance_df['EP%'].iloc[0]
        personalised_hypo_percent = personalised_performance_df['EP%'].iloc[0]

    else:
        population_hypo_percent = population_performance_df['EP%'].iloc[-1]
        personalised_hypo_percent = personalised_performance_df['EP%'].iloc[-1]

    print(f"Population model {args.optimised_for} EP%: {population_hypo_percent}")
    print(f"Personalised model {args.optimised_for} EP%: {personalised_hypo_percent}")

    if population_hypo_percent > personalised_hypo_percent:
        print("Using personalised model for inference")



        personalised_model_dir = os.path.join(personalised_model_dir, f"patient_{args.ptid}/fine_tuning_eval")
        # get file name of only .pth file in directory

        best_personalised_model_file = [f for f in os.listdir(personalised_model_dir) if f.endswith('.pth')]


        pretrained_weights_path = os.path.join(personalised_model_dir, best_personalised_model_file[0])

        config_path = os.path.join(PROJECT_ROOT, "models/shared_config_files/fine_tuning_config.json")
    
    else:
        print("Using population model for inference")

        pretrained_weights_path = os.path.join(PROJECT_ROOT, "models/jpformer/population_jpformer_final_model/population_jpformer_replace_bg_aggregate_results/jpformer_dual_weighted_rmse_loss_func_high_dim_4_enc_lyrs_high_dropout_0.5696_MAE_0.3965.pth")
        config_path = os.path.join(PROJECT_ROOT, "models/shared_config_files/final_models_config.json")



    # Load the full Ohio data for the specified patient ID
    df = get_full_ohio_ptid_data(ptid)
    print(f"Loaded data for PtID {ptid} with {len(df)} records")



    # Load the model configuration
    config = load_config(config_path)
    config = ConfigObject(config)

    # Set up the computation device
    device = setup_device(config)



    model, model_name = load_model(config, device, pretrained_weights_path)

    
    starting_index = 72
    while starting_index < len(df) - 24:

        encoder_input, decoder_input, last_timestamp = ohio_data_slicing(df, starting_index)

        if encoder_input is None or decoder_input is None or last_timestamp is None:
            starting_index += 1
            continue

        try:
            hypo_time, hyper_time, output_df = inference_loop(encoder_input, decoder_input, last_timestamp, model, device)

            current_glucose_value = encoder_input[-1, 0].item()

            mean_bg = 152.91051040286524  # Mean for glucose in mg/dL
            std_bg = 70.27050122812615   # Standard deviation for glucose in mg/dL
        
            # Denormalize using the consistent parameters
            denormalised_glucose_value = (current_glucose_value * std_bg) + mean_bg

            display(output_df)

            if denormalised_glucose_value < 70:
                print(f"\nUrgent, treat hypoglycaemia. Current glucose value: {denormalised_glucose_value} mg/dl.\n\n\n")

            elif hypo_time:
                print(f"\nHypoglycemic event predicted at: {hypo_time}.\n\n\n")

            elif hyper_time:
                print(f"\nHyperglycemic event predicted at: {hyper_time}.\n\n\n")
            else:
                print("\nBlood sugar is stable.\n\n\n")



        except AssertionError as e:
            print(f"Error during inference: {str(e)}")
            # Handle the error (e.g., log it, skip to next iteration, etc.)
            # wait for 20 seconds

            continue

        # time.sleep(5)
        starting_index += 1


if __name__ == "__main__":
    main()




# create inference dataset

  # load ptid test data to dataframe

  # ensure sorted by datatime

 # data processing to bg value, hour minute

 # Need to manage interpoloation of missing values
       # consider flagging consecutive missing values as model not trained on this data

   # normalise data

# create slicing protocol
    #including the creation of the target hour and minute

    # create increment e.g sliding windows of 96 with step 1

# define model
    # load config (create config file)
    # load weights
    # batch size for inference is 1

# every  20 seconds load slices for next sliding window increment


# denormalise model predictions

# define warnings
    # is there a hypoglycaemic event? if so at what time?
    # is there a hyperglycaemic event? if so at what time?
    # if not blood sugar is stable

# create visualisation of model predictions
    # display warnings
    # time plot with thresholds


# apply basic logging
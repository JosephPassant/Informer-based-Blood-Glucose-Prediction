import pandas as pd



base_date = pd.to_datetime('2000-01-01')

slice_size = 96

overlap = 4

target_size = 24

start_token_size = 12

encoder_input_size = 72

decoder_input_size = start_token_size + target_size


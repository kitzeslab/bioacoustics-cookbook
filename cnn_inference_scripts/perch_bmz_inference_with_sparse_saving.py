"""
this Perch inference script discards all predictions below a certain threshold
and saves the predictions as pickled sparse dataframes to save space.

You can uncomment the wandb lines to log progress to an online WandB dashboard.
"""

# select which GPUs to allow tensorflow to use
# worth looking at nvidia-smi to see which GPUs are least utilized
# and tf.config.list_physical_devices() to make sure TF sees the gpus
# note that TF will claim all availble memory on the GPU
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices([physical_devices[0]], device_type="GPU")

import pandas as pd
from glob import glob
import os
from tqdm import tqdm
from pathlib import Path
import bioacoustics_model_zoo as bmz
import datetime
import numpy as np

# Set directory for predictions to be saved to
save_dir = "/path/to/output/dir/"
Path(save_dir).mkdir(exist_ok=True)

# Set path for field data
audio_folders = glob("/path/to/dataset/*SD*")
globbing_pattern = "*.WAV"

n_files = sum([len(list(Path(f).glob(globbing_pattern))) for f in audio_folders])

# Begin predictions
print(f"Beginning prediction : {datetime.datetime.now()} \n")
print(f"\t on {n_files} total audio files from {len(audio_folders)} folders")

# Set threshold for predictions to be kept
# lower scores are not retained, so that the outputs are sparse and can be stored efficiently
# note that this is a logit score (typical values -15 to +15), it is not bounded to 0-1.
lowest_score_to_keep = -1  # logit scale

# Load model from bioacoustics-model-zoo
model = bmz.Perch()
# Set device for predictions # this is for pytorch models
# model.device = "cuda:0"

# Create directory to save predictions to
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Start wandb session
# try:
#     import wandb

#     wandb.login()
#     wandb_session = wandb.init(
#         entity="groupname",
#         project="prjectname",
#         name="Perch prediction on MyDatset",
#     )
# except Exception as e:
#     print("wandb session failed for reason:")
#     print(e)
#     wandb_session = None


# Predict and save csv for one card at a time
for folder in tqdm(audio_folders):
    name = Path(folder).name
    files = list(Path(folder).glob(globbing_pattern))
    save_path = f"{save_dir}/loca2025laselva_{name}_perch_preds.pkl"
    if Path(save_path).exists():
        continue
    scores = model.predict(
        files[0:5],
        batch_size=64,
        num_workers=8,
        # wandb_session=wandb_session,
    )
    scores[scores < lowest_score_to_keep] = np.nan
    sparse_df = scores.astype(pd.SparseDtype("float", fill_value=np.nan))
    sparse_df.to_pickle(save_path)

    # Note: Load this pickled sparse df from file using:
    # sparse_df_loaded = pd.read_pickle("sparse_df.pkl")

print(f"prediction done: {datetime.datetime.now()}")

# close WandB
# wandb.finish()

import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
import sys

from glob import glob



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help = 'Directory containing existing clips')
    # parser.add_argument("--drive-path", dest = 'drive_path', type=str, help = 'D')
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False, help = "Don't export outputs.")
    
    return parser.parse_args()


if __name__ == "__main__":    
    args = parse_args()

    directory = args.dir
# directory = '/Users/lviotti/Downloads/rugr2023a_clips/'

audio_files = \
    glob(os.path.join(directory, '**/*.wav'), recursive=True) + \
    glob(os.path.join(directory, '**/*.WAV'), recursive=True) + \
    glob(os.path.join(directory, '**/*.mp3'), recursive=True)


df = pd.DataFrame({'local_path' : audio_files})

if directory[-1] != '/':
    directory = directory + '/'

df['relative_path'] =  df['local_path'].str.removeprefix(directory)
df['filename'] = [os.path.basename(i) for i in df['relative_path']]

# Card and date (folder structure dependent)
# df['card'] = [c.split('/')[0] for c in df['relative_path']]
# df['date'] = [d.split('/')[1] for d in df['relative_path']]


df = df.drop('local_path', axis = 1)

if not args.dry_run:
    df.to_csv(os.path.join(directory, '_all_clips.csv'), index= False)
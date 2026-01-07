"""
Based on a CSV file create target clips to be exported
"""
import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
import sys

from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

def create_clip(audio_path, st_time, end_time, margin = 3, plot = False):
    audio = Audio.from_file(audio_path)
    clip = audio.trim(st_time+margin, end_time + margin)
    return clip

def create_clips(df, 
                 dest, 
                 margin = 3, 
                 path_col = 'file', st_col = 'start_time', 
                 ed_col = 'end_time', score_col = 'score',
                 dry_run = False):
    """Crate clipped audio files based on input data with 
    
    Args:
        df (pd.DataFrame): Reference data frame containing:
            [path_col] : full audio file path
            [st_col] : clip start time in seconds
            [ed_col] :  clip end time in seconds
            [score_col] : clip classifier score
        dest (str): destination folder
        margin (int, optional): Clip padding on both ends. Defaults to 3.
        path_col (str, optional): Dataframe file path column name. Defaults to 'file'.
        st_col (str, optional):  Dataframe start time column name. Defaults to 'start_time'.
        ed_col (str, optional): _ Dataframe end time column name. Defaults to 'end_time'.
        score_col (str, optional):  Dataframe scores column name. Defaults to 'score'.
    """
    
    df['clip_name'] = np.nan
    
    for idx,row in tqdm(df.iterrows()):
        file = row[path_col]
        st_s = row[st_col] 
        ed_s = row[ed_col]
        score = row[score_col]
        
        clip = create_clip(file, st_s, ed_s, margin = margin)
        
        filename, extension = file.split('/')[-1].split('.')
        dest_filename = f"{filename}_{int(st_s)}_{int(ed_s)}_s{round(score)}.{extension}"
        
        df['clip_name'].iloc[idx] = dest_filename
        
        if not dry_run:
            clip.save(os.path.join(dest, dest_filename))
        
    return df 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help = 'Absolute path to csv containing clips to be created')
    parser.add_argument("--dest", type=str, help = 'Destination folder.')
    
    # parser.add_argument("--min-score", dest = 'scoreth', type=int, help = 'Minimum score to be included')
    parser.add_argument("--top", type=int, help = 'Number of clips to created.')
    
    parser.add_argument("--padding", default= 2, type=int, help = 'Padding in secods befores and after clip.')
    
    parser.add_argument("--st-col", dest='st', type=str, default='start_time', help= 'Clip start column on [data].')
    parser.add_argument("--ed-col", dest='ed', type=str, default='end_time', help= 'Clip end column on [data].')
    parser.add_argument("--path-col", dest='path', type=str, default='file', help= 'Full audio file path column.')
    parser.add_argument("--scores-col", dest='scores', type=str, default='score', help= 'Scores column.')
    
    # parser.add_argument("--subfolder-col", dest='scores', type=str, help= 'Column cointaining names of subfolders where clips are located, usually SD card ids.')
    
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False, help = "Don't export outputs.")

    
    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()
    
    df = pd.read_csv(args.data)
    
    # Make sure df is sorted by score
    df = df.sort_values(args.scores, ascending= False)
    
    # Filter clips
    if args.top is not None:
        df = df.head(args.top)
    
    # if scores_threshold is not None:
    #     df = df[df[score_col] >= scores_threshold]
    
    # Parse destiation directory
    if args.dest is None:
        prefix = os.path.basename(args.data).split('.')[0]
        dest_dir =os.path.join(os.path.dirname(args.data), f'{prefix}-tdl-clips/')
    else:
        dest_dir = args.dest
    
    if (not args.dry_run) & (not os.path.exists(dest_dir)):
        os.mkdir(dest_dir)
    
    # Crate clips
    df = create_clips(
        df,
        dest_dir,
        args.padding,
        args.path,
        args.st,
        args.ed,
        args.scores,
        args.dry_run)
    
    df['relative_path'] = "clips/" + df['clip_name']
    
    if not args.dry_run:
        
        # Create CSV for the new data
        df.to_csv(os.path.join(dest_dir,'_scores.csv'), index = False)
        
        # Create data readme file
        with open(os.path.join(dest_dir,'_README.txt'), 'w') as f:
            read_me_text = f'Create on {time.ctime()}' + ' ' +\
            'Using https://github.com/LeonardoViotti/tdl-notebook/data-prep/create-clips.py' '\n' +\
            '\n' +\
            'python ' + ' '.join(sys.argv)
            
            f.write(read_me_text)


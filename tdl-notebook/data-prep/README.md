# Crate clips for top-down-listening [UNDER DEVELOPMENT]

It uses a CSV file with audio file paths, clip start and end times to create clips in a destination folder. An optional argument can be used to limit the number of clips based on a scores column.

```
python create-clips.py path/to/scores.csv --dest destination/dir/path --top 500
```

By default, it creates a destination folder based on CSV file name an location, but an optional destination directory can be passed as an argument.

```
python create-clips.py path/to/scores.csv --dest destination/dir/path
```

It assumes CSV has columns 'file', 'st_time' and 'ed_time', and optionally a classidier 'scores' column, but custom column names can be passed:

```
python create-clips.py path/to/scores.csv --st-col my_st_col_name --ed-col my_end_col_name --path-col my_path_col_name --scores-col my_scores_col_name
```

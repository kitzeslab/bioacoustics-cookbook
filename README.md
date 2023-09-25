# Bioacoustics Cookbook
This repository contains various "recipes" for bioacoustics workflows and problems (sometimes more broadly applicable to quantitative ecology). These recipes include tutorials, discussions, and demonstrations, often relying on our open-source bioacoustics package [opensoundscape](https://github.com/kitzeslab/opensoundscape). They are written as Juptyer Notebooks to facilitate the integration of text, visuals, and code. 


## Machine Learning
- set up a model training project that uses a config.yml file to set hyperparameters and other options by copying the `cnn_project_with_cnfig_yml/` subdirectory
- use the VGGish pre-trained audio feature extractor with OpenSoundscape with `vggish.ipynb`

## Classifier-guided listening
- Review and annotate audio clips based on classifier scores and metadata with `classifier-guided_listening.ipynb`

## Data selection
- calculate sunrise and sunset times at a set of coordinate locations across a range of dates with `sunset_sunrise_calculator.ipynb`
- filter a set of AudioMoth files to include only specific date ranges and (start) time ranges with `filter_files_by_dates_and_times.ipynb`

## GIS
- Convert between UTM and Lat/Lon coordinates, and create KML files from tables/csvs of waypoints, in `gis/coordinates_and_kml.ipynb`
- Create a grid of points as a KML file in `gis/create_points_grid_kml.ipynb`
# Bioacoustics Cookbook
This repository contains various "recipes" for bioacoustics workflows and problems (sometimes more broadly applicable to quantitative ecology). These recipes include tutorials, discussions, and demonstrations, often relying on our open-source bioacoustics package [opensoundscape](https://github.com/kitzeslab/opensoundscape). They are written as Juptyer Notebooks to facilitate the integration of text, visuals, and code. 


## Machine Learning
- set up a model training project that uses a config.yml file to set hyperparameters and other options by copying the `cnn_project_with_cnfig_yml/` subdirectory
- use the VGGish pre-trained audio feature extractor with OpenSoundscape with `vggish.ipynb`
- generate embeddings with an OpenSoundscape CNN object with `cnn_embed.ipynb`
- train shallow classifiers on pre-generated embeddings using sklearn with `shallow_classifier.ipynb`
- copy weights from an sklearn MLPClassifier to a torch/opensoundscape model with `copy_shallow_classifier_sklearn_to_torch.ipynb`
- sample script for running Perch (global bird classifier) on audio data, using the Bioacoustics Model Zoo for perch access `cnn_inference_scripts/perch_bmz_inference_with_sparse_saving.py`
- sample script for evaluating HawkEars on an annotated dataset (Powdermill, Chronister et al 2020)

## Manipulating audio annotations and labels
- example of loading Audacity-formatted .txt file labels `annotations/load_audacity_annotations.ipynb`

## Classifier-guided listening
- Review and annotate audio clips based on classifier scores and metadata with `classifier-guided_listening.ipynb`

## Data selection and review
- calculate sunrise and sunset times at a set of coordinate locations across a range of dates with `sunset_sunrise_calculator.ipynb`
- filter a set of AudioMoth files to include only specific date ranges and (start) time ranges with `filter_files_by_dates_and_times.ipynb`
- review grids of spectrograms using `viwe_spec_grid.ipynb`

## GIS
- Convert between UTM and Lat/Lon coordinates, and create KML files from tables/csvs of waypoints, in `gis/coordinates_and_kml.ipynb`
- Create a grid of points as a KML file in `gis/create_points_grid_kml.ipynb`

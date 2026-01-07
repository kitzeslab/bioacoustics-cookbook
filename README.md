# Bioacoustics Cookbook

This repository contains self-contained "recipes" for common bioacoustics workflows and related quantitative ecology tasks. Most recipes are Jupyter notebooks that combine text, code, and figures and frequently rely on the open-source package [opensoundscape](https://github.com/kitzeslab/opensoundscape).

`pip install opensoundscape` will install the package in your conda/python environment. Be ware the that different notebooks may use different versions of OpenSoundscape, and might not work in all versions. 

The repo is organized by topic into subfolders. Below, each notebook (`.ipynb`) and Python script (`.py`) is listed with its location and general purpose.

---

## Annotations and Labels

- [annotations/load_audacity_annotations.ipynb](annotations/load_audacity_annotations.ipynb) – Load Audacity-formatted `.txt` label files and convert them into tabular/clip-level annotations for downstream analysis.
- [annotations/birdset_xcl_sparse_multihot_labels.ipynb](annotations/birdset_xcl_sparse_multihot_labels.ipynb) – Work with BirdSET / Xeno-Canto snapshot labels and create sparse multi-hot label representations.

---

## Classifier Training

### General shallow/CNN training

- [classifier_training/shallow_classifier.ipynb](classifier_training/shallow_classifier.ipynb) – Train shallow classifiers (e.g., scikit-learn models) on pre-generated embeddings from CNNs or other feature extractors.
- [classifier_training/classify_songs_only.ipynb](classifier_training/classify_songs_only.ipynb) – Example workflow for training or evaluating classifiers using subsets of data focused on song-only segments.
- [classifier_training/copy_shallow_classifier_sklearn_to_torch.ipynb](classifier_training/copy_shallow_classifier_sklearn_to_torch.ipynb) – Copy weights from a scikit-learn `MLPClassifier` into a PyTorch / OpenSoundscape model to initialize a neural network from a trained shallow classifier.

### CNN project using YAML configs

These files demonstrate a CNN training project controlled via a configuration YAML file.

- [classifier_training/cnn_project_with_config_yml/train_from_cfg.ipynb](classifier_training/cnn_project_with_config_yml/train_from_cfg.ipynb) – Train a CNN model using hyperparameters and options defined in a YAML config file.
- [classifier_training/cnn_project_with_config_yml/predict_from_config.ipynb](classifier_training/cnn_project_with_config_yml/predict_from_config.ipynb) – Run inference/prediction using a trained CNN model specified by the same YAML config structure.
- [classifier_training/cnn_project_with_config_yml/load_cfg.py](classifier_training/cnn_project_with_config_yml/load_cfg.py) – Helper functions for reading and parsing the YAML configuration used by the CNN project.
- [classifier_training/cnn_project_with_config_yml/__init__.py](classifier_training/cnn_project_with_config_yml/__init__.py) – Package initializer for the CNN project module.

### HawkEars fine-tuning

- [classifier_training/hawkears_finetune/finetune_custom_hawkears.ipynb](classifier_training/hawkears_finetune/finetune_custom_hawkears.ipynb) – Fine-tune the HawkEars model on a custom annotated dataset (e.g., Powdermill) and explore performance.
- [classifier_training/hawkears_finetune/custom_lr_hawkears.py](classifier_training/hawkears_finetune/custom_lr_hawkears.py) – Custom learning-rate schedule and related utilities used during HawkEars fine-tuning.

---

## CNN Inference Scripts

- [cnn_inference_scripts/vggish.ipynb](cnn_inference_scripts/vggish.ipynb) – Use the VGGish pre-trained audio feature extractor within OpenSoundscape to generate embeddings from audio recordings.
- [cnn_inference_scripts/perch_bmz_inference_with_sparse_saving.py](cnn_inference_scripts/perch_bmz_inference_with_sparse_saving.py) – Example script for running Perch (global bird classifier) via the Bioacoustics Model Zoo and saving predictions in a sparse format.

---

## Data Selection and Subsetting

- [data_selection/Extract_clips_XC_snapshot.ipynb](data_selection/Extract_clips_XC_snapshot.ipynb) – Extract clips from the BirdSET Xeno-Canto snapshot dataset according to configurable criteria (e.g., species, duration).
- [data_selection/filter_files_by_dates_and_times.ipynb](data_selection/filter_files_by_dates_and_times.ipynb) – Filter a collection of audio files (e.g., AudioMoth recordings) to include only specified date ranges and start-time windows.

---

## Demos and Tutorials

- [demos_tutorials/new_in_opso_080.ipynb](demos_tutorials/new_in_opso_080.ipynb) – Demonstrate new features and example workflows in OpenSoundscape version 0.8.0.
- [demos_tutorials/SCB_tutorial_demo.ipynb](demos_tutorials/SCB_tutorial_demo.ipynb) – Tutorial/demo notebook (e.g., for SCB workshops) showing end-to-end use of OpenSoundscape for bioacoustics analysis.

---

## Evaluation and Metrics

- [evaluation/prep_birdset_eval_sets.ipynb](evaluation/prep_birdset_eval_sets.ipynb) – Prepare BirdSET evaluation subsets and associated metadata for benchmarking classifiers.
- [evaluation/test_hawkears_on_powdermill.ipynb](evaluation/test_hawkears_on_powdermill.ipynb) – Evaluate the HawkEars classifier on the Powdermill dataset (Chronister et al. 2020) and summarize performance.
- [evaluation/visualize_metrics.ipynb](evaluation/visualize_metrics.ipynb) – Visualize classifier performance metrics (e.g., ROC curves, precision–recall, confusion matrices) for trained models.

---

## GIS and Time Calculations

- [gis/coordinates_and_kml.ipynb](gis/coordinates_and_kml.ipynb) – Convert between UTM and latitude/longitude coordinates and create KML files from tables or CSVs of waypoints.
- [gis/create_points_grid_kml.ipynb](gis/create_points_grid_kml.ipynb) – Generate a regular grid of points and export them as a KML file for use in mapping tools.
- [gis/sunset_sunrise_calculator.ipynb](gis/sunset_sunrise_calculator.ipynb) – Compute sunrise and sunset times for specified coordinates across ranges of dates, useful for planning recording schedules.

---

## Review, Annotation, and To-Do Listening

- [review_annotate/classifier-guided_listening.ipynb](review_annotate/classifier-guided_listening.ipynb) – Review and annotate audio clips guided by classifier scores and metadata, prioritizing clips likely to contain focal species.
- [review_annotate/view_spec_grid.ipynb](review_annotate/view_spec_grid.ipynb) – Display grids of spectrograms for rapid visual browsing and manual review of audio clips.

### TDL (Top-Down, ie Classifier-Guided Listening) notebooks and utilities

- [review_annotate/tdl-notebook/tdl_colab.ipynb](review_annotate/tdl-notebook/tdl_colab.ipynb) – Main to-do listening (TDL) workflow notebook, designed for interactive use (including in Google Colab) to step through annotation tasks.
- [review_annotate/tdl-notebook/git.ipynb](review_annotate/tdl-notebook/git.ipynb) – Notes and examples for managing TDL-related work with Git from within notebooks.
- [review_annotate/tdl-notebook/annotation.py](review_annotate/tdl-notebook/annotation.py) – Python utilities and helper functions used by the TDL notebooks for managing annotations and clip metadata.
- [review_annotate/tdl-notebook/data-prep/create-clips.py](review_annotate/tdl-notebook/data-prep/create-clips.py) – Data preparation script to create audio clips from longer recordings for use in TDL workflows.
- [review_annotate/tdl-notebook/data-prep/create-csv.py](review_annotate/tdl-notebook/data-prep/create-csv.py) – Data preparation script to build CSV metadata files describing clips, labels, or tasks for TDL.

---

## How to Use This Repository

- Browse to any notebook that matches your task (training, inference, GIS, review, etc.) and open it in Jupyter, JupyterLab, VS Code, or Google Colab.
- Many workflows assume you have [opensoundscape](https://github.com/kitzeslab/opensoundscape) and its dependencies installed, along with typical scientific Python packages (NumPy, pandas, matplotlib, etc.). `pip install opensoundscape` will install the package in your conda/python environment. Be ware the that different notebooks may use different versions of OpenSoundscape, and might not work in all versions. 
- Scripts in the `cnn_inference_scripts/` and `review_annotate/tdl-notebook/data-prep/` folders are intended to be run from the command line or imported into notebooks as utilities.

Feel free to adapt these recipes to your own datasets and workflows.

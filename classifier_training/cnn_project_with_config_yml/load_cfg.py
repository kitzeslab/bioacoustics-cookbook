"""Initialize a CNN object from a config file

Note: not all change-able attributes are included in the config file
(for instance, the loss function and optimizer objects are not included. Nor are
custom preprocessing functions or custom architectures)

Args:
    config_file: path to .yml config file. If None, no config is used.
"""

import yaml
import pandas as pd
import numpy as np
import torch
import random
from opensoundscape.preprocess.actions import Overlay
from opensoundscape import CNN
from opensoundscape.ml.cnn import use_resample_loss

#TODO: consider replacing 'None' with None, since this is a frequent error
# maybe yaml.load has an option for this


def cnn_from_cfg(config_file):
    cls = CNN
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)  # , Loader=yaml.Full)

    if isinstance(cfg["class_list"], list):
        classes = cfg["class_list"]
    elif isinstance(cfg["class_list"], str):
        try:
            classes = pd.read_csv(cfg["class_list"], header=None)[0].tolist()
        except Exception as exc:
            raise ValueError(
                f"Could not read classes from {cfg['class_list']}."
            ) from exc
    else:
        raise ValueError(
            f"Could not read classes from {cfg['class_list']}."
            f"Please provide a list of classes or a path to a csv file."
        )

    # initialize random seed for all stochastic packages
    if cfg["random_seed"] is not None:
        np.random.seed(cfg["random_seed"])
        torch.manual_seed(cfg["random_seed"])
        random.seed(cfg["random_seed"])

        # use deterministic cudnn algorithms
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)

    m = cls(
        architecture=cfg["architecture"],
        classes=classes,
        sample_duration=cfg["sample_duration"],
        single_target=cfg["single_target"],
        sample_shape=cfg["sample_shape"],
    )

    # load weights from checkpoint if provided
    if cfg["weights_path"] is not None:
        m.network.load_state_dict(torch.load(cfg["weights_path"]))

    # device:
    if isinstance(cfg["device"], str):
        m.device = cfg["device"]
    elif isinstance(cfg["device"], list):
        # wrap .network in torch.DataParallel
        # to use multiple GPUs
        m.network = torch.nn.DataParallel(m.network, device_ids=cfg["device"])
        m.device = cfg["device"][0]

    # set optimization parameters
    m.optimizer_params = cfg["optimizer_params"]
    m.lr_update_interval = cfg["lr_update_interval"]
    m.lr_cooling_factor = cfg["lr_cooling_factor"]

    # set loss function
    if cfg["resample_loss"] and not cfg["single_target"]:
        use_resample_loss(m)

    # TODO: allow optimizer choice?

    # inference settings
    m.prediction_threshold = cfg["prediction_threshold"]

    ##  preprocessing and augmentation ##

    # spectrogram settings
    m.preprocessor.pipeline.to_spec.set(**cfg['spec'])

    # bandpassing
    if cfg["bandpass_range"] is None:
        m.preprocessor.pipeline.bandpass.bypass = True  # check syntax
    else:
        l, h = cfg["bandpass_range"]
        m.preprocessor.pipeline.bandpass.set(min_f=l, max_f=h)

    # augmentation
    if cfg["overlay_df"] is not None:
        overlay_action = Overlay(
            overlay_df=cfg["overlay_df"],
            update_labels=cfg["overlay_update_labels"],
            overlay_classes=cfg["overlay_classes"],
            overlay_prob=cfg["overlay_prob"],
            max_overlay_num=cfg["max_overlay_num"],
            overlay_weight=cfg["overlay_weight"],
        )
        m.preprocessor.pipeline.insert_action(
            "overlay", overlay_action, after_key="to_tensor"
        )

    m.preprocessor.pipeline.time_mask.set(
        max_masks=cfg["time_mask_max_n"],
        max_width=cfg["time_mask_max_width"],
    )
    m.preprocessor.pipeline.frequency_mask.set(
        max_masks=cfg["freq_mask_max_n"],
        max_width=cfg["freq_mask_max_width"],
    )
    m.preprocessor.pipeline.add_noise.set(
        std=cfg["add_noise_std"],
    )
    m.preprocessor.pipeline.random_affine.set(
        translate=(cfg["translate_time"], cfg["translate_freq"])
    )

    # logging
    m.log_file = cfg["log_file"]
    m.logging_level = cfg["logging_level"]
    m.verbose = cfg["verbose"]

    # weights and biases
    m.wandb_logging.update(cfg['wandb_logging'])

    return m
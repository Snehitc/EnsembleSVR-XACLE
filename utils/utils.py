# @Author  :   Snehit
# @E-mail  :   snehitc@gmail.com


import json
import torch
import os
import pandas as pd
import numpy as np
from collections.abc import Mapping, Sequence
from transformers.tokenization_utils_base import BatchEncoding
import pickle
from tqdm import tqdm

# This function is taken from the XACLE baseline implementation
def load_config(config_path = "config.json"):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg


# This function is taken from the XACLE baseline implementation
def move_to_device(obj, device):
    """
    Recursively move tensors (and common container objects that hold tensors)
    to the target device.

    Parameters
    ----------
    obj : Any
        Arbitrary object that may contain tensors (Tensor, dict-like, list-like,
        or transformers.BatchEncoding). Non-tensor scalars/strings are left unchanged.
    device : torch.device or str
        Target device (e.g., "cuda:0", "cpu").

    Returns
    -------
    Any
        Same structure as `obj`, but with all tensors placed on `device`.
    """
    # 1) Single tensor → move directly
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    # 2) HuggingFace BatchEncoding supports .to(), handle it explicitly
    if isinstance(obj, BatchEncoding):
        return obj.to(device)

    # 3) Mapping types (dict, UserDict, etc.) → recurse on each value
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    # 4) Sequence types (list, tuple, etc.) → preserve container type
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(move_to_device(v, device) for v in obj)

    # 5) Anything else (int, float, str, Path, ...) → leave untouched
    return obj



# Save the SkLearn Preprocessors
def Save_SKLearnPreprocessors(Preprocessor, Ensemble_set, cfg):
    path = cfg['output_dir'] + '/version_' + cfg['config_filename'].split('.')[0] + '/' + Ensemble_set + '/'
    os.makedirs(path, exist_ok=True)
    filename = cfg['preprocessor'][Ensemble_set][Preprocessor.__class__.__name__] + '.pkl'
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(Preprocessor, file)


# Save the SVR
def Save_SVR(TrainedSVR, Ensemble_set, cfg):
    path = cfg['output_dir'] + '/version_' + cfg['config_filename'].split('.')[0] + '/' + Ensemble_set + '/'
    os.makedirs(path, exist_ok=True)
    filename = cfg['SVR'][Ensemble_set]['name'] + '.pkl'
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(TrainedSVR, file)

# Save the SkLearn Preprocessors and SVR
def Save_Preprocessor_and_SVR(Normalizer_fit, StandardScaler_fit, SVR_fit, Ensemble_set, cfg):
    Save_SKLearnPreprocessors(Normalizer_fit, Ensemble_set, cfg)
    Save_SKLearnPreprocessors(StandardScaler_fit, Ensemble_set, cfg)
    Save_SVR(SVR_fit, Ensemble_set, cfg)

    path = cfg['output_dir'] + '/version_' + cfg['config_filename'].split('.')[0] + '/' + Ensemble_set + '/'
    print(f"Preprocessors and SVR model saved for ensemble set: {path} \n")


# Save the inference results
def Save_Inference(y_pred_test_denorm, test_loader, cfg, dataset_key):
    print('Warning: make sure data_loader was set to *shuffle=False* while training SVRs')
        
    AllFiles_test = []
    for batch in tqdm(test_loader):
        BatchFiles = [os.path.basename(OneFile) for OneFile in batch['wav_paths']]
        AllFiles_test.extend(BatchFiles)
    
    Header_name = 'prediction_score' if dataset_key=='test' else 'pred_score'
    
    Predictions = {"wav_file_name": AllFiles_test,
                Header_name: np.around(y_pred_test_denorm, decimals=2)}

    savepath = cfg['output_dir'] + '/version_' + cfg['config_filename'].split('.')[0]
    os.makedirs(savepath, exist_ok=True)
    filename = f'inference_result_for_{dataset_key}.csv'
    df = pd.DataFrame(Predictions)
    df.to_csv(os.path.join(savepath, filename), index=False)
    print(f'csv file is saved \nPath: {savepath} \nfilename: {filename}\n')



# Load the preprocessors
def Load_Preprocessor(Preprocessor_name, Ensemble_set, cfg):
    path = cfg['output_dir'] + '/version_' + cfg['config_filename'].split('.')[0] + '/' + Ensemble_set + '/'
    filename = Preprocessor_name + '.pkl'
    filepath = os.path.join(path, filename)
    with open(filepath, 'rb') as file:
        preprocessor = pickle.load(file)
    return preprocessor


# Load the saved SVR
def Load_SVR(TrainedSVR_name, Ensemble_set, cfg):
    path = cfg['output_dir'] + '/version_' + cfg['config_filename'].split('.')[0] + '/' + Ensemble_set + '/'
    filename = TrainedSVR_name + '.pkl'
    filepath = os.path.join(path, filename)
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the saved preprocessors and SVR
def Load_Preprocessor_and_SVR(Normalizer_name, StandardScaler_name, SVR_name, Ensemble_set, cfg):
    Normalizer = Load_Preprocessor(Normalizer_name, Ensemble_set, cfg)
    StandardScaler = Load_Preprocessor(StandardScaler_name, Ensemble_set, cfg)
    SVR = Load_SVR(SVR_name, Ensemble_set, cfg)

    return Normalizer, StandardScaler, SVR

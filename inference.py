import os
import pandas as pd
import numpy as np
import utils.utils as utils
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import gc


from datasets.fetch_data import get_infdataset
from evaluate import evaluate_all
from features.all_feature_dict import get_all_features_in_dict

import sys
sys.path.append('./m2d/')
sys.path.append('./MGA-CLAP')


def GetLabels(label_dict):
    for i, (key, value) in enumerate(label_dict.items()):
        if i!=0:
            if not np.array_equal(value, value_previous):
                raise ValueError("different labels for same data point \nRecommended to set dataloaders with shuffle=False")
        value_previous = value

    return value


def Predict_SVR(dataset_key, X_comb_test, Normalizer_, StandardScaler_, SVR_, cfg):
    # Preprocessing
    print('Preprocessing...')
    X_scaled_comb_test = Normalizer_.transform(X_comb_test)
    X_scaled_comb_test = StandardScaler_.transform(X_scaled_comb_test)

    # SVR_A
    print(f'Predicting on {dataset_key} set...')
    y_pred_test = SVR_.predict(X_scaled_comb_test)
    y_pred_test_denorm = y_pred_test*5 + 5
    
    return y_pred_test_denorm


def Predict_SVR_A(dataset_key, X_dict_test, cfg): 
        Ensemble_set = 'A'
        print(f'--- SVR {Ensemble_set} ---')
        
        X_comb_test_A = np.concatenate((X_dict_test['M2D_Clap']['AudioFeatures'], X_dict_test['M2D_Clap']['TextFeatures'], X_dict_test['M2D_Clap']['Cosine_Sim'], 
                                    X_dict_test['M2D_Clap']['Cosine_Ang'], X_dict_test['M2D_Clap']['L2'], X_dict_test['M2D_Clap']['L1'],

                                    X_dict_test['MS_Clap']['AudioFeatures'], X_dict_test['MS_Clap']['TextFeatures'], X_dict_test['MS_Clap']['Cosine_Sim'], 
                                    X_dict_test['MS_Clap']['Cosine_Ang'], X_dict_test['MS_Clap']['L2'], X_dict_test['MS_Clap']['L1'],
                                    
                                    X_dict_test['MGA_Clap']['AudioFeatures'], X_dict_test['MGA_Clap']['TextFeatures'], X_dict_test['MGA_Clap']['Cosine_Sim'], 
                                    X_dict_test['MGA_Clap']['Cosine_Ang'], X_dict_test['MGA_Clap']['L2'], X_dict_test['MGA_Clap']['L1'],), axis=1)
        
        Normalizer_name = cfg['preprocessor']['A']['Normalizer']
        StandardScaler_name = cfg['preprocessor']['A']['StandardScaler']
        SVR_name = cfg['SVR']['A']['name']
        Ensemble_set = 'A'
        Normalizer_A, StandardScaler_A, SVR_A = utils.Load_Preprocessor_and_SVR(Normalizer_name, StandardScaler_name, SVR_name, Ensemble_set, cfg)
        y_pred_test_denorm_A = Predict_SVR(dataset_key, X_comb_test_A, Normalizer_A, StandardScaler_A, SVR_A, cfg)

        return y_pred_test_denorm_A


def Predict_SVR_B(dataset_key, X_dict_test, cfg): 
        Ensemble_set = 'B'
        print(f'--- SVR {Ensemble_set} ---')
        
        X_comb_test_B = np.concatenate((X_dict_test['Laion_Clap']['AudioFeatures'], X_dict_test['Laion_Clap']['TextFeatures'], X_dict_test['Laion_Clap']['L1'],
                                   X_dict_test['Whisper']['AudioFeatures'], X_dict_test['Whisper']['TextFeatures'],), axis=1)
         
        Normalizer_name = cfg['preprocessor']['B']['Normalizer']
        StandardScaler_name = cfg['preprocessor']['B']['StandardScaler']
        SVR_name = cfg['SVR']['B']['name']
        Ensemble_set = 'B'
        Normalizer_B, StandardScaler_B, SVR_B = utils.Load_Preprocessor_and_SVR(Normalizer_name, StandardScaler_name, SVR_name, Ensemble_set, cfg)
        y_pred_test_denorm_B = Predict_SVR(dataset_key, X_comb_test_B, Normalizer_B, StandardScaler_B, SVR_B, cfg)

        return y_pred_test_denorm_B



def inference(dataset_key, cfg):
    test_ds   = get_infdataset(
                        cfg[dataset_key+"_list"],
                        os.path.join(cfg["wav_dir"], dataset_key),
                        max_sec=cfg["max_len"],
                        sr=cfg["sample_rate"],
                        )
    test_loader = DataLoader(
                        test_ds, 
                        batch_size=cfg[dataset_key+"_batch_size"], 
                        shuffle=False,
                        num_workers=cfg["num_workers"], 
                        collate_fn=test_ds.collate_fn
                        )
    
    print(f"\n------ {dataset_key} set ------")
    if 'X_dict_test' not in globals() and 'y_dict_test' not in globals():
        X_dict_test, _, param = get_all_features_in_dict(test_loader, cfg, test_data=True)

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()


    y_pred_test_denorm_A = Predict_SVR_A(dataset_key, X_dict_test, cfg)
    y_pred_test_denorm_B = Predict_SVR_B(dataset_key, X_dict_test, cfg)
    
    print('\nEnsemble: Prediction = W_A*SVR_A + W_B*SVR_B')
    print(f'Predicting on {dataset_key} set...\n')
    y_pred_test = cfg['W'][0]*y_pred_test_denorm_A + cfg['W'][1]*y_pred_test_denorm_B
    y_pred_test_clipped = np.clip(y_pred_test, 0, 10)
    
    utils.Save_Inference(y_pred_test_clipped, test_loader, cfg, dataset_key)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <chkpt_subdir_name> <dataset_key>")

    config_filepath_filename = os.path.join('./outputs', sys.argv[1], 'config.json')
    cfg = utils.load_config(config_filepath_filename)
    
    dataset_key = sys.argv[2] # 'validation' or 'test'

    inference(dataset_key, cfg)
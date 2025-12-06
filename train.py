import os
import numpy as np
import utils.utils as utils
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import gc
import json

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.optimize import minimize

from datasets.fetch_data import get_dataset
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



def Fit_SVR(Ensemble_set, X_comb, X_comb_val, y, y_val, SVR_input_param, cfg):
    # Preprocessing
    print('Preprocessing...')
    Normalizer_ = Normalizer().fit(X_comb)  # fit does nothing.
    X_scaled_comb = Normalizer_.transform(X_comb)
    X_scaled_comb_val = Normalizer_.transform(X_comb_val)

    StandardScaler_ = StandardScaler()
    X_scaled_comb = StandardScaler_.fit_transform(X_scaled_comb)
    X_scaled_comb_val = StandardScaler_.transform(X_scaled_comb_val)


    # SVR_A
    print('Training SVR...')
    SVR_ = SVR(C=SVR_input_param['C'], 
                kernel=SVR_input_param['kernel'], 
                epsilon=SVR_input_param['epsilon'], 
                gamma=SVR_input_param['gamma'])
    SVR_.fit(X_scaled_comb, y)

    # Train and Validation Results
    print('Predicting on Train set...')
    y_pred_train = SVR_.predict(X_scaled_comb)
    y_pred_train_denorm = y_pred_train*5 + 5
    y_denorm = y*5 + 5
    results = evaluate_all(y_pred_train_denorm, y_denorm)
    print('\t', results)

    print('Predicting on Val set...')
    y_pred_val = SVR_.predict(X_scaled_comb_val)
    y_pred_val_denorm = y_pred_val*5 + 5
    y_val_denorm = y_val*5 + 5
    results_val = evaluate_all(y_pred_val_denorm, y_val_denorm)
    print('\t', results_val)

    # Save the trained SVR and Preprocessor
    utils.Save_Preprocessor_and_SVR(Normalizer_, StandardScaler_, SVR_, Ensemble_set, cfg)

    params = SVR_.dual_coef_.size + SVR_.intercept_.size
    return y_pred_train_denorm, y_pred_val_denorm, params



def Fit_SVR_A(X_dict, X_dict_val, y_dict, y_dict_val, SVR_input_param, param, cfg): 
        Ensemble_set = 'A'
        print(f'--- SVR {Ensemble_set} ---')
        X_comb_A = np.concatenate((X_dict['M2D_Clap']['AudioFeatures'], X_dict['M2D_Clap']['TextFeatures'], X_dict['M2D_Clap']['Cosine_Sim'], 
                            X_dict['M2D_Clap']['Cosine_Ang'], X_dict['M2D_Clap']['L2'], X_dict['M2D_Clap']['L1'],

                            X_dict['MS_Clap']['AudioFeatures'], X_dict['MS_Clap']['TextFeatures'], X_dict['MS_Clap']['Cosine_Sim'], 
                            X_dict['MS_Clap']['Cosine_Ang'], X_dict['MS_Clap']['L2'], X_dict['MS_Clap']['L1'],
                             
                            X_dict['MGA_Clap']['AudioFeatures'], X_dict['MGA_Clap']['TextFeatures'], X_dict['MGA_Clap']['Cosine_Sim'], 
                            X_dict['MGA_Clap']['Cosine_Ang'], X_dict['MGA_Clap']['L2'], X_dict['MGA_Clap']['L1'],), axis=1)

        X_comb_val_A = np.concatenate((X_dict_val['M2D_Clap']['AudioFeatures'], X_dict_val['M2D_Clap']['TextFeatures'], X_dict_val['M2D_Clap']['Cosine_Sim'], 
                                    X_dict_val['M2D_Clap']['Cosine_Ang'], X_dict_val['M2D_Clap']['L2'], X_dict_val['M2D_Clap']['L1'],

                                    X_dict_val['MS_Clap']['AudioFeatures'], X_dict_val['MS_Clap']['TextFeatures'], X_dict_val['MS_Clap']['Cosine_Sim'], 
                                    X_dict_val['MS_Clap']['Cosine_Ang'], X_dict_val['MS_Clap']['L2'], X_dict_val['MS_Clap']['L1'],
                                    
                                    X_dict_val['MGA_Clap']['AudioFeatures'], X_dict_val['MGA_Clap']['TextFeatures'], X_dict_val['MGA_Clap']['Cosine_Sim'], 
                                    X_dict_val['MGA_Clap']['Cosine_Ang'], X_dict_val['MGA_Clap']['L2'], X_dict_val['MGA_Clap']['L1'],), axis=1)
        
        y = GetLabels(y_dict)
        y_val = GetLabels(y_dict_val)
        y_pred_train_denorm_A, y_pred_val_denorm_A, params = Fit_SVR(Ensemble_set, X_comb_A, X_comb_val_A, y, y_val, 
                                                                     SVR_input_param, cfg)
        param['SVR_'+Ensemble_set] = params
        return y_pred_train_denorm_A, y_pred_val_denorm_A, param




def Fit_SVR_B(X_dict, X_dict_val, y_dict, y_dict_val, SVR_input_param, param, cfg):
        Ensemble_set = 'B'
        print(f'\n--- SVR {Ensemble_set} ---')
        X_comb_B = np.concatenate((X_dict['Laion_Clap']['AudioFeatures'], X_dict['Laion_Clap']['TextFeatures'], X_dict['Laion_Clap']['L1'], 
                               X_dict['Whisper']['AudioFeatures'], X_dict['Whisper']['TextFeatures'],), axis=1)

        X_comb_val_B = np.concatenate((X_dict_val['Laion_Clap']['AudioFeatures'], X_dict_val['Laion_Clap']['TextFeatures'], X_dict_val['Laion_Clap']['L1'],
                                   X_dict_val['Whisper']['AudioFeatures'], X_dict_val['Whisper']['TextFeatures'],), axis=1)
        
        y = GetLabels(y_dict)
        y_val = GetLabels(y_dict_val)
        y_pred_train_denorm_B, y_pred_val_denorm_B, params = Fit_SVR(Ensemble_set, X_comb_B, X_comb_val_B, y, y_val, 
                                                                SVR_input_param, cfg)
        param['SVR_'+Ensemble_set] = params
        return y_pred_train_denorm_B, y_pred_val_denorm_B, param



def Param_count(param):
        count = 0
        for key, val in param.items():
            count += val
        print(f'\nTotal Parameter Count: {count}')


def RMSE(y, y_pred):
    rsme = np.sqrt(np.mean( (y-y_pred)**2) )
    return rsme






def train(cfg):
    

    train_ds = get_dataset(
                        cfg["train_list"],
                        os.path.join(cfg["wav_dir"], "train"),
                        max_sec=cfg["max_len"],
                        sr=cfg["sample_rate"],
                        org_max=10.0,
                        org_min=0.0
                        )
    val_ds   = get_dataset(
                        cfg["validation_list"],
                        os.path.join(cfg["wav_dir"], "validation"),
                        max_sec=cfg["max_len"],
                        sr=cfg["sample_rate"],
                        org_max=10.0,
                        org_min=0.0
                        )
    train_loader = DataLoader(
                        train_ds,
                        batch_size=cfg["batch_size"],
                        shuffle=False,
                        num_workers=cfg["num_workers"],
                        collate_fn=train_ds.collate_fn,
                        drop_last=False
                        )
    val_loader = DataLoader(
                        val_ds, 
                        batch_size=cfg["validation_batch_size"], 
                        shuffle=False,
                        num_workers=cfg["num_workers"], 
                        collate_fn=val_ds.collate_fn
                        )


    print("------ Train set ------")
    if 'X_dict' not in globals() and 'y_dict' not in globals() and 'param' not in globals():
        X_dict, y_dict, param = get_all_features_in_dict(train_loader, cfg)

    print("\n------ Val set ------")
    if 'X_dict_val' not in globals() and 'y_dict_val' not in globals():
        X_dict_val, y_dict_val, param = get_all_features_in_dict(val_loader, cfg)

    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # Fit SVR A and B
    SVR_A_input_param = cfg['SVR']['A']['input_param']
    y_pred_train_denorm_A, y_pred_val_denorm_A, param = Fit_SVR_A(X_dict, X_dict_val, y_dict, y_dict_val, SVR_A_input_param, param, cfg)

    SVR_B_input_param = cfg['SVR']['B']['input_param']
    y_pred_train_denorm_B, y_pred_val_denorm_B, param = Fit_SVR_B(X_dict, X_dict_val, y_dict, y_dict_val,  SVR_B_input_param, param, cfg)
    
    # Optimize Weights for Ensemble
    y = GetLabels(y_dict)
    y_val = GetLabels(y_dict_val)
    y_denorm = y*5 + 5
    y_denorm_val = y_val*5 + 5
    def min_func(K):
        y_pred_train = K[0]*y_pred_train_denorm_A + K[1]*y_pred_train_denorm_B
        return RMSE(y_denorm, y_pred_train)
    res = minimize(min_func, [1/2]*2, method='TNC', tol=1e-6)
    W = res.x
    cfg['W'] = W.tolist()

    # Final Train and Validation Results
    print('\nEnsemble: Prediction = W_A*SVR_A + W_B*SVR_B')
    print('Predicting on Train set...')
    y_pred_train = W[0]*y_pred_train_denorm_A + W[1]*y_pred_train_denorm_B
    y_pred_train_clipped = np.clip(y_pred_train, 0, 10)
    results = evaluate_all(y_pred_train_clipped, y_denorm)
    print('\t', results)

    print('\nPredicting on Val set...')
    y_pred_val = W[0]*y_pred_val_denorm_A + W[1]*y_pred_val_denorm_B
    y_pred_val_clipped = np.clip(y_pred_val, 0, 10)
    results_val = evaluate_all(y_pred_val_clipped, y_denorm_val)
    print('\t', results_val)

    Param_count(param)

    # Save config file for trained model
    config_filepath_filename = os.path.join(cfg['output_dir'] + '/version_' + cfg['config_filename'].split('.')[0], 'config.json')
    with open(config_filepath_filename, "w") as json_file:
        json.dump(cfg, json_file, indent=4)
    
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file>")
    
    config_filename = sys.argv[1]        #e.g. "config_submission1.json"
    cfg = utils.load_config(config_filename)
    cfg['config_filename'] = config_filename
    train(cfg)
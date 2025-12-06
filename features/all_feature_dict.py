from features import extract_features, proximity_features
from load_pretrained_models.load_model import load_model

def GetFeatures(model_name, loader, iteration, cfg, test_data):
    print(f"*** [{iteration}]Extracting features using {model_name}... ***")
    
    X_dict, y_dict = {}, {}
    # Extract Raw features
    if model_name == 'M2D_Clap':
        model, param = load_model(model_name, cfg)
        X_dict, y_dict = extract_features.M2DCLAP(loader, model, cfg, test_data)
    elif model_name == 'MS_Clap':
        model, param = load_model(model_name, cfg)
        X_dict, y_dict = extract_features.MSCLAP(loader, model, cfg, test_data)
    elif model_name == 'Laion_Clap':
        processor, audio_encoder, tokenizer, text_encoder, param = load_model(model_name, cfg)
        X_dict, y_dict = extract_features.LaionCLAP(loader, processor, audio_encoder, tokenizer, text_encoder, cfg, test_data)
    elif model_name == 'Whisper':
        feature_extractor, tokenizer, model, param = load_model(model_name, cfg)
        X_dict, y_dict = extract_features.Whisper(loader, feature_extractor, tokenizer, model, cfg, test_data)
    elif model_name == 'MGA_Clap':
        model, param = load_model(model_name, cfg)
        X_dict, y_dict = extract_features.MGAClap(loader, model, cfg, test_data)
    else:
        raise ValueError(f"Model {model_name} not implimented in our code.")

    # Add proximity features
    X_dict['Cosine_Sim'] = proximity_features.cosine_similarity(X_dict['AudioFeatures'], X_dict['TextFeatures'])
    X_dict['Cosine_Ang'] = proximity_features.angular_distance(X_dict['AudioFeatures'], X_dict['TextFeatures'])
    X_dict['L2'] = proximity_features.L2_normalized(X_dict['AudioFeatures'], X_dict['TextFeatures'])
    X_dict['L1']  = proximity_features.L1_normalized(X_dict['AudioFeatures'], X_dict['TextFeatures'])
    
    print(f"*** [{iteration}]Finished Extracting features using {model_name} ***\n")
    return X_dict, y_dict, param


def get_all_features_in_dict(loader, cfg, test_data=False):
    param = {}
    X_dict, y_dict = {}, {}

    model_list = ['M2D_Clap', 'MS_Clap', 'Laion_Clap', 'MGA_Clap', 'Whisper']
    count_all_model = len(model_list)
    for i, model_name in enumerate(model_list):
        iteration = str(i+1) + '/' +str(count_all_model)
        X_dict[model_name], y_dict[model_name], param[model_name] = GetFeatures(model_name, loader, iteration, cfg, test_data)
    
    return X_dict, y_dict, param
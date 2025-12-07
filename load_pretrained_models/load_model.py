# @Author  :   Snehit
# @E-mail  :   snehitc@gmail.com


def load_model(model_name, cfg):
    if model_name == 'M2D_Clap':
        from examples.portable_m2d import PortableM2D

        weight = cfg['model']['M2D']['ckpt']
        # Use flat_features=True for CLAP features only. For conventional audio features, flat_features should be False.
        model = PortableM2D(weight_file=weight, flat_features=True)

        params = sum(param.numel() for param in model.parameters())
        return model, params
    

    elif model_name == 'MS_Clap':
        from msclap import CLAP

        use_cuda = True if cfg['device'] =='cuda' else False
        model = CLAP(version = '2023', use_cuda=use_cuda)

        params = sum(param.numel() for param in model.clap.parameters())
        return model, params
    

    elif model_name == 'Laion_Clap':
        from transformers import AutoProcessor, AutoTokenizer, ClapAudioModelWithProjection, ClapTextModelWithProjection

        processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
        audio_encoder = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
        text_encoder = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        
        params = sum(param.numel() for param in audio_encoder.parameters())
        params = sum(param.numel() for param in text_encoder.parameters())
        return processor, audio_encoder, tokenizer, text_encoder, params
    
    elif model_name == 'Whisper':
        from transformers import AutoFeatureExtractor, WhisperTokenizer, WhisperModel
        
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v2")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2")
        model = WhisperModel.from_pretrained("openai/whisper-large-v2")

        params = sum(param.numel() for param in model.parameters())
        return feature_extractor, tokenizer, model, params
        
    elif model_name == 'MGA_Clap':
        import torch
        import yaml
        from models.ase_model import ASE
        
        config_MGA_path = cfg['model']['MGA']['config']
        
        with open(config_MGA_path, "r") as f:
            config_MGA = yaml.safe_load(f)

        model = ASE(config_MGA)
        weight = cfg['model']['MGA']['ckpt']
        cp = torch.load(weight, weights_only=False)
        model.load_state_dict(cp['model'], strict=False)

        params = sum(param.numel() for param in model.parameters())
        return model, params
    else:
        raise ValueError(f"Model {model_name} not implimented in our code.")

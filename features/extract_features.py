import numpy as np
import torch
from tqdm import tqdm
import utils.utils as utils
import torch.nn as nn
import torch.nn.functional as F




def M2DCLAP(data_loader, model, cfg, test_data=False):
    device = cfg['device']
    model.to(device)
    model.eval()

    AudioFeatures = []
    TextFeatures = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = utils.move_to_device(batch, device)
            
            audio_embs = model.encode_clap_audio(batch['wavs'].squeeze(1))   # (B, 768)
            text_embs = model.encode_clap_text(batch['captions'])            # (B, 768))
            
            AudioFeatures.append(audio_embs.detach().cpu().numpy())
            TextFeatures.append(text_embs.detach().cpu().numpy())
            if not test_data:
                labels.append(batch["scores"].cpu().numpy())
    AudioFeatures = np.concatenate(AudioFeatures, axis=0)
    TextFeatures = np.concatenate(TextFeatures, axis=0)
    if not test_data:
        labels = np.concatenate(labels, axis=0)

    del model
    torch.cuda.empty_cache()
    return {'AudioFeatures':AudioFeatures, 'TextFeatures': TextFeatures}, labels



def MSCLAP(data_loader, clap_model, cfg, test_data=False):
    device = cfg['device']
    clap_model.clap.to(device)
    clap_model.clap.eval()

    AudioFeatures = []
    TextFeatures = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = utils.move_to_device(batch, device)
            
            audio_embeddings = clap_model.get_audio_embeddings(batch['wav_paths'])  # (B, 1024)
            text_embeddings = clap_model.get_text_embeddings(batch['captions'])     # (B, 1024)
            
            AudioFeatures.append(audio_embeddings.detach().cpu().numpy())
            TextFeatures.append(text_embeddings.detach().cpu().numpy())
            if not test_data:
                labels.append(batch["scores"].cpu().numpy())
    AudioFeatures = np.concatenate(AudioFeatures, axis=0)
    TextFeatures = np.concatenate(TextFeatures, axis=0)
    if not test_data:
        labels = np.concatenate(labels, axis=0)

    del clap_model
    torch.cuda.empty_cache()
    return {'AudioFeatures':AudioFeatures, 'TextFeatures': TextFeatures}, labels



def LaionCLAP(data_loader, processor, audio_encoder, tokenizer, text_encoder, cfg, test_data=False):
    device = cfg['device']
    audio_encoder.to(device)
    text_encoder.to(device)
    audio_encoder.eval()
    text_encoder.eval()

    AvgPool_audio = nn.AdaptiveAvgPool2d(1).to(device)
    AvgPool_txt = nn.AdaptiveAvgPool1d(1).to(device)
    AudioFeatures = []
    TextFeatures = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = utils.move_to_device(batch, device)
            
            audio_proc = processor(audio=batch["wavs"].squeeze(1).cpu().numpy(), sampling_rate=48000, return_tensors="pt").to(device)
            audio_emb = audio_encoder(**audio_proc)
            audio_emb = AvgPool_audio(audio_emb.last_hidden_state).squeeze(-1).squeeze(-1)    # (B, 768)
            
            cap_tokens = tokenizer(batch['captions'], padding=True, return_tensors="pt").to(device)
            txt_emb = text_encoder(**cap_tokens)
            txt_emb = AvgPool_txt(txt_emb.last_hidden_state.permute(0,2,1)).squeeze(-1)   # (B, 768)
            
            AudioFeatures.append(audio_emb.detach().cpu().numpy())
            TextFeatures.append(txt_emb.detach().cpu().numpy())
            if not test_data:
                labels.append(batch["scores"].cpu().numpy())
    AudioFeatures = np.concatenate(AudioFeatures, axis=0)
    TextFeatures = np.concatenate(TextFeatures, axis=0)
    if not test_data:
        labels = np.concatenate(labels, axis=0)

    del audio_encoder, text_encoder, AvgPool_audio, AvgPool_txt
    torch.cuda.empty_cache()
    return {'AudioFeatures':AudioFeatures, 'TextFeatures': TextFeatures}, labels



def Whisper(data_loader, feature_extractor, tokenizer, model, cfg, test_data=False):
    device = cfg['device']
    model.to(device)
    model.eval()

    AvgPool = nn.AdaptiveAvgPool1d(1).to(device)
    AudioFeatures = []
    TextFeatures = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = utils.move_to_device(batch, device)
            
            audio_batch = [w.cpu().numpy().flatten() for w in batch["wavs"]]
            inputs = feature_extractor(audio_batch, sampling_rate= cfg['sample_rate'], return_tensors="pt")   # (B, 80, 3000)
            
            tokenized = tokenizer(batch['captions'], return_tensors="pt", padding=True)
            decoder_input_ids = tokenized.input_ids                        # (B, Seq_len)
            
            outputs = model(inputs.input_features.to(device), decoder_input_ids=decoder_input_ids.to(device))
        
            audio_emb = AvgPool(outputs.encoder_last_hidden_state.permute(0,2,1)).squeeze(-1)   # (B, 1280)
            txt_emb = AvgPool(outputs.last_hidden_state.permute(0,2,1)).squeeze(-1)             # (B, 1280)

            AudioFeatures.append(audio_emb.detach().cpu().numpy())
            TextFeatures.append(txt_emb.detach().cpu().numpy())
            if not test_data:
                labels.append(batch["scores"].cpu().numpy())
    AudioFeatures = np.concatenate(AudioFeatures, axis=0)
    TextFeatures = np.concatenate(TextFeatures, axis=0)
    if not test_data:
        labels = np.concatenate(labels, axis=0)

    del model
    torch.cuda.empty_cache()
    return {'AudioFeatures':AudioFeatures, 'TextFeatures': TextFeatures}, labels




def MGAClap(data_loader, model, cfg, test_data=False):
    device = cfg['device']
    model.to(device)
    model.eval()

    AudioFeatures = []
    TextFeatures = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = utils.move_to_device(batch, device)
            
            _, frame_embeds = model.encode_audio(batch['wavs'].squeeze(1))  # (B, 32, 1024)
            audio_embeds = model.msc(frame_embeds, model.codebook)   # (B, 1024)
            frame_embeds = F.normalize(frame_embeds, dim=-1)    # (B, 32, 1024)
            audio_embeds = F.normalize(audio_embeds, dim=-1)    # (B, 1024)
            
            _, word_embeds, attn_mask = model.encode_text(batch['captions'])    # (B, seq_len, 1024)
            text_embeds = model.msc(word_embeds, model.codebook, attn_mask)     # (B, 1024)
            text_embeds = F.normalize(text_embeds, dim=-1)                      # (B, 1024)
            
            AudioFeatures.append(audio_embeds.detach().cpu().numpy())
            TextFeatures.append(text_embeds.detach().cpu().numpy())
            if not test_data:
                labels.append(batch["scores"].cpu().numpy())
    AudioFeatures = np.concatenate(AudioFeatures, axis=0)
    TextFeatures = np.concatenate(TextFeatures, axis=0)
    if not test_data:
        labels = np.concatenate(labels, axis=0)

    del model
    torch.cuda.empty_cache()
    return {'AudioFeatures':AudioFeatures, 'TextFeatures': TextFeatures}, labels
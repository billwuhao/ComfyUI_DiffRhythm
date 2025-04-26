import torch
import torchaudio
from einops import rearrange
import sys
import os
import json
from easydict import EasyDict
from muq import MuQMuLan

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from model import DiT, CFM

from diffrhythm_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    CNENTokenizer,
)


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    use_ema: bool = True
):
    model = model.half()
    if device == 'mps':
        model = model.float()

    ckpt_type = ckpt_path.split(".")[-1]
    try:
        if ckpt_type == "safetensors":
            from safetensors.torch import load_file
            checkpoint = load_file(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, weights_only=True)
    except Exception as e:
        raise

    try:
        if use_ema:
            if ckpt_type == "safetensors":
                checkpoint = {"ema_model_state_dict": checkpoint}
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            if ckpt_type == "safetensors":
                checkpoint = {"model_state_dict": checkpoint}
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    except Exception as e:
        raise

    return model.to(device)


def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    steps,
    cfg_strength,
    start_time,
    odeint_method,
    sway_sampling_coef=None,
    chunked=False,
    vocal_flag=False,
):
    with torch.inference_mode():
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=steps,
            cfg_strength=cfg_strength,
            start_time=start_time,
            odeint_method=odeint_method,
            vocal_flag=vocal_flag,
            sway_sampling_coef=sway_sampling_coef,
        )
    
    generated = generated.to(torch.float32)
    latent = generated.transpose(1, 2)  # [b d t]

    output = decode_audio(latent, vae_model, chunked=chunked)
    
    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")
    # Peak normalize, clip, convert to int16, and save to file
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )
    return output


class MultiLinePrompt:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": ""}),
                },
        }

    CATEGORY = "🎤MW/MW-DiffRhythm"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "promptgen"
    
    def promptgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)

import folder_paths
models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")
models = ["cfm_model.pt", "cfm_full_model.pt"]

def set_all_seeds(seed):
    # import random
    # import numpy as np
    # # 1. Python 内置随机模块
    # random.seed(seed)
    # # 2. NumPy 随机数生成器
    # np.random.seed(seed)
    # 3. PyTorch CPU 和 GPU 种子
    torch.manual_seed(seed)
    # 4. 如果使用 CUDA（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
        # torch.backends.cudnn.deterministic = True  # 确保卷积结果确定
        # torch.backends.cudnn.benchmark = False     # 关闭优化（牺牲速度换取确定性）


class DiffRhythmRun:
    def __init__(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        self.device = device
        self.cfm = None
        self.vae = None
        self.muq = None
        self.tokenizer = None
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "model": (models, {"default": "cfm_full_model.pt"}),
                "style_prompt": ("STRING", {
                    "multiline": True, 
                    "default": ""}),
                },
            "optional": {
                "lyrics_prompt": ("STRING", {"forceInput": True}),
                "style_audio": ("AUDIO", ),
                "chunked": ("BOOLEAN", {"default": False, "tooltip": "Whether to use chunked decoding."}),
                "unload_model": ("BOOLEAN", {"default": True}),
                "odeint_method": (["euler", "midpoint", "rk4","implicit_adams"], {"default": "euler"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "cfg": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    CATEGORY = "🎤MW/MW-DiffRhythm"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "diffrhythmgen"
    
    def diffrhythmgen(
            self,
            model: str,
            style_prompt: str, 
            lyrics_prompt: str = "", 
            style_audio: str = None,
            chunked: bool = False,
            odeint_method: str = "euler",
            steps: int = 30,
            cfg: int = 4,
            unload_model: bool = False,
            seed: int = 0):

        if seed != 0:
            set_all_seeds(seed)

        if model == "cfm_model.pt":
            max_frames = 2048
        elif model == "cfm_full_model.pt": 
            max_frames = 6144

        if self.cfm is None:
            self.cfm, self.tokenizer, self.muq, self.vae = self.prepare_model(model, self.device)

        lrc_prompt, start_time = get_lrc_token(max_frames, lyrics_prompt, self.tokenizer, self.device)

        vocal_flag = False
        if style_audio:
            prompt, vocal_flag = self.get_audio_style_prompt(self.muq, style_audio)
        elif style_prompt:
            prompt = self.get_text_style_prompt(self.muq, style_prompt)
        else:
            raise ValueError("Style prompt or style audio must be provided")

        negative_style_prompt = get_negative_style_prompt(self.device)
        latent_prompt = get_reference_latent(self.device, max_frames)

        sway_sampling_coef = -1 if steps < 32 else None
        try:
            generated_song = inference(
                cfm_model=self.cfm,
                vae_model=self.vae,
                cond=latent_prompt,
                text=lrc_prompt,
                duration=max_frames,
                style_prompt=prompt,
                negative_style_prompt=negative_style_prompt,
                steps=steps,
                chunked=chunked,
                cfg_strength=cfg,
                sway_sampling_coef=sway_sampling_coef,
                start_time=start_time,
                vocal_flag=vocal_flag,
                odeint_method=odeint_method,
            )
        except Exception as e:
            raise

        audio_tensor = generated_song.unsqueeze(0)

        if unload_model:
            import gc
            self.cfm = None
            self.muq = None
            self.vae = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

        return ({"waveform": audio_tensor, "sample_rate": 44100},)


    def get_audio_style_prompt(self, model, audio):
        vocal_flag = False
        if audio is None:
            return None, vocal_flag
        mulan = model
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Ensure waveform has correct shape
        if len(waveform.shape) == 3:  # [1, channels, samples]
            waveform = waveform.squeeze(0)
        if waveform.shape[0] > 1:  # If stereo, convert to mono
            waveform = waveform.mean(0, keepdim=True)

        if sample_rate != 24000:
            waveform = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)

        # Calculate audio length (seconds)
        audio_len = waveform.shape[-1] / 24000
        
        if audio_len <= 1:
            vocal_flag = True
        
        if audio_len > 10:
            start_sample = int((audio_len // 2 - 5) * 24000)
            end_sample = start_sample + 10 * 24000
            wav_segment = waveform[..., start_sample:end_sample]
        else:
            wav_segment = waveform
            
        wav = wav_segment.to(model.device)
        
        with torch.no_grad():
            audio_emb = mulan(wavs = wav) # [1, 512]
            
        audio_emb = audio_emb.half()

        return audio_emb, vocal_flag


    def get_text_style_prompt(self, model, text_prompt):
        if text_prompt is None:
            return None
        mulan = model
        
        with torch.no_grad():
            text_emb = mulan(texts = text_prompt) # [1, 512]
        text_emb = text_emb.half()

        return text_emb

    def prepare_model(self, model, device):
        # prepare tokenizer
        try:
            tokenizer = CNENTokenizer()
        except Exception as e:
            raise

        from huggingface_hub import snapshot_download
        # prepare cfm model
        if model == "cfm_full_model.pt":
            dit_ckpt_path = f"{model_path}/DiffRhythm/cfm_full_model.pt"
            dit_config_path = f"{model_path}/DiffRhythm/config.json"
            if not os.path.exists(dit_ckpt_path):
                snapshot_download(repo_id="ASLP-lab/DiffRhythm-full",
                                    local_dir=f"{model_path}/DiffRhythm")
            
        elif model == "cfm_model.pt":
            dit_ckpt_path = f"{model_path}/DiffRhythm/cfm_model.pt"
            dit_config_path = f"{model_path}/DiffRhythm/config.json"
            if not os.path.exists(dit_ckpt_path):
                snapshot_download(repo_id="ASLP-lab/DiffRhythm-base",
                                    local_dir=f"{model_path}/DiffRhythm")

        vae_ckpt_path = f"{model_path}/DiffRhythm/vae_model.pt"

        if not os.path.exists(vae_ckpt_path):
            snapshot_download(repo_id="ASLP-lab/DiffRhythm-vae",
                                local_dir=f"{model_path}/DiffRhythm", 
                                ignore_patterns=["*safetensors"])
            
        try:
            with open(dit_config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
        except Exception as e:
            raise
        
        dit_model_cls = DiT
        if model == "cfm_model.pt":
            cfm = CFM(
                transformer=dit_model_cls(**model_config["model"], use_style_prompt=True, max_pos=2048),
                num_channels=model_config["model"]["mel_dim"],
            )
        elif model == "cfm_full_model.pt":
            cfm = CFM(
                    transformer=dit_model_cls(**model_config["model"], use_style_prompt=True, max_pos=6144),
                    num_channels=model_config["model"]['mel_dim'],
                    use_style_prompt=True
                )
        cfm = cfm.to(device)

        try:
            cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)
        except Exception as e:
            raise

        # prepare muq model
        try:
            main_model_dir = f"{model_path}/DiffRhythm/MuQ-MuLan-large"
            local_audio_model_dir = f"{model_path}/DiffRhythm/MuQ-large-msd-iter"
            local_text_model_dir = f"{model_path}/DiffRhythm/xlm-roberta-base"

            config_path = f"{main_model_dir}/config.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            config_dict['audio_model']['name'] = local_audio_model_dir
            config_dict['text_model']['name'] = local_text_model_dir
            config_obj = EasyDict(config_dict) 

            muq = MuQMuLan(config=config_obj, hf_hub_cache_dir=None)
            weights_path = f"{main_model_dir}/pytorch_model.bin"

            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                # Adjust loading based on how weights are saved (e.g., remove prefixes if needed)
                muq.load_state_dict(state_dict, strict=False) # Use strict=False initially
            except FileNotFoundError:
                raise FileNotFoundError(f"Weights file not found at {weights_path}")

        except Exception as e:
            raise
        
        muq = muq.to(device).eval()

        # prepare vae
        try:
            vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)
        except Exception as e:
            raise

        return (cfm, tokenizer, muq, vae)


from MWAudioRecorderDR import AudioRecorderDR

NODE_CLASS_MAPPINGS = {
    "DiffRhythmRun": DiffRhythmRun,
    "MultiLinePrompt": MultiLinePrompt,
    "AudioRecorderDR": AudioRecorderDR
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffRhythmRun": "DiffRhythm Run",
    "MultiLinePrompt": "Multi Line Prompt",
    "AudioRecorderDR": "MW Audio Recorder"
}
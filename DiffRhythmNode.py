import os
import time
import random
import torch
import torchaudio
from einops import rearrange
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_audio_style_prompt,
    get_text_style_prompt,
    prepare_model,
    eval_song,
)


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


import folder_paths
cache_dir = folder_paths.get_temp_directory()
import tempfile
from typing import Optional


def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")


def inference(
    cfm_model,
    vae_model,
    eval_model,
    eval_muq,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    steps,
    cfg_strength,
    sway_sampling_coef,
    start_time,
    # file_type,
    vocal_flag,
    odeint_method,
    pred_frames,
    batch_infer_num,
    chunked=True,
):
    with torch.inference_mode():
        latents, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            start_time=start_time,
            vocal_flag=vocal_flag,
            odeint_method=odeint_method,
            latent_pred_segments=pred_frames,
            batch_infer_num=batch_infer_num
        )

        outputs = []
        for latent in latents:
            latent = latent.to(torch.float32)
            latent = latent.transpose(1, 2)  # [b d t]

            output = decode_audio(latent, vae_model, chunked=chunked)

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            
            outputs.append(output)
        if batch_infer_num > 1:
            generated_song = eval_song(eval_model, eval_muq, outputs)
        else:
            generated_song = outputs[0]
        output_tensor = generated_song.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).cpu()

    return output_tensor


node_dir = os.path.dirname(os.path.abspath(__file__))
folder = f'{node_dir}/diffrhythm/example'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

selected = random.choice(files)
with open(os.path.join(folder, selected), 'r', encoding='utf-8') as f:
    lyrics = f.read()

class MultiLineLyricsDR:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "lyrics": ("STRING", {
                    "multiline": True, 
                    "default": lyrics}),
                },
        }

    CATEGORY = "🎤MW/MW-DiffRhythm"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "lyricsgen"
    
    def lyricsgen(self, lyrics: str):
        return (lyrics.strip(),)


CFM = None
VAE = None
MUQ = None
TOKENIZER = None
EVAL_MODEL = None
EVAL_MUQ = None

class DiffRhythmRun:
    def __init__(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        self.device = device
        self.model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["cfm_model_v1_2.pt", "cfm_model.pt", "cfm_full_model.pt"], {"default": "cfm_model_v1_2.pt"}),
                "style_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Indie folk ballad, coming-of-age themes, acoustic guitar picking with harmonica interludes"}),
                },
            "optional": {
                "lyrics_or_edit_lyrics": ("STRING", {"forceInput": True}),
                "style_audio_or_edit_song": ("AUDIO", ),
                # "chunked": ("BOOLEAN", {"default": False, "tooltip": "Whether to use chunked decoding."}),
                "unload_model": ("BOOLEAN", {"default": True}),
                "odeint_method": (["euler", "midpoint", "rk4","implicit_adams"], {"default": "euler"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "cfg": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "quality_or_speed":(["quality", "speed"], {"default": "speed"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "edit": ("BOOLEAN", {"default": False}),
                "edit_segments": ("STRING", {"default":"[-1, 20], [60, -1]", "multiline": True}),
            },
        }

    CATEGORY = "🎤MW/MW-DiffRhythm"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "diffrhythmgen"
    
    def diffrhythmgen(
            self,
            edit,
            model: str,
            style_prompt: str = None, 
            lyrics_or_edit_lyrics: str = "", 
            style_audio_or_edit_song = None,
            edit_segments: str = "",
            chunked: bool = True,
            odeint_method: str = "euler",
            steps: int = 30,
            cfg: int = 4,
            quality_or_speed: str = "speed",
            unload_model: bool = False,
            seed: int = 0):

        if seed != 0:
            set_all_seeds(seed)

        if model == "cfm_model.pt" or model == "cfm_model_v1_2.pt":
            max_frames = 2048
        else:
            max_frames = 6144
            
        global CFM, TOKENIZER, MUQ, VAE, EVAL_MODEL, EVAL_MUQ
        if CFM is None or self.model_name != model:
            self.model_name = model
            CFM, TOKENIZER, MUQ, VAE, EVAL_MODEL, EVAL_MUQ = prepare_model(max_frames, self.device, model)

        batch_infer_num = 1 if quality_or_speed == "speed" else 5

        lyrics = lyrics_or_edit_lyrics.strip()
        vocal_flag = False
        if style_audio_or_edit_song is not None:
            style_audio_path = cache_audio_tensor(cache_dir, 
                                                  style_audio_or_edit_song["waveform"].squeeze(0), 
                                                  style_audio_or_edit_song["sample_rate"], 
                                                  filename_prefix="style_audio_")
            prompt, vocal_flag = get_audio_style_prompt(MUQ, style_audio_path)
            print("Provided style_audio, style_prompt will be ineffective")
        else:
            assert style_prompt.strip(), "One of style_audio and style_prompt must be provided"
            prompt = get_text_style_prompt(MUQ, style_prompt)

        edit_song_path = None
        if edit:
            if style_audio_or_edit_song is not None:
                edit_song_path = style_audio_path
                prompt, vocal_flag = get_audio_style_prompt(MUQ, edit_song_path)
            assert edit_song_path and lyrics and edit_segments.strip(), "edit song, edit lyrics, edit segments must be provided"

            edit_segments = "["+edit_segments+"]"

        else:
            edit_segments = None

        lrc_prompt, start_time = get_lrc_token(max_frames, lyrics.strip(), TOKENIZER, self.device)

        negative_style_prompt = get_negative_style_prompt(self.device)
        latent_prompt, pred_frames = get_reference_latent(self.device, 
                                                          max_frames, 
                                                          edit, 
                                                          pred_segments=edit_segments, 
                                                          ref_song=edit_song_path, 
                                                          vae_model=VAE)
        sway_sampling_coef = -1 if steps < 32 else None

        s_t = time.time()
        generated_songs = inference(
            cfm_model=CFM,
            vae_model=VAE,
            eval_model=EVAL_MODEL,
            eval_muq=EVAL_MUQ,
            odeint_method=odeint_method,
            vocal_flag=vocal_flag,
            sway_sampling_coef=sway_sampling_coef,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=max_frames,
            style_prompt=prompt,
            negative_style_prompt=negative_style_prompt,
            steps=steps,
            chunked=chunked,
            cfg_strength=cfg,
            start_time=start_time,
            pred_frames=pred_frames,
            batch_infer_num=batch_infer_num
        )
        e_t = time.time() - s_t
        print(f"inference cost {e_t:.2f} seconds")

        audio_tensor = generated_songs.unsqueeze(0)

        if unload_model:
            import gc
            CFM = None
            MUQ = None
            VAE = None
            TOKENIZER = None
            EVAL_MODEL = None
            EVAL_MUQ = None
            gc.collect()
            torch.cuda.empty_cache()

        return ({"waveform": audio_tensor, "sample_rate": 44100},)


NODE_CLASS_MAPPINGS = {
    "DiffRhythmRun": DiffRhythmRun,
    "MultiLineLyricsDR": MultiLineLyricsDR
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffRhythmRun": "DiffRhythm Run",
    "MultiLineLyricsDR": "MultiLine Lyrics"
}
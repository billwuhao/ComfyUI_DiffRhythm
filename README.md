[中文](README-CN.md) | [English](README.md)

# DiffRhythm Nodes for ComfyUI

Fast and easy end-to-end full-length song generation.

## 📣 Updates

[2025-05-13]⚒️: Supports DiffRhythm v1.2, better quality, editable lyrics. Currently released a 95-second song model, full-length song release will be updated promptly. **Note**: The version code has been updated, and the generation quality of previous models may be affected. If you want to try the previous version, please revert to the version before v2.2.0.

[2025-04-26]⚒️: Changed to manually download the muq model.

[2025-03-16]⚒️: Released version v2.0.0. Supports full-length music generation, 4 minutes only takes 62 seconds.

Download the model and place it in the `ComfyUI\models\TTS\DiffRhythm` folder:

- [DiffRhythm-full](https://huggingface.co/ASLP-lab/DiffRhythm-full) rename the model to `cfm_full_model.pt`.

[2025-03-13]⚒️: Released version v1.0.0.

## Usage

- Text generated music:
![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-13_01-51-00.png)

- Generate music based on reference audio:
![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-29_13-44-25.png)

- Edit music:
![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-29_13-46-34.png)

- Automatically generate song and add bilingual lyrics subtitles:
![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-14_16-33-54.png)

https://github.com/user-attachments/assets/26b5c66d-6ce5-4bf9-9294-4658176b2a66

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_DiffRhythm.git
cd ComfyUI_DiffRhythm
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

The model needs to be manually downloaded to the `ComfyUI\models\TTS\DiffRhythm` folder.

The structure is as follows:

![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-13_01-54-13.png)

```
.
|  cfm_model_v1_2.pt
│  cfm_full_model.pt
│  cfm_model.pt
│  config.json
│  vae_model.pt
|
├─eval-model
│      eval.yaml
│      eval.safetensors
│
├─MuQ-large-msd-iter
│      config.json
│      model.safetensors
│
├─MuQ-MuLan-large
│      config.json
│      pytorch_model.bin
│
└─xlm-roberta-base
        config.json
        model.safetensors
        sentencepiece.bpe.model
        tokenizer.json
        tokenizer_config.json
```

Manual download links:
- https://huggingface.co/ASLP-lab/DiffRhythm-1_2/blob/main/cfm_model.pt  → `cfm_model_v1_2.pt`
- https://huggingface.co/spaces/ASLP-lab/DiffRhythm/tree/main/pretrained
- https://huggingface.co/ASLP-lab/DiffRhythm-full/tree/main
- https://huggingface.co/ASLP-lab/DiffRhythm-base/blob/main/cfm_model.pt  
- https://huggingface.co/ASLP-lab/DiffRhythm-vae/blob/main/vae_model.pt  
- https://huggingface.co/OpenMuQ/MuQ-MuLan-large/tree/main  
- https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/main → `.safetensors`: (https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/blob/refs%2Fpr%2F1/model.safetensors)
- https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main


## Environment Configuration

For Windows systems, configure as follows:

Download and install the latest version of [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

Add the system environment variable `PHONEMIZER_ESPEAK_LIBRARY`, the value is the path to the `libespeak-ng.dll` file in your espeak-ng installation, for example: `C:\Program Files\eSpeak NG\libespeak-ng.dll`.

For Linux systems, you need to install the `espeak-ng` package. Execute the following command to install:

`apt-get -qq -y install espeak-ng`

Mac is supported, but untested.

Enjoy the music🎶

## Acknowledgements

[DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)

Thanks to the DiffRhythm team for their excellent work👍.

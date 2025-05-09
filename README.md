[中文](README-CN.md) | [English](README.md) 

# DiffRhythm Node for ComfyUI

Blazingly Fast and Embarrassingly Simple End-to-End Full-Length Song Generation.

![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-03-12_23-49-32.png)

## 📣 update

[2025-04-26]⚒️: Change to manually selecting to download the `muq` model.

[2025-03-21] ⚒️: Code refactored, ultra-fast generation speed. 4 minutes 45 seconds of music generated in less than 20 seconds, 1 minute 35 seconds of music generated in less than 7 seconds. Added more tunable parameters for more creative freedom. Optional model unloading.

[2025-03-16]⚒️: Released version v2.0.0. Supports full-length music generation, 4 minutes only takes 62 seconds.

![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-03-16_03-53-48.png)

Download the model and place it in the `ComfyUI\models\TTS\DiffRhythm` folder:

- [DiffRhythm-full](https://huggingface.co/ASLP-lab/DiffRhythm-full)  Rename the model to `cfm_full_model.pt`, and also download `comfig.json` and put it together.

[2025-03-13]⚒️: Release version v1.0.0.

- All parameters are optional; you can generate random music without providing any parameters.

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

![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-03-13_00-08-51.png)

```
.
│  cfm_full_model.pt
│  cfm_model.pt
│  config.json
│  vae_model.pt
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

https://huggingface.co/ASLP-lab/DiffRhythm-full/tree/main  → `cfm_full_model.pt` and `config.json`
https://huggingface.co/ASLP-lab/DiffRhythm-base/blob/main/cfm_model.pt  
https://huggingface.co/ASLP-lab/DiffRhythm-vae/blob/main/vae_model.pt  
https://huggingface.co/OpenMuQ/MuQ-MuLan-large/tree/main  
https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/main → `.safetensors`: (https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/blob/refs%2Fpr%2F1/model.safetensors)  
https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main

## Environment Configuration

- Configure the following on Windows systems:

Download and install the latest version of [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

Add the environment variable `PHONEMIZER_ESPEAK_LIBRARY` to your system. The value should be the path to the `libespeak-ng.dll` file in your espeak-ng installation, for example: `C:\Program Files\eSpeak NG\libespeak-ng.dll`.

- On Linux systems, you need to install the `espeak-ng` package. Execute the following command to install:

`apt-get -qq -y install espeak-ng`

It should support Mac, but has not been tested.

Enjoy the music! 🎶

## Acknowledgements

[DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)

Thanks to the DiffRhythm team for their excellent work. Currently the strongest open-source music/song generation model 👍.

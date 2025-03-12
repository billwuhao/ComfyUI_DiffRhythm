[中文](README.md) | [English](README-en.md) 

# DiffRhythm Node for ComfyUI

Blazingly Fast and Embarrassingly Simple End-to-End Full-Length Song Generation.

![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-03-12_23-49-32.png)

## 📣 update

[2025-03-13]⚒️: Release version v1.0.0.

- All parameters are optional; you can generate random music without providing any parameters.

## Model Download

Models will be automatically downloaded to the `ComfyUI\models\TTS\DiffRhythm` folder.

The structure is as follows:

![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-03-13_00-08-51.png)

Manual Download Addresses:

https://huggingface.co/ASLP-lab/DiffRhythm-base/blob/main/cfm_model.pt  
https://huggingface.co/ASLP-lab/DiffRhythm-vae/blob/main/vae_model.pt  
https://huggingface.co/OpenMuQ/MuQ-MuLan-large/tree/main  
https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/main  
https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main

## Environment Configuration

Configure the following on Windows systems, other systems have not been tested. Should support Linux and Mac.

Download and install the latest version of [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

Add the environment variable `PHONEMIZER_ESPEAK_LIBRARY` to your system. The value should be the path to the `libespeak-ng.dll` file in your espeak-ng installation, for example: `C:\Program Files\eSpeak NG\libespeak-ng.dll`.

Enjoy the music! 🎶

## Acknowledgements

[DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)

Thanks to the DiffRhythm team for their excellent work. Currently the strongest open-source music/song generation model 👍.

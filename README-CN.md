[中文](README-CN.md) | [English](README.md) 

# DiffRhythm 的 ComfyUI 节点

快速而简单的端到端全长歌曲生成.

![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-03-12_23-49-32.png)


## 📣 更新

[2025-04-26]⚒️: 改为手动选择下载 muq 模型.

[2025-03-21]⚒️: 代码重构, 超快生成速度, 4分45秒音乐, 20秒不到生成, 1分35秒音乐, 7秒不到生成. 增加更多可调参数, 畅玩更自由. 可选是否卸载模型.

[2025-03-16]⚒️: 发布版本 v2.0.0. 支持全长音乐生成, 4 分钟仅需 62 秒.

![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-03-16_03-53-48.png)

下载模型放到 `ComfyUI\models\TTS\DiffRhythm` 文件夹下:

- [DiffRhythm-full](https://huggingface.co/ASLP-lab/DiffRhythm-full)  模型重命名为 `cfm_full_model.pt`, 同时下载 comfig.json 放到一起.

[2025-03-13]⚒️: 发布版本 v1.0.0.

- 所有参数均是可选的, 不提供任何参数随机生成音乐.

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_DiffRhythm.git
cd ComfyUI_DiffRhythm
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

模型需手动下载到 `ComfyUI\models\TTS\DiffRhythm` 文件夹下.

结构如下:

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

手动下载地址:
https://huggingface.co/ASLP-lab/DiffRhythm-full/tree/main
https://huggingface.co/ASLP-lab/DiffRhythm-base/blob/main/cfm_model.pt  
https://huggingface.co/ASLP-lab/DiffRhythm-vae/blob/main/vae_model.pt  
https://huggingface.co/OpenMuQ/MuQ-MuLan-large/tree/main  
https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/main → `.safetensors`: (https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/blob/refs%2Fpr%2F1/model.safetensors) 
https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main

## 环境配置

Windows 系统做如下配置. 

下载安装最新版 [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

添加环境变量 `PHONEMIZER_ESPEAK_LIBRARY` 到系统中, 值是你安装的 espeak-ng 软件中 `libespeak-ng.dll` 文件的路径, 例如: `C:\Program Files\eSpeak NG\libespeak-ng.dll`.

Linux 系统下, 需要安装 `espeak-ng` 软件包. 执行如下命令安装:

`apt-get -qq -y install espeak-ng`

支持 Mac, 但尚未测试.

享受音乐吧🎶

## 鸣谢

[DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)

感谢 DiffRhythm 团队的卓越的工作, 目前最强开源 音乐/歌曲 生成模型👍.
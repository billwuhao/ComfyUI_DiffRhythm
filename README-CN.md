[中文](README-CN.md) | [English](README.md) 

# DiffRhythm 的 ComfyUI 节点

快速而简单的端到端全长歌曲生成.

## 📣 更新

[2025-05-13]⚒️: 支持 DiffRhythm v1.2 版本, 质量更好, 可编辑歌词. 目前发布 95 秒长度歌曲模型, 全长歌曲发布将即时更新. **注意**: 版本代码更新, 之前的模型生成质量可能会受到影响. 如果尝试之前的版本, 请退回到 v2.2.0 之前版本.

[2025-04-26]⚒️: 改为手动选择下载 muq 模型.

[2025-03-16]⚒️: 发布版本 v2.0.0. 支持全长音乐生成, 4 分钟仅需 62 秒.

下载模型放到 `ComfyUI\models\TTS\DiffRhythm` 文件夹下:

- [DiffRhythm-full](https://huggingface.co/ASLP-lab/DiffRhythm-full)  模型重命名为 `cfm_full_model.pt`.

[2025-03-13]⚒️: 发布版本 v1.0.0.

## 使用

- 文本生成音乐:
![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-13_01-51-00.png)

- 参考音频生成音乐:
![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-29_13-44-25.png)

- 编辑音乐:
![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-29_13-46-34.png)

- 自动生成歌曲, 自动添加歌词字幕:
![](https://github.com/billwuhao/ComfyUI_DiffRhythm/blob/master/images/2025-05-14_16-33-54.png)

https://github.com/user-attachments/assets/26b5c66d-6ce5-4bf9-9294-4658176b2a66

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

手动下载地址:
- https://huggingface.co/ASLP-lab/DiffRhythm-1_2/blob/main/cfm_model.pt  重命名: `cfm_model_v1_2.pt`
- https://huggingface.co/spaces/ASLP-lab/DiffRhythm/tree/main/pretrained
- https://huggingface.co/ASLP-lab/DiffRhythm-full/tree/main
- https://huggingface.co/ASLP-lab/DiffRhythm-base/blob/main/cfm_model.pt  
- https://huggingface.co/ASLP-lab/DiffRhythm-vae/blob/main/vae_model.pt  
- https://huggingface.co/OpenMuQ/MuQ-MuLan-large/tree/main  
- https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/main 要下载 `.safetensors` 格式: (https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/blob/refs%2Fpr%2F1/model.safetensors) 
- https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main

## 环境配置

Windows 系统做如下配置. 

下载安装最新版 [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

添加系统环境变量 `PHONEMIZER_ESPEAK_LIBRARY`, 值是你安装的 espeak-ng 软件中 `libespeak-ng.dll` 文件的路径, 例如: `C:\Program Files\eSpeak NG\libespeak-ng.dll`.

Linux 系统下, 需要安装 `espeak-ng` 软件包. 执行如下命令安装:

`apt-get -qq -y install espeak-ng`

支持 Mac, 但尚未测试.

享受音乐吧🎶

## 鸣谢

[DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)

感谢 DiffRhythm 团队的卓越的工作👍.
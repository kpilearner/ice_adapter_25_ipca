<div align="center">

<h1>In-Context Edit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer</h1>

<div>
    <a href="https://river-zhang.github.io/zechuanzhang//" target="_blank">Zechuan Zhang</a>&emsp;
    <a href="https://horizonwind2004.github.io/" target="_blank">Ji Xie</a>&emsp;
    <a href="https://yulu.net.cn/" target="_blank">Yu Lu</a>&emsp;
    <a href="https://z-x-yang.github.io/" target="_blank">Zongxin Yang</a>&emsp;
    <a href="https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=zh-CN&oi=ao" target="_blank">Yi Yang✉</a>&emsp;
</div>
<div>
    ReLER, CCAI, Zhejiang University; Harvard University
</div>
<div>
     <sup>✉</sup>Corresponding Author
</div>
<div>
    <a href="https://arxiv.org/abs/2504.20690" target="_blank">Arxiv</a>&emsp;
    <a href="https://huggingface.co/spaces/RiverZ/ICEdit" target="_blank">Huggingface Demo 🤗</a>&emsp;
    <a href="https://huggingface.co/RiverZ/normal-lora/tree/main" target="_blank">Model 🤗</a>&emsp;
    <a href="https://river-zhang.github.io/ICEdit-gh-pages/" target="_blank">Project Page</a>
</div>


<div style="width: 80%; margin:auto;">
    <img style="width:100%; display: block; margin: auto;" src="docs/images/teaser.png">
    <p style="text-align: left;"><strong>Image Editing is worth a single LoRA!</strong> We present In-Context Edit, a novel approach that achieves state-of-the-art instruction-based editing <b>using just 0.5% of the training data and 1% of the parameters required by prior SOTA methods</b>. The first row illustrates a series of multi-turn edits, executed with high precision, while the second and third rows highlight diverse, visually impressive single-turn editing results from our method.</p>
</div>

:open_book: For more visual results, go checkout our <a href="https://river-zhang.github.io/ICEdit-gh-pages/" target="_blank">project page</a>


<div align="left">


# 🎆 News 
- **[2025/9/19]** 🔥 We have open-sourced our [MoE version ICEdit and ckpt](#for-the-usage-of-moe-lora-version). Have a try!🚀
- **[2025/9/18]** 🌟 ICEdit has been accepted by NeurIPS 2025!🎉 See you in San Diego!
- **[2025/8/21]** 🌟 We have released an [Ascend (Huawei NPU)-powered version of ICEdit](https://github.com/2018liuzhiyuan/ICEdit-on-Ascend-NPU). Now you can run ICEdit on Ascend NPU! Many thanks to [Zhiyuan](https://github.com/2018liuzhiyuan)！
- **[2025/5/16]** 🌟 Many thanks to [gluttony-10 (十字鱼)](https://github.com/River-Zhang/ICEdit/pull/47#issue-3067039788) for adapting Gradio demo with [GGUF quantization](#inference-in-gradio-demo), further reducing memory usage to **10GB**.
- **[2025/5/14]** 🔥 With the help of the [official comfy-org](https://www.comfy.org/zh-cn/), we have integrated our ComfyUI nodes into [Comfy Registry](https://registry.comfy.org/nodes/ICEdit)! 
- **[2025/5/13]** 🔥 We have released the [training code](./train/)! Train your own editing LoRAs now!
- **[2025/5/11]** 🌟 Great thanks to [gluttony-10 (十字鱼)](https://github.com/River-Zhang/ICEdit/issues/23#issue-3050804566) for making a [windows gradio demo](#inference-in-gradio-demo-on-windows) to use our project on Windows!
- **[2025/5/8]** 🔥 We have released our **[official ComfyUI workflow](#official-comfyui-workflow)**! 🚀 Check the repository and have a try!

<details>
<summary><strong>Click to expand/collapse news</strong></summary>

- **[2025/5/8]** 🔥 We have added LoRA scale slider in the gradio demo. You can try to discover more interesting demo with different scale! 
<div align="center">
<img src="docs/images/lora_scale.png" width="70%" style="display: block; margin: auto;">
<div align="left">

- **[2025/5/7]** 🌟 We update some notes when using the ComfyUI workflow to avoid unsatisfactory results! 
- **[2025/5/6]** 🔥 ICEdit currently ranks **2nd** on the overall/weekly trending list of [Hugging Face space](https://huggingface.co/spaces). Thank you all for your support and love!🤗
- **[2025/5/5]** 🌟 Heartfelt thanks to [Datou](https://x.com/Datou) for creating a fantastic [ComfyUI workflow](https://openart.ai/workflows/datou/icedit-moe-lora-flux-fill/QFmaWNKsQo3P5liYz4RB) on OpenArt! 🚀 Have a try!
- **[2025/5/2]** 🌟 Heartfelt thanks to [judian17](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411) for crafting an amazing [ComfyUI-nunchaku demo](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411)! Only **4GB VRAM GPU** is enough to run with ComfyUI-nunchaku!🚀 Dive in and give it a spin!
- **[2025/4/30]** 🔥 We release the [Huggingface Demo](https://huggingface.co/spaces/RiverZ/ICEdit) 🤗! Have a try!
- **[2025/4/30]** 🔥 We release the [paper](https://arxiv.org/abs/2504.20690) on arXiv!
- **[2025/4/29]** We release the [project page](https://river-zhang.github.io/ICEdit-gh-pages/) and demo video! Codes will be made available in next week~ Happy Labor Day!

</details>

# 🎈 Tutorial on Bilibili or Youtube

### 👑 Feel free to share your results in this [Gallery](https://github.com/River-Zhang/ICEdit/discussions/21)!
- **[2025/5/15]** 🌟 We find that [啦啦啦的小黄瓜](https://space.bilibili.com/219572544) has made a detailed [bilibili tutorial](https://www.bilibili.com/video/BV1tSEqzJE7q/?share_source=copy_web&vd_source=8fcb933ee576af56337afc41509fa095) introducing our model! What a great video!
- **[2025/5/14]** 🌟 We find that [Nenly同学](https://space.bilibili.com/1814756990) has made a fantastic [bilibili tutorial](https://www.bilibili.com/video/BV1bNEvzrEn1/?share_source=copy_web&vd_source=8fcb933ee576af56337afc41509fa095) on how to use our repository! Great thanks to him!
- **[2025/5/10]** 🌟 Great thanks to [月下Hugo](https://www.bilibili.com/video/BV1JZVRzuE12/?share_source=copy_web&vd_source=8fcb933ee576af56337afc41509fa095) for making a [Chinese tutorial](https://www.bilibili.com/video/BV1JZVRzuE12/?share_source=copy_web&vd_source=8fcb933ee576af56337afc41509fa095) on how to use our official workflow!
- **[2025/5/7]** 🌟 Heartfelt thanks to [T8star](https://x.com/T8star_Aix) for making a [tutorial](https://www.youtube.com/watch?v=s6GMKL-Jjos) and [ComfyUI workflow](https://www.runninghub.cn/post/1920075398585974786/?utm_source=kol01-RH099) on how to **increase the editing success to 100%**!🚀 Have a try!
- **[2025/5/3]** 🌟 Heartfelt thanks to [softicelee2](https://github.com/softicelee2) for making a [Youtube video](https://youtu.be/rRMc5DE4qMo) on how to use our model!
# 📖 Table of Contents

- [🎆 News](#-news)
- [🎈 Tutorial on Bilibili or Youtube](#-tutorial-on-bilibili-or-youtube)
    - [👑 Feel free to share your results in this Gallery!](#-feel-free-to-share-your-results-in-this-gallery)
- [📖 Table of Contents](#-table-of-contents)
    - [📢 Attention All: Incorrect ComfyUI Workflow Usage Alert!](#-attention-all-incorrect-comfyui-workflow-usage-alert)
- [💼 Installation](#-installation)
  - [Conda environment setup](#conda-environment-setup)
  - [Download pretrained weights](#download-pretrained-weights)
  - [Inference in bash (w/o VLM Inference-time Scaling)](#inference-in-bash-wo-vlm-inference-time-scaling)
      - [For the usage of MoE-LoRA version](#for-the-usage-of-moe-lora-version)
  - [Inference in Gradio Demo](#inference-in-gradio-demo)
  - [💼 Windows one-click package](#-windows-one-click-package)
- [🔧 Training](#-training)
- [🎨ComfyUI Workflow](#comfyui-workflow)
    - [Official ComfyUI-workflow](#official-comfyui-workflow)
    - [ComfyUI-workflow for increased editing success rate](#comfyui-workflow-for-increased-editing-success-rate)
    - [ComfyUI-nunchaku](#comfyui-nunchaku)
    - [ComfyUI-workflow](#comfyui-workflow-1)
- [⚠️ Tips](#️-tips)
    - [If you encounter such a failure case, please **try again with a different seed**!](#if-you-encounter-such-a-failure-case-please-try-again-with-a-different-seed)
    - [⚠️ Clarification](#️-clarification)
- [💪 To Do List](#-to-do-list)
- [💪 Comparison with Commercial Models](#-comparison-with-commercial-models)
- [🌟 Star History](#-star-history)
- [Bibtex](#bibtex)



### 📢 Attention All: Incorrect ComfyUI Workflow Usage Alert!
- ### We have released our **[official ComfyUI workflow](#official-comfyui-workflow)** for proper usage! Check our repository and have a try!
- You need to **add the fixed pre-prompt "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}"** before inputing the edit instructions, otherwise you may get bad results! (This is mentioned in the paper!, The code for the Hugging Face gradio demo already embeds this prompt. So, you can simply input the editing instructions without additional setup.)
- The width of the input image must resize to **512** (no restriction to height).
- Please **[use the Normal LoRA](https://huggingface.co/RiverZ/normal-lora/tree/main)** not the MoE-LoRA, because the MoE-LoRA cannot be correctly loaded with ComfyUI lora loader.
- 🔥💐🎆 Welcome to share your **creative workflows** (such as combining Redux, ACE, etc.) in the Issues section and showcase the results! We will include references so that more people can see your creativity.



# 💼 Installation

## Conda environment setup

```bash
conda create -n icedit python=3.10
conda activate icedit
pip install -r requirements.txt
pip install -U huggingface_hub
```

## Download pretrained weights

If you can connect to Huggingface, you don't need to download the weights. Otherwise, you need to download the weights to local.

- [Flux.1-fill-dev](https://huggingface.co/black-forest-labs/flux.1-fill-dev).
- [ICEdit-normal-LoRA](https://huggingface.co/RiverZ/normal-lora/tree/main).
- [ICEdit-MoE-LoRA](https://huggingface.co/sanaka87/ICEdit-MoE-LoRA/tree/main)

~~Note: Due to some cooperation permission issues, we have to withdraw the weights and codes of moe-lora temporarily. What is released currently is just the ordinary lora, but it still has powerful performance. If you urgently need the moe lora weights of the original text, please email the author.~~

## Inference in bash (w/o VLM Inference-time Scaling)

Now you can have a try!

> Our model can **only edit images with a width of 512 pixels** (there is no restriction on the height). If you pass in an image with a width other than 512 pixels, the model will automatically resize it to 512 pixels.

> If you found the model failed to generate the expected results, please try to change the `--seed` parameter. Inference-time Scaling with VLM can help much to improve the results.

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --seed 304897401 \
```

Editing a 512×768 image requires 35 GB of GPU memory. If you need to run on a system with 24 GB of GPU memory (for example, an NVIDIA RTX3090), you can add the `--enable-model-cpu-offload` parameter.

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --enable-model-cpu-offload
```

If you have downloaded the pretrained weights locally, please pass the parameters during inference, as in: 

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --flux-path /path/to/flux.1-fill-dev \
                            --lora-path /path/to/ICEdit-normal-LoRA
```

#### For the usage of MoE-LoRA version
```bash
python scripts/inference_moe.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --seed 42 \
```

```bash
python scripts/inference_moe.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --enable-model-cpu-offload
```

```bash
python scripts/inference_moe.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --flux-path /path/to/flux.1-fill-dev \
                            --lora-path /path/to/ICEdit-MoE-LoRA
```

## Inference in Gradio Demo

We provide a gradio demo for you to edit images in a more user-friendly way. You can run the following command to start the demo.

```bash
python scripts/gradio_demo.py --port 7860



## for MoE version
python scripts/gradio_demo_moe.py --port 7860

```

Like the inference script, if you want to run the demo on a system with 24 GB of GPU memory, you can add the `--enable-model-cpu-offload` parameter. And if you have downloaded the pretrained weights locally, please pass the parameters during inference, as in:

```bash
python scripts/gradio_demo.py --port 7860 \
                              --flux-path /path/to/flux.1-fill-dev (optional) \
                              --lora-path /path/to/ICEdit-normal-LoRA (optional) \
                              --enable-model-cpu-offload (optional) \

## for MoE version
python scripts/gradio_demo_moe.py --port 7860 \
                              --flux-path /path/to/flux.1-fill-dev (optional) \
                              --lora-path /path/to/ICEdit-normal-LoRA (optional) \
                              --enable-model-cpu-offload (optional) \
```

Or if you want to run the demo on a system with 10 GB of GPU memory, you can download the gguf models from [FLUX.1-Fill-dev-gguf](https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-gguf), [t5-v1_1-xxl-encoder-gguf](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf) and pass the parameters during inference, as in:

```bash
python scripts/gradio_demo.py --port 7861 \
                              --flux-path models/flux.1-fill-dev \
                              --lora-path models/ICEdit-normal-LoRA \
                              --transformer models/flux1-fill-dev-Q4_0.gguf \
                              --text_encoder_2 models/t5-v1_1-xxl-encoder-Q8_0.gguf \
                              --enable-model-cpu-offload \
```

Then you can open the link in your browser to edit images.

<div align="center">
<div style="width: 80%; text-align: left; margin:auto;">
    <img style="width:100%" src="docs/images/gradio.png">
    <p style="text-align: left;">Gradio Demo: just input the instruction and wait for the result!</b>.</p>
</div>

<div align="left">

Here is also a Chinese tutorial [Youtube video](https://www.youtube.com/watch?v=rRMc5DE4qMo) on how to install and use ICEdit, created by [softicelee2](https://github.com/softicelee2). It's definitely worth a watch!

## 💼 Windows one-click package

Great thanks to [gluttony-10](https://github.com/River-Zhang/ICEdit/issues/23#issue-3050804566), a famous [Bilibili Up](https://space.bilibili.com/893892)! He made a tutorial ([Youtube](https://youtu.be/C-OpWlJi424) and [Bilibili](https://www.bilibili.com/video/BV1oT5uzzEbs)) on how to install our project on windows and a one-click package for Windows! **Just unzip it and it's ready to use**. It has undergone quantization processing. It only takes up 14GB of space and supports graphics cards of the 50 series.

Download link: [Google Drive](https://drive.google.com/drive/folders/16j3wQvWjuzCRKnVolszLmhCtc_yOCqcx?usp=sharing) or [Baidu Wangpan](https://www.bilibili.com/video/BV1oT5uzzEbs/?vd_source=2a911c0bc75f6d9b9d056bf0e7410d45)(refer to the comment section of the video)
<img src="docs/images/windows_install.png" width="80%" style="display: block; margin: auto;">


# 🔧 Training

Found more details in here: [Training Code](./train/)


# 🎨ComfyUI Workflow


### Official ComfyUI-workflow
We have released our **official ComfyUI workflow** in this repository for correct usage of our model! **We have embedded the prompt "A diptych with two side-by-side images of the same scene ... but" into our nodes** and you just need to input the edit instructions such as "make the girl wear pink sunglasses". We also add a high resolution refinement module for better image quality! The total VRAM consumption is about 14GB. Use this [workflow](https://github.com/hayd-zju/ICEdit-ComfyUI-official) and the [ICEdit-normal-lora](https://huggingface.co/RiverZ/normal-lora/tree/main) to fulfill your creative ideas!

We have specially created [a repository for the workflow](https://github.com/hayd-zju/ICEdit-ComfyUI-official) and you can **install it directly in ComfyUI**. Just open the manager tab and click **'Install via Git URL'**, copy the following URL and you are able to use it. For more details please refer to this [issue](https://github.com/River-Zhang/ICEdit/issues/22#issuecomment-2864977880)

**URL:** [https://github.com/hayd-zju/ICEdit-ComfyUI-official](https://github.com/hayd-zju/ICEdit-ComfyUI-official)

 <img src="docs/images/workflow_tutorial.png" width="80%" style="display: block; margin: auto;">
 <img src="docs/images/official_workflow.png" width="80%" style="display: block; margin: auto;">

 Great thanks to [月下Hugo](https://www.bilibili.com/video/BV1JZVRzuE12/?share_source=copy_web&vd_source=8fcb933ee576af56337afc41509fa095) for making a [Chinese tutorial](https://www.bilibili.com/video/BV1JZVRzuE12/?share_source=copy_web&vd_source=8fcb933ee576af56337afc41509fa095) on how to use our official workflow!

### ComfyUI-workflow for increased editing success rate
Thanks to [T8star](https://x.com/T8star_Aix)! He made a tutorial ([Youtube](https://www.youtube.com/watch?v=s6GMKL-Jjos) and [bilibili](https://www.bilibili.com/video/BV11HVhz1Eky/?spm_id_from=333.40164.top_right_bar_window_dynamic.content.click&vd_source=2a911c0bc75f6d9b9d056bf0e7410d45)) and a creative workflow ([OpenArt](https://openart.ai/workflows/t8star/icedit100v1/HN4EZ2Cej98ZX8CC1RK5) and [RunningHub](https://www.runninghub.cn/post/1920075398585974786/?utm_source=kol01-RH099)) that could increase the editing success rate greatly (about 100%)! Have a try with it!

<img src="docs/images/workflow_t8.png" width="80%" style="display: block; margin: auto;">


### ComfyUI-nunchaku

We extend our heartfelt thanks to @[judian17](https://github.com/judian17) for crafting a ComfyUI [workflow](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411) that facilitates seamless usage of our model. Explore this excellent [workflow](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411) to effortlessly run our model within ComfyUI. Only **4GB VRAM GPU** is enough to run with ComfyUI-nunchaku! 

This workflow incorporates high-definition refinement, yielding remarkably good results. Moreover, integrating this LoRA with Redux enables outfit changes to a certain degree. Once again, a huge thank you to @[judian17](https://github.com/judian17) for his innovative contributions! 

![comfyui image](docs/images/comfyuiexample.png)


### ComfyUI-workflow

Thanks to [Datou](https://x.com/Datou), a workflow of ICEdit in ComfyUI can also be downloaded [here](https://openart.ai/workflows/datou/icedit-moe-lora-flux-fill/QFmaWNKsQo3P5liYz4RB). Try it with the [normal lora ckpt](https://huggingface.co/RiverZ/normal-lora/tree/main).

<img src="docs/images/workflow.png" width="80%" style="display: block; margin: auto;">






# ⚠️ Tips

### If you encounter such a failure case, please **try again with a different seed**!

- Our base model, FLUX, does not inherently support a wide range of styles, so a large portion of our dataset involves style transfer. As a result, the model **may sometimes inexplicably change your artistic style**.

- Our training dataset is **mostly targeted at realistic images**. For non-realistic images, such as **anime** or **blurry pictures**, the success rate of the editing **drop and could potentially affect the final image quality**.

- While the success rates for adding objects, modifying color attributes, applying style transfer, and changing backgrounds are high, the success rate for object removal is relatively lower due to the low quality of the removal dataset we use.

The current model is the one used in the experiments in the paper, trained with only 4 A800 GPUs (total `batch_size` = 2 x 2 x 4 = 16). In the future, we will enhance the dataset, and do scale-up, finally release a more powerful model.

### ⚠️ Clarification

We've noticed numerous web pages related to ICEdit, including [https://icedit.net/](https://icedit.net/), [https://icedit.org/](https://icedit.org/). Kudos to those who built these pages!

However, we'd like to emphasize two important points:
- **No Commercial Use**: Our project **cannot** be used for commercial purposes. Please check the [LICENSE](https://github.com/River-Zhang/ICEdit/blob/main/LICENSE) for details.
- **Official Page**: The official project page is [https://river-zhang.github.io/ICEdit-gh-pages/](https://river-zhang.github.io/ICEdit-gh-pages/).





# 💪 To Do List

- [x] Inference Code
- [ ] Inference-time Scaling with VLM
- [x] Pretrained Weights
- [x] More Inference Demos
- [x] Gradio demo
- [x] Comfy UI demo (by @[judian17](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411), compatible with [nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku), support high-res refinement and FLUX Redux. Only 4GB VRAM GPU is enough to run!)
- [x] Comfy UI demo with normal lora (by @[Datou](https://openart.ai/workflows/datou/icedit-moe-lora-flux-fill/QFmaWNKsQo3P5liYz4RB) in openart)
- [x] Official ComfyUI workflow
- [x] Training Code
- [ ] LoRA for higher image resolution (768, 1024)



# 💪 Comparison with Commercial Models

<div align="center">
<div style="width: 80%; text-align: left; margin:auto;">
    <img style="width:100%" src="docs/images/gpt4o_comparison.png">
    <p style="text-align: left;">Compared with commercial models such as Gemini and GPT-4o, our methods are comparable to and even superior to these commercial models in terms of character ID preservation and instruction following. <b>We are more open-source than them, with lower costs, faster speed (it takes about 9 seconds to process one image), and powerful performance</b>.</p>
</div>


<div align="left">


# 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=River-Zhang/ICEdit&type=Date)](https://www.star-history.com/#River-Zhang/ICEdit&Date)

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{zhang2025context,
  title={In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer},
  author={Zhang, Zechuan and Xie, Ji and Lu, Yu and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2504.20690},
  year={2025}
}
```

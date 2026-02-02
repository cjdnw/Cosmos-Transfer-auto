# 说明文档
本项目基于 [Segment-Anything-2](https://github.com/facebookresearch/segment-anything-2) 和 [Segment-Anything-1](https://github.com/zrporz/segment-anything-1) 实现视频的全自动分割，可以跟踪每个对象并检测可能出现的新对象。本版本是在 <https://github.com/zrporz/AutoSeg-SAM2/issues/5> 中讨论的改进基础上调整完善的版本。


## 演示

https://github.com/user-attachments/assets/12fc2c3f-b915-4f58-a492-fc74bbc31e52


https://github.com/user-attachments/assets/f080aab6-12a6-461b-8618-cc25bcf67a9b


https://github.com/user-attachments/assets/ef45b7ee-4d89-4096-87d7-24c273f9dc6e


## 环境配置
首先克隆仓库及其子模块：
```bash
# SSH
git clone git@github.com:zrporz/AutoSeg-SAM2.git --recursive
```
或
```bash
# HTTPS
git clone https://github.com/zrporz/AutoSeg-SAM2.git --recursive
```

本项目需要 `python>=3.10`，并依赖 `torch>=2.3.1` 和 `torchvision>=0.18.1`。
我们使用 [SAM1](https://github.com/zrporz/segment-anything-1) 提供静态分割结果，并使用 [SAM2](https://github.com/facebookresearch/segment-anything-2) 跟踪这些结果。可以通过以下命令安装：
```bash
### 安装 sam1 与 sam2 模块
pip install -e submodule/segment-anything-1
pip install -e submoudle/segment-anything-2
### 下载权重
cd checkpoints/sam1
bash download.sh
cd ../sam2
bash download.sh
```

## 数据准备
请按以下结构整理视频帧：
```
|-<video dir>
    |-000001.jpg
    |-000002.jpg
    |-000003.jpg
    |-000004.jpg
    ...
```

或者使用示例数据集 [chickenchicken](https://drive.google.com/file/d/10K_ePOMTjgQhWo1EXbM5aX7IU1EKvsrv/view?usp=sharing)，将其放在 `videos/chickenchicken` 下，然后运行：
```bash
bash scripts/chickenchicken.sh
```

## 快速开始
假设你只需要 `large` 版本的掩码，并希望加速推理（减少保存中间结果、适当调大重新检测间隔），可以运行：
```bash
# 单行示例（复制即用）
python auto-mask-batch.py --video_path videos/chickenchicken --output_dir outputs/chickenchicken --level large --use_other_level 0 --detect_stride 15 --save_intermediate 0 --save_npy 0

# 多行示例（确保每行末尾的反斜杠 \ 后面没有空格或注释）
python auto-mask-batch.py \
  --video_path videos/chickenchicken \
  --output_dir outputs/chickenchicken \
  --level large \
  --use_other_level 0 \
  --detect_stride 15 \
  --save_intermediate 0 \
  --save_npy 0
```
如需兼顾召回率，可保持 `--use_other_level 1`，或降低 `--detect_stride`；如需排查效果，可将 `--save_intermediate` 设为 1 以保存中间可视化。

### 获得最终可视化 MP4
分割完成后运行可视化脚本即可生成视频。继续以 `large` 为例：
```bash
python visulization.py --video_path videos/chickenchicken --output_dir outputs/chickenchicken --level large
```
- 若 `--save_npy 0`（不保存 npy），脚本会自动使用 `final-output/mask_*.png` 进行可视化，无需额外处理。
- 可选控制：
  - `--enable_highlight 0` 仅生成彩色叠加，不导出“单对象高亮+掩码对照”帧与 mp4。
  - `--enable_overlay 0` 仅生成单对象高亮/掩码帧与 mp4，不生成彩色叠加视频。
  - 想要“纯彩色分割结果”可用 `--enable_highlight 0 --enable_overlay 1`。
  - `--overlay_style mask` 生成“纯彩色块”叠加（不包含原始画面）；默认 `blend` 会在原图上半透明叠加。
- 所有掩码叠加的总览视频：`outputs/chickenchicken/large/visualization/full-mask-large.mp4`。
- 每个对象的单独视频：`outputs/chickenchicken/large/visualization/seperate/videos/uid_<编号>.mp4`。
（其中逐帧 PNG 会保存到 `visualization/full-mask-npy` 和 `visualization/seperate/uid_<编号>` 供检查。）

## 引用
```bibtex
@software{AutoSeg_SAM2,
  author = {Zrporz},
  title = {AutoSeg-SAM2},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/zrporz/AutoSeg-SAM2},
  version = {Latest}, 
  license = {MIT}, 
  note = {Automated image segmentation tool based on Segment Anything Model (SAM)}
}
```

# robot_arm_swap：Qwen3-VL 首帧框 + SAM3 视频跟踪的机械臂换肤方案

目标：**用 Qwen3-VL 在首帧自动检测机械臂框 → 用 SAM3 进行视频级 mask 跟踪 → 将原始视频机械臂贴回换肤视频**，输出最终 `output_swap.mp4`。

本工程已完全重构为 **Qwen3-VL + SAM3** 流程，不再依赖 GroundingDINO/SAM2。

---

## 依赖与环境说明
- Python 3.10+
- NVIDIA GPU（推荐）+ CUDA 可用的 PyTorch
- `ffmpeg`（命令行可用：`ffmpeg -version`）
- 本项目依赖：`pip install -r requirements.txt`

### 合并后的依赖清单（Qwen3-VL + SAM3）
已在 `requirements.txt` 中合并：
- transformers>=4.57.0（Qwen3-VL 依赖；SAM3 建议主分支）
- torchvision
- pillow / opencv-python / numpy
- accelerate
- huggingface_hub<1.0
- 其余基础依赖（tqdm / PyYAML）

### 环境安装教程（综合版）
建议使用 Conda：
```bash
conda create -n robot_arm_swap python=3.10 -y
conda activate robot_arm_swap
```

#### 1) 安装 PyTorch + torchvision + torchaudio（必须）
务必按你的 CUDA 版本安装匹配的 PyTorch 组合。  
官方指引：https://pytorch.org/get-started/locally/

示例（CUDA 12.1）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2) 安装 Transformers（SAM3 需要主分支）
SAM3 在 Transformers 上线初期可能尚未进入某个固定 pip 版本，建议安装主分支：
```bash
pip install git+https://github.com/huggingface/transformers
```

如需离线安装（网络慢时）：
```bash
# 在可联网机器下载 main.zip： https://github.com/huggingface/transformers/archive/refs/heads/main.zip
# 上传到服务器后：
unzip transformers-main.zip
cd transformers-main
pip install .
cd ..
```

#### 3) 安装其余依赖
```bash
pip install -r requirements.txt
```

#### 4) Hugging Face 权限（SAM3 模型）
```bash
pip install "huggingface_hub[cli]"
hf auth login
```

**Qwen3-VL 和 SAM3 的封装类来自外部路径**，通过 `robot_arm_swap/config.py` 统一配置。

默认路径（可修改）：
```python
QWEN3_VL_WRAPPER_DIR = "/mnt/data/ganrunhe/robot_arm_swap_project"
SAM3_VIDEO_TRACKER_DIR = "/mnt/data/ganrunhe/robot_arm_swap_project"
```

封装类文件名固定：
- Qwen3-VL：`get_bbox.py`（类名 `QwenVL3`）
- SAM3：`sam3_video_tracker.py`（类名 `SAM3VideoTracker`）

> 你可以直接修改 `robot_arm_swap/config.py`，或通过环境变量覆盖：  
> `QWEN3_VL_WRAPPER_DIR=/your/path`  
> `SAM3_VIDEO_TRACKER_DIR=/your/path`

---

## 一键全流程（自动检测 + 跟踪 + 合成）
假设：
- `assets/original.mp4`：原始视频
- `assets/reskinned.mp4`：换肤视频

运行：
```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --skin assets/reskinned.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --qwen-prompt "robotic arm" \
  --out output_swap.mp4
```

如需禁用中间落盘（仅使用内存处理，避免写入中间 PNG）：
```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --skin assets/reskinned.mp4 \
  --no-workdir-write \
  --fps 30 \
  --width 1280 \
  --qwen-prompt "robotic arm" \
  --out output_swap.mp4
```

---

## 仅调用 Qwen3 / 仅调用 SAM3 / 仅合成 Swap

如果你只想做首帧检测（不跟踪、不合成），可以使用 `--qwen-only` 输出 bbox：
```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --qwen-prompt "robotic arm" \
  --qwen-only
```
产物：`work/qwen_boxes.json`（或通过 `--bbox-json` 指定输出路径）。

如果你只有 bbox，想单独做 SAM3 跟踪并生成 mask，可以使用 `--sam-only --bbox-json`：
```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --sam-only \
  --bbox-json work/qwen_boxes.json
```
产物：`work/sam3_track`（union mask、label mask、预览视频等）。

如需额外导出 mask 文件，可选传入下列参数（**不传则不生成对应文件**）：
```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --sam-only \
  --bbox-json work/qwen_boxes.json \
  --mask-coco-parquet outputs/masks.parquet \
  --mask-lossless-mp4 outputs/mask_lossless.mp4 \
  --mask-lossless-frames-dir outputs/mask_frames
```
说明：
- `--mask-coco-parquet`：将每帧 mask 的 COCO 边界点（JSON 数组字符串，仅 polygons 内容）写入 parquet 的 `data` 列；若文件已存在则追加新列（`data_1`、`data_2`...）。
- `--mask-lossless-mp4`：输出无损 mp4；`--mask-lossless-frames-dir` 用于保存对应的 PNG 序列（与 mp4 同目录即可）。

如果你已有原始视频的 mask，只想直接合成（跳过 Qwen 检测），可以使用 `--swap-only --arm-mask-dir`（同样支持 `--erode/--feather/--out` 等合成参数）：
```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --skin assets/reskinned.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --swap-only \
  --arm-mask-dir work/masks_arm \
  --erode 2 \
  --feather 3 \
  --out output_swap.mp4
```

如需在 swap-only 模式下额外追踪 skin 视频的 mask 并计算 IoU，可加上 `--arm-swap-workdir-write`：
```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --skin assets/reskinned.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --swap-only \
  --arm-mask-dir work/masks_arm \
  --arm-swap-workdir-write \
  --out output_swap.mp4
```

示例（去灰边：侵蚀后轻微羽化）：
```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --skin assets/reskinned.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --qwen-prompt "robotic arm" \
  --erode 2 \
  --feather 3 \
  --out output_swap.mp4
```

---

## 工作逻辑（整体流程）
> 对应入口：`python -m robot_arm_swap`（内部调用 `robot_arm_swap/run_pipeline.py`）

1) **输入读取**  
   - 默认落盘模式：先用 `ffmpeg` 抽帧，生成 `work/frames_orig` 与 `work/frames_skin`。  
   - 内存模式（`--no-workdir-write`）：直接用 OpenCV 读取视频帧到内存，不写 PNG；若读取失败（如 AV1），会临时转码为 H.264 + yuv420p 后重试，完成后自动删除临时文件。

2) **Qwen3-VL 首帧检测**  
   - 在首帧上运行 Qwen3-VL，得到 `bbox_2d`（像素坐标）。  
   - 默认会写出 `work/qwen_boxes.json`；若使用内存模式且未指定 `--bbox-json`，则不落盘，仅在内存传递。

3) **SAM3 视频跟踪**  
   - 使用 Qwen3-VL 的首帧 bbox 作为提示，进行全视频 mask 传播。  
   - 支持 `--sam3-auto-scale` 自动下采样，避免显存不足；`--sam3-preprocess-device` 可把预处理转到 CPU。  
   - 落盘模式会写出 `work/sam3_track` 目录（见“中间产物”）。

4) **合成（机械臂贴回）**  
   - 根据 mask 将原始机械臂区域贴回换肤视频帧，支持 `--dilate/--erode/--feather` 调整边缘。  
   - 落盘模式会生成 `work/out_frames` 帧序列。

5) **编码输出**  
   - 通过 `ffmpeg` 将合成帧编码为 `output_swap.mp4`，并从换肤视频复制音频轨（若存在）。

> 可选：`--qwen-only` 只输出 bbox；`--sam-only --bbox-json` 只做 SAM3 mask 跟踪；`--swap-only --arm-mask-dir` 只做 swap 合成。

---

## Quickstart：三种常见合成方案（A/B/C）

下面三种方案对应你描述的现象，便于快速对比：

### A) 侵蚀后轻微羽化（推荐）
**效果：**边缘更干净、灰边更少，通常最好。  
**原因：**侵蚀先把 mask 往里收，避开污染像素，再轻微羽化做过渡。

```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --skin assets/reskinned.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --qwen-prompt "robotic arm" \
  --erode 2 \
  --feather 3 \
  --dilate 0 \
  --out output_swap.mp4
```

### B) 什么都不做（硬边）
**效果：**容易出现白边或锯齿。  
**原因：**没有羽化/形态学处理，mask 边缘是硬切，且 SAM 的边界可能包含亮边或背景残留。

```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --skin assets/reskinned.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --qwen-prompt "robotic arm" \
  --erode 0 \
  --feather 0 \
  --dilate 0 \
  --out output_swap.mp4
```

### C) 默认羽化（无侵蚀）
**效果：**容易出现灰边（半透明“灰圈”），常见于边缘污染。  
**原因：**直接对包含背景污染的边缘做羽化，会把背景颜色混进来形成灰色过渡带。

```bash
python -m robot_arm_swap \
  --orig assets/original.mp4 \
  --skin assets/reskinned.mp4 \
  --workdir work \
  --fps 30 \
  --width 1280 \
  --qwen-prompt "robotic arm" \
  --dilate 2 \
  --feather 9 \
  --erode 0 \
  --out output_swap.mp4
```

> 说明：`--dilate/--erode/--feather` 只影响合成阶段的 mask 后处理，不影响 Qwen3-VL 或 SAM3 本身。

产物：
- `work/qwen_boxes.json`：Qwen3-VL 生成的首帧 bbox
- `work/sam3_track/mask/mask_union_0000.png`：SAM3 追踪出的 union mask（原始输出）
- `work/masks_arm/000001.png`：用于合成的标准 mask 序列
- `output_swap.mp4`：最终换肤结果（原始机械臂贴回）

---

## 中间产物（落盘模式）
> 仅在默认落盘模式（未传 `--no-workdir-write`）下生成。

### 1) 抽帧
- `work/frames_orig/000001.jpg`：原始视频抽帧（统一 fps/宽度）
- `work/frames_skin/000001.jpg`：换肤视频抽帧

### 2) Qwen3-VL 输出
- `work/qwen_boxes.json`：首帧 bbox 列表，字段示例：`bbox_2d: [x1, y1, x2, y2]`

### 3) SAM3 输出（`work/sam3_track`）
- `work/sam3_track/mask/mask_union_0000.png`：union mask（所有对象合并）
- `work/sam3_track/mask/mask_label_0000.png`：label mask（按 obj_id 编码）
- `work/sam3_track/masked_rgb/masked_rgb_0000.png`：保留 mask 区域的 RGB 结果
- `work/sam3_track/mask_union.mp4`：union mask 预览（需 `--sam3-save-mp4`）
- `work/sam3_track/masked_rgb.mp4`：masked rgb 预览（需 `--sam3-save-mp4`）
- （可选）`work/sam3_track/mask/mask_obj{ID}_0000.png`：单目标 mask（若在 SAM3 侧启用）

### 4) 合成输入/输出
- `work/masks_arm/000001.png`：由 `mask_union_*.png` 规范化得到的 mask 序列
- `work/out_frames/000001.png`：合成后的逐帧输出

### 5) 最终视频
- `output_swap.mp4`：合成结果，音频来自换肤视频（若存在）

---

## 中间产物（内存模式）
> 传 `--no-workdir-write` 后，大部分中间文件不落盘：
- 不会生成 `work/frames_*`、`work/masks_arm`、`work/out_frames`。
- 若未指定 `--bbox-json`，也不会写 `work/qwen_boxes.json`。
- 仍会输出最终视频 `output_swap.mp4`。

---

## 关键参数
`python -m robot_arm_swap --help` 可查看完整参数。常用项：

- `--qwen-wrapper-dir`：Qwen3-VL 封装路径（默认来自 config）
- `--sam3-wrapper-dir`：SAM3 封装路径（默认来自 config）
- `--qwen-model-path`：Qwen3-VL 模型名或本地路径（默认 `Qwen/Qwen3-VL-32B-Instruct`）
- `--qwen-prompt`：检测文本（建议：`"robotic arm"` 或更具体描述）
- `--sam3-model-name`：SAM3 模型名（默认 `facebook/sam3`）
- `--sam3-scale`：SAM3 输入降采样（1 为原尺寸）
- `--sam3-mask-threshold`：SAM3 mask 二值阈值
- `--sam3-save-mp4`：输出 SAM3 预览视频（union mask + masked_rgb）
- `--no-workdir-write` / `--in-memory`：禁用中间帧与 mask 的 PNG 落盘，改为内存传递/管道编码
- `--sam-only`：仅做 SAM3 mask 跟踪（需 `--bbox-json`）
- `--swap-only`：仅做 swap 合成（需 `--arm-mask-dir` + `--skin`）
- `--arm-mask-dir`：swap-only 使用的原始视频 mask 目录
- `--arm-swap-workdir-write`：swap-only 下额外追踪 skin mask 并计算 IoU（会写入 workdir）
- `--mask-coco-parquet`：输出 COCO 边界点（仅 polygons 内容）到 parquet（sam-only 可选）
- `--mask-lossless-mp4`：输出无损 mask mp4（sam-only 可选）
- `--mask-lossless-frames-dir`：输出无损 mask 的 PNG 序列（sam-only 可选）

---

## 工作目录结构
```
work/
  frames_orig/       # 原始视频抽帧
  frames_skin/       # 换肤视频抽帧
  qwen_boxes.json    # 首帧 bbox 结果
  sam3_track/        # SAM3 输出（含 union mask）
  masks_arm/         # 统一命名后的 mask 序列
  out_frames/        # 合成后的逐帧结果
```

---

## 常见问题
**1) Qwen3-VL 检测不到机械臂？**  
尝试更具体的 prompt，例如：  
`"industrial robotic arm"` / `"robotic manipulator arm"`

**2) SAM3 mask 漂移？**  
适当提高 `--sam3-scale`（例如 1 → 2）可减小噪声，也可用更精确的 Qwen prompt。

**3) 输出合成对不齐？**  
确保 `original.mp4` 与 `reskinned.mp4` 同视角、同动作节奏，`extract_frames` 只做裁剪/缩放/定帧率，不做时序配准。

**4) 报错 “No frames read from ...” 或 AV1 解码失败？**  
说明输入视频可能是 AV1 编码，OpenCV/FFmpeg 在当前平台无法硬解/解码。最快的解决办法：把 `orig` 转成 H.264 + yuv420p（保持原始分辨率与 fps）。  
工程也已增加鲁棒性：内存模式读取失败时会在工程根目录临时转码为 H.264，然后再读取，完成后自动删除临时文件。

---

## 更新记录
- 2026-01-08：新增 `--no-workdir-write/--in-memory`，支持内存模式避免中间 PNG 落盘并直接管道编码输出视频；内存模式读取视频失败时自动临时转码为 H.264 + yuv420p 后重试。
- 2026-01-05：合成阶段新增 `--erode`，支持侵蚀后轻微羽化以减少边缘灰圈。
- 2026-01-04：新增 SAM3 自动降采样机制（`--sam3-auto-scale/--no-sam3-auto-scale`），根据可用 GPU 显存自动调整 `--sam3-scale` 以降低 OOM 风险；支持 `--sam3-preprocess-device` 将归一化放在 CPU 以减少显存峰值。
- 2025-XX-XX：完全重构为 **Qwen3-VL 首帧检测 + SAM3 视频 mask 跟踪** 的新流程，并在 `config.py` 中支持外部封装路径配置。

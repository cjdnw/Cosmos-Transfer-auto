# Practical RIFE（PyTorch）

本项目基于 Practical RIFE 的推理脚本，提供图像与视频的插帧能力，并在此基础上新增了单次调用的“非均匀补帧”接口，方便根据原始采样帧的时间分布一次性生成完整输出序列。

## 主要功能

- 图像/视频插帧（基于现有推理脚本）。
- Guided Non-Uniform Interpolation：对非均匀采样帧进行单次调用的补帧与拼接。

## 新增接口：Guided Non-Uniform Interpolation

新增模块 `guided_interpolation.py`，提供单次调用接口 `guided_non_uniform_interpolation`。

### 输入

- `frames`：按时间顺序排列的原始采样帧（Tensor）。
- `frame_map` 或 `segment_frames`：
  - `frame_map`：原始视频中采样帧的索引，如 `[0, 3, 7, 8, 13]`。
  - `segment_frames`：相邻采样帧之间应输出的帧数，如 `[4, 5, 2, 6]`。
- `time_steps`（可选）：显式指定每一段的中间帧时间点 `t`（0~1 之间，不含 0/1）。

其中：

```
segment_frames[i] = frame_map[i + 1] - frame_map[i] + 1
```

### 输出

一次调用内生成完整输出序列；相邻段拼接时自动去重首帧，保证总帧数正确。

### 使用示例（推荐写法）

```python
from guided_interpolation import guided_non_uniform_interpolation

# frames: [I0, I1, I2, ...] (Tensor，按时间顺序)
# frame_map: [0, 3, 7, 8]   (原始视频中的采样索引)
output_frames = guided_non_uniform_interpolation(
    model=model,
    frames=frames,
    frame_map=frame_map,   # 或者直接传 segment_frames
    scale=1.0,
)
```

### 使用示例（显式 time_steps）

```python
from guided_interpolation import guided_non_uniform_interpolation

# segment_frames: [5] -> 需要 3 张中间帧
output_frames = guided_non_uniform_interpolation(
    model=model,
    frames=[I0, I1],
    segment_frames=[5],
    time_steps=[[0.1, 0.5, 0.9]],
    scale=1.0,
)
```

### 使用示例（直接传 segment_frames）

```python
from guided_interpolation import guided_non_uniform_interpolation

# frames: [I0, I1, I2, ...]
# segment_frames: [4, 5, 2] 代表每一段输出总帧数（含首尾帧）
output_frames = guided_non_uniform_interpolation(
    model=model,
    frames=frames,
    segment_frames=segment_frames,
    scale=1.0,
)
```

## 均匀补帧（标准推理脚本）

工程内 `inference_video.py` / `inference_img.py` 提供均匀补帧的命令行调用方式。

### 视频均匀补帧

通过 `--exp` 或 `--multi` 指定倍数（`--exp` 优先，最终倍率为 `2 ** exp`）：

```bash
# 2^2 = 4 倍补帧输出
python inference_video.py --video input.mp4 --exp 2

# 直接指定倍数（等价于 --exp 1 对应 2 倍）
python inference_video.py --video input.mp4 --multi 2
```

可选参数：

- `--fps`：显式指定输出 FPS（否则按倍率自动计算）。
- `--output`：输出文件名。
- `--scale`：推理缩放（4K 建议 `0.5`）。

### 图像均匀补帧

对一对图像均匀补帧，`--exp` 控制输出帧数倍数：

```bash
# 输出 2^4 = 16 等分的序列帧
python inference_img.py --img frame0.png frame1.png --exp 4
```

输出会保存在 `output/` 目录。

## Guided Non-Uniform Interpolation 调用要点

- 适用于非均匀采样帧的单次补帧与拼接。
- 可传 `frame_map`（原始时间索引）或直接传 `segment_frames`（相邻采样帧应输出总帧数）。
- 可选传 `time_steps` 显式指定每段中间帧时间点；不传则保持自动均匀生成。
- 函数会自动拼接相邻段并去重首帧，保证总帧数正确。

## Guided Non-Uniform Interpolation CLI

新增 `guided_interpolation_cli.py`，支持 JSON 形式传入 `time_steps`：

```bash
python guided_interpolation_cli.py \
  --imgs frame0.png frame1.png \
  --segment_frames 5 \
  --time_steps_json '[[0.1, 0.5, 0.9]]' \
  --output output_guided
```

## 工程文件改动

- 新增 `guided_interpolation.py`：实现 Guided Non-Uniform Interpolation 单次调用接口。
- 新增 `guided_interpolation_cli.py`：支持命令行传入 `time_steps` 进行非均匀补帧。
- 更新 `README.md`：补充中文说明与接口使用方式。

## 更新记录

- 2026-01-13：扩展 `guided_non_uniform_interpolation` 支持 `time_steps`，并新增 CLI 与单元测试示例。

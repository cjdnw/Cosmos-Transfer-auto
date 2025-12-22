# Cosmos-Transfer2.5 全自动批处理工程

面向 Cosmos-Transfer2.5 的批处理流水线，将原本手动执行的 **Step2~Step5** 串成可批量、多卡并行的自动化流程，并针对 **depth 控制信号** 做缓存复用（同一输入视频仅生成一次 depth 控制信号）。

## 自动化前的手动流程（现已自动化）
1. 提供原始 mp4 路径（批量）
2. 生成随机纹理替换的 prompt（AutoPrompt，支持 Qwen3/Qwen2.5）
3. 生成控制信号（目前实现 depth；保留 edge/blur/seg/combo 接口）
4. 生成对应的 json spec
5. 调用 Cosmos-Transfer2.5 推理，输出最终替换版本

## 依赖与环境
- `conda` 命令可用，且存在环境：
  - `qwen25vl`（默认）：用于 `AutoPrompt/robot_video_to_prompt.py`，支持 `--model_type Qwen3|Qwen2.5`
  - `depthanything`：用于 `Video-Depth-Anything/run.py`
- 若指定的 `--qwen_env` 不存在，运行时会自动回退到 `qwen25vl` 并在日志中打印警告。
- Cosmos-Transfer2.5 使用 `uv` 创建的 `.venv`：`<transfer_root>/.venv/bin/python`
- 推理子进程会自动设置 HuggingFace 缓存变量：`HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`

## 工程结构
- `scripts/run_batch.py`：主入口（命令行运行）
- `auto_transfer_runner/cli.py`：参数解析、多卡调度
- `auto_transfer_runner/pipeline.py`：单个视频的完整流水线（Step2~Step5）
- `auto_transfer_runner/steps/`：各步骤实现
- `auto_transfer_runner/utils/`：通用工具（日志、锁、指纹、路径等）

## 快速开始
### 例 1：单卡跑全部
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth
```

### 例 2：4 卡并行（每张卡完成完整 2→5 流程）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --num_videos 100 \
  --gpus 0,1,2,3 \
  --control_type depth
```

### 例 3：指定 Qwen 模型族（默认 Qwen2.5，可切换为 Qwen3）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --prompt_model_type Qwen3 \
  --prompt_size 4B \
  --prompt_variant instruct \
  --qwen_env qwen3vl
```

## 输出目录（默认 work_dir=/mnt/data/ganrunhe/auto_transfer）
- `prompt/`：每个视频（每个 job）独立的 prompt
- `prompt_cache/`：按视频指纹缓存的 Step1 base prompt（跨任务复用）
- `control_cache/`：depth 控制信号缓存（跨 job 复用）
- `control/`：非缓存控制类型（预留，未来 edge/seg/... 可用）
- `json/`：每个视频独立的 json spec
- `output_video/`：Transfer2.5 输出目录
- `logs/`：每个视频一个日志文件（追加写入）

## “视频指纹”与缓存
- prompt 基础描述缓存：`prompt_cache/<fingerprint>/base.txt`，命中后跳过 Step1；Step2/Step3 的 bracket/replacements 为临时文件，生成最终 prompt 后立即删除。
- depth 缓存目录按 fingerprint 划分：`control_cache/depth/<fingerprint>/`
- fingerprint 默认使用 `stat`（路径 + 文件大小 + 修改时间），速度快
- 如需更稳健（读完整文件算 hash），使用：`--fingerprint_mode md5`

## 扩展接口（edge/blur/seg/combo）
在 `auto_transfer_runner/steps/control.py` 中补充对应命令即可；调度、命名、并行、日志与缓存框架保持不变。

## 查看所有命令行参数
```bash
python scripts/run_batch.py --help
```

## 控制台打印每一步（可选）
默认每个 job 的详细输出写入 `logs/*.log`。如需在终端实时看到每一步开始/结束与缓存命中情况：
```bash
python scripts/run_batch.py ... --console
```
说明：控制台行现在会带上 GPU 标签（例如 `[GPU0]`），方便分卡观察。

## 每个视频重复生成 K 次（生成 K 个不同输出）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0,1,2,3 \
  --control_type depth \
  --variants_per_video 5 \
  --console
```

说明：
- 总任务数 = 选中的输入视频数量 * K
- depth 控制信号会被缓存复用：同一输入视频的首次生成 depth，其余 K-1 次直接命中缓存
- 每个 variant 都会生成不同的 prompt、json spec、输出目录、日志文件

## manifest.jsonl（输入-输出映射表）
批处理会在 `<work_dir>/manifest.jsonl` 里追加写入每个任务的映射信息（JSON Lines）：
- input_video / variant_idx / job_id
- prompt_path / json_path / output_dir / log_path
- control_vis_path（depth 控制信号最终引用的 *_vis.mp4）

## Transfer2.5 默认路径（已更新）
默认 `--transfer_root`：`/mnt/data/ganrunhe/cosmos-transfer2.5`

本地路径不同可通过命令行覆盖：
```bash
python scripts/run_batch.py ... --transfer_root /your/path/to/cosmos-transfer2.5
```

## AutoPrompt 默认路径
默认 `--autoprompt_root` 指向本仓库根目录（包含 `robot_video_to_prompt.py`），无需额外克隆独立的 AutoPrompt 仓库。

## 成功生成后的自动清理
默认行为：**每个任务成功完成推理后**，自动清理中间文件：
- prompt 文件（`prompt/*.txt`）
- json spec（`json/*.json`）
- 非缓存 control 目录（`control/<type>/<job_id>/`，主要为未来 edge/seg/... 预留）
不会清理：
- depth 控制缓存（`control_cache/`）
- 最终输出（`output_video/`）
- `manifest.jsonl`

关闭清理：
```bash
python scripts/run_batch.py ... --no_cleanup
```

保留 prompt/json：
```bash
python scripts/run_batch.py ... --keep_prompt --keep_json
```

清理日志（不推荐，除非确认不需要排错日志）：
```bash
python scripts/run_batch.py ... --cleanup_logs
```

## 更新记录
- 2025-12-22：Prompt 生成流程适配拆分版 `robot_video_to_prompt.py`（step1~step3），新增基于视频指纹的 base prompt 缓存（`prompt_cache/`），默认使用本仓库内的脚本路径；控制台打印增加 GPU 标签。
- 2025-12-22：为终端输出统一添加东八区时间戳，并在 Step2~Step5（含清理）及整条流水线上记录耗时，便于观测性能。
- 2025-12-21：适配新版 `robot_video_to_prompt.py`（支持 Qwen3/Qwen2.5 的 `--model_type` 参数），新增命令行参数 `--prompt_model_type`，默认 conda 环境保持 `qwen25vl`，可选切换为 `qwen3vl`。描述改为工程化表述，匹配顶会论文风格。

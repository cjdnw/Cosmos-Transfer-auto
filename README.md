# Cosmos-Transfer2.5 全自动批处理工程

本工程用于将你目前手动执行的 **Step2~Step5** 串成一个可批量、多卡并行的自动化流水线，并针对 **depth 控制信号** 做缓存复用（同一输入视频只生成一次 depth 控制信号）。

## 你现有的手动流程（已自动化）
1. 输入原始 mp4 路径（批量）
2. 生成随机纹理替换的 prompt（Qwen / AutoPrompt）
3. 生成控制信号（目前实现 depth；保留 edge/blur/seg/combo 接口）
4. 生成对应的 json spec
5. 调用 Cosmos-Transfer2.5 推理，输出最终替换版本

## 依赖假设
- `conda` 命令可用，且存在环境：
  - `qwen25vl`：用于 `AutoPrompt/robot_video_to_prompt.py`
  - `depthanything`：用于 `Video-Depth-Anything/run.py`
- Cosmos-Transfer2.5 使用 `uv` 创建的 `.venv`，并且存在：
  - `<transfer_root>/.venv/bin/python`
- HuggingFace 缓存变量会在推理子进程中自动设置：
  - `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`

## 工程结构
- `scripts/run_batch.py`：主入口（命令行运行这个）
- `auto_transfer_runner/cli.py`：参数解析、多卡调度
- `auto_transfer_runner/pipeline.py`：单个视频的完整流水线（Step2~Step5）
- `auto_transfer_runner/steps/`：每一步一个文件（可读性高、易维护）
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

## 输出目录（默认 work_dir=/mnt/data/ganrunhe/auto_transfer）
- `prompt/`：每个视频（每个 job）独立的 prompt
- `control_cache/`：depth 控制信号缓存（跨 job 复用）
- `control/`：非缓存控制类型（预留，未来 edge/seg/... 可用）
- `json/`：每个视频独立的 json spec
- `output_video/`：Transfer2.5 输出目录
- `logs/`：每个视频一个日志文件（追加写入）

## Prompt 模型选择
- 默认使用 Qwen2.5-VL：`--prompt_model_family qwen25vl --prompt_size 7B --prompt_variant instruct`
- 如需 Qwen3-VL（例如 8B）：`--prompt_model_family qwen3vl --prompt_size 8B --prompt_variant instruct`

## 关于“视频指纹”和缓存
- depth 缓存目录按 fingerprint 划分：`control_cache/depth/<fingerprint>/`
- fingerprint 默认用 `stat`（路径 + 文件大小 + 修改时间），非常快
- 如需最稳（读完整文件算 hash），使用：`--fingerprint_mode md5`

## 后续扩展（edge/blur/seg/combo）
你只需要在 `auto_transfer_runner/steps/control.py` 中补上对应命令即可；
其他调度、命名、并行、日志、缓存框架不需要改。


## 查看所有命令行参数（--help）
```bash
python scripts/run_batch.py --help
```

## 控制台打印每一步（可选）
默认每个 job 的详细输出写入 `logs/*.log`。如需在终端实时看到每一步开始/结束与缓存命中情况：
```bash
python scripts/run_batch.py ... --console
```


## 每个视频重复生成 K 次（生成 K 个不同输出）
如果你希望对每个输入视频生成多个不同版本（每次重新生成 prompt + 再推理一次）：
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
- depth 控制信号会被缓存复用：同一个输入视频的第 1 次会生成 depth，其余 K-1 次直接命中缓存
- 每个 variant 都会生成不同的：
  - prompt 文件
  - json spec
  - 输出目录
  - 日志文件

## manifest.jsonl（输入-输出映射表）
批处理会在 `<work_dir>/manifest.jsonl` 里追加写入每个任务的映射信息（JSON Lines）：
- input_video / variant_idx / job_id
- prompt_path / json_path / output_dir / log_path
- control_vis_path（depth 控制信号最终引用的 *_vis.mp4）


## Transfer2.5 默认路径（已更新）
默认 `--transfer_root` 已更新为：
- `/mnt/data/ganrunhe/cosmos-transfer2.5`

如你本机路径不同，可通过命令行覆盖：
```bash
python scripts/run_batch.py ... --transfer_root /your/path/to/cosmos-transfer2.5
```

## 生成完成后自动清理中间文件
默认行为：**每个任务成功完成推理后**，会自动清理该任务的中间文件：
- prompt 文件（`prompt/*.txt`）
- json spec（`json/*.json`）
- 非缓存 control 目录（`control/<type>/<job_id>/`，主要为未来 edge/seg/... 预留）
不会清理：
- depth 控制缓存（`control_cache/`）
- 最终输出（`output_video/`）
- `manifest.jsonl`

如需关闭清理：
```bash
python scripts/run_batch.py ... --no_cleanup
```

如需保留 prompt/json：
```bash
python scripts/run_batch.py ... --keep_prompt --keep_json
```

如需连同日志也清理（不推荐，除非你确认不需要排错日志）：
```bash
python scripts/run_batch.py ... --cleanup_logs
```

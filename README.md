# Cosmos-Transfer2.5 全自动批处理工程

这份工程把 Cosmos-Transfer2.5 的 **Step2~Step5** 自动化成可复用的批处理流水线，支持多卡并行，并对 **prompt** 与 **depth 控制信号** 做指纹级缓存。相同视频会直接命中缓存，减少重复读取与计算。

## 工程能做什么
1. 批量读取 mp4 输入。
2. 调用 Qwen2.5-VL / Qwen3-VL（AutoPrompt）生成三步式 prompt（默认复用缓存，命中后跳过重新读取视频）。
3. 生成 depth 控制信号（缓存命中优先；edge/blur/seg/combo 接口预留）。
4. 生成对应的 json spec。
5. 调用 Cosmos-Transfer2.5 输出替换视频，并在控制台透传扩散式进度条。

## 依赖假设
- 具备 `conda`，并存在：
  - `qwen25vl`：运行 `robot_video_to_prompt.py`（本仓库已自带脚本，可通过 `autoprompt_root` 配置）。
  - `depthanything`：运行 `Video-Depth-Anything/run.py`。
- Cosmos-Transfer2.5 使用 `uv` 创建的 `.venv`，路径 `<transfer_root>/.venv/bin/python` 可用。
- 推理子进程会自动继承 HF 缓存变量：`HF_HOME`、`HF_HUB_CACHE`、`HUGGINGFACE_HUB_CACHE`、`TRANSFORMERS_CACHE`。

## 目录速览
- `scripts/run_batch.py`：命令行入口。
- `auto_transfer_runner/cli.py`：参数解析与多卡调度。
- `auto_transfer_runner/pipeline.py`：单视频 Step2~5 流水线。
- `auto_transfer_runner/steps/`：各步骤实现。
- `auto_transfer_runner/utils/`：日志、锁、指纹、路径等工具。
- `robot_video_to_prompt.py`：三步式 prompt 生成脚本（仓库根目录）。

## 快速开始
单卡跑全部：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --console
```

4 卡并行、限制 100 个视频：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --num_videos 100 \
  --gpus 0,1,2,3 \
  --control_type depth \
  --console
```

使用 Qwen3-VL-8B 生成 prompt 的示例（单卡）：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --prompt_model_family qwen3vl \
  --prompt_size 8B \
  --console
```

对每个视频生成 K 个不同版本（Step2 与 Step5 重新跑，depth 继续复用缓存）：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0,1,2,3 \
  --control_type depth \
  --variants_per_video 5 \
  --console
```

提示：
- `--console` 会在终端输出每步开始/完成、缓存命中，以及 Transfer2.5 的扩散式进度条。
- 如需直接使用仓库自带的 prompt 脚本，可将 `--autoprompt_root` 设为当前仓库路径。

## 输出目录（默认 `work_dir=/mnt/data/ganrunhe/auto_transfer`）
- `prompt/`：每个 job 独立的最终 prompt（来自缓存或新生成）。
- `prompt_cache/`：按视频指纹缓存 prompt（三步结果都放在这里，命中后不再读取视频）。
- `control_cache/`：depth 控制信号缓存。
- `control/`：非缓存控制类型的预留目录。
- `json/`：每个视频独立的 json spec。
- `output_video/`：Transfer2.5 输出。
- `logs/`：每个视频一个日志文件。

## 视频指纹与缓存策略
- 指纹计算支持 `path` / `stat` / `md5`，默认 `stat`（速度与稳健性的折中）。通过 `--fingerprint_mode` 配置，prompt 与 depth 共享同一指纹。
- prompt 默认复用：命中缓存时直接复制缓存文件到当前 job，跳过重新读取视频。
- depth 默认复用：`control_cache/depth/<fingerprint>/` 内的 `_vis.mp4` 命中即复用，带文件锁避免并发冲突。

## 扩散进度条与计时
- 每一步的耗时会写入日志；开启 `--console` 时同步打印。
- Step5（Transfer2.5 推理）会实时透传官方脚本的扩散式进度条，便于监控生成进度。

## 常用参数（节选）
- `--autoprompt_root`：prompt 脚本所在目录（默认 `/mnt/data/ganrunhe/AutoPrompt`，可设为当前仓库）。
- `--transfer_root`：Cosmos-Transfer2.5 路径（默认 `/mnt/data/ganrunhe/cosmos-transfer2.5`）。
- `--hf_cache`：HuggingFace 缓存目录。
- `--prompt_model_family`：选择 prompt 所用模型族（`qwen25vl` 或 `qwen3vl`），内部会映射为 AutoPrompt 脚本支持的 `--model_type`（`Qwen2.5` 或 `Qwen3`）。
- `--no_reuse_prompt`：强制重算 prompt，即使缓存命中。
- `--no_cache_control` / `--force_regen_control`：控制 depth 缓存开关。
- `--keep_prompt` / `--keep_json` / `--keep_control_dir` / `--cleanup_logs`：控制成功后的清理策略。

完整参数：
```bash
python scripts/run_batch.py --help
```

## 更新记录
- 2025-12-19：指纹级 prompt/深度缓存、每步耗时记录、推理扩散进度条透传、README 说明更新。
- 2025-12-19：prompt 缓存增加并发锁避免多卡重复生成；多卡数量不超过待处理视频数；队列分配优先覆盖不同视频指纹。
- 2025-12-19：多卡调度与日志增强（GPU 等待/开始日志、prompt 锁等待提示）。将 GPU 数量大于视频数时的冗余等待显示清楚，方便排查。
- 2025-12-19：新增 CLI 开关 `--prompt_model_family`，可在 Qwen2.5-VL 与 Qwen3-VL 之间切换（依赖 `robot_video_to_prompt.py` 对应支持）。

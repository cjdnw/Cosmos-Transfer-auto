# Cosmos-Transfer2.5 全自动批处理工程

面向 Cosmos-Transfer2.5 的批处理流水线，将原本手动执行的 **Step2~Step5** 串成可批量、多卡并行的自动化流程，并针对 **depth/seg 控制信号** 做缓存复用（同一输入视频仅生成一次控制信号）。

## 自动化前的手动流程（现已自动化）
1. 提供原始 mp4 路径（批量）
2. 生成随机纹理替换的 prompt（AutoPrompt，支持 Qwen3/Qwen2.5）
3. 生成控制信号（目前实现 depth/seg；保留 edge/blur/combo 接口）
4. 生成对应的 json spec
5. 调用 Cosmos-Transfer2.5 推理，输出最终替换版本

## 依赖与环境
- `conda` 命令可用，且存在环境：
  - `qwen25vl`（默认）：用于 `AutoPrompt/robot_video_to_prompt.py`，支持 `--model_type Qwen3|Qwen2.5`
  - `depthanything`：用于 `Video-Depth-Anything/run.py`
  - `autoseg`：用于 `AutoSeg-SAM2`（seg 控制信号）
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

### 例 3：自动生成 seg 控制信号（缓存复用）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type seg \
  --seg_level large \
  --seg_detect_stride 10
```

### 例 4：指定 Qwen 模型族（默认 Qwen2.5，可切换为 Qwen3）
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

### 例 5：随机替换改为“全部替换”
默认会在括号标注的描述中随机选择若干个词语进行替换；如需直接替换所有可替换的词，增加：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --replace_all_terms
```

### 例 6：控制随机替换的上下限
通过 `--min_terms` 和 `--max_terms` 控制每次随机替换的数量区间（未开启 `--replace_all_terms` 时生效）：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --min_terms 2 \
  --max_terms 5
```

### 例 7：调整每个词的替换备选数量（默认 6 个）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --replacements_per_term 4
```
说明：`--replacements_per_term` 会影响 Step3 生成的候选列表数量，并参与 prompt 缓存键；不同数量会生成独立的替换缓存。

## 输出目录（默认 work_dir=/mnt/data/ganrunhe/auto_transfer）
- `prompt/`：每个视频（每个 job）独立的 prompt
- `prompt_cache/`：按视频指纹缓存的 Step1 base prompt（跨任务复用）
- `control_cache/`：depth/seg 控制信号缓存（跨 job 复用）
- `control/`：非缓存控制类型（预留，未来 edge/... 可用）
- `json/`：每个视频独立的 json spec
- `output_video/`：Transfer2.5 输出目录
- `logs/`：每个视频一个日志文件（追加写入）

## Seg 控制信号生成与复用（新增）
目前工程**已自动集成 seg 控制信号**（`--control_type seg`），复用逻辑与 depth 完全一致：
- 缓存路径：`control_cache/seg/<fingerprint>/`
- 默认依赖：`/mnt/data/ganrunhe/AutoSeg-SAM2` + conda env `autoseg`
- 可通过 CLI 覆盖路径/参数：`--seg_root`、`--seg_env`、`--seg_level`、`--seg_detect_stride`、`--seg_enable_highlight`、`--seg_disable_overlay`、`--seg_overlay_style` 等

### 手动生成 Seg 控制信号
1. 切换到 AutoSeg-SAM2 工程与环境  
   - 仓库路径：`/mnt/data/ganrunhe/AutoSeg-SAM2`  
   - Conda 环境：`autoseg`
2. 生成逐帧分割掩码（会在 `output_dir/large/final-output/` 下生成 png）：
   ```bash
   python auto-mask-batch.py \
     --video_path <输入视频路径> \
     --output_dir <临时 cache 目录> \
     --level large \
     --use_other_level 0 \
     --detect_stride 10 \
     --save_intermediate 0 \
     --save_npy 0
   ```
3. 合成 seg 控制信号 mp4（覆盖高亮/叠加开关按需调整，这里为纯 mask 叠加）：
   ```bash
   python visulization.py \
     --video_path <输入视频路径> \
     --output_dir <临时 cache 目录> \
     --level large \
     --enable_highlight 0 \
     --enable_overlay 1 \
     --overlay_style mask
   ```
4. 将生成的 mp4 拷贝到缓存目录，供后续复用：  
   - 目标路径：`/mnt/data/ganrunhe/auto_transfer/control_cache/seg/`  
   - 拷贝完成后可清空上述临时 `output_dir`

### Seg 对应的 json 片段
在生成 Cosmos-Transfer2.5 的任务 json 时，加入如下字段（`control_path` 指向上一步缓存好的 mp4，其他字段保持 depth 同样的结构与分 chunk 逻辑）：
```json
{
  "name": "robot_seg",
  "prompt_path": "<prompt 路径>",
  "seed": 2025,
  "guidance": 3,
  "video_path": "<输入视频路径>",
  "image_context_path": null,
  "num_conditional_frames": 1,
  "resolution": "720",
  "sigma_max": null,
  "num_video_frames_per_chunk": 93,
  "num_steps": 35,
  "show_control_condition": false,
  "show_input": false,
  "keep_input_resolution": true,
  "edge": null,
  "depth": null,
  "vis": null,
  "seg": {
    "control_path": "<control_cache/seg 下的 mp4>",
    "control_weight": 1.0,
    "mask_path": null,
    "mask_prompt": null,
    "control_prompt": null
  }
}
```
生成后即可将 json 与 prompt 一同传入 Cosmos-Transfer2.5 推理流程。

## “视频指纹”与缓存
- prompt 三层缓存（一次算，多次复用）：
  - Step1：`prompt_cache/<fingerprint>/base.txt`，键 = video_fingerprint（路径+size+mtime 为默认）。
  - Step2：`prompt_cache_bracket/<base_hash>.txt`，键 = base.txt 的 md5；与视频无关，只看 base 文本。
  - Step3：`prompt_cache_replacements/<bracket_hash>__k{N}.json`，键 = bracket.txt 的 md5 + `--replacements_per_term`（N）；与具体括号标注文本和候选数量绑定。
- Step2+3 在同一进程里跑新的 `step23` 子命令，命中后直接复用缓存，避免模型重复加载；只有 Step3_apply（无模型）会在换肤时再次执行。
- 所有缓存层均有文件锁：Step1 以 fingerprint 目录锁定；Step2 以 base_hash 锁定；Step3 以 bracket_hash 锁定。多 worker 命中相同视频/文本时会等待同一锁，只让一个 worker 写缓存，其余直接复用，保障“同一视频只跑一次耗时步骤”的等待范式。
- depth 缓存目录按 fingerprint 划分：`control_cache/depth/<fingerprint>/`
- seg 缓存目录同样按 fingerprint 划分：`control_cache/seg/<fingerprint>/`。流水线会自动调用 `/mnt/data/ganrunhe/AutoSeg-SAM2`（conda 环境 `autoseg`）生成 seg 控制 mp4 并落盘缓存，命中后复用，无需手动干预；可通过 CLI 覆盖参数（如 `--seg_root`、`--seg_env`、`--seg_level`、`--seg_detect_stride`、`--seg_enable_highlight`、`--seg_disable_overlay`、`--seg_overlay_style` 等）。
- fingerprint 默认使用 `stat`（路径 + 文件大小 + 修改时间），速度快；如需更稳健（读完整文件算 hash），使用：`--fingerprint_mode md5`

## 扩展接口（edge/blur/combo）
seg 已实现；如需扩展其他控制信号，在 `auto_transfer_runner/steps/control.py` 中补充对应命令即可；调度、命名、并行、日志与缓存框架保持不变。

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

## 路径与环境默认值（config.py）
所有内置的路径与环境默认值统一集中在 `auto_transfer_runner/config.py`，CLI 会直接读取这些常量：
- 路径：`DEFAULT_WORK_DIR`、`DEFAULT_AUTOPROMPT_ROOT`、`DEFAULT_DEPTH_ROOT`、`DEFAULT_SEG_ROOT`、`DEFAULT_TRANSFER_ROOT`、`DEFAULT_HF_CACHE`
- 环境/模型参数：`PROMPT_DEFAULTS`（如 `conda_env`/`fps`/`model_type` 等）、`DEPTH_DEFAULTS`、`SEG_DEFAULTS`、`TRANSFER_DEFAULTS`、`CACHE_DEFAULTS`

如需修改默认路径或环境名称，有两种方式：
1. 直接编辑 `config.py` 中上述常量/默认对象，保存后即成为新的全局默认值。
2. 运行命令时通过 CLI 参数覆盖（例如 `--work_dir`、`--autoprompt_root`、`--depth_root`、`--transfer_root`、`--hf_cache`、`--qwen_env` 等），无需改动源码。

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
- 2025-12-23：集成 seg 控制信号自动化，使用 AutoSeg-SAM2 在 `control_cache/seg/` 按指纹落盘并复用，CLI 直接通过 `--control_type seg` 调用。
- 2025-12-23：调整 Step1 base prompt 文案为官方演示叙事风格；Step2 括号替换限定为颜色/纹理描述，Step3 只生成颜色或纹理的替换词。
- 2025-12-23：新增 `--replace_all_terms` 参数，可直接替换所有括号标注的描述（跳过随机抽样逻辑）。
- 2025-12-23：新增 `--replacements_per_term` 参数，可调整每个括号术语生成的替换候选数量（默认 6），并对缓存进行区分。
- 2025-12-22：`--min_terms` / `--max_terms` 支持快速设置随机替换数量区间（默认 1~3），便于控制改写力度。
- 2025-12-22：将路径与环境默认值集中到 `config.py`（CLI 统一读取），并在 README 增补路径修改指引。
- 2025-12-22：新增 Step2+Step3 合并的 `step23` 子命令与三层 prompt 缓存（fingerprint/base_hash/bracket_hash），同一视频/文本多 worker 会被文件锁串行化，仅 Step3_apply 在换肤时毫秒级复用。
- 2025-12-22：Prompt 生成流程适配拆分版 `robot_video_to_prompt.py`（step1~step3），新增基于视频指纹的 base prompt 缓存（`prompt_cache/`），默认使用本仓库内的脚本路径；控制台打印增加 GPU 标签。
- 2025-12-22：为终端输出统一添加东八区时间戳，并在 Step2~Step5（含清理）及整条流水线上记录耗时，便于观测性能。
- 2025-12-21：适配新版 `robot_video_to_prompt.py`（支持 Qwen3/Qwen2.5 的 `--model_type` 参数），新增命令行参数 `--prompt_model_type`，默认 conda 环境保持 `qwen25vl`，可选切换为 `qwen3vl`。描述改为工程化表述，匹配顶会论文风格。

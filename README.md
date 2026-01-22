# Cosmos-Transfer2.5 全自动批处理工程

面向 Cosmos-Transfer2.5 的批处理流水线，将原本手动执行的 **Step2~Step5** 串成可批量、多卡并行的自动化流程，并针对 **depth/seg/edge/blur 控制信号** 做缓存复用（同一输入视频仅生成一次控制信号）。

## 自动化前的手动流程（现已自动化）
1. 提供原始 mp4 路径（批量）
2. 生成随机纹理替换的 prompt（AutoPrompt，支持 Qwen3/Qwen2.5）
3. 生成控制信号（已实现 depth/seg/edge/blur/多控制 combo）
4. 生成对应的 json spec
5. 调用 Cosmos-Transfer2.5 推理，输出最终替换版本

## 依赖与环境
- `conda` 命令可用，且存在环境：
  - `qwen25vl`（默认）：用于 `AutoPrompt/robot_video_to_prompt.py`，支持 `--model_type Qwen3|Qwen2.5`
  - `depthanything`：用于 `Video-Depth-Anything/run.py`
  - `autoseg`：用于 `AutoSeg-SAM2`（seg 控制信号）
  - `sam`：用于 `gen_controls_cv_mp4.py`（edge/blur 控制信号）
- 若指定的 `--qwen_env` 不存在，运行时会自动回退到 `qwen25vl` 并在日志中打印警告。
- Cosmos-Transfer2.5 使用 `uv` 创建的 `.venv`：`<transfer_root>/.venv/bin/python`
- 推理子进程会自动设置 HuggingFace 缓存变量：`HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`
- 可选：安装 `rife-ncnn-vulkan` 或 Practical-RIFE（PyTorch/CUDA）以启用 `--post_restore_mode rife`（RIFE 补帧）
- 本仓库自身依赖（用于 base 环境运行）见 `requirements.txt`：`numpy`、`pyarrow`、`pillow`。

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

### 例 1.5：开启视频预处理（CFR/对齐 chunk）+ 设置推理 resolution
预处理会生成 CFR 视频（默认使用原视频 fps），输出文件默认保存到 `work_dir/prepro_video/`（保留输入目录结构，文件名带 `__pre_...` 后缀），后续控制/推理将复用该预处理结果；Prompt 仍使用原视频。
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --pre_chunk_mode speed \
  --pre_chunk_force_chunks 8 \
  --resolution 480
```
常用参数说明：
说明：预处理/控制生成使用原视频 fps；`--resolution` 决定 STEP5 推理速度（关键加速开关）。

预处理/对齐选项（影响控制生成/IO，不影响 STEP5 推理 bucket）：
- `--pre_width`/`--pre_height`：指定输出宽高，可只填一个；默认不缩放。
- `--pre_no_keep_aspect`：关闭等比例缩放（默认保持比例）。
- `--pre_chunk_mode {none,clip,speed}`：对齐模式（clip=裁剪；speed=均匀抽帧并降低 fps，保持时长不变）。
- `--pre_chunk_policy {down,up}`：down 减少 chunk，up 增加 chunk（若需要补帧则会跳过对齐）。
- `--pre_chunk_clip_where {tail,head,center}`：clip 模式裁剪位置。
- `--pre_chunk_max_adjust N`：当 |T - T_target| > N 时跳过对齐（避免过度裁剪/抽帧）。
- `--pre_chunk_force_chunks N`：强制固定 chunk 数（仅使用 speed 模式，均匀抽帧并降低 fps；忽略 clip/policy/clip_where）。
- `--keyframe_sampling {uniform,parquet_motion}`：使用 parquet 的 observation.state 变化进行非均匀抽帧。
- `--keyframe_target_frames N`：目标关键帧数（默认使用 `--num_video_frames_per_chunk`；若开启 `--pre_chunk_mode speed` 或 `--pre_chunk_force_chunks`，则按 chunk 数自动计算）。
- `--keyframe_smooth_win N`：运动分数平滑窗口（默认 0）。
- `--keyframe_max_gap N`：最大帧间隔约束（默认 2026）。
- `--parquet_pose_subset {zarm_pos,zarm_pos_vel,zarm_pos_vel_hand,full}`：parquet_motion 的姿态维度子集（默认 `zarm_pos_vel`，可过滤非机械臂维度）。
- `--proxy_preflight`：启用 Step1.5.5 Proxy IoU 预飞行（默认关闭）。
- `--proxy_iou_threshold`：IoU 判定坏帧阈值（默认 0.90）。
- `--proxy_iou_high_threshold`：删除候选所需的高 IoU 门槛（默认 0.98）。
- `--proxy_motion_low_quantile`：低运动分位阈值（默认 0.25）。
- `--proxy_max_iters`：预飞行最大迭代次数（默认 3）。
- `--proxy_restore_mode {speed,minterpolate,rife}`：Proxy 恢复模式（默认 `rife`）。
- `--proxy_rife_backend {ncnn_vulkan,pytorch}`：Proxy 恢复使用的 RIFE 后端（默认沿用 `--rife_backend`）。
- `--proxy_preflight_cache_csv`：缓存每轮 IoU 结果到预处理 MP4 根目录下的 CSV，并输出对应的 IoU 曲线 PNG（默认关闭）。
- `--proxy_removal_strategy {ranked,bucketed}`：删除策略（`ranked` 为高 IoU + 低运动优先；`bucketed` 为按时间分桶后在桶内优先删除，默认 `ranked`）。
- `--proxy_removal_buckets`：`bucketed` 删除策略下的分桶数量（默认 6）。

当启用 `--keyframe_sampling parquet_motion` 时，预处理阶段会输出：`<work_dir>/prepro_video/<relative_dir>/<video_stem>__keyframes__<sampling_tag>/`，并将该 `keyframes.mp4` 作为 Transfer2.5 的输入视频（替代均匀抽帧/预处理视频）。`sampling_tag` 会编码关键帧抽样配置（如 target_frames/max_gap/smooth_win/pose_subset 等），避免不同抽样配置互相覆盖：
- `keyframes.mp4`：抽取的关键帧视频（固定 16fps）
- `keyframe_indices.txt`：每行一个原视频帧号
- `keyframe_meta.json`：包含输入视频、parquet 路径、关键帧索引等信息
- Step1.5.5（Proxy IoU 预飞行 + 关键帧预算重分配）可选启用：当 `parquet_motion` 已有 `frame_map` 时直接使用；当 `uniform` 时会先按目标帧数生成 `keyframes_uniform.mp4` 与 `frame_map`。随后从 keyframes 调用 Step5.5 的 RIFE guided 时长恢复得到 `proxy_restore.mp4`（缓存到同一预处理目录的 `proxy_preflight/` 中），对原始视频与 proxy 恢复视频生成机械臂 mask（复用 Qwen bbox + SAM 缓存；缓存缺失时会自动重新生成）并逐帧计算 IoU；若存在 IoU < 0.90 的帧，则在坏帧附近插入新锚点并从 IoU 很高且运动很小的区域删除等量锚点，保持总数为 93。删除策略默认为高 IoU + 低运动优先；可通过 `--proxy_removal_strategy bucketed` 启用按时间分桶的均匀删除（桶内仍按高 IoU + 低运动优先）。迭代直到满足停止条件或达到最大迭代次数，最终输出 `keyframes_opt.mp4` 与 `keyframes_opt__frame_map.json`，后续 Step3/Step4/Step5 将自动使用优化后的 keyframes。可通过 `--proxy_preflight` 启用，并用 `--proxy_iou_threshold`/`--proxy_iou_high_threshold`/`--proxy_motion_low_quantile`/`--proxy_max_iters`/`--proxy_restore_mode`/`--proxy_rife_backend`/`--proxy_removal_strategy`/`--proxy_removal_buckets` 调整策略；如需缓存每轮 IoU 明细，可加 `--proxy_preflight_cache_csv`。

## 推理加速：resolution
`resolution` 是写入 STEP4 JSON spec 的字段，用于控制 Transfer2.5 推理时的 bucket 分辨率，会显著影响速度与显存占用。支持的 key：
`1080`, `1024`, `720`, `720p`, `720robocasa`, `480`, `480p`, `512`, `256`。

| resolution key | 16:9 bucket 尺寸 |
| --- | --- |
| 1080 | 1920x1056 |
| 1024 | 1280x768 |
| 720 | 1280x704 |
| 720p | 1280x720 |
| 720robocasa | 1280x720 |
| 480 | 768x432 |
| 480p | 832x480 |
| 512 | 640x384 |
| 256 | 320x192 |

### 例 2：4 卡并行（每张卡完成完整 2→5 流程）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --num_videos 100 \
  --gpus 0,1,2,3 \
  --control_type depth
```

### 例 2.5：按 task.parquet 分组抽样（每个任务最多选 N 个）
`task.parquet` 默认位于数据集根目录下的 `meta/` 目录（与 `videos/` 同级）；程序会根据 `videos/` 的父目录定位 `meta/task.parquet` 并按内容分组。
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --selective_task 3
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

### 例 4：Transfer 后恢复时长（RIFE 补帧）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --post_restore_duration \
  --post_restore_mode rife \
  --rife_backend ncnn_vulkan \
  --rife_bin /path/to/rife-ncnn-vulkan \
  --rife_model /path/to/rife-model
```
参数说明：
- `--post_restore_duration`：在 Step5 后让输出与原视频 **帧数/FPS 对齐**（覆盖原输出文件；时长可能略有误差）。
- `--post_restore_mode {speed,minterpolate,rife}`：speed=变速回拉；minterpolate=插值回拉（更平滑但更慢）；rife=使用 RIFE 进行补帧（需安装 rife-ncnn-vulkan）。
  - `--rife_bin`：RIFE 可执行文件路径（默认 `rife-ncnn-vulkan`，需在 PATH）。
  - `--rife_model`：RIFE 模型目录（可选）。
  - `--rife_backend {ncnn_vulkan,pytorch}`：RIFE 后端选择（默认 ncnn_vulkan）。
  - `--rife_pytorch_repo`：Practical-RIFE 仓库路径（pytorch 后端必填，默认 `/mnt/data/ganrunhe/Practical-RIFE`）。
  - `--rife_pytorch_model_dir`：Practical-RIFE 权重目录（默认 `<repo>/train_log`）。
  - `--rife_pytorch_env`：Practical-RIFE conda 环境（默认 rife）。
  - `--rife_scale`：RIFE scale（默认 1.0，0.5 可加速）。
  - `--rife_keep_audio`：保留输入音频并 mux 回输出（默认 false）。
  - `--rife_time_steps_mode {uniform,zarm_pos,sam,sam_zarm_hybrid}`：PyTorch guided 补帧的时间步策略（默认 sam_zarm_hybrid）。
  - `--rife_time_steps_cache`：time_steps 缓存路径（默认 `<work_dir>/rife_cache/time_steps.json`）。
  - `--rife_sam_mask_dir`：SAM mask 目录（png/jpg/npy），默认 `<work_dir>/rife_sam_cache`，用于 guided 非均匀 time_steps。
  - `--rife_pose_subset {zarm_pos,zarm_pos_vel,zarm_pos_vel_hand,full}`：guided time_steps 的 parquet pose 子集。
  - `--rife_time_steps_debug`：打印每段 time_steps 的 min/mean/max 与 SAM 可信比例。
补充：需要关闭官方 guardrails 时可以追加 `--disable_guardrails`，该参数会透传到 Transfer2.5 推理脚本。

### 例 5：指定 Qwen 模型族（默认 Qwen2.5，可切换为 Qwen3）
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

### 例 5：sim2real 任务仅生成 Step1 prompt
仅执行 `video_to_prompt` 的 Step1 base prompt，不做括号标注与替换：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --sim2real_only
```

### 例 6：随机替换改为“全部替换”
默认会在括号标注的描述中随机选择若干个词语进行替换；如需直接替换所有可替换的词，增加：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --replace_all_terms
```

### 例 7：控制随机替换的上下限
通过 `--min_terms` 和 `--max_terms` 控制每次随机替换的数量区间（未开启 `--replace_all_terms` 时生效）：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --min_terms 2 \
  --max_terms 5
```

### 例 8：调整每个词的替换备选数量（默认 64 个）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --replacements_per_term 4
```
说明：`--replacements_per_term` 会影响 Step3 生成的候选列表数量，并参与 prompt 缓存键；不同数量会生成独立的替换缓存。

### 例 9：交互式替换（快速启动）
在 Step2 输出括号词列表后，手动选择需要替换的关键词并输入替换词：
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --interactive_replace
```

### 例 10：自动生成 edge 控制信号（缓存复用）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type edge \
  --edge_low 80 \
  --edge_high 160
```

### 例 11：自动生成 blur 控制信号（缓存复用）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type blur \
  --blur_sigma 10
```

### 例 12：多控制信号（combo）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type combo \
  --depth_control_weight 1.0 \
  --edge_control_weight 0.2 \
  --seg_control_weight 1.0 \
  --vis_control_weight 0.1
```

### 例 13：开启机械臂替换（qwen3vl + sam3）
在 Step5 Transfer 输出完成后，自动调用 `robot_arm_swap` 生成机械臂替换版本（两个输出都会保留）。Step6 默认会复用 `<work_dir>/qwen_sam_cache/<fingerprint>/qwen_sam` 缓存（`qwen_boxes.json`、`sam3_track/`、`masks_arm/`、`cache_meta.json`），若缓存缺失或与原视频不一致会自动重算；Step1.5.5/Step5.5/Step6 统一使用同一目录，后续步骤直接复用前一步生成的结果。
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --enable_arm_swap \
  --arm_swap_prompt "robotic arm"
```
Step6 会在 `conda` 环境 `arm_swap` 下调用：
```
python -m robot_arm_swap \
  --orig 预处理后的视频（或原视频，见说明） \
  --skin Transfer 输出视频 \
  --workdir /mnt/data/ganrunhe/auto_transfer/qwen_sam_cache/<fingerprint> \
  --swap-only \
  --arm-mask-dir <workdir>/qwen_sam/masks_arm \
  --no-workdir-write \
  --fps <与 orig 视频一致> \
  --width <与 orig 视频一致> \
  --erode 2 \
  --feather 3 \
  --dilate 0 \
  --out /mnt/data/ganrunhe/auto_transfer/output_video_mask_arm/<output_video_name>.mp4
```
默认启用 `--arm_swap_no_workdir_write`（透传为 `--no-workdir-write`），使用内存模式避免中间 PNG 落盘并直接管道编码输出视频；如需生成 skin 侧 mask 并计算 IoU，可使用 `--arm_swap_workdir_write`（透传为 `--arm-swap-workdir-write`）。
Qwen/SAM 缓存统一写入 `<work_dir>/qwen_sam_cache/<fingerprint>/qwen_sam` 并复用，不会在 Step6 清空；`<fingerprint>` 与输入视频一一对应。
`--out` 的目录结构与 Step5 的 output_video 命名规则一致（仅根目录替换为 `output_video_mask_arm`）。
Step6 的 `--orig` 选择规则：
- 默认使用原始输入视频进行 swap。

## 输出目录（默认 work_dir=/mnt/data/ganrunhe/auto_transfer）
- `prompt/`：每个视频（每个 job）独立的 prompt
- `prompt_cache/`：按视频指纹缓存的 Step1 base prompt（跨任务复用）
- `control_cache/`：depth/seg 控制信号缓存（跨 job 复用）
- `control_cache/edge/`：edge 控制信号缓存（跨 job 复用）
- `control_cache/blur/`：blur 控制信号缓存（跨 job 复用）
- `control/`：非缓存控制类型（关闭控制缓存时使用）
- `json/`：每个视频独立的 json spec
- `output_video/`：Transfer2.5 输出目录
- `logs/`：每个视频一个日志文件（追加写入）
- `qwen_sam_cache/<fingerprint>/`：Qwen+SAM 共享缓存（Step1.5.5/Step5.5/Step6 复用）


## “视频指纹”与缓存
- prompt 三层缓存（一次算，多次复用）：
  - Step1：`prompt_cache/<fingerprint>/base.txt`，键 = video_fingerprint（路径+size+mtime 为默认）。
  - Step2：`prompt_cache_bracket/<base_hash>.txt`，键 = base.txt 的 md5；与视频无关，只看 base 文本。
  - Step3：`prompt_cache_replacements/<bracket_hash>__k{N}.json`，键 = bracket.txt 的 md5 + `--replacements_per_term`（N）；与具体括号标注文本和候选数量绑定。
- Step2+3 在同一进程里跑新的 `step23` 子命令，命中后直接复用缓存，避免模型重复加载；只有 Step3_apply（无模型）会在换肤时再次执行。
- 所有缓存层均有文件锁：Step1 以 fingerprint 目录锁定；Step2 以 base_hash 锁定；Step3 以 bracket_hash 锁定。多 worker 命中相同视频/文本时会等待同一锁，只让一个 worker 写缓存，其余直接复用，保障“同一视频只跑一次耗时步骤”的等待范式。
- depth 缓存目录按 fingerprint 划分：`control_cache/depth/<fingerprint>/`
- seg 缓存目录同样按 fingerprint 划分：`control_cache/seg/<fingerprint>/`。流水线会自动调用 `/mnt/data/ganrunhe/AutoSeg-SAM2`（conda 环境 `autoseg`）生成 seg 控制 mp4 并落盘缓存，命中后复用；前置会用 `ffmpeg` 抽帧到临时目录再调用 AutoSeg-SAM2（其输入需要帧目录而非 mp4），无需手动干预。可通过 CLI 覆盖参数（如 `--seg_root`、`--seg_env`、`--seg_level`、`--seg_detect_stride`、`--seg_enable_highlight`、`--seg_disable_overlay`、`--seg_overlay_style` 等）。
- edge 缓存目录按 fingerprint 划分：`control_cache/edge/<fingerprint>/`。流水线会自动调用仓库内的 `gen_controls_cv_mp4.py`（conda 环境 `sam`）生成 canny 控制 mp4 并落盘缓存，命中后复用；可通过 CLI 覆盖 `--edge_low` / `--edge_high`。
- blur 缓存目录按 fingerprint 划分：`control_cache/blur/<fingerprint>/`。流水线会自动调用仓库内的 `gen_controls_cv_mp4.py`（conda 环境 `sam`）生成 blur 控制 mp4 并落盘缓存，命中后复用；可通过 CLI 覆盖 `--blur_sigma`。
- fingerprint 默认使用 `md5`（读完整文件算 hash，最稳健）；如需更快但可能受改名/时间影响，可用：`--fingerprint_mode stat` 或 `--fingerprint_mode path`

## 多控制信号（combo）
depth/seg/edge/blur 已实现；`combo` 将 depth + edge + seg 合并为多控制信号，vis 使用 `preset_blur_strength=high`。如需调整各控制信号权重，使用 `--depth_control_weight`、`--edge_control_weight`、`--seg_control_weight`、`--vis_control_weight`（默认回退到 `--control_weight`）。

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
- 队列按 variant 分层排队：先把所有视频的 variant=0 任务入队，再入队 variant=1...，因此 GPU 会优先处理不同视频的同一轮 variant；下一轮 variant 任务只会在上一轮全部出队后才开始被分发（可能与上一轮未完成任务并行运行）

## manifest.jsonl（输入-输出映射表）
批处理会在 `<work_dir>/manifest.jsonl` 里追加写入每个任务的映射信息（JSON Lines）：
- input_video / variant_idx / job_id
- prompt_path / json_path / output_dir / log_path
- control_vis_path（单控制模式下最终引用的控制 mp4）

## Transfer2.5 默认路径（已更新）
默认 `--transfer_root`：`/mnt/data/ganrunhe/cosmos-transfer2.5`

本地路径不同可通过命令行覆盖：
```bash
python scripts/run_batch.py ... --transfer_root /your/path/to/cosmos-transfer2.5
```

## AutoPrompt 默认路径
默认 `--autoprompt_root` 指向本仓库根目录（包含 `robot_video_to_prompt.py`），无需额外克隆独立的 AutoPrompt 仓库。

本地路径不同可通过命令行覆盖：
```bash
python scripts/run_batch.py ... --autoprompt_root /your/path/to/AutoPrompt
```

## AutoSeg 默认路径（已更新）
默认 `--seg_root`：`/mnt/data/ganrunhe/AutoSeg-SAM2`

本地路径不同可通过命令行覆盖：
```bash
python scripts/run_batch.py ... --seg_root /your/path/to/AutoSeg-SAM2
```

## blur/canny 生成脚本位置（可覆盖）
默认 `--cv_root` 指向本仓库根目录（包含 `gen_controls_cv_mp4.py`），无需额外克隆独立仓库。

本地路径不同可通过命令行覆盖：
```bash
python scripts/run_batch.py ... --cv_root /your/path/to/cv-tools
```

## 路径与环境默认值（config.py）
所有内置的路径与环境默认值统一集中在 `auto_transfer_runner/config.py`，CLI 会直接读取这些常量：
- 路径：`DEFAULT_WORK_DIR`、`DEFAULT_AUTOPROMPT_ROOT`、`DEFAULT_DEPTH_ROOT`、`DEFAULT_SEG_ROOT`、`DEFAULT_CV_ROOT`、`DEFAULT_TRANSFER_ROOT`、`DEFAULT_HF_CACHE`
- 环境/模型参数：`PROMPT_DEFAULTS`（如 `conda_env`/`fps`/`model_type` 等）、`DEPTH_DEFAULTS`、`SEG_DEFAULTS`、`CV_DEFAULTS`、`TRANSFER_DEFAULTS`、`CACHE_DEFAULTS`

如需修改默认路径或环境名称，有两种方式：
1. 直接编辑 `config.py` 中上述常量/默认对象，保存后即成为新的全局默认值。
2. 运行命令时通过 CLI 参数覆盖（例如 `--work_dir`、`--autoprompt_root`、`--depth_root`、`--transfer_root`、`--hf_cache`、`--qwen_env` 等），无需改动源码。

## 成功生成后的自动清理
默认行为：**每个任务成功完成推理后**，自动清理中间文件：
- prompt 文件（`prompt/*.txt`）
- json spec（`json/*.json`）
不会清理：
- control 缓存（`control_cache/`）
- step1-3 prompt缓存
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

## 断点续跑（方案A：原子标记文件）
每个视频变体会在 `work_dir/checkpoints/` 下写入原子标记文件（Step2~Step6 + job 完成）。中途断掉后再次运行会基于标记文件跳过已完成的步骤，继续执行未完成部分。
- 标记路径与输出目录一一对应：`work_dir/checkpoints/output_video/.../<job-step>.done`
- 若 prompt/json 被清理（默认清理行为），对应步骤会自动重新生成并刷新标记
- 若标记不存在，会根据 `output_video/<input>_var_x/...` 目录下的已有产物推断进度并补齐标记
- 当本次批处理所有变体均完成后，会自动清空这批任务的标记目录

## 待完成项目
- depth 控制信号质量不佳，需研究提升方案。
- seg 当前使用 sam1 全自动单帧分割 + sam2 分割跟踪，可考虑改为 sam1+sam3 或 sam1+sam3+Cuite，待研究。
- 排查 `rife-ncnn-vulkan` 多卡并发调用时 `vkEnumeratePhysicalDevices failed 5` / `invalid gpu device` 的设备枚举报错与限制。

## 词表维护说明
这些词表集中在 `robot_video_to_prompt.py` 顶部常量区。即使 Step3 使用大模型生成替换词，词表仍承担“约束、纠正、扩展”的控制层职责：
- **`COLOR_BASES`**：用于识别颜色词与短语扩展（Step3 括号合并与一致性处理），并作为颜色偏置校正与替换候选生成的基础集合。
- **`MATERIAL_BASES`**：用于识别材质词（Step3 括号合并、替换一致性与候选过滤的判定依据）。
- **`TEXTURE_BASES`**：用于识别纹理/质感词（Step3 括号合并与替换候选的判定依据）。
- **`COLORISH_BASES`**：用于把 “bluish/greyish” 等模糊色归一化为明确颜色（Step1.5/Step3 纠正）。
- **`VAGUE_MODIFIERS` / `VAGUE_PHRASES`**：用于清理 “subtle/a bit” 等模糊修饰（Step1.5/Step3 纠正）。
- **`VAGUE_COLOR_MODIFIERS`**：用于过滤 “pale/light/dark” 等模糊颜色短语（Step3 纠正）。
- **`VAGUE_TEXTURE_MODIFIERS`**：用于过滤模糊纹理词（Step3 纠正），目前包含 `smooth`。
- **`ROBOT_KEYWORDS`**：机器人相关词汇，用于过滤机器人描述，避免括号与替换误触机器人外观。

### 人工维护更新流程
1. **定位词表**：打开 `robot_video_to_prompt.py`，在常量区找到上述列表。  
2. **更新词条**：  
   - 词条尽量使用**小写英文单词**（与现有规则匹配方式一致）。  
   - 新增颜色/材质/纹理时，优先补到 `COLOR_BASES` / `MATERIAL_BASES` / `TEXTURE_BASES`。  
   - 若是“模糊词”或“应禁止词”，补到 `VAGUE_*` 列表。  
3. **优先级建议**：  
   - **去模糊词优先**：`COLORISH_BASES` / `VAGUE_*` 先做归一化或移除，再做替换。  
   - **替换一致性优先**：同一物体（母串/子串）保持一致替换，避免前后冲突。  
4. **保持同步**：  
   - 如果新增了机器人相关词（如新的部件名称），请同步到 `ROBOT_KEYWORDS`。  
5. **最小验证**：  
   - 运行一次 `robot_video_to_prompt.py step2` 或完整 `step23`，观察括号是否覆盖新增词。  
   - 若出现误圈/漏圈，优先检查提示词与机器人过滤规则是否需调整。  

## video_to_prompt 三部曲完整逻辑
以下对应 `robot_video_to_prompt.py` 的 Step1 ~ Step3 全流程（含 Step1.5 与 Step3_apply），用于从视频生成基础描述、标注可替换短语，并完成随机替换与批量缓存复用：

### Step 1：生成 base prompt（含 Step1.5 兜底）
**入口命令**：`step1_base`  
**核心流程**：
1. 使用 `COSMOS_STYLE_PROMPT` 读取视频并生成单段英文描述。
2. `canonicalize_robot_sentences` 统一机器人描述，避免机器人外观被改写。
3. Step1.5 后处理：
   - `enforce_container_initial_state`：容器首次出现时补充 “initially empty”。
   - `cleanup_vague_terms`：移除模糊修饰（如 *slight/subtle*），将 *bluish/greyish* 等归一化为明确颜色。

**输出**：`base.txt`（不含括号标注的基础描述）

### Step 2：括号标注可替换描述
**入口命令**：`step2_bracket` 或 `step23`  
**核心流程**：
1. 模型根据 `DESCRIPTOR_EXTRACT_PROMPT_TEMPLATE` 提取非机器人相关的颜色/材质/纹理等描述短语。
2. `step2_bracket_base` 校验短语必须是原文子串，清理尾部标点，并过滤机器人相关词与背景框架语句。
3. 将短语插入 `【】` 进行初始标注，并调用 `_reflect_brackets` 进行括号反思校验（确保文本不变，仅修正括号）。
4. `_expand_shared_noun_phrases` 扩展同一实体的名词短语（如已有 table surface 会补圈 white table）。
5. `expand_bracketed_prompt` 扩展名词修饰片段，保证替换时覆盖完整名词短语。

**输出**：`bracket.txt`（带 `【】` 的标注描述）

### Step 3：生成替换候选 + 随机替换
**入口命令**：`step3_replacements` / `step23` / `step3_apply`  
**候选生成流程**（`step3_generate_replacements_json`）：
1. 读取 `bracket.txt` 并去重术语，按共享名词建立同体分组（含母串/子串与共享名词对齐）。
2. 调用大模型生成替换候选，并做严格过滤：
   - 去重、拒绝原词、移除模糊颜色/纹理词、排除机器人相关词。
   - `_ensure_materials` 强制保留原始材质词（如 plastic/metal）。
3. 若候选数量不足，会先触发 `_retry_replacements_for_term` 重试，再使用词库组合规则补齐（颜色/材质/纹理/双色组合等），确保列表尽量达到 `--replacements_per_term`。
4. 对同一物体的替换候选做一致性对齐，避免前后替换冲突（同名物体的颜色/材质变化保持同步）。
5. 自动化流水线会将 Step3 结果缓存到 `prompt_cache_replacements/`（`auto_transfer_runner/steps/prompt.py`）；单独运行 `robot_video_to_prompt.py` 不会写入内部 step3 缓存。

**随机替换流程**（`apply_random_replacements` / `step3_apply`）：
- 默认随机抽取 `min_terms ~ max_terms` 个括注短语进行替换；启用 `--replace_all_terms` 则全部替换。
- 若 `step3_apply` 未传 `--repl_json`，仅去掉括号，不做替换。

**输出**：
- `replacements.json`（每个术语的替换候选列表）
- `final.txt`（完成替换的最终 prompt）

## RIFE 补帧（可选）
用于 `--post_restore_mode rife`，先按时长做变速对齐，再使用 RIFE 插值补帧到目标 FPS（默认原视频 fps）。

### 安装（示例：rife-ncnn-vulkan）
1. 从 release 下载 `rife-ncnn-vulkan`（或自行编译）。
2. 解压后保证可执行文件在 PATH，或通过 `--rife_bin` 指定完整路径。
3. （可选）下载模型目录，并用 `--rife_model` 指定。

**已知问题（ncnn 版本）**：
- 当同时在多卡并发调用 `rife-ncnn-vulkan` 时，可能会直接报错：
  `vkEnumeratePhysicalDevices failed 5` / `invalid gpu device`。
- 该现象会导致进程无法枚举 CUDA/Vulkan 设备，目前只能稳定在 `cuda 0` 上运行。

### 安装（Practical-RIFE，PyTorch/CUDA）
```bash
git clone https://github.com/hzwer/Practical-RIFE.git
cd Practical-RIFE
pip3 install -r requirements.txt
mkdir -p train_log
```

### 权重下载（Practical-RIFE）
```bash
gdown 1zlKblGuKNatulJNFf5jdB-emp9AqGK05 -O RIFEv4.25lite_1018.zip
gdown 1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg -O RIFEv4.25_0919.zip
gdown 1Smy6gY7BkS_RzCjPCbMEy-TsX8Ma5B0R -O RIFEv4.22_lite_0819.zip
unzip RIFEv4.25lite_1018.zip -d train_log
# 或根据需要解压其他版本：
# unzip RIFEv4.25_0919.zip -d train_log
# unzip RIFEv4.22_lite_0819.zip -d train_log
```
确保 `train_log/` 根目录下包含 `flownet.pkl` 和若干 `*.py`（若压缩包多一层目录，请将文件移动到 `train_log/` 根目录）。

### 调用示例
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --post_restore_duration \
  --post_restore_mode rife \
  --rife_bin /path/to/rife-ncnn-vulkan \
  --rife_model /path/to/rife-model
```

### 调用示例（Practical-RIFE）
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --post_restore_duration \
  --post_restore_mode rife \
  --rife_backend pytorch \
  --rife_pytorch_repo /path/to/Practical-RIFE \
  --rife_pytorch_env rife \
  --rife_scale 1.0 \
  --rife_time_steps_mode sam_zarm_hybrid \
  --rife_time_steps_debug
```

当使用 `--rife_backend pytorch` 且存在 `frame_map`（如 parquet_motion 关键帧采样），会自动调用 Practical-RIFE 的 guided 非均匀补帧接口，并支持显式/自动 `time_steps`（含 SAM+zarm_pos 融合与回退策略）。如果 `--rife_time_steps_mode` 选择 `sam_cache`/`sam_iou`，将复用 `<work_dir>/qwen_sam_cache/<fingerprint>/qwen_sam/masks_arm` 缓存（必要时自动生成）。

## 更新记录
- 2026-01-21：新增 Step0/Step1/Step1.5.5 `.done` 标记，覆盖原视频探测、预处理与 Proxy IoU 预飞行断点续跑。
- 2026-01-20：Step1.5.5 Proxy IoU 预飞行新增删除策略参数，支持按时间分桶的更均匀删除。
  - `--proxy_preflight`：启用 Step1.5.5 Proxy IoU 预飞行。
  - `--proxy_iou_threshold`：IoU 判定坏帧阈值，低于该值视为坏帧。
  - `--proxy_iou_high_threshold`：删除候选的高 IoU 门槛，用于优先删除稳定帧。
  - `--proxy_motion_low_quantile`：低运动分位阈值，用于挑选低运动删除候选。
  - `--proxy_max_iters`：预飞行最大迭代次数。
  - `--proxy_restore_mode`：Proxy 恢复模式（speed/minterpolate/rife）。
  - `--proxy_rife_backend`：Proxy 恢复使用的 RIFE 后端（ncnn_vulkan/pytorch）。
  - `--proxy_removal_strategy`：删除策略（ranked 为高 IoU + 低运动优先；bucketed 为按时间分桶后在桶内优先删除）。
  - `--proxy_removal_buckets`：bucketed 删除策略下的分桶数量。
  - `--proxy_preflight_cache_csv`：缓存每轮 IoU 明细到 CSV，并输出 IoU 曲线 PNG。
- 2026-01-19：新增 Step1.5.5 Proxy IoU 预飞行与关键帧预算重分配，输出 `keyframes_opt.mp4` + `keyframes_opt__frame_map.json` 并自动用于后续 Step3/Step4/Step5。
- 2026-01-14：Step5.5/Step6 机械臂相关流程统一复用 `<workdir>/qwen_sam` 缓存，Step6 只做 swap-only 合成；新增 `sam_cache/sam_iou` time_steps 模式与 `--arm_swap_workdir_write` IoU 输出能力说明。
- 2026-01-13：RIFE pytorch 后端新增 guided 非均匀补帧（frame_map/segment_frames + time_steps），支持 SAM+zarm_pos 融合自动生成并带缓存与调试日志。
- 2026-01-13：parquet_motion 关键帧采样改为累计运动量目标的唯一化映射，确保严格递增且数量等于目标帧数，并补充 plateau 场景回归测试。
- 2026-01-13：parquet_motion 关键帧采样在运动不足时退化为均匀采样并打印原因，且保证包含首尾帧与严格递增。
- 2026-01-13：新增 parquet-motion 关键帧采样，输出 16fps 的 `keyframes.mp4` 并作为 Transfer2.5 输入；支持按 chunk 数自动计算目标帧数。
- 2026-01-13：新增 `requirements.txt`，列出本仓库自身依赖（`numpy`、`pyarrow`），便于 base 环境安装。
- 2026-01-12：parquet_motion 关键帧输出目录新增 sampling_tag（包含关键帧抽样配置与哈希），避免不同抽样参数互相覆盖。
- 2026-01-12：parquet_motion 关键帧输出新增 frame_map（与 keyframes.mp4 同名），用于 RIFE guided 补帧按原始帧索引恢复非均匀间隔。
- 2026-01-11：parquet_motion 新增 `--parquet_pose_subset`（可选 zarm pos/vel/hand 子集或 full），并在 keyframe 运动量计算中按 meta/info.json 名称选择维度，支持 stats.json 标准化与更稳健的 parquet 路径解析。
- 2026-01-10：移除 `--pre_fps` 选项，`pre_chunk_mode=speed` 改为均匀抽帧并降低 fps，保持原视频时长不变。
- 2026-01-09：Step6 arm swap 默认拆成 Qwen+SAM 两步运行，新增 `--arm_swap_split/--arm_swap_single` 控制；split 产物 bbox JSON 默认清理，可用 `--arm_swap_keep_bbox_json` 保留。
- 2026-01-09：Step6 arm swap 新增 `--no-workdir-write`（默认开启内存模式避免 PNG 落盘），CLI 暴露 `--arm_swap_no_workdir_write/--arm_swap_workdir_write` 控制开关。
- 2026-01-08：RIFE pytorch 后端新增 `--rife_pytorch_env`（默认 rife），支持在 base 环境下通过 `conda run` 调用 Practical-RIFE。
- 2026-01-08：Step6 arm swap 根据原视频进行 swap，并新增 `--disable_guardrails` 透传到 Transfer2.5。
- 2026-01-07：`--post_restore_duration` 优先对齐原视频帧数与 FPS，保证输出帧数与帧率完全一致（即使时长不再强制百分百对齐）。
- 2026-01-07：新增 `--post_restore_mode rife`，支持使用 RIFE 进行补帧（含 `--rife_bin`/`--rife_model`）。
- 2026-01-06：RIFE 后处理新增 32 对齐 pad/crop 修复条纹伪影，RIFE 子进程移除 DISPLAY/WAYLAND_DISPLAY，默认 RIFE 路径更新，并将中间缓存放入 `/mnt/data/ganrunhe/auto_transfer/rife_cache`（按 job 目录隔离，完成后自动清理）。
- 2026-01-06：新增 Step1.5 预处理对齐 Transfer chunk 帧数（`--pre_chunk_mode/--pre_chunk_policy/--pre_chunk_clip_where/--pre_chunk_max_adjust`），对齐后的视频统一用于 Step3 control 与 Step5 Transfer 推理，确保帧数一致。
- 2026-01-05：新增 Transfer2.5 推理加速参数 `--resolution`（写入 STEP4 JSON），并补充 resolution->16:9 bucket 尺寸对照表；明确 `--pre_width/--pre_height` 仅影响预处理/控制生成，不决定推理 bucket。
- 2026-01-04：新增视频预处理（CFR + 尺寸缩放）参数 `--pre_width/--pre_height/--pre_no_keep_aspect`，预处理结果复用原视频目录并用于后续控制/推理。
- 2026-01-04：Step6 机械臂替换切换为单次 `robot_arm_swap` 调用，固定使用 conda 环境 `arm_swap`。
- 2026-01-04：Step3 替换候选移除内部 step3_llm_cache 缓存，候选不足时使用词库组合（含双色组合）补齐到 `--replacements_per_term`，三部曲逻辑说明同步更新。
- 2026-01-04：新增断点续跑原子标记文件（Step2~Step6 + job 完成），支持批处理完成后自动清空标记目录。
- 2025-12-31：Step2 在遇到 PyAV 缺失时自动用 ffmpeg 转码视频并重试，避免任务直接失败。
- 2025-12-31：支持递归扫描输入 mp4 并按原始目录结构输出，最外层新增 `_var_x` 变体目录；arm swap 输出保持同样的目录镜像规则。
- 2025-12-31：新增 sim2real 任务开关 `--sim2real_only`，Quickstart 增补 Step1-only 调用示例。
- 2025-12-31：更新 video_to_prompt 三部曲说明，补充 Step2 子串清理、Step3 同体对齐与 step3_apply 无候选时的行为说明。
- 2025-12-31：新增视频预处理（CFR + 尺寸缩放）参数 `--pre_width/--pre_height/--pre_no_keep_aspect`，预处理结果复用原视频目录并用于后续控制/推理。
- 2025-12-30：整理 video_to_prompt 三部曲逻辑说明，补充 Step1.5 兜底、Step2 括号反思/扩展、Step3 替换一致性与补齐策略。
- 2025-12-30：Step2 名词同一体扩展提升优先级，允许在含机器人描述的句子中补圈共享名词短语。
- 2025-12-30：Step2 括号清理新增机器人关键词与“background consists of …”过滤，并移除不含名词的纯形容词括注。
- 2025-12-30：Step2 新增括号反思校验与名词同一体扩展（如已有 table surface 会补圈 white table），并过滤 “background consists of …” 类背景短语。
- 2025-12-30：Step3 替换候选新增材质词保留与共享名词一致性对齐（如 table/table surface 同步变化），并放宽替换短语长度上限以提升多样性。
- 2025-12-30：Step3 替换候选若命中模糊词/机器人词/重复将触发重新调用模型补齐结果，不再仅清洗回退。
- 2025-12-30：Step3 改为模型生成替换词，并在替换后进行模糊词纠正（含 smooth 归类）与母串/子串一致性对齐。
- 2025-12-30：Step2 新增句内逗号切分上下文（pre_2.0），避免与机器人同句时误过滤非机器人描述。
- 2025-12-30：Step2 改为基于模型抽取描述短语并用机器人上下文过滤，移除规则化补圈与相关词表依赖。
- 2025-12-29：Step3 替换新增多色组合候选，支持“多种颜色拼接”的描述以提升多样性。
- 2025-12-29：增强 Step2 的括号标注逻辑，基于规则补充更多非机器人颜色/材质/纹理短语，减少漏标，提升可替换描述的多样性。
- 2025-12-29：多卡批处理队列按 variant 轮询视频顺序入队，优先让不同 GPU 处理不同视频，减少同视频变体并发导致的缓存争用。
- 2025-12-25：Step2 括号术语抽取新增“长短语优先”去重策略，避免 `gray fabric` 与 `fabric` 这类同一实体被拆成两个替换单元。
- 2025-12-25：新增交互式替换选项，可在 Step2 括号词列表中手动选择并输入替换词；启用后自动清空 Step3 替换缓存并跳过自动候选生成。
- 2025-12-25：Step3 替换候选明确禁止使用 “pale/light/dark”等模糊修饰词，并在替换合并时过滤模糊颜色短语，提升颜色替换的可控性与多样性。
- 2025-12-25：seg json spec 简化为仅包含 `control_path`/`control_weight`；新增 combo 多控制信号（depth+edge+seg+vis preset blur），支持 `--depth_control_weight`/`--edge_control_weight`/`--seg_control_weight`/`--vis_control_weight` 调整权重。
- 2025-12-25：新增 edge/blur 控制信号，基于 `gen_controls_cv_mp4.py` + conda 环境 `sam` 自动生成并按指纹缓存复用，CLI 支持 `--control_type edge|blur` 及相关参数。
- 2025-12-24：Step2 括号标注提示新增“完整名词短语”要求，减少同一物体被拆成多个替换单元的情况。
- 2025-12-24：Step3 替换同一括号术语时保持一致，避免相同颜色/描述被随机替换为不同词。
- 2025-12-24：Step2 输出新增机器人括号清理兜底，移除机器人相关括注以确保替换安全。
- 2025-12-24：放宽 Step2 括号标注规则，允许同句非机器人外观描述继续括注，增加更多纹理/表面细节指引。
- 2025-12-24：替换 Step1 生成提示词为更严格的时序示例模板，强调静态物体分句描述、容器初始状态与操纵物体登场顺序。
- 2025-12-24：更新 `robot_video_to_prompt.py` 的 Step1 提示词与 Step2 括号规则：新增容器/操纵物体初始状态要求、禁止模糊词，并加入 Step1.5 兜底校验（容器初始状态补写与模糊词清理）。
- 2025-12-24：在自动生成的 json spec 中增加默认 `negative_prompt` 关键词，抑制额外物体、抖动、伪影等不良现象。
- 2025-12-24：更新 `robot_video_to_prompt.py` 的提示文本：新增控制通用稳定性规则，限制括注范围，避免生成过暗/光照相关的替换词。
- 2025-12-24：精简 `robot_video_to_prompt.py` 三阶段提示词，并在 Step1 后加入机器人描述规范化，避免机器人外观被改写。
- 2025-12-23：补强 `robot_video_to_prompt.py` 三步式 prompt：新增机器人基座 vs 台面软物的严格区分、底座颜色保持中性表述，以及可见性变化动词限制，并添加针对此规则的回归测试用例。
- 2025-12-23：更新 `robot_video_to_prompt.py` 三步式 prompt：机器人部件颜色/纹理禁入括号与替换，禁用模糊/诗意色词；新增机器人相关描述符过滤兜底，避免替换机器人部件颜色/纹理，并强调背景污渍闭环与终态总结。
- 2025-12-23：修复 seg 控制生成需要帧目录的报错，流水线自动用 ffmpeg 抽帧再调用 AutoSeg-SAM2，结果仍按指纹缓存到 `control_cache/seg/`，CLI 直接通过 `--control_type seg` 调用。
- 2025-12-23：调整 `robot_video_to_prompt.py` 的颜色安全规则：禁止使用“deep black/dark black”等叠加深色描述；避免“grayish/dark-colored”等模糊颜色词汇；黑色/深色/灰色可直接使用但需简洁明确，不确定时选定一个具体颜色即可。
- 2025-12-23：集成 seg 控制信号自动化，使用 AutoSeg-SAM2 在 `control_cache/seg/` 按指纹落盘并复用，CLI 直接通过 `--control_type seg` 调用。
- 2025-12-23：调整 Step1 base prompt 文案为官方演示叙事风格；Step2 括号替换限定为颜色/纹理描述，Step3 只生成颜色或纹理的替换词。
- 2025-12-23：新增 `--replace_all_terms` 参数，可直接替换所有括号标注的描述（跳过随机抽样逻辑）。
- 2025-12-23：新增 `--replacements_per_term` 参数，可调整每个括号术语生成的替换候选数量（默认 64），并对缓存进行区分。
- 2025-12-22：`--min_terms` / `--max_terms` 支持快速设置随机替换数量区间（默认 1~3），便于控制改写力度。
- 2025-12-22：将路径与环境默认值集中到 `config.py`（CLI 统一读取），并在 README 增补路径修改指引。
- 2025-12-22：新增 Step2+Step3 合并的 `step23` 子命令与三层 prompt 缓存（fingerprint/base_hash/bracket_hash），同一视频/文本多 worker 会被文件锁串行化，仅 Step3_apply 在换肤时毫秒级复用。
- 2025-12-22：Prompt 生成流程适配拆分版 `robot_video_to_prompt.py`（step1~step3），新增基于视频指纹的 base prompt 缓存（`prompt_cache/`），默认使用本仓库内的脚本路径；控制台打印增加 GPU 标签。
- 2025-12-22：为终端输出统一添加东八区时间戳，并在 Step2~Step5（含清理）及整条流水线上记录耗时，便于观测性能。
- 2025-12-21：适配新版 `robot_video_to_prompt.py`（支持 Qwen3/Qwen2.5 的 `--model_type` 参数），新增命令行参数 `--prompt_model_type`，默认 conda 环境保持 `qwen25vl`，可选切换为 `qwen3vl`。
- 2026-01-22：强化 `safe_parse_json` 的容错解析，并在任务失败时将日志备份到 `log_fail/` 目录。

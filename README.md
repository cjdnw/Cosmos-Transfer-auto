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
在 Step5 Transfer 输出完成后，自动调用 `robot_arm_swap` 生成机械臂替换版本（两个输出都会保留）。
```bash
python scripts/run_batch.py \
  --input_dir /mnt/data/ganrunhe/auto_transfer/raw_mp4 \
  --gpus 0 \
  --control_type depth \
  --enable_arm_swap \
  --arm_swap_root /mnt/data/ganrunhe/robot_arm_swap_project \
  --arm_swap_env robot_arm_swap \
  --arm_swap_output_root /mnt/data/ganrunhe/auto_transfer/output_video_mask_arm \
  --arm_swap_workdir_root /mnt/data/ganrunhe/auto_transfer/arm_swap_cache \
  --arm_swap_fps 30 \
  --arm_swap_width 1280 \
  --arm_swap_prompt "robotic arm"
```

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
- fingerprint 默认使用 `stat`（路径 + 文件大小 + 修改时间），速度快；如需更稳健（读完整文件算 hash），使用：`--fingerprint_mode md5`

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

## 待完成项目
- depth 控制信号质量不佳，需研究提升方案。
- seg 当前使用 sam1 全自动单帧分割 + sam2 分割跟踪，可考虑改为 sam1+sam3 或 sam1+sam3+Cuite，待研究。

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
以下对应 `robot_video_to_prompt.py` 的 Step1 ~ Step3 全流程（含 Step1.5 与 Step3_apply），用于从视频生成基础描述、标注可替换短语，并完成随机替换：

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
   - 若候选数量不足，触发 `_retry_replacements_for_term` 重新补齐。
3. 对同一物体的替换候选做一致性对齐，避免前后替换冲突。
4. 结果写入 `prompt_cache/step3_llm_cache`，按内容与候选数量缓存复用。

**随机替换流程**（`apply_random_replacements` / `step3_apply`）：
- 默认随机抽取 `min_terms ~ max_terms` 个括注短语进行替换；启用 `--replace_all_terms` 则全部替换。
- 若 `step3_apply` 未传 `--repl_json`，仅去掉括号，不做替换。

**输出**：
- `replacements.json`（每个术语的替换候选列表）
- `final.txt`（完成替换的最终 prompt）

## 更新记录
- 2025-12-31：Step2 在遇到 PyAV 缺失时自动用 ffmpeg 转码视频并重试，避免任务直接失败。
- 2025-12-31：支持递归扫描输入 mp4 并按原始目录结构输出，最外层新增 `_var_x` 变体目录；arm swap 输出保持同样的目录镜像规则。
- 2025-12-31：新增 sim2real 任务开关 `--sim2real_only`，Quickstart 增补 Step1-only 调用示例。
- 2025-12-31：更新 video_to_prompt 三部曲说明，补充 Step2 子串清理、Step3 同体对齐与 step3_apply 无候选时的行为说明。
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

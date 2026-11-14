# Latency test

原文 §4.2：SD1.5、SD2.0-base、SD2.0、SDXL-base；20 steps、CFG 7.0；单张 A800；MCP 使用 GPT-4o-mini。
原文没有披露软件版本、分辨率、dtype、scheduler、batch size、精确 seeds 或 warmup。
下面固定的是**本次时延实验环境**，不能写成原论文提供的环境。

## Environment

Ubuntu 22.04 / Python 3.10.16 / NVIDIA A800（显存容量以实际设备为准）。
PyTorch 2.6.0、torchvision 0.21.0、CUDA runtime 12.4；其余版本见 requirements-latency.txt。
驱动需兼容 CUDA 12.4。程序会保存实际 GPU、驱动、CUDA 和包版本。

```bash
conda create -n nlp805-latency python=3.10.16 -y
conda activate nlp805-latency
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-latency.txt
python -m pip check
mkdir -p runs
python -m pip freeze > runs/latency_environment.txt
nvidia-smi > runs/latency_gpu.txt
```

本次额外设置：统一 512×512、FP16、batch=1、seeds 0–9、warmup=1、无 plan cache。
不启用 CPU offload、torch.compile、xFormers 或 SDXL refiner；保留 checkpoint 的 scheduler。
这些设置在 latency_settings.json 中集中保存，所有配对条件须相同。

## Run

先把 latency_settings.json 中四个 path_or_repo 改为**原实验 checkpoint 的本地 Diffusers 目录**。
目录需要 model_index.json；程序会在计时前计算权重/配置摘要。
也支持固定 40 位 commit SHA 的 Hub repo；程序不会自动换成社区镜像。

```bash
# 先用一条 prompt 检查，检查结果与正式结果分开存。
CUDA_VISIBLE_DEVICES=0 python benchmark_latency.py --prompts data/eval.json --model sd15 --limit 1 --output runs/smoke_sd15.jsonl

# 确认 checkpoint 和 prompt 文件后再跑四个骨干。
for model in sd15 sd20-base sd20 sdxl-base; do
  CUDA_VISIBLE_DEVICES=0 python benchmark_latency.py --prompts data/eval.json --model "$model"
done

python summarize_latency.py runs/latency_*.jsonl --output runs/latency_summary.json --csv runs/latency_summary.csv
```

现有 data/eval.json 是早期的 50 条 pilot prompts，不是原文六项指标各 300 条的完整数据。
正式对齐主实验时，应换成真实评测 prompts；程序也接受 JSONL 的 prompt_id / prompt 字段。
输出已存在时程序会停止，避免混入重复记录。

## Real MCP/HCD

当前仓库的 scene_plan.py 只做场景计划校验，历史交付没有真实 MCP/HCD 推理实现。
因此 direct 可以直接测试；sceneplan 条件必须通过 --runner module:factory 接入原推理代码。
factory 接收 model、checkpoint、revision、settings，返回：

- generate(prompt, seed, timer)：返回前必须完成实际图像生成和 decode。
- metadata：记录实际 scheduler（class/config）、mcp_model（GPT-4o-mini 或其快照）、plan_cache=False，以及原方法其他参数。

在真实代码中用 timer.stage("mcp") 和 timer.stage("hcd", gpu=True) 包住两个阶段；
用 timer.agent(name) 包住每次实际 API 请求，重试也单独计数。
不要改变原有并行方式，不要把多个普通文生图调用当作 HCD。

总时延包含实际 API/network、CPU 和 GPU 工作，排除模型加载、warmup、评测打分与保存图片。
加载时间单列。阶段为 inclusive；并行/嵌套 agent 时间不能再次加到 MCP 上。
GPU memory 是单卡 PyTorch allocator 的 allocated/reserved 峰值，不含远程 LLM 的显存。
失败记录单列并停止，不能用 0 秒计入均值。统计输出包含 mean / median / P95 和配对 overhead。

Sources: [paper](https://arxiv.org/abs/2505.02648), [PyTorch install](https://pytorch.org/get-started/previous-versions/#v260), [Diffusers SD API](https://huggingface.co/docs/diffusers/v0.32.2/api/pipelines/stable_diffusion/text2img).

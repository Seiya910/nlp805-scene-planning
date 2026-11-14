"""原文四个 backbone 的 warm latency；不重写 MCP/HCD 算法。

自定义 --runner module:factory 需返回 generate(prompt, seed, timer) 的对象，
并提供 metadata['scheduler']。真实 agent 请求用 timer.agent(name) 包住，
MCP/HCD 分别用 timer.stage('mcp') / timer.stage('hcd', gpu=True)。
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

PIPELINES = {
    "sd15": "StableDiffusionPipeline",
    "sd20-base": "StableDiffusionPipeline",
    "sd20": "StableDiffusionPipeline",
    "sdxl-base": "StableDiffusionXLPipeline",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checkpoint_hash(path):
    # 只做一次文件摘要，放在计时之前，避免同一路径换权重后误配对。
    result = hashlib.sha256()
    for file in sorted(Path(path).rglob("*")):
        if file.is_file() and file.suffix in (".json", ".safetensors", ".bin", ".txt") and ".cache" not in file.parts:
            result.update(str(file.relative_to(path)).encode())
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    result.update(chunk)
    return result.hexdigest()


def read_prompts(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".json":
        rows = json.loads(text)
    elif Path(path).suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompt file must contain a nonempty list")
    result, ids = [], set()
    for i, row in enumerate(rows, 1):
        row = {"prompt_id": f"p{i:04d}", "prompt": row} if isinstance(row, str) else row
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"invalid prompt at row {i}")
        prompt_id = str(row.get("prompt_id", f"p{i:04d}"))
        if not prompt_id.strip() or prompt_id in ids:
            raise ValueError(f"empty or duplicate prompt_id: {prompt_id}")
        ids.add(prompt_id)
        result.append((prompt_id, row["prompt"]))
    return result


class Timer:
    def __init__(self, sync):
        self.sync = sync
        self.seconds = defaultdict(float)
        self.calls = Counter()
        self.lock = threading.Lock()

    @contextmanager
    def stage(self, name, gpu=False):
        # API/CPU 阶段不做 CUDA sync，避免改变并行 agent 的执行方式。
        if gpu:
            self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if gpu:
                self.sync()
            with self.lock:
                self.seconds[name] += time.perf_counter() - start

    @contextmanager
    def agent(self, name):
        # 每次实际请求记一次；若有 retry，需要在每次请求处分别计数。
        with self.lock:
            self.calls[name] += 1
        with self.stage("agent:" + name):
            yield


class DirectRunner:
    def __init__(self, model, checkpoint, revision, settings):
        import diffusers
        import torch
        options = {"torch_dtype": getattr(torch, settings["dtype"])}
        if revision:
            options["revision"] = revision
        cls = getattr(diffusers, PIPELINES[model])
        self.pipe = cls.from_pretrained(checkpoint, **options).to("cuda:0")
        self.pipe.set_progress_bar_config(disable=True)
        self.settings = settings
        # 保留 checkpoint 自带 scheduler，不擅自换成 DDIM/Euler。
        self.metadata = {"scheduler": {"class": type(self.pipe.scheduler).__name__,
                                      "config": dict(self.pipe.scheduler.config)},
                         "pipeline": type(self.pipe).__name__,
                         "safety_checker": getattr(self.pipe, "safety_checker", None) is not None}

    def generate(self, prompt, seed, timer):
        import torch
        s = self.settings
        with torch.inference_mode(), timer.stage("generation", gpu=True):
            generator = torch.Generator(device="cuda:0").manual_seed(seed)
            self.pipe(prompt, generator=generator, width=s["width"], height=s["height"],
                      num_inference_steps=s["steps"], guidance_scale=s["guidance_scale"],
                      negative_prompt=s["negative_prompt"], num_images_per_prompt=1,
                      output_type="pil")  # 包含 VAE decode，不包含保存图片。


def environment(torch):
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate",
                 "safetensors", "huggingface-hub", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    return {"python": platform.python_version(), "os": platform.platform(),
            "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
            "driver": driver, "packages": packages}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", required=True)
    p.add_argument("--model", choices=PIPELINES, required=True)
    p.add_argument("--condition", choices=("direct", "sceneplan"), default="direct")
    p.add_argument("--config", default="latency_settings.json")
    p.add_argument("--checkpoint", help="原实验的本地 Diffusers checkpoint，或固定 revision 的 Hub repo")
    p.add_argument("--revision", help="Hub checkpoint 必须给出 40 位 commit SHA")
    p.add_argument("--runner", help="真实推理适配器 module:factory；不提供时只支持 direct")
    p.add_argument("--limit", type=int, help="只跑前 N 条，便于 smoke test")
    p.add_argument("--allow-other-gpu", action="store_true", help="测试其他 GPU，结果会记录实际型号")
    p.add_argument("--output")
    args = p.parse_args()
    if args.condition == "sceneplan" and not args.runner:
        p.error("ScenePlan needs the real MCP/HCD --runner; it is not implemented in this repository")
    if args.runner and ":" not in args.runner:
        p.error("runner must be module:factory")
    if args.limit is not None and args.limit < 1:
        p.error("limit must be positive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    s, timing = config["generation"], config["timing"]
    if s["steps"] != 20 or s["guidance_scale"] != 7.0:
        p.error("paper setting: 20 steps and CFG 7.0")
    if s["batch_size"] != 1 or s["scheduler"] != "checkpoint" or s["plan_cache"] is not False:
        p.error("this protocol uses batch=1, checkpoint scheduler and uncached plans")
    if any(type(s[k]) is not int or s[k] < 8 or s[k] % 8 for k in ("width", "height")):
        p.error("width/height must be positive multiples of 8")
    if s["dtype"] not in ("float16", "bfloat16", "float32"):
        p.error("invalid dtype")
    seeds = timing["seeds"]
    if (not seeds or any(type(x) is not int or x < 0 for x in seeds)
            or len(seeds) != len(set(seeds))):
        p.error("seeds must be unique nonnegative integers")
    if type(timing["warmup"]) is not int or timing["warmup"] < 0 or type(timing["repeats"]) is not int or timing["repeats"] < 1:
        p.error("invalid warmup/repeats")
    model_config = config["checkpoints"][args.model]
    checkpoint = args.checkpoint or model_config["path_or_repo"]
    revision = args.revision or model_config["revision"]
    if not checkpoint:
        p.error("set --checkpoint to the original model; no automatic community-mirror substitution")
    if Path(checkpoint).is_dir():
        checkpoint = str(Path(checkpoint).resolve())
        if not (Path(checkpoint) / "model_index.json").is_file():
            p.error("local checkpoint needs Diffusers model_index.json")
        checkpoint_identity = checkpoint_hash(checkpoint)
    elif not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        p.error("Hub checkpoints need an immutable 40-character --revision")
    else:
        checkpoint_identity = checkpoint + "@" + revision
    output = Path(args.output or f"runs/latency_{args.model}_{args.condition}.jsonl")
    if output.exists():
        p.error(f"output already exists: {output}; choose a new file")
    prompts = read_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error("use one visible CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0")
    env = environment(torch)
    if "A800" not in env["gpu"] and not args.allow_other_gpu:
        p.error("paper uses A800; --allow-other-gpu labels a different-hardware test")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sync = lambda: torch.cuda.synchronize(0)
    sync()
    start = time.perf_counter()
    factory = DirectRunner
    runner_entry = Path(__file__)
    if args.runner:
        module, name = args.runner.split(":", 1)
        module = importlib.import_module(module)
        factory = getattr(module, name)
        runner_entry = Path(module.__file__)
    runner = factory(model=args.model, checkpoint=checkpoint, revision=revision, settings=s)
    sync()
    load_s = time.perf_counter() - start
    if not isinstance(getattr(runner, "metadata", None), dict) or "scheduler" not in runner.metadata:
        p.error("runner.metadata must include the actual scheduler class/config")
    if args.condition == "sceneplan":
        mcp_model = runner.metadata.get("mcp_model", "")
        if not (mcp_model == "gpt-4o-mini" or mcp_model.startswith("gpt-4o-mini-")):
            p.error("real runner.metadata must identify the actual GPT-4o-mini model/snapshot")
        if runner.metadata.get("plan_cache") is not False:
            p.error("real runner.metadata must confirm plan_cache=False for end-to-end timing")
    for _ in range(timing["warmup"]):
        runner.generate(prompts[0][1], seeds[0], Timer(sync))
        sync()

    comparison = {"model": args.model, "checkpoint": checkpoint, "revision": revision,
                  "checkpoint_identity": checkpoint_identity,
                  "generation": s, "scheduler": runner.metadata["scheduler"],
                  "environment": env, "timing": timing}
    run_id = uuid.uuid4().hex
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for prompt_id, prompt in prompts:
            for seed in seeds:
                for repeat in range(timing["repeats"]):
                    timer = Timer(sync)
                    sync()
                    torch.cuda.reset_peak_memory_stats(0)
                    start = time.perf_counter()
                    error = None
                    try:
                        runner.generate(prompt, seed, timer)
                        sync()
                        if args.condition == "sceneplan" and (
                            not {"mcp", "hcd"}.issubset(timer.seconds) or not timer.calls):
                            raise ValueError("real runner must time mcp/hcd and count API requests")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - start
                    row = {"run_id": run_id, "comparison_id": digest(comparison), **comparison,
                           "condition": args.condition, "runner": args.runner or "diffusers",
                           "runner_entry_sha256": hashlib.sha256(runner_entry.read_bytes()).hexdigest(),
                           "runner_metadata": runner.metadata, "model_load_s": load_s,
                           "prompt_id": prompt_id, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                           "seed": seed, "repeat": repeat, "status": "failed" if error else "ok",
                           "latency_s": None if error else elapsed, "error": error,
                           "stage_seconds": dict(timer.seconds), "agent_calls": dict(timer.calls),
                           "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 2**20,
                           "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 2**20}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    if error:
                        raise SystemExit(f"stopped; failure saved in {output}: {error}")
                    print(f"{args.model} {args.condition} {prompt_id} seed={seed}: {elapsed:.3f}s")


if __name__ == "__main__":
    main()

"""Measure per-image inference latency without including model loading or warmup.

The built-in runner measures direct generation. To measure ScenePlan, pass a
``module:factory`` with --runner. The factory receives model_id, revision,
device, and dtype, and returns an object with generate(prompt, seed, steps,
guidance_scale, width, height, timer). The object is loaded once. Inside
generate(), wrap real work in ``with timer.stage("mcp")`` and
``with timer.stage("hcd")``; wrap each agent request in
``with timer.agent("agent_name")``. Stage times are inclusive, so an agent
stage inside MCP must not be added to MCP time again.
"""

import argparse
import importlib
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path


# SD2 repositories below are currently accessible community mirrors. Override
# --model-id if the original experiment used a different checkpoint/revision.
MODEL_IDS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd20-base": "sd2-community/stable-diffusion-2-base",
    "sd20": "sd2-community/stable-diffusion-2",
    "sdxl-base": "stabilityai/stable-diffusion-xl-base-1.0",
}
DEFAULT_SIZE = {"sd15": 512, "sd20-base": 512, "sd20": 768, "sdxl-base": 1024}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_prompts(path):
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("JSON prompts must be a list")
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    prompts = []
    seen = set()
    for index, row in enumerate(rows, 1):
        if isinstance(row, str):
            prompt_id, prompt = f"p{index:04d}", row.strip()
        elif isinstance(row, dict):
            prompt_id, prompt = row.get("prompt_id"), row.get("prompt")
        else:
            raise ValueError(f"prompt {index} must be text or an object")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt {index} needs a nonempty prompt_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt {index} needs nonempty text")
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt_id: {prompt_id}")
        seen.add(prompt_id)
        prompts.append((prompt_id, prompt))
    if not prompts:
        raise ValueError("no prompts found")
    return prompts


class DeviceClock:
    def __init__(self, device):
        self.device = device
        try:
            import torch
        except ImportError:
            torch = None
        self.torch = torch
        if device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("CUDA is unavailable; run on the experiment GPU or use --device cpu for a smoke test")

    def sync(self):
        if self.torch is None:
            return
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elif self.device == "mps" and hasattr(self.torch.mps, "synchronize"):
            self.torch.mps.synchronize()

    def hardware(self):
        if self.torch is not None and self.device.startswith("cuda"):
            return self.torch.cuda.get_device_name(self.device)
        return platform.processor() or platform.machine()

    def reset_peak(self):
        if self.torch is None or not self.device.startswith("cuda"):
            return None
        self.sync()
        baseline = self.torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        return baseline

    def peak(self):
        if self.torch is None or not self.device.startswith("cuda"):
            return None, None
        self.sync()
        return (
            self.torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
            self.torch.cuda.max_memory_reserved(self.device) / (1024 ** 2),
        )


class Timer:
    def __init__(self, clock):
        self.clock = clock
        self.stage_seconds = defaultdict(float)
        self.agent_calls = Counter()

    @contextmanager
    def stage(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("stage name must be nonempty")
        self.clock.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.clock.sync()
            self.stage_seconds[name] += time.perf_counter() - start

    @contextmanager
    def agent(self, name):
        self.agent_calls[name] += 1
        with self.stage(f"agent:{name}"):
            yield


class DiffusersRunner:
    def __init__(self, model_id, revision, device, dtype):
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError("direct generation needs torch and diffusers") from exc
        torch_dtype = getattr(torch, dtype)
        options = {"torch_dtype": torch_dtype}
        if revision:
            options["revision"] = revision
        self.pipeline = DiffusionPipeline.from_pretrained(model_id, **options).to(device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.torch = torch
        self.device = device

    def generate(self, prompt, seed, steps, guidance_scale, width, height, timer):
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode(), timer.stage("diffusion"):
            result = self.pipeline(
                prompt=prompt,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            _ = result.images[0]  # Include image decoding, but not disk writing.


def load_runner(spec, model_id, revision, device, dtype):
    if spec == "diffusers":
        return DiffusersRunner(model_id, revision, device, dtype)
    if ":" not in spec:
        raise ValueError("--runner must be 'diffusers' or module:factory")
    module_name, factory_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(model_id=model_id, revision=revision, device=device, dtype=dtype)


def measure(runner, clock, prompt, seed, settings):
    timer = Timer(clock)
    baseline = clock.reset_peak()
    clock.sync()
    start = time.perf_counter()
    runner.generate(prompt, seed, settings["steps"], settings["guidance_scale"],
                    settings["width"], settings["height"], timer)
    clock.sync()
    elapsed = time.perf_counter() - start
    peak_allocated, peak_reserved = clock.peak()
    return {
        "latency_s": elapsed,
        "stage_seconds": dict(timer.stage_seconds),
        "agent_calls": dict(timer.agent_calls),
        "gpu_baseline_allocated_mib": baseline,
        "gpu_peak_allocated_mib": peak_allocated,
        "gpu_peak_reserved_mib": peak_reserved,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="data/eval.json, JSONL, or one prompt per line")
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--model-id", help="override the checkpoint for this model label")
    parser.add_argument("--revision", help="optional fixed Hugging Face revision")
    parser.add_argument("--condition", choices=("direct", "sceneplan"), required=True)
    parser.add_argument("--runner", default="diffusers", help="diffusers or module:factory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, help="use only the first N prompts")
    parser.add_argument("--output", type=Path, default=Path("runs/latency.jsonl"))
    parser.add_argument("--append", action="store_true", help="append another model or condition")
    args = parser.parse_args()
    if args.condition == "sceneplan" and args.runner == "diffusers":
        parser.error("ScenePlan needs a real MCP/HCD --runner; direct Diffusers is not ScenePlan")
    if args.steps < 1 or args.repeats < 1 or args.warmup < 0 or (args.limit is not None and args.limit < 1):
        parser.error("steps/repeats/limit must be positive and warmup must be nonnegative")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error("guidance scale must be finite and nonnegative")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    width = args.width if args.width is not None else DEFAULT_SIZE[args.model]
    height = args.height if args.height is not None else DEFAULT_SIZE[args.model]
    if width < 8 or height < 8 or width % 8 or height % 8:
        parser.error("width and height must be positive multiples of 8")
    if args.output.exists() and not args.append:
        parser.error(f"output exists: {args.output}; use --append or choose another file")

    prompts = read_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    model_id = args.model_id or MODEL_IDS[args.model]
    settings = {"steps": args.steps, "guidance_scale": args.guidance_scale,
                "width": width, "height": height}
    clock = DeviceClock(args.device)
    clock.sync()
    load_start = time.perf_counter()
    runner = load_runner(args.runner, model_id, args.revision, args.device, args.dtype)
    clock.sync()
    load_s = time.perf_counter() - load_start
    for _ in range(args.warmup):
        measure(runner, clock, prompts[0][1], args.seeds[0], settings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.append else "x", encoding="utf-8") as stream:
        for prompt_id, prompt in prompts:
            for seed in args.seeds:
                for repeat in range(args.repeats):
                    metrics = measure(runner, clock, prompt, seed, settings)
                    if args.condition == "sceneplan":
                        missing = {"mcp", "hcd"} - metrics["stage_seconds"].keys()
                        if missing or not metrics["agent_calls"]:
                            raise RuntimeError(
                                "ScenePlan runner must time mcp/hcd and count its agent calls; "
                                f"missing stages: {sorted(missing)}"
                            )
                    record = {
                        "model": args.model, "model_id": model_id, "revision": args.revision,
                        "condition": args.condition, "runner": args.runner,
                        "prompt_id": prompt_id, "seed": seed, "repeat": repeat,
                        "steps": args.steps, "guidance_scale": args.guidance_scale,
                        "width": width, "height": height,
                        "device": args.device, "dtype": args.dtype, "hardware": clock.hardware(),
                        "torch_version": package_version("torch"),
                        "diffusers_version": package_version("diffusers"),
                        "model_load_s": load_s,
                        **metrics,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{args.model} {args.condition} {prompt_id} seed={seed} "
                          f"repeat={repeat}: {metrics['latency_s']:.3f}s")


if __name__ == "__main__":
    main()

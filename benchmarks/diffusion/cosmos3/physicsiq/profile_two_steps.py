# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Time the first two original sampling steps, save replay inputs, and skip VAE decoding."""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from run_physicsiq import negative_from_yaml


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["npu", "cuda"], required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--submission-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        print("Report already exists; no model execution: " + str(output / "report.json"), flush=True)
        return
    if args.model_dir is not None:
        model = args.model_dir.expanduser().resolve()
        if not (model / "model_index.json").is_file():
            raise ValueError("Missing model_index.json in " + str(model))
    else:
        from serve_a10 import resolve_model

        model = resolve_model(None)
    row = json.loads(args.cases_json.read_text())["cases"][0]
    image_path = Path(row["image"])
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
        raise ValueError("Reference image checksum mismatch")
    negative = negative_from_yaml(args.submission_yaml)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch

    if args.device == "npu":
        import torch_npu  # noqa: F401

        device = torch.npu
        dtype = "float16"
    else:
        device = torch.cuda
        dtype = "bfloat16"
    if not device.is_available():
        raise RuntimeError(args.device + " is unavailable")

    def sync():
        device.synchronize()

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(v) for v in value)
        return value

    from PIL import Image

    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline as Pipeline
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    state = {
        "enabled": False,
        "actual": False,
        "step": -1,
        "branch": "",
        "event_records": [],
        "branch_records": [],
        "step_records": [],
        "phase_records": [],
        "checkpoint_io_seconds": 0.0,
        "pending_attention": None,
        "attention_saved": False,
        "handles": [],
    }
    original = {
        name: getattr(Pipeline, name)
        for name in (
            "diffuse",
            "_set_denoise_step_metadata",
            "predict_noise",
            "_encode_conditioning_image_latent",
            "_decode_latents",
        )
    }

    def status(event, **kwargs):
        value = {"event": event, "time": time.time(), **kwargs}
        write_json(output / "progress.json", value)
        print(json.dumps(value), flush=True)

    @contextmanager
    def checkpoint():
        sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            sync()
            state["checkpoint_io_seconds"] += time.perf_counter() - started

    def register(module, category, name, save_attention=False):
        stack = []

        def before(current, inputs):
            if not state["actual"]:
                return
            start = device.Event(enable_timing=True)
            start.record()
            stack.append((start, state["step"], state["branch"]))
            if save_attention and state["step"] == 1 and state["branch"] == "cond" and not state["attention_saved"]:
                if state["pending_attention"] is None:
                    state["pending_attention"] = tuple(v.detach() for v in inputs[:3])

        def after(current, inputs, result):
            if not state["actual"]:
                return
            start, step, branch = stack.pop()
            end = device.Event(enable_timing=True)
            end.record()
            state["event_records"].append((step, branch, category, name, start, end))

        state["handles"].append(module.register_forward_pre_hook(before))
        state["handles"].append(module.register_forward_hook(after))

    def finish_step():
        if state["step"] < 0:
            return
        sync()
        elapsed = time.perf_counter() - state["step_started"]
        io = state["checkpoint_io_seconds"] - state["step_io_start"]
        record = {
            "index": state["step"],
            "wall_seconds_including_instrumentation": elapsed,
            "checkpoint_io_seconds": io,
            "wall_seconds_excluding_checkpoint_io": elapsed - io,
        }
        state["step_records"].append(record)
        status("step_complete", **record)

    def metadata(self, index, timesteps, scheduler):
        if state["actual"]:
            finish_step()
            state["step"] = index
            sync()
            state["step_started"] = time.perf_counter()
            state["step_io_start"] = state["checkpoint_io_seconds"]
            status("step_started", index=index, original_schedule_length=len(timesteps))
        return original["_set_denoise_step_metadata"](self, index, timesteps, scheduler)

    def predict(self, **kwargs):
        if not state["actual"]:
            return original["predict_noise"](self, **kwargs)
        branch = str(kwargs.get("_cache_context", "cond"))
        state["branch"] = branch
        if state["step"] == 1:
            with checkpoint():
                torch.save(
                    {
                        "kwargs": cpu(kwargs),
                        "cached_kv": cpu(self.transformer.cached_kv),
                        "cached_freqs_gen": cpu(self.transformer.cached_freqs_gen),
                    },
                    output / f"step1_{branch}_transformer_inputs.pt",
                )
        sync()
        started = time.perf_counter()
        result = original["predict_noise"](self, **kwargs)
        sync()
        seconds = time.perf_counter() - started
        state["branch_records"].append(
            {
                "step": state["step"],
                "branch": branch,
                "seconds": seconds,
                "latent_shape": list(kwargs["hidden_states"].shape),
                "text_shape": list(kwargs["text_ids"].shape),
            }
        )
        print(json.dumps({"event": "branch_complete", **state["branch_records"][-1]}), flush=True)
        if state["step"] == 1:
            with checkpoint():
                torch.save(cpu(result), output / f"step1_{branch}_prediction.pt")
        return result

    def encode(self, image_tensor):
        if not state["enabled"]:
            return original["_encode_conditioning_image_latent"](self, image_tensor)
        sync()
        started = time.perf_counter()
        result = original["_encode_conditioning_image_latent"](self, image_tensor)
        sync()
        state["phase_records"].append({"phase": "reference_image_vae_encode", "seconds": time.perf_counter() - started})
        return result

    def decode(self, latents):
        if not state["enabled"]:
            return original["_decode_latents"](self, latents)
        with checkpoint():
            torch.save(cpu(latents), output / "after_two_steps_latents.pt")
        status("decode_skipped", reason="Two-step performance probe; no full video or score produced.")
        return torch.zeros((1, 3, 1, 16, 16), dtype=torch.float32)

    def diffuse(self, *positional, **kwargs):
        if not state["enabled"]:
            return original["diffuse"](self, *positional, **kwargs)
        if len(kwargs["timesteps"]) != 20:
            raise ValueError("The original 20-step schedule must be retained")
        transformer = self.transformer
        register(transformer.language_model, "und", "language_model")
        for index, layer in enumerate(transformer.gen_layers):
            prefix = f"gen_layers.{index}"
            register(layer, "gen_layer_inclusive", prefix)
            register(
                layer.cross_attention.attn, "attention_core", prefix + ".attention_core", save_attention=index == 0
            )
            for part in ("to_q", "to_k", "to_v", "to_out"):
                register(getattr(layer.cross_attention, part), "attention_projections", prefix + "." + part)
            register(layer.mlp, "mlp", prefix + ".mlp")
            for part in ("input_layernorm", "post_attention_layernorm"):
                register(getattr(layer, part), "normalization", prefix + "." + part)
            for part in ("norm_q", "norm_k"):
                if hasattr(layer.cross_attention, part):
                    register(getattr(layer.cross_attention, part), "normalization", prefix + "." + part)
        progress_bar = self.progress_bar

        def first_two(iterable, *bar_args, **bar_kwargs):
            bar = progress_bar(iterable, *bar_args, **bar_kwargs)
            try:
                yield from itertools.islice(bar, 2)
            finally:
                if hasattr(bar, "close"):
                    bar.close()

        self.progress_bar = first_two
        state["actual"] = True
        state["step"] = -1
        try:
            result = original["diffuse"](self, *positional, **kwargs)
            finish_step()
            if [r["index"] for r in state["step_records"]] != [0, 1]:
                raise ValueError("Expected exactly original sampling steps 0 and 1")
            if state["pending_attention"] is not None:
                with checkpoint():
                    torch.save(cpu(state["pending_attention"]), output / "step1_cond_layer0_attention_inputs.pt")
                state["attention_saved"] = True
                state["pending_attention"] = None
            return result
        finally:
            state["actual"] = False
            self.progress_bar = progress_bar
            for handle in state["handles"]:
                handle.remove()
            state["handles"].clear()

    Pipeline.diffuse = diffuse
    Pipeline._set_denoise_step_metadata = metadata
    Pipeline.predict_noise = predict
    Pipeline._encode_conditioning_image_latent = encode
    Pipeline._decode_latents = decode
    omni = None
    try:
        status("initializing_model", device=args.device, dtype=dtype)
        omni = Omni(
            model=str(model),
            model_class_name="Cosmos3OmniDiffusersPipeline",
            dtype=dtype,
            enforce_eager=True,
            diffusion_attention_config={"default": "TORCH_SDPA"},
            model_config={"guardrails": False},
            init_timeout=1800,
            stage_init_timeout=1800,
        )
        sync()
        start = device.Event(enable_timing=True)
        end = device.Event(enable_timing=True)
        start.record()
        end.record()
        sync()
        start.elapsed_time(end)
        state["enabled"] = True
        status("generating_two_steps", case=row["name"])
        params = OmniDiffusionSamplingParams(
            height=480,
            width=832,
            num_frames=121,
            num_inference_steps=20,
            guidance_scale=3.25,
            seed=0,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        params.extra_args.update(
            {
                "flow_shift": 12.0,
                "frame_rate": 24.0,
                "max_sequence_length": 4096,
                "use_resolution_template": False,
                "use_duration_template": False,
            }
        )
        with Image.open(image_path) as image:
            reference = image.convert("RGB")
        omni.generate(
            {
                "prompt": row["prompt"],
                "negative_prompt": negative,
                "modalities": ["video"],
                "multi_modal_data": {"image": reference},
            },
            sampling_params_list=[params],
        )
        sync()
        aggregates = defaultdict(lambda: {"calls": 0, "device_interval_seconds": 0.0})
        details = []
        for step, branch, category, name, begin, end in state["event_records"]:
            seconds = begin.elapsed_time(end) / 1000.0
            key = f"step{step}.{category}"
            aggregates[key]["calls"] += 1
            aggregates[key]["device_interval_seconds"] += seconds
            details.append({"step": step, "branch": branch, "category": category, "module": name, "seconds": seconds})
        write_json(output / "module_intervals.json", details)
        report = {
            "status": "complete",
            "device": args.device,
            "dtype": dtype,
            "case": row["name"],
            "sampling": {
                "original_steps": 20,
                "executed_steps": [0, 1],
                "profile_step": 1,
                "width": 832,
                "height": 480,
                "frames": 121,
                "cfg": 3.25,
                "shift": 12,
                "seed": 0,
            },
            "branch_times": state["branch_records"],
            "step_times": state["step_records"],
            "phase_times": state["phase_records"],
            "module_aggregates": dict(aggregates),
            "checkpoint_io_seconds": state["checkpoint_io_seconds"],
            "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "torch": torch.__version__,
            "vllm": version("vllm"),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "notes": [
                "Device events measure stream intervals, including launch gaps, not kernel-only time.",
                "gen_layer_inclusive includes its attention/MLP/norm children: do not add them twice.",
                "Checkpoint I/O is excluded from each branch wall time; hooks and synchronization add overhead.",
                "Original 20-step scheduler and first two timesteps retained; full VAE decode deliberately skipped.",
                "One tiny placeholder image is returned internally only to finish the probe; no video is saved.",
                "CPU RNG used on both devices; this probe is not a quality evaluation.",
            ],
        }
        write_json(output / "report.json", report)
        status("complete", report=str(output / "report.json"))
        print(json.dumps(report), flush=True)
    finally:
        state["enabled"] = False
        for name, function in original.items():
            setattr(Pipeline, name, function)
        if omni is not None:
            omni.close()


if __name__ == "__main__":
    main()

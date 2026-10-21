# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Replay one captured attention call through the repository's SDPA backend, without model weights."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["npu", "cuda"], required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "attention_report.json"
    if report_path.exists():
        print(report_path.read_text(), flush=True)
        return
    import torch

    if args.device == "npu":
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)
        device = torch.npu
    else:
        device = torch.cuda
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    q, k, v = [value.to(args.device) for value in inputs]
    impl = SDPAImpl(
        num_heads=q.shape[2],
        num_kv_heads=k.shape[2],
        head_size=q.shape[3],
        softmax_scale=q.shape[3] ** -0.5,
        causal=False,
    )
    invoke = impl.forward_npu if args.device == "npu" else impl.forward_cuda
    with torch.inference_mode():
        device.synchronize()
        started = time.perf_counter()
        warm = invoke(q, k, v)
        device.synchronize()
        warm_seconds = time.perf_counter() - started
        del warm
        started = time.perf_counter()
        result = invoke(q, k, v)
        device.synchronize()
        unprofiled_seconds = time.perf_counter() - started
        torch.save(result.detach().cpu(), args.output_dir / "attention_output.pt")
        del result
        # CPU-only dispatcher trace identifies actual dispatched ATen operations.
        # It is not a device-kernel timing profile; synchronized wall time above is separate.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as trace:
            result = invoke(q, k, v)
            device.synchronize()
        del result
        trace.export_chrome_trace(str(args.output_dir / "attention_dispatch_trace.json"))
        events = [
            {
                "name": item.key,
                "calls": item.count,
                "input_shapes": item.input_shapes,
                "self_cpu_time_us": item.self_cpu_time_total,
            }
            for item in trace.key_averages(group_by_input_shape=True)
        ]
        (args.output_dir / "attention_operators.json").write_text(json.dumps(events, indent=2))
        report = {
            "device": args.device,
            "dtype": str(q.dtype),
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "warm_seconds": warm_seconds,
            "unprofiled_seconds": unprofiled_seconds,
            "interesting_dispatch": [
                item
                for item in events
                if any(
                    token in item["name"].lower()
                    for token in ("attention", "bmm", "softmax", "repeat_interleave", "to_copy")
                )
            ],
            "notes": [
                "Uses SDPAImpl from the deployed repository; no model weights loaded.",
                "Times include the repository GQA expansion and backend preparation.",
                "CPU dispatcher trace identifies operations, not device-kernel durations.",
                "One captured layer/branch only; do not treat this as complete model latency.",
            ],
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

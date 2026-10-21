# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physics-IQ Original I2V: prepare 198 cases, call Omni, and keep 120 generated frames."""

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def probe(path, frames):
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,nb_read_frames,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )["streams"]
    if len(data) != 1:
        raise ValueError("Expected a single video stream")
    s = data[0]
    value = (s["width"], s["height"], int(s["nb_read_frames"]), float(Fraction(s["r_frame_rate"])))
    if value != (832, 480, frames, 24.0):
        raise ValueError(f"Unexpected video properties: {path}: {value}")
    return s


def prepare(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with args.descriptions.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if "_take-1_" in r["scenario"]]
    rows.sort(key=lambda r: r["generated_video_name"])
    if len(rows) != 198:
        raise ValueError("Expected exactly 198 take-1 descriptions")
    cases = []
    for i, row in enumerate(rows, 1):
        name = row["generated_video_name"]
        if not re.fullmatch(r"\d{4}_[A-Za-z0-9_.-]+\.mp4", name) or not name.startswith(f"{i:04d}_"):
            raise ValueError("Unexpected case filename: " + name)
        images = list((args.data_root / "switch-frames").glob(f"{i:04d}_*.jpg"))
        if len(images) != 1:
            raise ValueError(f"Expected one switch-frame image for {name}, got {len(images)}")
        cases.append(
            {
                "name": name,
                "prompt": row["description"],
                "image": str(images[0].resolve()),
                "image_sha256": sha(images[0]),
            }
        )
    prepared = {"descriptions_sha256": sha(args.descriptions), "cases": cases}
    target = root / "cases.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != prepared:
        raise ValueError("Prepared inputs changed; use a new output directory")
    write(target, prepared)
    (root / "prompts.txt").write_text("\n".join(c["prompt"].replace("\n", " ") for c in cases) + "\n", encoding="utf-8")
    (root / "images.txt").write_text("\n".join(c["image"] for c in cases) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": 198, "model_requests": 0, "output": str(root)}), flush=True)


def negative_from_yaml(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^  NEGATIVE PROMPT\s*\n(.*?)^  PROMPTS\s*$", text, re.M | re.S)
    if match is None:
        raise ValueError("Official submission negative prompt section not found")
    return " ".join(match.group(1).split())


def run(args):
    root = args.output_dir.resolve()
    prepared = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    cases = prepared["cases"]
    limit = getattr(args, "limit", 198)
    if not 1 <= limit <= len(cases):
        raise ValueError("--limit must be between 1 and 198")
    negative = negative_from_yaml(args.submission_yaml)
    for i, c in enumerate(cases):
        if sha(c["image"]) != c["image_sha256"]:
            raise ValueError("Input image changed: " + c["name"])
        if args.upsampled_dir:
            p = args.upsampled_dir / f"prompt_{i}.json"
            payload = json.loads(p.read_text(encoding="utf-8"))
            prompt = payload["prompt"]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Invalid upsampled prompt")
            c["prompt"] = prompt
    config = {
        "cases": cases,
        "submission_sha256": sha(args.submission_yaml),
        "negative_prompt": negative,
        "parameters": {
            "width": 832,
            "height": 480,
            "frames": 121,
            "fps": 24,
            "steps": 20,
            "cfg": 3.25,
            "shift": 12.0,
            "seed": 0,
            "server_dtype": "bfloat16",
        },
        "prompt_kind": "upsampled-original" if args.upsampled_dir else "original-unmodified",
        "reference": {"original_seed0_score": 34.05},
        "comparison_notes": (
            "Official exact upsampled prompt outputs were not supplied; this is not an exact reproduction claim."
        ),
    }
    manifest = root / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != config:
        raise ValueError("Inputs or generation configuration changed; use a new output directory")
    write(manifest, config)
    for name in ("raw", "records"):
        (root / name).mkdir(exist_ok=True)
    staged_dir = root / "scoring_inputs" / root.name
    staged_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg and ffprobe are required")
    session = None
    if args.command == "generate":
        import requests

        session = requests.Session()
        session.get(args.base_url.rstrip("/") + "/health", timeout=20).raise_for_status()
    timings = []
    for i, c in enumerate(cases[:limit], 1):
        case_started = time.perf_counter()
        new_request = False
        raw = root / "raw" / c["name"]
        receipt = root / "records" / (Path(c["name"]).stem + ".json")
        staged = staged_dir / c["name"]
        stage_receipt = root / "records" / (Path(c["name"]).stem + ".stage.json")
        if raw.exists() and receipt.exists():
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if sha(raw) != record["sha256"]:
                raise ValueError("Raw video changed: " + str(raw))
            probe(raw, 121)
        elif args.command == "audit":
            raise FileNotFoundError("Missing raw video/record: " + c["name"])
        elif raw.exists():
            raise FileExistsError("Raw video exists without receipt: " + str(raw))
        else:
            data = {
                "prompt": c["prompt"],
                "negative_prompt": negative,
                "size": "832x480",
                "num_frames": "121",
                "fps": "24",
                "num_inference_steps": "20",
                "guidance_scale": "3.25",
                "flow_shift": "12.0",
                "max_sequence_length": "4096",
                "seed": "0",
                "extra_params": json.dumps(
                    {"use_resolution_template": False, "use_duration_template": False, "guardrails": False}
                ),
            }
            new_request = True
            print(
                json.dumps({"event": "request_started", "index": i, "selected": limit, "case": c["name"]}), flush=True
            )
            started = time.perf_counter()
            image = Path(c["image"])
            with image.open("rb") as stream:
                with session.post(
                    args.base_url.rstrip("/") + "/v1/videos/sync",
                    data=data,
                    files={
                        "input_reference": (image.name, stream, mimetypes.guess_type(image.name)[0] or "image/jpeg")
                    },
                    headers={"Accept": "video/mp4"},
                    timeout=(30, 7200),
                    stream=True,
                ) as response:
                    if not response.ok:
                        raise RuntimeError(f"{c['name']}: HTTP {response.status_code}: {response.text[:2000]}")
                    if "video/mp4" not in response.headers.get("Content-Type", ""):
                        raise ValueError("Expected raw MP4 response")
                    pending = raw.with_suffix(".partial.mp4")
                    with pending.open("wb") as f:
                        for block in response.iter_content(1024 * 1024):
                            f.write(block)
                    headers = dict(response.headers)
            probe(pending, 121)
            os.replace(pending, raw)
            record = {
                "sha256": sha(raw),
                "request": data,
                "response_headers": headers,
                "seconds": time.perf_counter() - started,
            }
            write(receipt, record)
        if staged.exists() and stage_receipt.exists():
            saved = json.loads(stage_receipt.read_text(encoding="utf-8"))
            if saved["raw_sha256"] != sha(raw) or saved["sha256"] != sha(staged):
                raise ValueError("Staged video changed")
            probe(staged, 120)
        elif args.command == "audit":
            raise FileNotFoundError("Missing staged video/record: " + c["name"])
        elif staged.exists():
            raise FileExistsError("Staged video without receipt: " + str(staged))
        else:
            pending = staged.with_suffix(".partial.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(raw),
                    "-vf",
                    "trim=start_frame=1:end_frame=121,setpts=PTS-STARTPTS",
                    "-frames:v",
                    "120",
                    "-r",
                    "24",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    str(pending),
                ],
                check=True,
            )
            probe(pending, 120)
            os.replace(pending, staged)
            write(
                stage_receipt,
                {
                    "sha256": sha(staged),
                    "raw_sha256": sha(raw),
                    "frames": 120,
                    "first_raw_frame": 1,
                    "last_raw_frame": 120,
                    "encoding": "libx264 lossless, crf=0",
                },
            )
        if new_request:
            record["case_seconds"] = time.perf_counter() - case_started
            write(receipt, record)
        timing = {
            "case": c["name"],
            "reused": not new_request,
            "request_seconds": record["seconds"],
            "case_seconds": record.get("case_seconds"),
        }
        timings.append(timing)
        print(json.dumps({"completed": i, "selected": limit, "full_total": len(cases), **timing}), flush=True)
    summary = {
        "selected": limit,
        "full_total": len(cases),
        "new_requests": sum(not t["reused"] for t in timings),
        "cases": timings,
        "measurement": (
            "request_seconds includes HTTP generation, video transfer and validation; "
            "case_seconds also includes local trimming. Service startup excluded. "
            "Reused rows retain original timings."
        ),
    }
    warm = [t["case_seconds"] for t in timings[1:] if t["case_seconds"] is not None]
    if warm:
        mean = sum(warm) / len(warm)
        summary.update(
            {
                "warm_mean_case_seconds": mean,
                "projected_198_seconds": timings[0]["case_seconds"] + (len(cases) - 1) * mean
                if timings[0]["case_seconds"] is not None
                else len(cases) * mean,
                "projected_remaining_seconds": (len(cases) - limit) * mean,
                "estimate_note": (
                    "Rough extrapolation excluding first request from warm mean; "
                    "two cases do not establish throughput distribution. "
                    "Excludes dataset download, service startup and scoring."
                ),
            }
        )
    write(root / "timing_summary.json", summary)
    print(json.dumps({"event": "timing_summary", **summary}), flush=True)
    expected = {c["name"] for c in cases}
    actual = {p.name for p in staged_dir.glob("*.mp4")}
    if not {c["name"] for c in cases[:limit]}.issubset(actual) or not actual.issubset(expected):
        raise ValueError("Unexpected or missing staged videos")
    if limit < len(cases):
        write(
            root / "pilot_complete.json",
            {
                "status": "pilot_complete",
                "videos_checked": limit,
                "full_total": len(cases),
                "manifest_sha256": sha(manifest),
                "timing_summary": str(root / "timing_summary.json"),
            },
        )
        print("PHYSICSIQ_PILOT_COMPLETE (not a full benchmark or score)", flush=True)
        return
    write(
        root / "generation_complete.json",
        {
            "status": "complete",
            "videos": 198,
            "manifest_sha256": sha(manifest),
            "staged_sha256": {c["name"]: sha(staged_dir / c["name"]) for c in cases},
        },
    )
    print("PHYSICSIQ_GENERATION_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--descriptions", required=True, type=Path)
    prep.add_argument("--data-root", required=True, type=Path)
    prep.add_argument("--output-dir", required=True, type=Path)
    for command in ("generate", "audit"):
        q = sub.add_parser(command)
        q.add_argument("--output-dir", required=True, type=Path)
        q.add_argument("--submission-yaml", required=True, type=Path)
        q.add_argument("--upsampled-dir", type=Path)
        q.add_argument("--base-url", default="http://127.0.0.1:8000")
        q.add_argument(
            "--limit",
            type=int,
            default=198,
            help="Process only the first N cases, including reused cases; omit to finish all 198.",
        )
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

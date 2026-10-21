# Cosmos3-Edge：A10 两条计时与 Physics-IQ 生成

本次在 A10 先测两条，请直接按 [本次操作步骤](A10_TIMING_RUNBOOK.md) 从拉取分支到回传结果依次执行。

在同一个服务上顺序生成前两条完整视频，按实测耗时估算 198 条总时长。视频保存后可直接续跑其余 196 条。默认 `generate` 会运行全部 198 条，试跑必须显式传 `--limit 2`。

固定输入：Physics-IQ Original 的 take-1，832×480、121 帧、24fps、20 步、CFG3.25、shift12、seed0。GPU 服务使用 BF16。这里先测生成耗时，不执行精度评分。

## 1. 输入准备

以下命令在 A10 的 Linux Bash、当前 fork 仓库根目录执行。使用已有 vLLM-Omni Conda 环境；将示例环境名替换为实际名称，无须另 clone 上游或重装已有的 PyTorch。

```bash
conda activate cosmos-edge-gpu
conda install -c conda-forge ffmpeg -y
python -m pip install requests

python benchmarks/diffusion/cosmos3/physicsiq/prepare_timing_inputs.py \
  --output-dir "$HOME/cosmos-physicsiq"

python benchmarks/diffusion/cosmos3/physicsiq/run_physicsiq.py prepare \
  --descriptions "$HOME/cosmos-physicsiq/descriptions_original.csv" \
  --data-root "$HOME/cosmos-physicsiq/datasets/physics-IQ-benchmark" \
  --output-dir "$HOME/cosmos-physicsiq/results/a10_original_seed0"
```

仅下载官方 CSV、提交配置和 198 张输入图片，建立完整清单以支持续跑。不下载真实未来视频、运动掩码或评分模型。`prepared=198` / `model_requests=0` 表示输入准备完成，尚未运行模型。图片按大小与 MD5 验证并复用。

## 2. 终端 A：启动一次 GPU 服务

已存在同配置服务时直接复用。MODEL_DIR 改为实际的 Cosmos3-Edge Diffusers 权重目录，应包含 transformer、vae、scheduler、text_tokenizer 和 model_index.json。沿用先前固定的模型版本 `344d602b128d1bbdacb43b08d0a3626f46343e29`。

```bash
conda activate cosmos-edge-gpu
export MODEL_DIR="/path/to/Cosmos3-Edge"
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
set -o pipefail

vllm serve "$MODEL_DIR" --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --dtype bfloat16 --enforce-eager \
  --diffusion-attention-backend TORCH_SDPA \
  --no-guardrails --host 127.0.0.1 --port 8000 \
  --num-gpus 1 --init-timeout 1800 --stage-init-timeout 1800 \
  2>&1 | tee "$HOME/cosmos-physicsiq/server_a10_timing.log"
```

不要每条请求各重启一次服务。测试不启用量化、采样缓存或 CPU 卸载；如发生 OOM，先保留日志，不将卸载或减帧后的耗时视为同配置结果。

## 3. 终端 B：只生成两条

在同一个 fork 仓库根目录执行：

```bash
conda activate cosmos-edge-gpu
curl -f http://127.0.0.1:8000/health
set -o pipefail

python benchmarks/diffusion/cosmos3/physicsiq/run_physicsiq.py generate \
  --output-dir "$HOME/cosmos-physicsiq/results/a10_original_seed0" \
  --submission-yaml "$HOME/cosmos-physicsiq/EDGE_IQVERIFIED_SUBMISSION.yaml" \
  --limit 2 \
  2>&1 | tee "$HOME/cosmos-physicsiq/timing_two.log"
```

`--limit 2` 是清单前两条（包括已完成条目），不是每次额外生成两条。完成打印 `PHYSICSIQ_PILOT_COMPLETE` 后客户端退出，不会进入第三条；服务仍在终端 A 运行，不需要时按 Ctrl+C 关闭。

每条保存 121 帧原始视频、去掉首帧后的 120 帧视频及请求记录。仅在真实解码验证尺寸、帧数、帧率并保存哈希后才算完成；续跑复用有效记录，不重新请求模型或裁剪。

## 4. 读取计时与估算

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path.home() / "cosmos-physicsiq/results/a10_original_seed0/timing_summary.json"
d = json.loads(p.read_text())
for i, case in enumerate(d["cases"], 1):
    print(f"第{i}条：请求 {case['request_seconds']:.2f} 秒，完整处理 {case['case_seconds']:.2f} 秒，复用={case['reused']}")
print(f"198条预计：{d['projected_198_seconds'] / 3600:.2f} 小时")
print(f"剩余196条预计：{d['projected_remaining_seconds'] / 3600:.2f} 小时")
PY
```

- `request_seconds`：HTTP 请求、模型生成、视频传输与原视频校验。
- `case_seconds`：还包括本地裁剪、哈希及输出校验。模型内部采样、VAE 时间另见服务日志。
- 总时长初估：`T1 + 197 × T2`；剩余初估：`196 × T2`。第二条作为启动后较稳定耗时的初步样本；两条不足以覆盖不同提示词长度和机器负载。
- `reused=true` 表示使用以前已保存的计时，本次没有新模型请求。
- 估算不含输入下载、服务启动与最终评分。试跑只写 `pilot_complete.json`；覆盖完整 198 条才写 `generation_complete.json`。

回传 `timing_two.log`、`timing_summary.json` 和服务日志的采样/解码计时即可。不预设 A10 实测速度。

## 后续全量

确认时间可接受后，保持输入和输出目录不变，去掉 `--limit 2`：

```bash
python benchmarks/diffusion/cosmos3/physicsiq/run_physicsiq.py generate \
  --output-dir "$HOME/cosmos-physicsiq/results/a10_original_seed0" \
  --submission-yaml "$HOME/cosmos-physicsiq/EDGE_IQVERIFIED_SUBMISSION.yaml"
```

前两条直接复用，从第三条开始。只读检查使用 `audit` 子命令，加 `--limit 2` 可仅检查试跑范围。

## 来源与评测边界

- [Physics-IQ 官方仓库](https://github.com/google-deepmind/physics-IQ-benchmark)，输入描述固定版本 `c2bca8a5bf452d24e177e9975641af1888eb1492`。
- [NVIDIA Edge 正式提交](https://github.com/google-deepmind/physics-IQ-benchmark/issues/76)及[提交配置](https://github.com/user-attachments/files/31664307/EDGE_IQVERIFIED_SUBMISSION.yaml)。

本流程使用公开原始描述，记录为 original-unmodified。官方 Original 参考 34.05 使用重写提示词，不能将本流程称为严格复现；这里尚未计算任何质量分数。脚本的 BF16 配置记录依赖按上述命令启动服务，不通过 HTTP 检查实际服务精度。
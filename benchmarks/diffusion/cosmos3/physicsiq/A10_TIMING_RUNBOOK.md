# 本次操作：A10 先跑两条，估算 198 条耗时

日期：2026-10-21。目的：在你已安装好的 A10 环境上，实测前两条完整视频的耗时，再决定是否跑满 198 条。本次两条结束后停止客户端，不自动接全量或评分。

所有命令在 **A10 机器的 Linux Bash** 执行。用现有 fork 和 Conda 环境，不需要手动传文件。需要按实际情况修改：仓库目录、Conda 环境名、权重目录。

## 一、拉取本次分支

进入你之前安装 vLLM-Omni 的 fork 仓库：

```bash
cd /path/to/your/vllm-omni

git fetch origin
git switch cosmos3-edge-a10-timing
git pull --ff-only origin cosmos3-edge-a10-timing

ls benchmarks/diffusion/cosmos3/physicsiq
```

目录中应有 `run_physicsiq.py`、`prepare_timing_inputs.py` 和本操作文档。若切换时 Git 提示本地修改会被覆盖，先保留这些修改，不执行 reset 或强制切换。

这次新增的是独立脚本，拉取分支后不需要重装 vLLM-Omni。下面的 `cosmos-edge-gpu` 都替换为你之前已经装好 Omni 的 Conda 环境名。

## 二、准备依赖与输入

继续在仓库根目录执行：

```bash
conda activate cosmos-edge-gpu

# 已有这两个命令和 requests 时可跳过安装。
conda install -c conda-forge ffmpeg -y
python -m pip install requests

mkdir -p "$HOME/cosmos-physicsiq"

python benchmarks/diffusion/cosmos3/physicsiq/prepare_timing_inputs.py \
  --output-dir "$HOME/cosmos-physicsiq"

python benchmarks/diffusion/cosmos3/physicsiq/run_physicsiq.py prepare \
  --descriptions "$HOME/cosmos-physicsiq/descriptions_original.csv" \
  --data-root "$HOME/cosmos-physicsiq/datasets/physics-IQ-benchmark" \
  --output-dir "$HOME/cosmos-physicsiq/results/a10_original_seed0"
```

预期最后显示 `prepared: 198`、`model_requests: 0`。这是准备 198 条可续跑的输入清单，尚未生成视频。仅下载提示词、提交配置和 198 张输入图片，不下载评分用的真实未来视频、运动掩码，也不安装评分器。

如下载报错，停在这一步，保留终端错误；不要把没有准备完整的目录继续用于生成。已校验正确的图片会在重跑时复用。

## 三、终端 A：启动服务并保持运行

在仓库根目录、已有的 Omni 环境中执行。将 `MODEL_DIR` 改成你已经下载好的 **Cosmos3-Edge Diffusers 权重目录**：

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

沿用之前的权重版本 `344d602b128d1bbdacb43b08d0a3626f46343e29`。目录应包含 `model_index.json`、`transformer/`、`vae/`、`scheduler/` 和 `text_tokenizer/`。如果相同配置的服务已在 8000 端口运行，直接复用，不要另起一个服务占显存。

这里使用 BF16、eager 和 TORCH_SDPA，不开启量化、CPU 卸载或采样缓存。若发生显存错误，把服务日志保留下来；本次先不通过减帧、减步等方式改变工作量。

服务启动后，这个终端先保持运行。两条请求共用同一个服务，不要每条重新加载模型。

## 四、终端 B：只生成前两条

另开一个终端，同样进入已有 fork 仓库并激活同一个环境：

```bash
cd /path/to/your/vllm-omni
conda activate cosmos-edge-gpu

curl -f http://127.0.0.1:8000/health
```

确认 health 请求成功后，再执行：

```bash
set -o pipefail

python benchmarks/diffusion/cosmos3/physicsiq/run_physicsiq.py generate \
  --output-dir "$HOME/cosmos-physicsiq/results/a10_original_seed0" \
  --submission-yaml "$HOME/cosmos-physicsiq/EDGE_IQVERIFIED_SUBMISSION.yaml" \
  --limit 2 \
  2>&1 | tee "$HOME/cosmos-physicsiq/timing_two.log"
```

**保留 `--limit 2`**。省略它会跑全部 198 条。当前配置是：

| 参数 | 本次取值 |
|---|---|
| 样本 | Physics-IQ Original 清单前两条，I2V |
| 分辨率 | 832×480 |
| 视频帧数、帧率 | 121 帧、24 fps |
| 采样 | 20 步、CFG 3.25、shift 12、seed 0 |
| 服务精度 | BF16 |

两条完成后应打印 `PHYSICSIQ_PILOT_COMPLETE`，客户端自动退出，不进入第三条。服务还会留在终端 A，后面查看完结果后再关闭。

若同一个输出目录已经有成功的两条，重跑会显示 `reused: true`，表示没有重新计时，而是读取原来的结果。若希望重新测一轮，准备输入和生成时都换一个新的结果目录名，例如 `a10_original_seed0_repeat1`，保留原结果。

## 五、查看实际耗时和全量估算

在终端 B 执行：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path.home() / "cosmos-physicsiq/results/a10_original_seed0"
data = json.loads((root / "timing_summary.json").read_text())

for index, case in enumerate(data["cases"], 1):
    print(f"第{index}条：")
    print(f"  请求与原视频校验：{case['request_seconds']:.2f} 秒")
    print(f"  包含本地裁剪的完整处理：{case['case_seconds']:.2f} 秒")
    print(f"  是否复用以前结果：{case['reused']}")

print(f"198条预计：{data['projected_198_seconds'] / 3600:.2f} 小时")
print(f"剩余196条预计：{data['projected_remaining_seconds'] / 3600:.2f} 小时")
PY
```

这里的计时含模型生成、视频下载和校验；完整处理还包含本地裁剪。服务日志中另外有采样循环和 VAE 解码的计时，可以区分哪一段慢。

以第一条耗时 T1、第二条耗时 T2，粗估 **198 条总耗时 = T1 + 197×T2**，剩余耗时 = 196×T2。第一条可能含首次执行开销，因此用第二条作后续每条的初估。两条样本不足以保证所有提示词耗时相同，估计不包含输入下载、服务启动和最终评分。

## 六、结束本次测试，回传结果

在终端 A 按 **Ctrl+C** 关闭生成服务。输出文件均保留：

```text
~/cosmos-physicsiq/
├── server_a10_timing.log
├── timing_two.log
└── results/a10_original_seed0/
    ├── timing_summary.json
    ├── pilot_complete.json
    ├── raw/                         # 两条121帧视频
    ├── records/                     # 请求、耗时及哈希
    └── scoring_inputs/
        └── a10_original_seed0/      # 去掉首帧后的120帧视频
```

把以下内容发回来即可：

1. `timing_summary.json` 的内容。
2. `timing_two.log` 最后几行。
3. `server_a10_timing.log` 中两条请求的采样、解码和总耗时，以及存在的警告或错误。

本次到这里结束，尚不计算精度分数。是否继续全量，等看到两条耗时后再决定；已成功的两条届时可直接复用。通用脚本与续跑说明见 [README](README.md)。
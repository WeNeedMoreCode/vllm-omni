# 本次操作：A10 先跑两条，估算 198 条耗时

日期：2026-10-21。目的：在你已安装好的 A10 环境上，实测前两条完整视频的耗时，再决定是否跑满 198 条。本次两条结束后停止客户端，不自动接全量或评分。

所有命令在 **A10 机器的 Linux Bash** 执行。用现有 fork 和 Conda 环境，不需要手动传文件。需要按实际情况修改：仓库目录、Conda 环境名、权重目录。

<a id="paths"></a>
## 开始前：路径、变量和示例值

### `$HOME` 是什么

`$HOME` 是 Linux 系统为**当前登录用户**设置的家目录环境变量。Bash 执行命令时会自动将它展开成实际路径，通常不需要手动填写，也不要修改 `HOME` 本身。

| 你在 A10 上使用的用户 | `$HOME` 常见值 | `"$HOME/cosmos-physicsiq"` 展开后 |
|---|---|---|
| 普通用户，例如 `syx` | `/home/syx` | `/home/syx/cosmos-physicsiq` |
| root | `/root` | `/root/cosmos-physicsiq` |

这是常见示例，实际值以机器输出为准。在 **A10 的终端**执行：

```bash
whoami
echo "$HOME"
pwd
conda env list
```

分别显示当前用户、家目录、当前所在目录和已有 Conda 环境。文档中的 `~` 也表示当前用户的家目录，例如 `~/cosmos-physicsiq` 与 `$HOME/cosmos-physicsiq` 指向同一位置。Python 中的 `Path.home()` 通常也是这个用户的家目录。第四、五节客户端使用已经确认的 `PHYSICSIQ_WORKDIR` 读取输入和结果，避免改动 HOME 后指向另一份目录。

终端 A、终端 B 使用同一台 A10 机器上的同一个 Linux 用户，才能按默认路径找到同一份输入和结果。`$HOME` 不表示 vLLM-Omni 仓库目录，也不表示权重目录。

### 本文用到的目录和需要修改的值

| 文档写法 | 含义 | 怎么处理 |
|---|---|---|
| `/path/to/your/vllm-omni` | 你已经 clone 并安装的 fork 仓库的绝对路径 | **必须改成真实路径**。在已有仓库终端执行 `pwd` 可查看，例如 `/data/projects/vllm-omni` |
| `cosmos-baseline` | 已装好 vLLM-Omni 的 Conda 环境名 | **改成真实环境名**，用上面的 `conda env list` 查看 |
| `/path/to/Cosmos3-Edge` | 已下载的 Cosmos3-Edge Diffusers 权重目录 | **必须改成真实路径**，例如 `/data/models/Cosmos3-Edge` |
| `MODEL_DIR`、`$MODEL_DIR` | 本文设置的权重路径变量，以及读取这个变量的写法 | 修改 `export MODEL_DIR="..."` 等号后的路径；后面的 `"$MODEL_DIR"` 自动读取该值 |
| `$HOME/cosmos-physicsiq` | 本次输入、视频、计时记录和日志的工作目录 | 默认自动展开，`mkdir -p` 会创建它；它与代码仓库、权重目录分开 |
| `benchmarks/diffusion/cosmos3/physicsiq/...` | 仓库内脚本的相对路径 | 保持原样，执行命令前先进入仓库根目录 |
| `a10_original_seed0` | 本次结果子目录名 | 首次测试保持原样，准备输入、生成和读取结果用同一个名字 |

例如你的仓库在 `/data/projects/vllm-omni`，文档里的进入仓库命令就改成 `cd /data/projects/vllm-omni`。如果权重在 `/data/models/Cosmos3-Edge`，就改成 `export MODEL_DIR="/data/models/Cosmos3-Edge"`。这两个示例路径也不是自动存在的目录，要以你自己的机器为准。

### 如果想把结果放到其他磁盘

默认输出会放在家目录所在磁盘。若要改到例如 `/data/cosmos-physicsiq`：

1. 将本文所有 Bash 命令里的 `$HOME/cosmos-physicsiq` 统一替换成 `/data/cosmos-physicsiq`，包括下载、准备清单、生成和日志路径。
2. 第四节客户端设置 `export PHYSICSIQ_WORKDIR="/data/cosmos-physicsiq"`；第五节从这个变量读取结果，保持同一个终端即可。
3. 第六节查看和回传文件也使用新的目录。这个目录需要有写权限和足够空间，首次生成前确定好位置；已经准备好的清单包含图片绝对路径，之后不要直接移动输入目录。

无需修改 Linux 的 `HOME` 环境变量。下面正文使用默认的家目录方案。
## 一、拉取本次分支

进入你之前安装 vLLM-Omni 的 fork 仓库：

```bash
cd /home/syx/cosmos-baseline/vllm-omni-baseline

git fetch origin
git switch cosmos3-edge-a10-timing
git pull --ff-only origin cosmos3-edge-a10-timing

ls benchmarks/diffusion/cosmos3/physicsiq
```

目录中应有 `run_physicsiq.py`、`prepare_timing_inputs.py` 和本操作文档。若切换时 Git 提示本地修改会被覆盖，先保留这些修改，不执行 reset 或强制切换。

这次新增的是独立脚本，拉取分支后不需要重装 vLLM-Omni。下面的 `cosmos-baseline` 都替换为你之前已经装好 Omni 的 Conda 环境名。

## 二、准备依赖与输入

继续在仓库根目录执行：

```bash
conda activate cosmos-baseline

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

## 三、终端 A：检查真实权重目录并启动服务

根据你回传的终端，本次环境名是 `cosmos-baseline`，仓库为 `/home/syx/cosmos-baseline/vllm-omni-baseline`。当前登录用户是 root，因此 `$HOME` 通常为 `/root`；仓库在 `/home/syx` 下不会改变这个值。先执行下面几条命令，每条各占一行：

```bash
cd /home/syx/cosmos-baseline/vllm-omni-baseline
conda activate cosmos-baseline
git pull --ff-only origin cosmos3-edge-a10-timing
python benchmarks/diffusion/cosmos3/physicsiq/serve_a10.py --dry-run
```

启动器会依次检查 `MODEL_DIR`（如果已设为真实目录）、仓库旁的 `models/Cosmos3-Edge`、仓库内的 `models/Cosmos3-Edge`、家目录下的旧下载位置以及固定版本的 Hugging Face 本地缓存。会检查 `model_index.json`、`transformer/`、`vae/`、`scheduler/`、`text_tokenizer/` 是否齐全；不会下载模型。

如果输出 `Model directory: ...`，并打印完整服务命令，说明找到了可用的本地目录。再单独执行下面**一行命令**启动：

```bash
python benchmarks/diffusion/cosmos3/physicsiq/serve_a10.py
```

脚本自动设置 worker 启动方式、超时时间，并将日志写到 `$HOME/cosmos-physicsiq/server_a10_timing.log`；保持终端打开，Ctrl+C 停止服务。配置仍为 BF16、eager、TORCH_SDPA、单卡、关闭 guardrails，不启用量化、CPU 卸载或采样缓存。如要把输出放到其他磁盘，在检查和启动这两条命令后都加 `--work-dir /data/cosmos-physicsiq`，与本文其他步骤的数据目录保持一致。

如果提示找不到完整权重，先只查你此前的工作目录：

```bash
find /home/syx/cosmos-baseline -maxdepth 6 -name model_index.json -print
```

输出的 `model_index.json` 所在目录才是模型根目录。例如实际输出为 `/data/models/Cosmos3-Edge/model_index.json`，则把检查和启动命令后都加 `--model-dir /data/models/Cosmos3-Edge`。这个路径仍要以机器实际输出为准；如果没有找到，说明该搜索范围内尚无 Diffusers 权重，先核对此前的下载位置，不能把 `/path/to/Cosmos3-Edge` 原样传给服务。

### 本次报错的含义

- `HFValidationError ... '/path/to/Cosmos3-Edge'`：占位路径不存在，服务将它当作 Hugging Face 仓库 ID 继续校验后失败；模型还没有开始加载。新启动器会在本地直接指出缺失的目录。
- 多行粘贴中出现 `&#x20;`、`\--`、`$(MODEL_DIR`、`)$HOME`，都不是正确的 Bash 写法。复制代码块中的原始代码，不合并多条命令；现在启动服务只需上述一行命令。
- vLLM-Omni `0.1.dev...` 与 vLLM `0.30.0` 的版本警告不是本次路径异常的直接原因；开发版版本号也可能受构建元数据影响。脚本会打印环境版本，暂不因这条警告盲目重装。若修好路径后仍报错，保留完整堆栈，再核对当前提交、包来源和依赖。

沿用先前固定权重版本 `344d602b128d1bbdacb43b08d0a3626f46343e29`。如果相同配置的服务已在 8000 端口运行，直接复用；两条请求共用同一服务，不要每条重新加载模型。
## 四、终端 B：只生成前两条

另开一个终端，激活同一个环境。**当前目录如果是 `/sgl-workspace/sglang`，相对路径 `benchmarks/...` 就会被解释成 sglang 下的目录，找不到这里的脚本。** 以下直接使用已确认的 vLLM-Omni 仓库绝对路径，从其他目录也可运行。

先激活环境并检查服务：

```bash
conda activate cosmos-baseline
curl -f http://127.0.0.1:8000/health
```

然后定位已经准备好的数据，不修改 Linux 的 `HOME`。下面优先检查你指定的 `/home/syx/cosmos-physicsiq`，其次检查先前 root 用户可能使用的 `/root/cosmos-physicsiq`，只选择同时存在清单和提交配置的目录。不会移动或重新下载数据。

```bash
if [ -z "${PHYSICSIQ_WORKDIR:-}" ]; then
  for PHYSICSIQ_CANDIDATE in /home/syx/cosmos-physicsiq /root/cosmos-physicsiq; do
    if [ -f "$PHYSICSIQ_CANDIDATE/results/a10_original_seed0/cases.json" ] && [ -f "$PHYSICSIQ_CANDIDATE/EDGE_IQVERIFIED_SUBMISSION.yaml" ]; then
      export PHYSICSIQ_WORKDIR="$PHYSICSIQ_CANDIDATE"
      break
    fi
  done
fi
: "${PHYSICSIQ_WORKDIR:?未找到已准备的数据，请先完成第二节；若数据在其他磁盘，请设置 PHYSICSIQ_WORKDIR 为实际工作目录}"
test -f "$PHYSICSIQ_WORKDIR/results/a10_original_seed0/cases.json" && test -f "$PHYSICSIQ_WORKDIR/EDGE_IQVERIFIED_SUBMISSION.yaml" && printf '本次数据目录：%s\n' "$PHYSICSIQ_WORKDIR"
```

如果数据放在其他磁盘，先执行例如 `export PHYSICSIQ_WORKDIR="/data/cosmos-physicsiq"`，再运行上面的检查；路径要以已有数据为准。`PHYSICSIQ_WORKDIR` 只管理本次输入和输出，和 Linux 的 `HOME` 是两个不同的变量。修改 `HOME` 也不会自动搬移已有文件，早先启动的服务进程仍保留它启动时的环境。

确认上面打印真实数据目录且没有报错后，在同一个终端执行下面这**一整行**客户端命令，避免多行复制发生符号损坏：

```bash
set -o pipefail
python /home/syx/cosmos-baseline/vllm-omni-baseline/benchmarks/diffusion/cosmos3/physicsiq/run_physicsiq.py generate --output-dir "$PHYSICSIQ_WORKDIR/results/a10_original_seed0" --submission-yaml "$PHYSICSIQ_WORKDIR/EDGE_IQVERIFIED_SUBMISSION.yaml" --limit 2 2>&1 | tee "$PHYSICSIQ_WORKDIR/timing_two.log"
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

在第四节的同一个终端 B 执行（沿用已确认的 `PHYSICSIQ_WORKDIR`）：

```bash
python - "$PHYSICSIQ_WORKDIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "results/a10_original_seed0"
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
实际工作目录（第四节打印的 PHYSICSIQ_WORKDIR）/
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

客户端日志和结果在 `PHYSICSIQ_WORKDIR` 下；服务日志在终端 A 启动时打印的 `Server log` 路径，改过 HOME 时两者可能位于不同目录。把以下内容发回来即可：

1. `timing_summary.json` 的内容。
2. `timing_two.log` 最后几行。
3. `server_a10_timing.log` 中两条请求的采样、解码和总耗时，以及存在的警告或错误。

本次到这里结束，尚不计算精度分数。是否继续全量，等看到两条耗时后再决定；已成功的两条届时可直接复用。通用脚本与续跑说明见 [README](README.md)。
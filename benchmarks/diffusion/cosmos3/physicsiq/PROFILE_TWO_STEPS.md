# 粗分 310P 与 A10 的速度差：两步性能插桩

本次不重跑完整视频。先复用旧日志看大段耗时，再对相同 121 帧工作量只执行原 20 步时间表的前两步：第 0 步建立文本 K/V 等缓存，第 1 步观察较稳定执行的分项耗时。不是把 num_inference_steps 改成 2；原始 20 步的时刻和求解器配置保留。

## 已有记录说明了什么

310P 第一条完整生成约 1296 秒，其中采样约 1237 秒、VAE 解码 58.28 秒。因此先关注采样网络。A10 客户端两条完整处理约 70.99 / 69.29 秒，但客户端记录本身没有拆开内部采样和解码，不能直接据此认定具体算子慢了多少倍。

## 310P 已完成的实际分项

本轮只执行原 20 步中的第 0、1 步，未运行后续 18 步或完整 VAE 解码。第 1 步包含正、负两个 CFG 分支，扣除检查点 I/O 后为 61.403 秒：

| 第1步分项 | 时间 | 占单步比例 |
|---|---:|---:|
| 注意力核心（含后端准备与 GQA 处理） | 56.619 s | 92.2% |
| MLP | 2.429 s | 4.0% |
| Q/K/V 与输出投影 | 0.556 s | 0.9% |
| GEN 层归一化 | 0.643 s | 1.0% |
| 其他计算与发射间隙 | 1.156 s | 1.9% |

参考图像 VAE 编码 0.148 秒；第 0 步两个 UND 编码合计 0.198 秒，第 1 步没有重算 UND。检查点保存等 I/O 合计约 0.398 秒，另行记录。

用保存的首层真实 Q/K/V 单独重放，未加载模型权重：非插桩调用约 1.016 秒，分派记录明确出现 `aten::_scaled_dot_product_attention_math`、`repeat_interleave`、`_safe_softmax`、数据拷贝/转换。注意力矩阵形状为 `[1,16,12090,12155]`，约 23.5 亿元素。当前 310P 路径没有选到融合注意力实现；这已经是可确认的主要瓶颈。CPU 分派跟踪用于确认算子路径，不用于声称各设备 kernel 的独立耗时。

这不能代替 A10 分项实测：A10 的具体注意力内核与各项倍数仍待同脚本验证。已保存的完整 310P 参考为 `reference_310p_two_steps.json`，探针和单层重放结束后已核查 NPU 无运行进程，全量任务未恢复。
## 探针测哪些位置

- 参考图像的 VAE 编码。
- 正、负条件分支的 Transformer 前向时间，分别记录第 0、1 步。
- UND 理解分支；正常只应在第 0 步的两个条件分支各计算一次。
- GEN 28 层的注意力核心、Q/K/V 与输出投影、MLP、归一化。
- 每步总时间，以及保存检查点的单独 I/O 时间。

使用设备 Event 记录模块之间的时间间隔，只在步骤/分支边界同步；Event 间隔可能包含算子发射间隙，不等于纯 kernel 时间。GEN 层总时间包含注意力、MLP 等子项，不要重复相加。插桩及同步存在额外开销，生产耗时仍以之前无插桩记录为准。

保存的 `.pt`：第 1 步两个分支的 Transformer 输入、已缓存文本 K/V 与位置特征、预测结果，首层条件分支注意力的真实 Q/K/V，以及两步后的潜变量。后续只读加载或小模块重放可复用这些文件，避免再跑前序计算。

完整 VAE 解码故意跳过；内部只返回一个极小占位张量以结束探针，不保存为视频，不计为任何生成成功或精度样本。

## A10 上怎么跑

GPU 不可由代理直连，这部分由用户执行。先在运行服务的终端 A 按 Ctrl+C 关闭已有服务，释放显存。本探针会独立加载模型，不能和常驻服务重复占用同一张 A10。

在现有仓库运行：

```bash
cd /home/syx/cosmos-baseline/vllm-omni-baseline
conda activate cosmos-baseline
git pull --ff-only origin cosmos3-edge-a10-timing
```

按此次已完成两条的输入位置设置工作目录；若使用其他位置，填第四节已确认的数据目录，不修改 HOME：

```bash
export PHYSICSIQ_WORKDIR="/home/syx/cosmos-physicsiq"
test -f "$PHYSICSIQ_WORKDIR/results/a10_original_seed0/cases.json"
```

启动以下一行命令。权重会使用与 serve_a10.py 相同的本地查找方式；也可以增加 `--model-dir` 和真实权重路径。

```bash
set -o pipefail
python benchmarks/diffusion/cosmos3/physicsiq/profile_two_steps.py --device cuda --cases-json "$PHYSICSIQ_WORKDIR/results/a10_original_seed0/cases.json" --submission-yaml "$PHYSICSIQ_WORKDIR/EDGE_IQVERIFIED_SUBMISSION.yaml" --output-dir "$PHYSICSIQ_WORKDIR/profiles/a10_two_steps_20261021" 2>&1 | tee "$PHYSICSIQ_WORKDIR/profile_two_steps.log"
```

这里 A10 使用 BF16，310P 使用已有 FP16 路径，均选择 eager/TORCH_SDPA、CPU RNG、同一条描述与参考图像。探针用于性能分解，不把不同精度的输出直接当成精度对齐验收。

## 看哪些结果

先运行这个命令，直接打印易读表格，并与已实测的 310P 分项对照：

```bash
python benchmarks/diffusion/cosmos3/physicsiq/summarize_profile.py "$PHYSICSIQ_WORKDIR/profiles/a10_two_steps_20261021/report.json" --reference benchmarks/diffusion/cosmos3/physicsiq/reference_310p_two_steps.json
```

表中 Current 是这次 A10，Reference 是 310P，Ref/current 为对应分项的耗时倍数。两边精度不同（A10 BF16、310P FP16），这是当前部署方式的性能分解，不是精度验收。

如需查看原始记录：

```bash
python -m json.tool "$PHYSICSIQ_WORKDIR/profiles/a10_two_steps_20261021/report.json"
```

- `branch_times`：两个步骤内正、负分支各用了多久；不含检查点存盘时间。
- `step_times`：两步实际耗时，另列检查点 I/O 及扣除 I/O 后时间。
- `module_aggregates`：`step1.attention_core`、`step1.attention_projections`、`step1.mlp`、`step1.normalization` 是主要分项。`step1.gen_layer_inclusive` 是父项合计，不能与这些子项重复相加。
- `phase_times`：参考图像 VAE 编码耗时。
- `module_intervals.json`：具体到各层、各分支的记录，必要时再看。

回传 `report.json` 即可先粗分，异常时再附 `profile_two_steps.log`。`.pt` 文件留在原机，后续需要重放时再使用。同一输出目录已有 report.json 时脚本不会再次运行模型；要测新配置，用新的输出目录。

探针结束自动关闭加载的模型。310P 的 198 条全量任务继续保持用户暂停状态；本探针不会恢复下载、生成或评分流水线。
## 只重放已保存的一层注意力（需要进一步核实时）

无需加载模型权重，也无需重新计算前两步：

```bash
python benchmarks/diffusion/cosmos3/physicsiq/replay_attention.py --device cuda --inputs "$PHYSICSIQ_WORKDIR/profiles/a10_two_steps_20261021/step1_cond_layer0_attention_inputs.pt" --output-dir "$PHYSICSIQ_WORKDIR/profiles/a10_attention_replay_20261021"
```

会先暖机、再测一次无插桩调用，最后用 CPU dispatcher 跟踪再调用一次以识别实际算子（总共三次单层调用），保存 `attention_report.json`、`attention_operators.json` 和 Chrome trace。这不是全模型 profiler。调用仓库部署中的 SDPAImpl，不另写一份注意力算法。若要看是否出现 flash/fused 或 math 路径，检查 `interesting_dispatch` 即可。
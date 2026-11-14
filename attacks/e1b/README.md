# attacks/e1b — 目录说明

E1b 攻击实验(zkAgent 解码/注意力容差的下游利用)的代码、数据与报告。
总结果与结论在 `SUMMARY.md`;时间线在 `TIMELINE.md`;论文可用前沿在 `FRONTIERS.md`。
子目录大体按实验campaign排列(时间顺序见 `TIMELINE.md`):

```
attacks/e1b/
├── SUMMARY.md            总结果(锚点文档,不动)
├── TIMELINE.md           按时间排列的进展线(要求 → 达成)
├── FRONTIERS.md          当前前沿:可进论文的结果(英文;仅 paper 设计口径,不含 preliminary / free-r)
├── README.md             本文件
├── yield_oracle/         E1b-1  可达性 / yield(C++ oracle + logits 导出)
├── forced_extension/     E1b-2  强制不停(负面结论)
├── length_steering/      E1b-3  账单膨胀 + 共享 relation/reach 库
├── decision/             E1b-4 → E1b-5 → E1b-tree  决策操纵
├── attention/            E1b-attn 注意力杠杆(测量 + vLLM patch)
├── scale_harness/        上规模评测(H200 harness 的本仓库侧)
├── choice/               E1b-choice → m2w → τ-bench → Qwen3.5(决策轮攻击)
├── reports/              各阶段计划 / 结果 / 复盘报告(SUMMARY 之外)
└── data/                 choice 数据集与输出的本地快照
```

## 本地脚本 vs harness 快照

- **本地脚本**(本仓库可独立运行):`yield_oracle/`、`forced_extension/`、
  `length_steering/`、`decision/`、`attention/` 的主体,以及 `scale_harness/eval_at_scale.py`。
  依赖 torch + transformers;跨目录导入(e.g. `decision/decision_bestofN.py` 引用
  `length_steering/e1b_length_steer.py` 与 `attention/attn_reach.py`)通过每个脚本
  头部的 `sys.path` 垫片解析,**从任意 CWD 直接 `python3 <path>.py` 即可**。
- **harness 快照**(标注 ⧉):`attention/attn_slack_vllm.py`、`attention/attn_slack_validate.py`、
  `scale_harness/{vllm_bestofn,scale_probe,validate_simulation}.py`、`choice/` 全部。
  它们 `import experiments.*`,**正本在 `/root/zk_agent/experiments/`(H200 worker 仓库)**,
  必须从那个仓库根运行;此处副本仅供此仓库自含记录,可能落后于正本。

## 逐文件说明

### yield_oracle/ — E1b-1 每步可达 token(2026-10-30)
| 文件 | 说明 |
|---|---|
| `dump_logits.py` | 导出真实 logits 为 zkAgent 整数域文本(供 C++ oracle) |
| `yield_decode.cpp` | paper 模型(fixed-H + grind Θ)的可达性 oracle |
| `freer_reach.cpp` | free-r(释出实现)的可达性 oracle |
| `reach_both.cpp` | 两模型可达性对照(与 Python 逐位对齐校验用) |

### forced_extension/ — E1b-2 强制不停(2026-10-30,负面结论)
| 文件 | 说明 |
|---|---|
| `dump_until_stop.py` | 自然停止轨迹 + 每步终止符候选导出 |
| `e1b_forced.py` | 合并版强制延长实验(模型只加载一次) |
| `forced_extension.cpp` | 停止点是否可被挤过的判定 oracle(**不可** ⇒ 负面结论) |

### length_steering/ — E1b-3 账单膨胀(2026-10-30/31, 11-05)
| 文件 | 说明 |
|---|---|
| `e1b_length_steer.py` | P1/P2-lexical 注入续写词拉长推理(**兼共享库**:quant / active_and_shift / reach_freeR / reach_paper / prefix_coin,decision、attention、scale_harness 的脚本都导入它) |
| `bill_inflation.py` | 早期单 prompt 账单膨胀测量(E1b-2.5) |
| `train_length_probe.py` | 真·P2 value probe(隐状态 → 剩余推理长度,ridge;支持跨域训练/评估) |

### decision/ — E1b-4/5/tree 决策操纵(10-31 → 11-08)
| 文件 | 说明 |
|---|---|
| `decision_attack.py` | 盲注入(前几步塞可达错数字)— 失败,被自我纠正 |
| `decision_lookahead.py` | 贪心 lookahead(注入后 rollout 到答案)— 旧协议 |
| `decision_bestofN.py` | **E1b-5 修正**:哈希链重投 → 采样 best-of-N;greedy/sample/steer × attn on/off × `--search tree` |
| `tree_attack.py` | **E1b-tree**:可复用、torch-free 的可容许树搜索(每个 emitted token ∈ 当步可达集);`tests/test_tree_attack.py` 测它 |

### attention/ — E1b-attn 注意力容差杠杆(11-05 → 11-14)
| 文件 | 说明 |
|---|---|
| `attn_reach.py` | `AttnSlack`(逐层注意力输出缩放 hook,架构无关)+ `union_reach`(attn-on 占位)— 共享库 |
| `attn_slack_logits.py` | GPT-2 124M(= zkAgent artifact 模型)上量化容差 → logit 扰动 / 可达集增量 |
| `attn_slack_vllm.py` ⧉ | vLLM 运行时 monkeypatch(逐 (layer,head) 行输出 × 1±ρ;enforce-eager、V1 单进程) |
| `attn_slack_validate.py` ⧉ | vLLM patch vs HF eager 验证;bf16 不可分辨 / fp32 语义的实验现场 |

### scale_harness/ — 上规模(2026-10-31 → 11-14)
| 文件 | 说明 |
|---|---|
| `eval_at_scale.py` | 放大版主评测(本地:任意 HF 模型 × GSM8K/MATH-500 × decision+billing × τ-sweep × attn on/off × probe) |
| `vllm_bestofn.py` ⧉ | 批量 best-of-N runner(纯批量生成,单模型 9–13 min;greedy 臂 + 采样阶梯 + prompt_logprobs 严格审计) |
| `scale_probe.py` ⧉ | 尺寸缩放 Phase A:attn margin 位移 + paper/free-r 可达集测量 |
| `validate_simulation.py` ⧉ | E1b-simval:Python 关系模拟 vs 真 prover/verifier(配 `attacks/validate_simulation.cpp`) |

### choice/ — 无 gold 决策轮攻击(11-11 → 11-15,全部 ⧉)
| 文件 | 说明 |
|---|---|
| `choice_data.py` | schema / prompt 构造 / loader / 解析(torch-free) |
| `build_choice_data.py` | 从原始 CSV/JSON/parquet 构建 DailyDilemma / WebShop / Mind2Web / τ-bench choice 数据 |
| `choice_scoring.py` | choice 解析、null/ASR 汇总、AUC 工具 |
| `choice_tree.py` | 主 runner:vLLM 批量 step-loop,exhaustive/beam/rollout × any/directed |
| `choice_tree_hf.py` | HF eager + AttnSlack 的 attention 轴 runner |
| `choice_tree_attn.py` | attention-fork beam runner(S* 坐标上升 + config-keyed fork 菜单;fp32 / 逐行符号) |
| `choice_tree_audit.py` | 模型无关 witness 逐条 replay(全部运行 0 失败的审计器) |
| `choice_report.py` / `choice_feasible_table.py` | 汇总 / 每题搜索规模表 |
| `choice_rollout_stats.py` / `choice_margin_auc.py` | rollout 预算曲线 / forced-answer AUC |
| `choice_probe.py` / `choice_attn_stats.py` | 诊断探针(边界 forced-answer 概率 / attention 轴统计) |
| `choice_collect.py` | 多 run 汇总成一张对照表 |
| `test_choice_{data,scoring,tree}.py` | 逻辑测试(正本在 `/root/zk_agent/tests/`) |

### reports/ — 计划与结果报告
| 文件 | 阶段 |
|---|---|
| `E1b-NEXT.md` | E1b-5 机制修正(best-of-N / steer / attn)+ D1/D2/B 实验计划与状态 |
| `E1b-tree-attack.md` | E1b-tree 的任务书 + 完成状态 |
| `E1b-choice-NEXT.md` | E1b-choice 计划(无 gold × 枚举 × forced-answer beam)+ 实测驱动的 10 条修正 |
| `E1b-choice-RESULTS.md` | E1b-choice 运行记录(含 spawn bug 复盘 §4.11) |
| `E1b-choice-ANSWER.md` | E1b-choice 按用户 5 个方向逐条回答 |
| `M2W-ATTN-RUNS-2026-11-12.md` | Mind2Web + 注意力杠杆 campaign 计划与 run 清单 |
| `NEXT-STEPS-4LINES.md` | 2026-11-13 的四线后续计划 + session log(1a/1b/2/3/4) |

### data/ — 数据与输出快照
| 文件 | 说明 |
|---|---|
| `mind2web-choice-150.jsonl` | Mind2Web 决策轮(150,4 选 1,directed target) |
| `webshop-choice-150-directed.jsonl` | WebShop 静态四选一(150,directed) |
| `taubench-choice-119.jsonl` | τ-bench 高影响动作选择(119,directed = 代价最高干扰) |
| `E1b-choice-feasible-v2.jsonl` | E1b-choice 每题搜索规模(v2,spawn 修复后) |

## 测试

```bash
cd <repo-root>            # zkAgent-artifacts
python3 -m unittest tests.test_tree_attack     # tree_attack(移动后已更新 import 路径)
```
`choice/test_choice_*.py` 是 `/root/zk_agent/tests/` 的副本,在**那个**仓库根跑:
`python3 -m unittest tests.test_choice_tree` 等。

## C++ oracle 编译

```bash
./build.sh Release   # 仓库根,见顶层 README
LIB=cmake-build-release
g++ -std=c++14 -O2 -fopenmp -mcmodel=large -Isrc -I3rd -I3rd/mcl/include \
    attacks/e1b/yield_oracle/yield_decode.cpp \
    $LIB/src/libgpt_lib.a $LIB/lib/libmcl.a -lgmp -lcrypto -o yield_decode
# freer_reach.cpp / reach_both.cpp / forced_extension.cpp 同法
# attacks/validate_simulation.cpp(E1b-simval)也在 attacks/ 根,同法
```

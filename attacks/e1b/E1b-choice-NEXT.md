# E1b-NEXT2 — 无标准答案数据集 × 无限制枚举 × forced-answer beam

本文件把用户 2026-11-11 的方向落成可执行的实验计划。所有数字来自 `attacks/e1b/SUMMARY.md`
的 E1b-tree / E1b-5 结论；本计划只做 **paper** 关系。

## 0. 要回答的问题

1. **换掉有 gold 的数据集后，决策翻转率是多少？** GSM8K/MATH 的 ASR 分母被
   “honest 答对”筛过；DailyDilemma / WebShop 没有唯一正确答案，每一项的 honest
   choice 本身就是一个可攻击的决策，不需要 gold 筛选。
2. **完全枚举可行吗？** 现有 tree search 的启发式（只搜数字 token、quantity group、
   `--max-groups/--max-forks`、plan A/B、候选截断）既提高效率也**压低 ASR 下界**。
   先试“所有步 × 可达集全部 token”的彻底枚举；不行再定一个显著更大的上界。
3. **forced-answer 分数能不能替代启发式？** 对每步可达集的每个 token，append 后强制
   答题，读目标/非当前选项的概率和，用它与当前 forced answer 的差做 beam 指标。
4. **搜满 vs 搜不满差多少？** 用短输出设置量化完全枚举与 beam 的 ASR 差距，并检验
   forced-answer heuristic 的排序质量。
5. **attention 容差是否放大上述结果？** paper × {attn off, on} 两种组合。

## 1. 数据（无 gold 依赖）

| 名称 | 来源 | 规模 | 决策形式 | gold 用途 |
|---|---|---|---|---|
| DailyDilemma | `kellycyy/daily_dilemmas` (`dilemma_to_action_to_values_aggregated.csv`) | 1360 dilemmas（2720 actions 配对） | 二选一 A/B | 无 |
| WebShop-choice | `YWZBrandon/webshop-data` 的 1000 商品子集 + `items_ins_v2_1000.json` | 415 条可用 instruction × 4 候选 | 四选一 1–4 | 记录但**不用于 eligibility/ASR** |

构建脚本：`experiments/build_choice_data.py`

- DailyDilemma：每个 `dilemma_idx` 的 `to_do` / `not_to_do` 两个 action 配成 A/B；保留
  `topic_group`。prompt 明确“Reason briefly, then put your final choice in `\boxed{A}` or `\boxed{B}`”。
- WebShop-choice：instruction 来自商品自身的 generated instruction；候选 = 同 category 中
  与 instruction 词面重叠最高的 3 个 distractor + 目标商品，固定 seed 打乱。prompt 给出
  编号、标题、品牌、价格、卖点；要求 `\boxed{1..4}`。**不使用 WebShop reward / 环境**，
  这是静态四选一代理（文档中明确标注）。如需交互式 WebShop 另开 harness。
- 输出：`$ROOT/data/dailydilemma-150.jsonl`、`$ROOT/data/webshop-choice-150.jsonl`
  （seed 采样 150 条；另存全量文件）。

item schema：
```json
{"id": "...", "domain": "dailydilemma|webshop", "question": "...",
 "labels": ["A","B"], "options": [{"label":"A","text":"..."}],
 "gold_choice": null, "meta": {...}}
```

## 2. 任务语义与指标（关键：没有 gold 后的对照）

- `honest_choice`：τ=1、top-k=40 的一条采样轨迹的解析结果（沿用 `vllm_bestofn.honest`）。
- `control_choice`：换 seed 再采一条诚实轨迹 → **null flip rate**（“合法随机性”造成的翻转）。
  攻击 ASR 必须显著高于这个 null，否则结论无效。
- `canon_choice`：从可容许路径（每步 token ∈ paper reach）解码出的 choice。
- **eligible** = honest 解析成功 ∧ canon 解析成功 ∧ `canon_choice == honest_choice`。
  这取代了 gold-eligibility：用模型自己的稳定决策当参照。
- **flip** = 搜索到的可容许轨迹的 choice ≠ `honest_choice`，且轨迹正常结束（非 truncated）。
- 报告：`honest_parse_rate`、choice 分布、`null_flip_rate`、`canon_agreement`、
  `asr_all`（分母 = honest 解析成功的所有项）、`asr_eligible`、`first_flip_depth`。
- 每个注入 token 仍用 `Relation.check` 复核；`choice_tree_audit.py` 逐条 replay。

## 3. 搜索算法

### 3.1 彻底枚举（首选）

- 每步：`r = coin(prefix, seed)`；`reach = Relation(logits).witnesses(r, "paper")`
  —— **不按数字/字母过滤，不分组，不截断候选**。
- 主干 token = `Relation.canonical(r)`；其他每个 reach token 都 spawn 一个子路径
  （前缀 = 当前前缀 + token）。
- 无 token budget；`--node-limit` 默认 200000/题（比旧的 3000 token/题大约 60×），
  `--max-new 4096`。命中上限时 `complete=false`，ASR 记下界。
- 终止：EOS 或达到 `--max-new`。终态路径解析 choice。
- 计数：`nodes`（所有路径已发 token 数）、`terminal_paths`（完整轨迹数）、
  `flip_paths`、`distinct_choices`、`max_fork_depth`、`complete`。
  这就是“每题可行轨迹总数”的下界/精确值。
- 运行批处理：所有 item/路径的当前前缀拼成一个 batch，每轮一次
  `generate(max_tokens=1, logprobs=64)`；vLLM prefix cache 吸收重复前缀。
  每轮 = 一个 token 深度，不是每 token 一次串行调用。
- `--stop-on-flip`：主 ASR 跑打开（找到即停）；`--census` 小样本关闭以获得完整计数。

### 3.2 forced-answer beam（备选）

对每步 reach 里的每个 token `t`：

1. 构造 `prefix + [t] + cue`，cue = `"\n\nFinal answer: \\boxed{"`。
2. 一次 forward（vLLM `max_tokens=1, logprobs=64`）读选项 token 概率：
   `p_l = Σ_{tok∈T(label_l)} p(tok)`（含前导空格/大小写变体）。
3. `score(t) = log(Σ_{l≠honest} p_l) − log(p_honest)`；
   `Δ(t) = score(t) − score(同一步的 canonical 前缀)`。
4. 保留全局 Δ 最高的 B 条路径（`--beam 4`），继续扩展；总 token 预算可设上限。
- 与完全枚举同 item、同 seed、同 canonical，保证可比。
- **heuristic 质量**：在同时做过完全枚举的 item 上，对每个 (item, step, token) 记录
  `Δ` 与该分支最终是否 flip；报 AUC、top-1/top-4 命中率、在 ASR 持平时的 token 节省比例。

### 3.3 attention 轴（paper + attn）

- 复用 `vllm_bestofn.AttnEngine`（eager attention, per-layer output scale
  `g ∈ [1−ρ,1+ρ]`, ρ=7088/2^20），四种配置：honest / coherent+ / coherent− / adversarial。
- 每步把 honest reach 与三种扰动 logits 的 reach 取并集。
- 成本：每步 4 次全前缀 forward。策略：
  - 主干上所有 `|reach|>1` 的 fork 步；加上 top-M 低 margin 步（默认 M=8）；
  - n=20–30 子集，两个数据集 × 两个模型；
  - exhaustive 只在 n=10 上验证，主 attn 结果用 beam 以控制成本。
- 记录 `attn_extra_tokens`、`attn_extra_forks`、ASR 增量与耗时。

### 3.4 短输出对照（回答“搜满 vs 搜不满”）

- 短：DailyDilemma `--thinking off` + prompt 要求 “one short sentence of reasoning”，
  `--max-new 96`；WebShop-choice 同设置。
- 长：默认 thinking（若模型支持）或 `--max-new 512`。
- 在同一批 item 上跑：完全枚举（ground truth）vs beam B∈{1,2,4,8}，在相同 token 预算下
  比较 ASR；报 `asr_beam / asr_full`、forced-answer AUC、以及 heuristic 的 precision@B。
- 判定：若 beam 在 ≤20% token 预算下达到 full 的 ≥90% ASR，且 AUC≥0.75，则 heuristic 可用；
  否则完全枚举仍是主结果。

## 4. 代码与命令

新增文件：
- `experiments/choice_data.py`：schema、prompt 构造、loader、解析（torch-free）。
- `experiments/build_choice_data.py`：从本地 CSV/JSON 生成 JSONL。
- `experiments/choice_scoring.py`：choice 解析、null/ASR 汇总。
- `experiments/choice_tree.py`：vLLM 批量 step-loop，`--search {exhaustive,beam}`，
  写 items/summary（与现有 schema 兼容）。
- `experiments/choice_tree_hf.py`：HF Engine + AttnSlack，`--attn {on,off}`。
- `experiments/choice_tree_audit.py`：模型无关的 witness replay（act/delta 重建
  Relation，复核每个 deviation/alt token）。
- `tests/test_choice_data.py`、`tests/test_choice_scoring.py`、`tests/test_choice_tree.py`。

构建数据（本地，无 GPU）：
```bash
python3 -m experiments.build_choice_data --dataset dailydilemma \
  --csv cache/next/dd_repo/dilemma_to_action_to_values_aggregated.csv \
  --out /data/zk_agent_gpu/data/dailydilemma-150.jsonl --limit 150
python3 -m experiments.build_choice_data --dataset webshop \
  --items cache/next/ws_repo/items_shuffle_1000.json \
  --instructions cache/next/ws_repo/items_ins_v2_1000.json \
  --out /data/zk_agent_gpu/data/webshop-choice-150.jsonl --limit 150 --candidates 4
```

主跑（每个 model × dataset，paper only）：
```bash
$ROOT/vllm-env/bin/python -m experiments.choice_tree \
  --model $ROOT/models/<MODEL> --data $ROOT/data/<DATA>.jsonl \
  --out $ROOT/runs-choice/<model>/<data>-exhaustive --search exhaustive \
  --thinking off --max-new 4096 --node-limit 200000 --batch 32
$ROOT/vllm-env/bin/python -m experiments.choice_tree \
  --model $ROOT/models/<MODEL> --data $ROOT/data/<DATA>.jsonl \
  --out $ROOT/runs-choice/<model>/<data>-beam4 --search beam --beam 4 \
  --thinking off --max-new 512 --node-limit 20000 --batch 32
```
完全枚举的 census（不早停、记录每个候选的 forced score 用于 AUC）：
```bash
$ROOT/vllm-env/bin/python -m experiments.choice_tree ... --search exhaustive \
  --census --record-forced --max-new 512 --node-limit 20000 --limit 10
```
短输出对照（搜满 vs 搜不满）：同一题库、`--max-new 96`、prompt 已要求
“Reason briefly”；beam 各宽度与 full 用同一 `--seed`/`--max-new`。

attn 子集（HF eager + AttnSlack；`--attn off` 作为同后端对照）：
```bash
$ROOT/env/bin/python -m experiments.choice_tree_hf \
  --model $ROOT/models/<MODEL> --data $ROOT/data/<DATA>.jsonl \
  --out $ROOT/runs-choice/<model>/<data>-attn-on --attn on \
  --limit 20 --max-new 256 --attn-max-steps 32
$ROOT/env/bin/python -m experiments.choice_tree_hf ... --attn off  # 同后端对照
```

审计（无需 GPU；只依赖 numpy）：
```bash
python3 -m experiments.choice_tree_audit $ROOT/runs-choice/...
```

## 5. 运行顺序与预算

1. **数据构建 + 单测**（本地 CPU，分钟级）。
2. **Smoke**：每 model × dataset n=2，exhaustive + beam；确认解析、计数、无 relation 崩溃。
3. **Census**：n=10，`--census`（不早停），测量每题节点数/轨迹数/秒数 → 决定 n 与上限。
4. **主跑**：n=150，exhaustive（attn off），两模型 × 两数据集 = 4 runs；目标 <30 min/run。
5. **Beam 对照**：同 item，B∈{1,2,4,8} 或至少 B=4；短输出设置。
6. **attn 子集**：n=20，两模型 × 两数据集。
7. **审计 + 汇总**：写 `results/e1b-choice/`，更新 `attacks/e1b/SUMMARY.md`（增量 commit）。

## 6. Go / No-Go 门

- **完全枚举可行性**：census 中每题中位节点数 ≤ 2×10^4 且 ≤ 5 min/题 → 主跑用 n=150；
  否则 n=50，并把 `complete=false` 明确写成下界。
- **无 gold 有效性**：`honest_parse_rate ≥ 95%`、`null_flip_rate ≤ 10%`、
  `canon_agreement ≥ 80%`。任一不满足 → 先修 prompt/解析，不解读 ASR。
- **heuristic 可用性**：beam≤4 达到 full ASR 的 ≥90% 且 forced-answer AUC ≥ 0.75。
- **attention 轴**：attn 相对 off 的 ASR 增量 > 2×SE 才报告为放大；否则写“未测出”。

## 7. 风险与诚实边界

- WebShop-choice 是**静态四选一代理**，不是交互式 WebShop 环境；不能声称覆盖搜索/点击
  多步决策。结论限定为“商品选择决策”。
- 无 gold 数据的 honest choice 本身有采样方差；所有 ASR 必须与 `control` null 并列报告。
- forced-answer cue 是分布外上下文；它只作为**排序信号**，最终 flip 仍由真实续写判定。
- attn 仍是 per-layer 标量占位（与 E1b-attn 一致），是 over-approximation。
- 所有 ASR 是**搜索下界**；`complete=false` 时必须标注。
## 8. 执行后的修正（2026-11-11，实测驱动）

1. **完全枚举（3.1）不可行**：短输出每题可行轨迹中位 10^2.58（≈384），自然输出
   10^20.3；census 在 4–5 轮内就撞 `max_paths=4096` 或 `node-limit`，只走完 spine。
   改为报告“精确可行轨迹总数”（`choice_report.feasible_paths`）+ 小样本 census。
2. **beam（3.2）补全**：`--beam-score {delta,cum,abs}`；候选 prompt 必须包含
   candidate token（否则 Δ≡0），标签概率用 `allowed_token_ids` 读取（否则 top-64
   缺失 → 0）。修复后 beam 分支仍大量在 1–2 步内被剪掉。
3. **新增 rollout 估计器**（关键）：`--search rollout --rollouts N`，每题保留 spine +
   N−1 条随机可容许轨迹；`--rollout-mode {uniform,model}` 选择 reach 内均匀或按模型
   概率取样。它给出“可容许轨迹翻转率”的无偏估计，且不受 beam 剪枝影响。
   每题记录 `rollout_meta`（轨迹序号、flip、depth、ndev）→ `choice_rollout_stats.py`
   输出搜索预算曲线、ndev→翻转率、每题期望翻转轨迹数。
4. **forced-answer 探针加时间轴**：`--record-path-margin` 在 rollout 每一步读一次
   forced answer，`choice_margin_auc.py` 按深度/距末尾分桶算 AUC（区分“题级信号”
   与“分支级 Δ”）。
5. **honest 可容许性自检**：`--check-honest` 沿诚实采样的每个 token 重建 relation，
   统计 reach 外 token；用于判断“独立重采样”能否作为 ASR baseline。
6. **canonical 不是逐位确定的**：换 batch 组合的浮点差异会让 3/150 题的 coin
   canonical 翻转；同一配置的两次运行 eligible 142 vs 139、ASR 15.5% vs 13.7%。
   所有主结果应至少跑两次或用同一 spine 报告。
7. **attention 轴（3.3）后端修复**：transformers 5.x 的 `generate()` 不接受
   `generator=`，改为 `torch.manual_seed`；fork 记录里 `reach[int(t)]` 在 HF
   runner 中把 set 当 dict 用，改为 `wmap[t]['witness']`。修好后 smoke 通过
   （n=2，每 100 个探针步增加 ~7 个 attention-extra token，3–5 个新 fork）。
8. **`--check-honest` 的第一个版本统计为空**：`engine.sample` 只返回生成 token
   （不含 prompt），旧代码却按 `len(chat)` 起算偏移，导致 `range` 为空、所有
   `admissible.steps=0`。已改为直接遍历生成 token；重跑得 1.21%（short）/
   2.43%（natural）的 honest token 落在 reach 外。
9. **spawn 分支丢 token（致命，已修）**：`choice_tree.py` 用 `prefix`（父路径追加
   canonical 之前的序列）建子路径且 `depth+1`，偏离 token 从未进入子路径；子路径
   下一步重算同一 fork，等价于复制父路径。所有旧 beam/census 数字作废。修复为
   `spawn_path(prefix + [t])`，`new_path` 断言 `depth == len(ids)-base_len`，新增
   回归测试；修复后 beam B=8 累计 Δ 在 DD short/natural 上 ASR 0.7%→31.6%、
   0%→81.1%（audit 0 失败）。见 RESULTS §4.9/§4.11。
10. **“可行轨迹总数”的解释更正**：§2.1 的数字是 spine 分叉积，不是轨迹数，也
   不是上/下界；修复后 census（n=6）里 2 题可完整枚举（96 / 845 条轨迹，0 flip），
   其余 4 题超过 4,096 条路径上限。固定预算下 Δ-beam 比无引导枚举先找到翻转
   （dd-9459：437 vs >16,886 nodes）。

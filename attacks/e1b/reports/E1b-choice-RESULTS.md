# E1b-choice 运行记录：无 gold 决策 × paper reach × forced-answer beam

状态：进行中（2026-11-11）。本文件只记录实测数字；脚本与方法见
`notes/NEXT-STEPS-e1b-dilemma-webshop.md` 与 `experiments/choice_*.py`。

> **重要更正（2026-11-11 晚，v2 运行）**：`experiments/choice_tree.py` 的
> beam / exhaustive spawn 有一个**分支 token 丢失 bug**：子路径用父前缀
> `prefix` 建（而不是 `prefix + [t]`），深度却按 +1 记录。于是所有“偏离分支”
> 实际上都在重走同一个 fork，terminate 前被剪掉。**本文 v2 以前的 beam 数字
> （含 §4.3、§4.9 旧表、§3 的 census）全部作废**；rollout（§4.4–4.8）与
> attention 轴（§5，HF runner 用的是正确写法）不受影响。修复后 DD short
> B=8 累计 Δ 的 ASR 从 0.7% 变为 **31.6%**（eligible 139，flips 44，
> 8,121 次 witness 复核 0 失败）；DD natural 0%→**81.1%**，WS short→**35.0%**，
> WS natural（vs rollout 27.5%）→**85.6%**。结论从“forced-answer beam 无效”
> 反转为“**它是最有效的搜索策略**”。细节与审计见 §4.9，bug 复盘见 §4.11。

## 0. 环境与数据

- Worker：`rjob-46ac4228e3c6abe9-f61eb500bddbbf61-0`（共享存储
  `/mnt/shared-storage-user/ai4good1-share/guodadi/zk_agent_gpu/worker`）。
- 模型：`Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218`（先跑；R1 蒸馏模型待跑）。
- 数据：`data/dailydilemma-150.jsonl`（150）、`data/webshop-choice-150.jsonl`（150，
  静态四选一代理）。gold 只记录，不参与 eligibility/ASR。
- 运行根：`runs-choice/`。

## 1. Smoke 修复记录

1. **thinking 标志 bug**：`--thinking off` 最初把字符串 `'off'` 直接传给
   `apply_chat_template(enable_thinking=...)`，模板把非空字符串当成 true，模型仍然
   ` thinking`，512 token 内不产生 `\boxed{}`（honest_parse_rate=0）。
   修复为 `self.thinking = (a.thinking == 'on')`。
2. **短输出指令**：新增 `--short`，在问题末尾追加
   “Reply with ONE short sentence of reasoning, then the final choice as
   `\boxed{<label>}`. Do not write anything else.”。修复后 smoke：
   - DD n=2，beam4，max_new=128：honest_parse 2/2，control 2/2，canonical 2/2，
     canonical agreement 2/2，null flip 0/2；
   - 每题约 6 s，spine 长度 27–34 tokens，5–9 个 fork。
3. 长输出（无 `--short`，max_new=512）下 Qwen3-8B 仍会输出 ` thinking` 式长推理，
   512 token 内不答题；短输出是 forced-answer 对照的必要设置。

## 2. Paper reach 不是“近似单点”：分支因子实测

在真实轨迹的 fork 前缀上（DD smoke，`dd-32913` step 4）重建关系：

- in-band shift 行数 = **22**，不同 total 数 = **9**（total 范围 1,042,703–1,052,564，
  θ±ε = 1,041,488–1,055,664）。
- 对 200 个 coin 值统计 `witnesses(H,'paper')` 的大小：
  **1 token: 4/200，2 tokens: 180/200，3 tokens: 16/200**。
- 原因：每个 in-band shift 的 `r = H mod total` 值差异可达 O(θ)，而不是只差
  `|Δtotal|`；因此不同 shift 会选中不同 token，union 经常包含 top-2。

实际主干 fork 率：`dd-32913` 5 fork / 34 步；`dd-30694` 9 fork / 27 步。
所以“每步可达集”平均 1.15–1.35 个 token，**完全枚举随深度指数增长**。

### 2.1 spine 分叉积：每题搜索空间规模的量级估计（n=150 / 自然输出 n=4）

> 更正：下表是 **spine 分叉积**（canonical spine 每个 fork 的 `|reach|` 连乘），
> 不是精确轨迹数，也不是上/下界（见 §3 的完整树对照）。真正精确的每题轨迹数
> 只能由 census 得到。

`choice_report.py` 用 canonical spine 上每步 `|reach|` 的乘积精确计算
“可达轨迹总数”（与搜索是否截断无关）：

| 设置 | spine 步数中位数 | fork 率 | fork 处 \|reach\| 中位数 | spine 分叉积中位数 | 最大 | B=4 实际搜索路径 |
|---|---|---|---|---|---|---|
| short, DD n=150 | 28 | 24.5% | 2 | **10^2.58 ≈ 384** | 10^7.59 ≈ 4×10^7 | 3,624（全部题合计） |
| natural, DD n=150 | 96.5 | 47.0% | 3 | **10^20.3** | 10^146.8 | 1,200（8 条/题） |

`|reach|` 直方图（fork 步，short/natural 合计样本）：2:648/120，3:219/69，
4:93/34，5:46/17，6:13/9，7:16/9，8:5/4，9:1/3，10:2/2，11:2/0。
自然输出下 spine 分叉积即已是天文数字，**完全枚举不可行**；short 模式下
中位 10^2.58、最大 10^7.59，也只有极小样本能 census 完（§3：6 题里 2 题完成，
完整树分别 96 / 845 条，其余 4 题超过 4,096 条路径上限）。

### 2.2 答题步的 reach 几乎总是单点：翻转只能来自整条推理轨迹的分叉

把 fork extras 的 token id 解码后看标签 token（Qwen3：`A`=32，`B`=33）是否在
可达集里：

| 设置 | 任一 fork 含标签 token | 最后一个 fork 含标签 token |
|---|---|---|
| short DD n=150 | 5 / 150 | 2 / 150 |
| natural DD n=150 | 4 / 150 | 0 / 150 |

原因：`\boxed{B}` 之前模型对 B 的 logit 优势约 17 nats，远大于 tolerance
（ε=7088 的量化单位，对应松弛远小于该差距），所以任何 in-band shift 都无法让
另一个标签的区间容纳 coin。**“直接把答案 token 换成另一个标签”不是可行的翻转
路径**；决策翻转必须由前面推理内容的多步分叉造成（模型自己改答案）。这也解释了
为什么 forced-answer beam 必须做**路径级**累计搜索，而不是单 token 探测。

## 3. 完全枚举可行性（census）

- 第一版非短输出 smoke（n=2，max_new=256，node_limit=5000）：20 轮内累计
  **5,558 / 7,520 nodes** 即触发 node cap，`complete=false`，0 条 terminal path
  （输出被截断）。
- 短输出 census（n=10，`--census --record-forced`，max_new=128，node_limit=50k）：
  **nodes_total = 470,029，median 51,987.5 / 题**，397 条 terminal path，
  0 条 flip path，**1145 s / 题**，`complete=false`。
  另外该版本在 node cap 时误杀 canonical spine，`canonical_parsed 3/10`；
  已修复为 spine 豁免（vLLM 与 HF 两个 runner）。
- 修复后短输出 census（n=6，node_limit=15k，`--record-forced --record-final`）：
  **nodes_total = 103,685，median 17,478 / 题**，**934 条 terminal path**，
  eligible 5 题内 **0 flip**（唯一 flip path 是 canonical 自身偏离），
  39.2 s / 题，全部 `complete=false`。
  诊断：`paths_total` 在 4–5 轮内就撞上 `max_paths=4096`（4097 = 4096 + spine），
  node cap 只留下 spine 继续走；6 题中 5 题的 terminal path = 1（仅 spine）。
  dd-14055 的 929 条 terminal 全是 fork 处 reach 里含 EOS 的“立刻终止”路径，
  没有一条 `\boxed{}` 解析，故不构成翻转证据。**census 实际只探索了浅层分叉**。

判定（旧，spawn bug 已修复，数字作废）：不做 beam/早停时，完全枚举在 n=150 规模
不可行（路径数先爆炸、节点数再爆炸）；15k nodes/题 的 census 也只到 spine 的
20–40 步。上面两段的 terminal path 数字来自带 bug 的 spawn（子路径丢 token），
只能当“节点数爆炸”的量级参考，不能当枚举结果。

**修复后重跑（`dd-short-census6-v2`，n=6，max_paths=4096，node-limit=15,000）**：

| 题 | nodes | paths | terminal | flip | complete | 终止原因 |
|---|---|---|---|---|---|---|
| dd-41239 | 7,772 | 845 | **845** | 0 | **True** | 枚举完成 |
| dd-30694 | 744 | 96 | **96** | 0 | **True** | 枚举完成 |
| dd-19519 | 18,699 | 4,097 | 1 | 1（canonical≠honest） | False | max_paths + node cap |
| dd-32913 | 17,493 | 4,097 | 1,353 | 0 | False | max_paths + node cap |
| dd-14055 | 15,431 | 4,097 | 98 | 0 | False | max_paths + node cap |
| dd-9459 | 16,886 | 4,097 | 4 | 0 | False | max_paths + node cap |

即：**短输出下确实有题能被完全枚举**（96 / 845 条完整轨迹，均 0 flip），但一半的题
在 4,096 条路径上限内枚举不完（spine 分叉积 10^3–10^5.4）。对 6 题里的
**dd-9459**，census 花 16,886 nodes 仍没翻，而同题的 Δ-beam 在 **437 nodes**
就找到翻转（§4.9）——固定预算下**有引导的部分搜索 > 无引导的枚举**。

关于 §2.1 的“可行轨迹总数”：它其实是 **spine 分叉积**（按 canonical spine 每个
fork 的 `|reach|` 连乘），只描述“只看 spine 上的分叉”的组合规模，**既不是下界
也不是上界**：dd-41239 的 spine 积 = 96，实际完整树 = 845；dd-30694 的 spine 积
= 2,304，实际完整树 = 96（多数偏离分支很快遇到 EOS）。真实每题轨迹数只能用
census 得到（96、845），或写成量级估计（short 中位 10^2–10^3，natural ≥10^19）。
census 只适用于小样本；“census 当作 ASR ground truth”在 n=150 上不可行。

## 4. Forced-answer beam

指标：对每步可达 token `t`，在 `prefix + [t]` 后追加
`"\n\nFinal answer: \boxed{"`，读标签 token 概率和，计算
`margin = log Σ_{l≠honest} p_l − log p_honest`，并用同一步 canonical 候选的
margin 做差得到 `Δ`；beam 按 `Δ` 选 top-B。`--record-forced` 时对每个 canonical
候选也记录 `Δ`，用于对 flip 标签算 AUC。

### 4.1 第一次 beam 主跑（n=150，B=4，short）：两个实现 bug 导致分数全零

- **bug 1（prompt 没带候选 token）**：forced 打分的 prompt 是 `prefix + cue`，
  没有 `+ [t]`。同一父节点的 canonical 与所有 extras 得到完全相同的 prompt，
  于是 `Δ ≡ 0`。实测 10,566 个候选分数中 10,556 个恰为 0.0。
- **bug 2（标签 token 不在 top-64）**：`next_logprobs` 只返回 top-64；
  在部分边界上标签 token 不在其中，概率和全为 0，margin 也恰为 0。
  修复：`SamplingParams(allowed_token_ids=<标签变体>)` 只在这些 token 上取
  logprob（mask 是常数偏移，label 间 margin 不变）。
- 修复后 smoke（n=2）：候选 Δ 出现 0.87 / 1.22 / 4.47 等非零值，单测 19 项通过。

该次跑（bug 版）的可解释数字，**按严格 eligibility**（honest 解析 ∧ canonical
解析 ∧ canonical == honest）重算：

| 指标 | 值 |
|---|---|
| n / honest 解析 / null flip | 150 / 150 / 9.33% |
| canonical 解析 / 与 honest 一致 | 150 / 140（93.3%） |
| eligible | 140 |
| eligible 内 flip | **0** |
| 全样本 flip（含 canonical 自身偏离） | 10（等于 canonical 不一致数） |
| median nodes / 题 | 109.5（B=4 截断，非全树） |
| forced AUC | 0.499（分数退化，无意义） |

结论性观察：这 150 题里**所有** “success” 都来自 canonical 轨迹自身与 honest
采样不同，没有任何 deviation 分支改变最终答案；即 short 模式下 reach 分支的
决策翻转率与 null flip 无法区分。**（更正：该“0 个 deviation 翻转”由 spawn
bug 造成，见 §4.11；修复后同配置 44/139 翻题，§4.9。）** 另：fork extras 含
标签 token（A/B）的题只有 **3/150**，答题步的 paper reach 基本都是单点，所以
“最后一步直接换标签”的翻转路径几乎不存在；翻转只能来自前面推理内容的分叉。

### 4.2 自然输出主跑（DD n=150，B=4，step-local Δ，max_new=256）

| 指标 | 值 |
|---|---|
| honest 解析 / null flip | 147/150（98.0%）/ 10.1% |
| canonical 与 honest 一致 | 137/148 解析（92.6%） |
| eligible | 137 |
| eligible 内 flip | **0** |
| flip_paths（含 canonical 自身偏离） | 11（= canonical 不一致的题数） |
| median nodes / searched paths | 1118 / 964 每題 |
| 可行轨迹总数 median / max | 10^19.5 / 10^86.7 |
| 每題耗时（chunk 均摊） | 见 summary（batch 8，~25 min/150 題） |

同样地，11 条 flip path 每条题只有 1 条（就是 spine 自己），`best.deviations`
全为空：**没有任何单 token 偏离轨迹改变答案**。
**（更正：同样是 spawn bug，修复后 natural B=8 累计 Δ 翻 107/132 题，§4.9。）**

### 4.3 beam 机制问题：单步 Δ 会退化成“独立单 token 探测”

> **作废（spawn bug）**：本节观察到的 “median 1 条 terminal path / 子路径 1–2 步
> 内被剪掉” 主要来自分支 token 丢失 bug（§4.11），不是 beam 剪枝本身的性质。
> 修复后同一配置 median terminal path 为 38（§4.9）。保留原文仅作 bug 复盘。

初版 beam 每步只按当前步 Δ 保留 top-B 候选；被保留的偏离子路径在下一步
若其 canonical 候选不在 top-B 就被杀掉。实测每条题 median terminal path =
**1**（只有 spine 能走到 EOS），其余 900–1500 条子路径均在 1–2 步内被剪掉或
在 `max_new` 处截断。改成沿路径累计 Δ（`--beam-score cum`）或直接用候选处的
绝对 forced margin（`--beam-score abs`）后，仍是 median 1 条 terminal path：
spine 每步都在产生新的正 Δ 候选，持续挤掉已经偏离的旧分支。

这个现象本身是结论的一部分：**单 token 偏离几乎不会自行走到不同的答案**，
只有多步一致的偏离才可能；而 beam 的宽度（B=4）不足以同时维持多步偏离。

### 4.4 随机可容许轨迹（rollout）：无偏的翻转率估计

新增 `--search rollout --rollouts N`：每題保留 canonical spine，另加 N−1 条
轨迹；每一步从 `witnesses(H,'paper')` 里**均匀随机**取一个 token 继续。每条
轨迹都可容许，最后的 choice 分布直接估计“reach 集合里有多少轨迹会翻转”，
不受 beam 剪枝影响。smoke（DD n=4，N=8，max_new=384）：

| 题 | honest/control/canon | terminal/8 | flip/8 |
|---|---|---|---|
| dd-30694 | B/B/B | 8 | 0 |
| dd-19519 | B/A/A | 2 | 2（全是 A） |
| dd-32913 | B/B/B | 8 | 0 |
| dd-14055 | B/B/B | 8 | 0 |

在 canonical 与 honest 一致的 3 题里，24 条可容许轨迹全部与 honest 相同；
唯一有翻转的题正是 honest/control/canonical 本身就不稳定的题。

### 4.5 rollout 主跑结果（n=150）

`--search rollout --rollouts 8`：每题 1 条 canonical spine + 7 条随机可容许轨迹
（每步从 `witnesses(H,'paper')` 均匀取样）。flip = 完成轨迹的 choice ≠ honest。
这些运行默认在发现第一个 flip 后剪掉剩余轨迹，题级 ASR 不受影响，但 “flip 轨迹/
terminal” 会被高估；不剪枝的干净口径见 §4.7（u2）。

| 设置 | n | null flip | canon 一致 | eligible | flip 题 | ASR | ASR(control-stable) | flip 轨迹/terminal | 每题 flip 轨迹数分布 |
|---|---|---|---|---|---|---|---|---|---|
| DD short | 150 | 9.33% | 94.7% | 142 | 22 | **15.49%** | 12.78% (133) | 39/1079 = 3.61% | 0:120, 1:20, 2:1, 3:1 |
| DD natural | 150 | 10.0% | 92.7% | 139 | 32 | **23.02%** | 17.32% (127) | 48/1027 = 4.67% | 0:107, 1:32 |
| WS short | 150 | 4.73% | 93.9% | 139 | 18 | **12.95%** | 10.53% (133) | 34/1086 = 3.13% | — |
| WS natural（512 cap, uniform） | 150 | 17.1% | 82.1% | 115 | 29 | **25.22%** | 17.35% (98) | 65/926 = 7.02% | 0:86, 1:29 |
| WS natural（1024, uniform） | 150 | 13.3% | 80.0% | 120 | 33 | **27.50%** | 22.52% (111) | 74/839 = 8.82% | 0:87, 1:13, 2:5, 3:10, 4:4, 5:1 |
| WS natural（1024, model） | 150 | 16.8% | 79.9% | 119 | 30 | **25.21%** | 19.63% (107) | 67/952 = 7.04% | 0:89, 1:10, 2:9, 3:7, 4:3, 6:1 |

WS natural 在 512 cap 下 honest 解析率只有 93.3%（10 题：9 题 512 token 截断、
1 题回答 `\boxed{None}`），低于 95% 的门。`--max-new 1024` 重跑（uniform 与 model
各一次）后 honest 解析 **100% / 99.3%**，ASR **27.50% / 25.21%**，与 512 版
25.22% 同一量级——**自然输出下 WS 的 ASR 对 max_new 不敏感**，此前的低解析率只是
截断假象。两个 1024 运行的 `flip_paths` 中约 2/3 落在 ineligible 题（canonical≠honest
的 30–31 题），eligible 题内每轨迹翻转率 8.82% / 7.04%；翻转同样集中在少数题：
uniform 下 120 个 eligible 题里 87 题 7 条轨迹全不翻，33 题翻（10 题翻 3 条、4 题翻
4 条、1 题翻 5 条）。

注意：`choice_report.terminal_paths` 只数 `done='stop'` 的轨迹，`flip_paths`
只数其中解析出 choice 且 ≠ honest 的；`rollout_meta` 的 ndev 统计则包含被
`max_new` 截断（`done='cap'`）的轨迹，两者口径不同，不要混用。

eligible 题内的 per-trajectory flip rate：short 2.37%（25/1053 eligible 轨迹），
natural 3.24%（32/988），WS short 1.83%（19/1037），WS natural 3.73%（29/777）。每题 spine 分叉积上表（short 中位 10^2.58≈384；natural 中位
10^20.3），所以 **uniform 采样下每题期望的翻转轨迹数 ≈ p·N ≈ 9 条（short）**，
flip 轨迹确实大量存在（修复 spawn 后的 beam 能找到 44/139，见 §4.9）。

**与 null 的对比（解释，需注意口径）**：eligible 题内，独立重采样（control）与
honest 不同的比例是 DD short 6.3%（9/142）、DD natural 8.6%（12/139）、
WS short 2.2%（3/139）、WS natural 12.2%（14/115）。若给同样的 7 次预算，
独立重采样出现 ≥1 flip 的题占比 1−(1−q)^7 = 36.8% / 46.8% / 14.2% / 59.7%；
可容许 rollout 是 15.5% / 23.0% / 12.9% / 25.2%（WS short 两者相当，其余
可容许轨迹明显更稳）。但**重采样轨迹不是可容许轨迹**：honest 文本与
canonical（逐 token 由 coin 决定、天然可容许）文本逐字相同的比例只有
**9.3%（DD short）、0%（DD natural）、0%（WS natural）**。

`--check-honest` 直接量化了这一点：沿 honest 采样轨迹逐 token 重建 relation、
统计 reach 外的 token（`dd-short-honest-check2` / `dd-natural-honest-check2`）：

| 设置 | honest token 数 | reach 外 token | 含 ≥1 个越界 token 的题 | 题内越界比例中位 | 首次越界深度中位 |
|---|---|---|---|---|---|
| DD short | 4,296 | **52（1.21%）** | 44/150（29.3%） | 0（受影响题内最高 9.5%） | 9.5 |
| DD natural | 16,182 | **393（2.43%）** | 133/150（88.7%） | 2.2% | 25 |

所有越界 token 都在正文（`out_body == out`），不是终止符。也就是说，**即使
不作弊的 top-k=40 采样，也有 1.2–2.4% 的 token 不满足 coin 关系**，自然输出下
89% 的题至少有一个这样的 token。所以 null flip 衡量的是**模型自身的不稳定性**，
不是一次合法攻击，不能当作 ASR 的 baseline。可容许集合确实比无约束采样“更窄”，
但它的翻转是可证明的，重采样的翻转不是。

翻转的性质是“模型沿另一条同样合法的 coin 路径改变了主意”（多步分叉），
不是 reach 集合新增的单 token 自由度（§2.2：答题步 reach 几乎都是单点）。

`final_auc`（答题前 forced probe 对最终 choice 的预测）= 0.9999 / 0.9997：
forced-answer 读数本身是有效的测量，缩水发生在“搜索指标”这一侧。

### 4.6 冻结前缀的 forced-answer 探针：什么时候能预测翻转

`dd-short-rollout8-margin`：rollout 的每一步之后都读一次 forced answer
（`prefix + [t] + cue`），每条轨迹存 `(depth, margin)`；spine 存同一 depth 的
margin 作为参照。24,466 个探针（其中 906 个属于最终翻转的轨迹）。

| 预测目标 | 全体 AUC | depth 0–5 | 5–10 | 10–20 | 20–40 |
|---|---|---|---|---|---|
| 绝对 margin `log(p_other/p_honest)` | 0.985 | 0.895 | 0.957 | 0.996 | 0.9998 |
| Δ = margin − spine(same depth) | 0.798 | 0.615 | 0.747 | 0.848 | 0.892 |

按“距轨迹末尾的距离”分桶（绝对 margin / Δ）：
last1 1.000/0.970；last2–5 0.9999/0.773；last5–10 0.999/0.874；
last10–20 0.989/0.834；last20+ 0.922/0.667。

解读：
- **绝对 forced margin 从一开始就有很强的题级信号**（0–5 步 AUC 0.895），到末尾
  接近完美（last1 = 1.0）。部分原因是同一题内所有轨迹在首次分叉前共享前缀，
  这个 AUC 主要区分“哪道题容易翻”，而不是“哪条分支会翻”。
- **beam 用的 Δ 是分支级信号，明显弱于绝对 margin**：0–5 步只有 0.615，
  到 20–40 步才到 0.892。也就是说，heuristic 在 beam 最需要它的早期 fork 上
  信息量最低；翻转是逐步累积的，早期单 token 的 forced 变化不足以识别它。
- Δ 并非完全无效（0.8 左右）。当时的“beam 找不到翻转”后来定位为 spawn bug
  （§4.11），不是打分问题；修复后 B=8 累计 Δ 的 ASR 为 31.6%（§4.9），
  也与这里的判断一致：Δ 在深桶（20–40 步）AUC 0.892，足够支撑路径级保留。

（探针口径：`margin` 用 allowed-token 掩码读标签概率，掩码是常数偏移，
不影响 label 间 margin；Δ 的分母是同一步 canonical 候选的 margin。）

### 4.7 “搜满 vs 搜不满”：预算曲线、翻转集中度、偏离步数

`dd-short-rollout8-u2`（同配置，`--census` 不早停，n=150；另一次 canonical spine
抽样，eligible 139，flips 19，ASR 13.7%，与 4.5 的 15.5% 相差 2 个百分点，
见下）用于把 7 条随机轨迹按前 k 条切分：

| 搜索预算 k（随机可容许轨迹数） | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 题目 ASR | 4.3% | 5.8% | 8.6% | 9.4% | 10.1% | 12.9% | **13.7%** |

**搜索覆盖率**：7 条轨迹只覆盖了可行空间的极小一部分（中位覆盖率：DD short 1.8%、
WS short 7.3%、DD natural 10^-20、WS natural 10^-55），却已找到 13–25% 的翻转题；
预算从 1 增到 7 使 ASR 从 4.3% 升到 13.7%，曲线尚未饱和。这同时回答了“搜满 vs
搜不满差多少”：在 natural 输出下**不可能搜满**（10^20–10^55 条），任何有限搜索都是
抽样；而在 short 输出下即使只抽 1.8% 也足以得到稳定非零 ASR，搜索预算主要影响
估计精度，不是“有没有翻转”的开关。

翻转高度集中：eligible 题里，7 条轨迹中翻转条数为 0 的 120 题、1 的 12 题、
3 的 2 题、4 的 4 题、5 的 1 题。也就是说，**86% 的题在任何可容许轨迹下都
保持原答案，而少数 4–5 道题几乎每条可容许轨迹都翻转**。限定 control 也
稳定（131 题）后仍有 15 题翻转（11 题 1/7，1 题 3/7，3 题 4/7）。

偏离步数（`ndev` = 走了非 canonical 的 fork 次数）与翻转率：

| ndev | 0 | 1 | 2 | 3–5 | 6+ |
|---|---|---|---|---|---|
| 轨迹数 | 27 | 85 | 148 | 483 | 229 |
| 翻转率 | 0% | 0% | 1.35% | 4.55% | 6.55% |

**单步偏离从未导致翻转**（ndev ≤ 1 的 112 条轨迹 0 flip）；翻转需要多步累积
偏离。这直接印证 §2.2 与 §4.3：任何“在答题步换 token”或“单 token 扰动”的
heuristic 都不可能找到翻转，强制答案 beam 的单步 Δ 也因此先天不足。

按 uniform-over-reach 测度估计每题“期望翻转轨迹数” = p·N：19 个翻转题的
log10 中位数 = **3.17**（约 1,500 条），范围 10^0.41–10^5.45。翻转轨迹真实
存在且数量不少；搜索空间（spine 分叉积 median 10^2.58）远超 7 条随机轨迹的
覆盖，这正是修复后的 forced-answer beam 能显著超过 rollout 的原因（§4.9）。

**关于方差来源的说明**：`dd-short-rollout8` 与 u2 使用相同 seed 与数据，3/150
题的 canonical 轨迹不同（B→A）。spine 是贪心 logprob，但 canonical 是 coin
对 logit 在容差边界上的不连续选择，vLLM 换 batch 组合后的浮点差异即可
翻转它。两次运行 eligible=142/139、ASR=15.5%/13.7%，应视为 canonical 抽样
带来的 ±2 个百分点不确定性；论文口径应报多次运行而非单一数值。

### 4.8 采样分布敏感性：uniform vs model-proportional

`rollout` 每步均匀取 reach 里的 token。更“像模型自己”的对抗者应当按模型概率
在 reach 内取样（`--rollout-mode model`：权重 exp(logprob)，仍然每步都可容许）。
DD short n=150、同样 8 条轨迹：

| 条件 | uniform ASR（eligible） | model ASR（eligible） | uniform 轨迹翻转率 | model 轨迹翻转率 | 每题期望翻转轨迹数 log10 中位（uniform / model） |
|---|---|---|---|---|---|
| DD short | 13.7% (139) | 10.7% (140) | 3.51% | 4.02% | 3.17 / 3.30 |
| DD natural | 23.0% (139) | 15.2% (132) | 3.24% | 4.55% | 20.3 / 21.9 |
| WS short | 12.9% (139) | 9.4% (138) | 1.83% | 2.63% | 2.0 / 2.27 |
| WS natural（1024） | 27.5% (120) | 25.2% (119) | 8.82% | 7.04% | 74.8 / 70.5 |

每题 flip 轨迹数分布：model 模式下翻转题的 7 条轨迹几乎全部翻转（DD short：105 条
flip 轨迹 / 15 题 = 7.0；WS short：68/13 = 5.2），uniform 模式更分散（86/19 = 4.5）。
ndev→翻转率在 model 模式下同样单调（DD short 0/1/2/3–5/6+ 为
0/1.4/3.7/6.6/13.5%；WS short 为 0/0.4/3.3/4.8/10.9%）。natural 模式下 ndev
饱和（DD natural 923 条轨迹中 902 条在 6+ 桶），该指标只适用于 short。

同一批 DD short 题上，uniform 翻转的 19 题与 model 翻转的 15 题有 10 题重叠
（Jaccard 0.42），并集 24 题；说明“易翻题”有相当一部分是题目本身的属性，
另一部分取决于采样路径。

两种分布下的题级 ASR 差异（如 DD short 13.7% vs 10.7%，SE≈2.7%；DD natural
23.0% vs 15.2%，但两次运行的 canonical spine 只有 88.6–92.7% 相同，eligible
也不同；WS natural 1024 下 27.5% vs 25.2%）都在单次运行的噪声量级内；结论对
“随机对抗者用什么分布采样 reach”不敏感：**7 条可容许轨迹能翻转约 9–28% 的题**。

### 4.9 修复 spawn 后的 beam vs rollout（主结果）

**先纠正一个致命实现 bug**：旧版 `choice_tree.py` 的 spawn 用
`prefix`（父路径 *追加 canonical 之前* 的序列）建子路径，深度却 +1，等于把偏离
token 本身丢掉了（详见 §4.11）。所有 v2 之前的 beam / census 数字因此无效：
“median 1 条 terminal path”“beam 一个翻转都找不到”都是这个 bug 的直接后果。
修复为 `prefix + [t]` 后重跑（`*-v2`，同 seed、同数据、同预算），结论反转。

**DD short（n=150，max_new=128，stop_on_flip）**：

| 运行 | 排序 | B | eligible / flip 题 | ASR | stable ASR | 模型步数 median | terminal 轨迹 | 搜索路径总数 | eligible 轨迹翻转率 |
|---|---|---|---|---|---|---|---|---|---|
| `dd-short-beam8-cum-v2` | 累计 Δ | 8 | 139 / **44** | **31.6%** | 39/132 = 29.5% | 604 | 5,150 | 17,451 | 1.66% |
| `dd-short-beam16-cum-v2` | 累计 Δ | 16 | 140 / **47** | **33.6%** | 41/132 = 31.1% | 1,081 | 10,075 | 31,973 | 0.78% |
| `dd-short-beam4-abs-v2` | 绝对 margin | 4 | 142 / **47** | **33.1%** | 40/135 = 29.6% | 307 | 2,088 | 8,206 | 3.50% |
| `dd-short-beam8-delta-v2` | 单步 Δ | 8 | 139 / 35 | 25.2% | 29/132 = 22.0% | 633 | 6,096 | 19,551 | 0.81% |
| `dd-short-beam1-abs-v2` | 绝对 margin | 1 | 139 / 17 | 12.2% | 11/133 = 8.3% | **65** | 387 | 1,279 | 4.66% |
| rollout（u2） | 随机 | 8 条 | 139 / 19 | 13.7% | 15/131 = 11.5% | 224 | 1,200 | 1,200 | 2.37% |

**DD natural（n=150，max_new=384）**：

| 运行 | eligible / flip 题 | ASR | stable ASR | 模型步数 median | terminal 轨迹 | 搜索路径总数 | eligible 轨迹翻转率 |
|---|---|---|---|---|---|---|---|
| `dd-natural-beam8-cum-v2` | 132 / **107** | **81.1%** | 99/124 = 79.8% | 1,362 | 2,435 | 92,041 | 6.58% |
| rollout 7 条随机 | 139 / 32 | 23.0% | — | 779 | 1,200 | 1,200 | — |

**WS short（n=150，max_new=128）**：

| 运行 | eligible / flip 题 | ASR | stable ASR | 模型步数 median | terminal 轨迹 | 搜索路径总数 | eligible 轨迹翻转率 |
|---|---|---|---|---|---|---|---|
| `ws-short-beam8-cum-v2` | 140 / **49** | **35.0%** | 45/135 = 33.3% | 798 | 6,363 | 17,830 | 1.41% |
| rollout 7 条随机 | 139 / 18 | 12.9% | 10.5% | — | 1,086 | 1,086 | 1.83% |

**WS natural（n=150，max_new=1024，B=8 累计 Δ）**：

| 运行 | eligible / flip 题 | ASR | stable ASR | 模型步数 median | terminal 轨迹 | 搜索路径总数 | audit |
|---|---|---|---|---|---|---|---|
| `ws-natural-beam8-cum-v2` | 125 / **107** | **85.6%** | 97/112 = 86.6% | 5,304 | 610 | 305,703 | 317,455 检查 / 0 失败 |
| rollout 7 条随机（1024 uniform） | 120 / 33 | 27.5% | 25/111 = 22.5% | — | 839 | 839 | — |

WS natural 的 eligible 只有 125（canon_agree 83.3%），但翻转率最高：**85.6%**，
稳定子集 86.6%。代价也最高：median 5,304 步/题（rollout 的 6.8×）。

**修复前后同一配置的对比**（B=8 累计 Δ，DD short）：0.70% → **31.6%**；
natural：0.00% → **81.1%**；WS short（首跑）→ 35.0%。旧结论“forced-answer beam
不如随机 rollout”完全由 bug 造成。

**翻转集合**：beam 的翻转题几乎覆盖 rollout 的全部，并多出一大批——

| 对比 | beam 翻转题 | rollout 翻转题 | 交集 | beam 独有 |
|---|---|---|---|---|
| DD short B=8 cum vs rollout(u2) | 44 | 19 | 14 | **30** |
| DD short B=4 abs vs rollout(u2) | 47 | 19 | 15 | **32** |
| WS short B=8 cum vs rollout8 | 49 | 18 | 15 | **34** |

**代价**：B=8 累计 Δ 的 DD short 每题 median 604 步（rollout 224 步），换来 2.3×
的翻题数；natural 1,362 vs 779 步，换来 3.3× 的翻题数。B=1 绝对 margin 只要
65 步/题（rollout 的 0.29×）就已接近 rollout 的 ASR（12.2% vs 13.7%），B=4
绝对 margin 以 307 步/题达到 33.1%，是性价比最高的配置。**B 从 4 增到 8/16
只带来 0–2 个百分点的 ASR 增益**，说明宽度不是瓶颈，累计 Δ 的路径保留才是。

**与完全枚举对照（回答“搜满 vs 搜不满”）**：§3 的修复后 census（n=6，
node-limit 15k）里，dd-9459 用 16,886 nodes 仍未枚举完且没找到翻转，而
`dd-short-beam8-cum-v2` 在 **437 nodes** 就找到翻转；dd-41239 与 dd-30694 的
完整树（845 / 96 条轨迹）确实没有翻转，beam 也没有翻——**在能搜满的题上两者
一致；在搜不满的题上，Δ 引导的部分搜索用 1/40 的节点数先找到翻转**。因此
“搜满 vs 搜不满”的差距不是“知道有没有翻转”，而是枚举的预算全花在 EOS 早停的
浅层分支上；Δ 排序把预算集中在“forced answer 真的在动”的分支。

**机制**：修复后，偏离分支本身带着偏离 token 继续生成，Δ（或绝对 margin）在
后续步骤继续指示方向，累计 Δ 让一条需要 3–8 次连续偏离的轨迹（§4.7）不会因为
中间某一步 Δ≈0 就被淘汰。这正是旧版想实现但被 bug 破坏的行为。

（备注：v2 运行的 `best.ndev` 记的是 0，因为 spawn 分支当时没有维护该字段；
需要偏离次数时用 `len(best.deviations)`。之后的代码已在 spawn 时写入
`ndev = len(deviations)`，不影响本节的任何搜索行为。）

**关于候选级 AUC ≈ 0.51**：v2 beam 的 `forced_auc` 仍在 0.5 附近，但这不是打分
无效的证据——池化样本里绝大多数候选是 canonical token（按定义 Δ=0，标签由
所在轨迹决定），真正“分支级”的正例极少；v1 的 §4.6 已经显示 Δ 在深桶
AUC 0.89、绝对 margin 题级 AUC 0.985。判据因此改为端到端 ASR/节点数，而不是
这个池化 AUC。

### 4.10 审计

`choice_tree_audit` 在两次 rollout 主跑上复算所有 canonical fork 的 witness：
short 3,878 次检查，natural 34,807 次检查，**0 失败**；WS natural（1024, model）
95,904 次检查（209 条 trace，含 59 条 flip trace），**0 失败**。两次运行的所有
偏差 token 都带有效 witness。新增运行内自检 `witness_checks/witness_failures`：
每条 rollout 轨迹的每个 token 都用生成它的同一个 relation 重新 `check()`，计数写入
item JSON。
（说明：rollout 路径不保存每步 act/delta，因此模型无关的 audit 只能复核 spine
与 `best` 轨迹的 fork；轨迹本身的每步 witness 由运行内自检覆盖。）

**修复后 v2 beam 的 audit**（`choice_tree_audit`，每条 deviation 重放 witness）：

| 运行 | trace 数 | 检查数 | 失败 |
|---|---|---|---|
| `dd-short-beam8-cum-v2` | 205 | 8,121 | **0** |
| `dd-short-beam8-delta-v2` | 196 | 6,969 | **0** |
| `dd-short-beam16-cum-v2` | 207 | 7,025 | **0** |
| `dd-short-beam1-abs-v2` | 178 | 4,977 | **0** |
| `dd-short-beam4-abs-v2` | 205 | 6,653 | **0** |
| `dd-natural-beam8-cum-v2` | 274 | 89,855 | **0** |
| `ws-short-beam8-cum-v2` | 207 | 6,116 | **0** |
| `ws-natural-beam8-cum-v2` | 282 | 317,455 | **0** |
| `dd-short-census6-v2` | 7 | 182 | **0** |

修复前 `dd-short-beam8-fix` 有 188 条、`dd-short-beam4-abs` 有 45 条审计失败
（`deviation-token-match` / `deviation-witness`），这正是发现 bug 的入口（§4.11）。
attention-on 两臂的少量失败（dd 172 条、ws 130 条）是设计使然：扰动 relation 的
witness 不能由 honest act/delta 复核，不属于同一类错误。

### 4.11 bug 复盘：spawn 丢掉了偏离 token

- **现象**：`dd-short-beam8-fix` 的 `dd-36743` 审计失败：
  `{"step": 9, "check": "deviation-token-match"}`；`best` 轨迹出现 `depth=81`
  而 `path_len − base_len = 22`，59 条 deviation 的 step 最大到 69。
  加 `ZK_INV=1` 自检后打印
  `INV-spawn dd-25447 128 1 143 cap 126`：一条只有 1 个生成 token 的路径
  depth 却是 128，所有 deviation 越界（最深 drop 121 个 token）。
- **根因**：主循环里 `prefix = list(p['ids'])` 在 `p['ids'].append(canon)` **之前**
  捕获，`cand` 里每个候选（含 canonical）都用这份前缀；spawn 时写的是
  `new_path(p['item'], ids, base_len, depth=p['depth'])`，`ids` 就是 `prefix`，
  没有追加候选 token `t`，但 `p['depth']` 已经 +1。于是子路径的 `ids` 比
  `depth` 少一个 token，而且每次从这种路径再 spawn 就再少一个。
- **后果**：子路径下一步用同一 `prefix` 重新算 coin，会**重走同一个 fork**并再
  追加 canonical，等于偏离分支从未真正发生；同时 `max_paths` 很快被这些重复
  路径耗尽。旧版“beam median 1 条 terminal path”“spawned 885 条却全被剪掉”
  “natural 0 个翻转”都是这个原因。rollout 不 spawn，所以不受影响；HF attention
  runner 写的是 `prefix + [t]`，也不受影响。
- **修复**：新增 `spawn_path(item, prefix, token, base_len, depth)`，统一构造
  `prefix + [int(token)]`，两处（beam / exhaustive）都调用它；`new_path` 增加
  断言 `depth == len(ids) - base_len`；新增回归测试
  `test_spawn_carries_deviation_token`。修复后 n=10 冒烟运行
  （`dd-short-inv4`，`ZK_INV=1`）无任何 INV 输出、audit 310 检查 0 失败，
  且同一批题从 0 flip 变为 2 flip 题。
- **影响范围**：所有以 `choice_tree.py` 的 `--search beam|exhaustive` 跑出的
  数据（含 §3 的旧 census 数字、§4.1/§4.2/§4.3/§4.9 旧表）作废；rollout
  （§4.4–4.8）、attention（§5）、honest 自检不受影响。

## 5. Attention 轴

HF eager + AttnSlack（ρ=7088/2^20，3 个扰动配置的 union：coherent+ / coherent− /
adversarial sign），`--attn on/off` 同后端对照。探测只限 canonical spine
（`--attn-max-steps 32`，fork 步必探）；n=20，short，max_new=128，
node-limit=2000，max-paths=32，两臂预算完全相同。

| 数据 | 臂 | eligible | flip 题 | ASR | stable flip | probes | 额外 token | 额外 token/100 探针 | 额外 fork |
|---|---|---|---|---|---|---|---|---|---|
| DD | on | 20 | **7** | **35.0%** | 6/19 = 31.6% | 1,683 | 157 | 9.33 | 96（中位 5/题） |
| DD | off | 20 | **5** | 25.0% | 4/19 = 21.1% | — | — | — | — |
| WS | on | 17 | **3** | **17.6%** | 2/16 = 12.5% | 1,917 | 122 | 6.36 | 100（中位 5/题） |
| WS | off | 17 | **2** | 11.8% | 1/16 = 6.25% | — | — | — | — |

**翻转集合严格包含**：DD 的 on-flip 题 {dd-14224, dd-19519, dd-36743, dd-9459,
dd-9632} ∪ **{dd-25447, dd-42006}**，off 的 5 题全部在 on 里；WS 的 on-flip 集合
同样包含 off 的 5 题并多出 **ws-B09GK4XTJG**（eligible，h=2/c=2，best=1）。也就是说
attention 容忍度带来的额外可达 token **确实能多翻题，且没有让任何原本能翻的题
翻不了**（同样 seed、同样预算）。

幅度：每个被探针的步平均多出 6–9 个可接受 token、每题中位多 5 个新 fork；
在小样本（n=20）上换取 +10pp（DD 25→35%）/ +6pp（WS 11.8→17.6%）的 ASR。

注意：on 臂的额外 token 来自 eager 后端的三个扰动 logits，honest reach 仍来自
sdpa 后端，因此 on/off 差异里可能混有少量 kernel 数值差（未做 ρ=0 的 eager
对照）。但 (a) 所有 on-flip 都 ⊇ off-flip，(b) 额外 token 只在扰动下产生，
定性结论不受影响。

另一个旁证：这个 HF 穷举臂（无 forced-answer 排序，只按预算保留路径）在
DD short 上 n=20 就找到 5–7 个翻转题，ASR 25–35%，与修复后 vLLM beam 的
31.6%（§4.9）同量级，说明翻转不是某个后端的数值伪影。

**关于“剪枝”的旧结论已作废**：v2 之前认为“按 forced Δ 剪枝会导致找不到翻转”，
实际是 spawn bug（§4.11）。修复后，引导式 beam 在同一节点预算下反而**超过**
无引导枚举（dd-9459：437 vs >16,886 nodes，§3/§4.9）。现在能成立的说法是：
**路径记账必须正确；在此前提下，按 forced-answer 分数保留路径是有效的搜索策略。**

## 6. 审计汇总

`experiments/choice_tree_audit.py`（模型无关）逐条复核 canonical fork 与每个
deviation 的 witness。所有用于结论的运行：

| 类别 | 运行 | 检查数 | 失败 |
|---|---|---|---|
| rollout | dd-short / dd-natural | 3,878 / 34,807 | 0 |
| rollout | ws-natural 1024 model | 95,904 | 0 |
| beam v2 | 见 §4.10 表（9 个运行） | 8,121 + 6,969 + 7,025 + 4,977 + 6,653 + 89,855 + 6,116 + 317,455 + 182 | **0** |
| attention-on | dd-short / ws-short | 172 / 130 条设计性失败（跨 relation 复核，见 §5） | — |

结论：所有翻转轨迹的每个偏离 token 都有可复现的 coin witness；v2 beam 的
翻转不是记账伪影。审计也是发现 spawn bug 的入口（§4.11）。
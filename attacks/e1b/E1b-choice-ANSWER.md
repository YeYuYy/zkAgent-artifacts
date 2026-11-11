# 2026-11-11 方向执行总结（无 gold 决策 × paper × forced-answer beam）

本文件按你给的 5 个方向逐条回答“做了什么、结果是什么、在哪里”。所有数字都可从
`notes/RESULTS-e1b-choice.md` 与 run 目录复核；方法是实测驱动、过程中修正过两处
实现 bug（forced-answer 打分、spawn 丢 token），下文明确标注作废与修正。

## 0. 一句话结论

在 **DailyDilemma（二选一）与 WebShop-choice（静态四选一代理）** 上，只用
**paper 关系**（不跑 free-r）：

- **完全枚举不可行**：短输出下每题完整轨迹树从 96 条到 >10^5 条不等；6 题 census
  里只有 2 题能枚举完，其余撞 4,096 路径上限。
- **forced-answer beam 是最有效的搜索策略**（修复 spawn bug 后）：B=8 累计 Δ
  在 DD short / DD natural / WS short / WS natural 上把 eligible ASR 做到
  **31.6% / 81.1% / 35.0% / 85.6%**，同数据 7 条随机可容许轨迹只有
  **13.7% / 23.0% / 12.9% / 27.5%**。
- **attention 容差放大**：n=20 上 DD 25%→35%、WS 11.8%→17.6%，且 on 的翻转题
  严格包含 off（每 100 个探针步多 6–9 个可达 token）。
- 所有翻转轨迹的每个偏离 token 都通过 witness 审计（v2 beam 合计 44 万+ 次检查，
  0 失败）。

## 1. 数据集：不再使用有标准答案的数据集

- DailyDilemma：150 题 A/B 二选一，`data/dailydilemma-150.jsonl`；
  构建脚本 `experiments/build_choice_data.py`。
- WebShop-choice：150 题 1–4 四选一，`data/webshop-choice-150.jsonl`；**这是静态
  四选一代理**，不使用 WebShop 环境与 reward（文档中已标注）。
- gold 只记录，不参与任何筛选。参照改用模型自己的稳定决策：
  - `honest` = τ=1/top-k=40 采样；`control` = 换 seed 的第二条诚实轨迹
    → `null_flip_rate`；`canonical` = coin 决定的 paper 可容许主干。
  - **eligible** = honest 解析 ∧ canonical 解析 ∧ canonical == honest。
  - **flip** = 正常结束（`done='stop'`）且解析出的 choice ≠ honest。
- 关键对照：honest 采样本身有 **1.21%（short）/ 2.43%（natural）** 的 token
  落在 paper reach 之外（29% / 89% 的题至少一个），所以“独立重采样”不是合法
  攻击 baseline，ASR 必须与 null flip 并列报告。

## 2. 每题搜索 token 上限与“可行轨迹总数”

- 已取消旧的启发式上限（不再按数字/字母过滤、不分组），改为每步展开整个
  `witnesses(H,'paper')`；只保留 `--node-limit` / `--max-paths` 作安全阀。
- **完全枚举的可行性**（`dd-short-census6-v2`，n=6，node-limit=15,000）：
  - dd-30694：**完整枚举 96 条轨迹**，744 nodes，0 flip；
  - dd-41239：**完整枚举 845 条轨迹**，7,772 nodes，0 flip；
  - 其余 4 题在 4,096 条路径上限内枚举不完（其中 dd-9459 的 spine 分叉积
    10^5.4）。
- 因此“每题可行轨迹总数”只能给量级：short 的 spine 分叉积中位 10^2.58（384），
  natural 中位 10^19.5；**这些不是精确轨迹数，也不是上/下界**（完整树可能比它
  大或小）。精确值只能靠 census，且只能对小样本做。
- 结论：**n=150 上搜满不可行**；ASR 一律是搜索下界。

## 3. 只跑 paper × {attention 开, 关}

free-r 已按你的要求跳过。attention 轴（HF eager + AttnSlack，ρ=7088/2^20，
三种扰动的 union，只在 canonical spine 上探测）n=20、short、两臂同预算：

| 数据 | 臂 | eligible | flip 题 | ASR | stable flip | 额外 token/100 探针 | 额外 fork |
|---|---|---|---|---|---|---|---|
| DD | on | 20 | 7 | **35.0%** | 31.6% | 9.33 | 96（中位 5/题） |
| DD | off | 20 | 5 | 25.0% | 21.1% | — | — |
| WS | on | 17 | 3 | **17.6%** | 12.5% | 6.36 | 100（中位 5/题） |
| WS | off | 17 | 2 | 11.8% | 6.25% | — | — |

on 的翻转集合**严格包含** off（DD 多 dd-25447、dd-42006；WS 多 ws-B09GK4XTJG）。
注意 n=20 小样本，且 on 臂额外 logits 来自 eager 后端、honest reach 来自 sdpa，
有少量 kernel 数值混淆（未做 ρ=0 对照），定性结论成立、定量幅度需更大样本。

## 4. 去掉 flip-token 选择的启发式

### 4.1 先试“去掉所有限制、完全枚举”

不可行（§2）：路径数先爆炸（每步 reach 中位 2、最大 21），再撞节点上限。
15k nodes/题 的 census 也只走完 spine 附近的分支。

### 4.2 forced-answer 分数（按你的方案实现）

对每步 reach 里的每个 token `t`：构造 `prefix + [t] + "\n\nFinal answer: \boxed{"`，
用 `allowed_token_ids` 读标签 token 概率和，定义
`margin = log Σ_{l≠honest} p_l − log p_honest`，
`Δ(t) = margin(prefix+t) − margin(prefix+canonical)`；beam 按 Δ 保留路径
（`--beam-score cum` 沿路径累计，另有 `delta` 单步、`abs` 绝对 margin 两种）。

### 4.3 结果（v2，全部审计通过）

DD short（n=150，max_new=128）：

| 策略 | eligible / flip | ASR | stable ASR | 步数/题 |
|---|---|---|---|---|
| B=8 累计 Δ | 139 / 44 | **31.6%** | 29.5% | 604 |
| B=16 累计 Δ | 140 / 47 | **33.6%** | 31.1% | 1,081 |
| B=4 绝对 margin | 142 / 47 | **33.1%** | 29.6% | **307** |
| B=8 单步 Δ | 139 / 35 | 25.2% | 22.0% | 633 |
| B=1 绝对 margin | 139 / 17 | 12.2% | 8.3% | **65** |
| 7 条随机可容许轨迹 | 139 / 19 | 13.7% | 11.5% | 224 |

natural / WebShop：

| 数据 | B=8 累计 Δ ASR | rollout ASR | 步数/题（beam / rollout） |
|---|---|---|---|
| DD natural（384） | **81.1%**（132 eligible，107 flips） | 23.0% | 1,362 / 779 |
| WS short（128） | **35.0%**（140，49） | 12.9% | 798 / — |
| WS natural（1024） | **85.6%**（125，107） | 27.5% | 5,304 / — |

- beam 的翻转题覆盖 rollout 的绝大部分并多出一大批：DD short 44 vs 19，交集 14；
  B=4 abs 47 vs 19，交集 15；WS short 49 vs 18，交集 15。
- **B=1 绝对 margin 只要 65 步/题（rollout 的 0.29×）就接近 rollout 的 ASR**，
  B=4 绝对 margin 以 307 步/题达到 33.1%——性价比最高。加宽到 8/16 只多 0–2pp。

### 4.4 过程中发现并修复的致命 bug（必须记录）

旧版 vLLM runner 的 spawn 用父前缀 `prefix` 建子路径却把深度 +1，**偏离 token
从未进入子路径**，子路径下一步重算同一 fork，等价于复制父路径。于是“偏离分支”
全部无效，旧 beam 只找到 0–1 个翻转（0.70% / 0.00%），审计报
`deviation-token-match` 失败。修复为 `spawn_path(prefix + [t])`、在 `new_path`
加断言 `depth == len(ids)-base_len`、加回归测试；修复后同一配置 ASR 变为
31.6% / 81.1%，审计 0 失败。**§4.9 之前的 beam/census 数字全部作废**；
rollout 与 attention（HF runner 写法正确）不受影响。

### 4.5 heuristic 好不好用

- 绝对 forced margin 的**题级**预测很强（AUC 0.985，深度 0–5 也有 0.895）；
  分支级 Δ 早期弱（0–5 步 0.615）、后期强（20–40 步 0.892）。
- 候选级池化 AUC 约 0.5 是口径问题（绝大多数候选是 canonical、Δ=0），不能
  用它否定打分；端到端 ASR 已证明它有效。
- **结论：forced-answer 差值是好用的排序信号**；真正的失败原因是记账 bug，
  不是 heuristic。

## 5. 搜满 vs 搜不满 / 短输出 vs 自然输出

- 短输出下 beam 能在 **437 nodes** 翻 dd-9459，而 census 花 16,886 nodes 仍没
  枚举完也没翻；在能枚举完的两题（96 / 845 条轨迹）上两者一致（都不翻）。
  即：**在搜不满的题上，Δ 引导的部分搜索比无引导枚举更快找到翻转**。
- 输出长度：同一 DD 数据，short 的 B=8 cum 是 31.6%、natural 是 81.1%；
  rollout 对应 13.7% / 23.0%。自然输出给出更多多步分叉空间，翻转率显著更高，
  但成本也高（WS natural 5,304 步/题）。
- 7 条随机轨迹只覆盖可行空间的 1.8%（DD short）到 10^-55（WS natural），
  ASR 随预算 k 从 4.3% 单调升到 13.7% 仍未饱和——这是 rollout 的下界性质，
  也是 beam 能大幅超过它的原因。

## 6. 没有做 / 不建议现在做

- **RewardBench 未跑**：它带 gold 标签，与你“不要用有标准答案的数据集”的前提
  冲突；且本地无数据。若确实要跑，建议只用它做 forced-answer 打分的辅助校验，
  不作为 ASR 数据集。
- **R1 蒸馏模型未跑**：当前所有结果都是 Qwen3-8B；换模型前应先复现 v2 beam。
- **交互式 WebShop 未接**：现在是静态四选一代理，不能声称覆盖搜索/点击多步决策。
- attention 轴只有 n=20，且未做 ρ=0 的 eager 对照。

## 7. 文件与复现

| 内容 | 路径 |
|---|---|
| 结果与更正记录 | `notes/RESULTS-e1b-choice.md`（§4.9 主结果、§4.11 bug 复盘） |
| 实验计划与执行修正 | `notes/NEXT-STEPS-e1b-dilemma-webshop.md`（§8 第 9/10 条） |
| vLLM runner（已修复） | `experiments/choice_tree.py`（`spawn_path`、`new_path` 断言） |
| HF attention runner | `experiments/choice_tree_hf.py` |
| 审计 | `experiments/choice_tree_audit.py` |
| 汇总/统计 | `experiments/choice_report.py`、`choice_rollout_stats.py`、`choice_margin_auc.py`、`choice_attn_stats.py` |
| 单元测试 | `tests/test_choice_tree.py`（含 spawn 回归测试） |
| 每题搜索规模导出 | `notes/tables/feasible-v2.jsonl`（spine 分叉积、搜索/终止路径数、翻转） |
| v2 运行目录 | `$R/runs-choice/qwen3/{dd,ws}-{short,natural}-beam*-v2/` |

复现一条 beam（worker 上执行）：
```bash
R=/mnt/shared-storage-user/ai4good1-share/guodadi/zk_agent_gpu
$R/vllm-env/bin/python -m experiments.choice_tree \
  --model $R/models/Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218 \
  --data $R/data/dailydilemma-150.jsonl \
  --out $R/runs-choice/qwen3/dd-short-beam8-cum-v2 \
  --thinking off --short --search beam --beam 8 --beam-score cum \
  --max-new 128 --node-limit 40000 --batch 8 --record-forced --record-final
python3 -m experiments.choice_tree_audit $R/runs-choice/qwen3/dd-short-beam8-cum-v2
```
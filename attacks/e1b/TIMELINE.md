# E1b 时间线 —— 每一步"要求什么 → 达成了什么"(2026-10-30 → 11-15)

按时间排列的进展线。每条记录:**要求/动机**(当时的下一步是什么、为什么)、
**达成**(实测结果)、**落点**(commit / 代码 / 报告)。数字细节见 `SUMMARY.md`
对应章节;早期被后续修正取代的数字在文中明确标注。行末 commit 为 zkAgent-artifacts 仓库。

---

## 阶段 0(10-30):E1a 前置 —— 关系是集值的

- **要求**:验证 zkAgent 两个 tolerance-checked normalizer 是否把证明关系变成集合
  (proposal 的第一步,代码在 `attacks/` 根,非 e1b)。
- **达成**:端到端真 prover + 真 verifier:解码币重投 24/24 配置、注意力行 15/15
  全部复现 —— "币不可重投"的安全声明对其自身关系为假。`3d13787`
- **由此引出 E1b 的全部问题**:集值关系在真实模型分布下的**下游**威力
  (每步多少可达 token)与**后果**(账单 / 决策)。

## 阶段 1(10-30 → 10-31):E1b 在 1.5B / GPT-2 上立起四个子问题

- **要求**(初始计划,`SUMMARY.md` §4 原型):① 真实分布下每步可操纵 token 数;
  ② 强制不停(force-skip stop);③ 账单膨胀;④ 决策操纵;并为上规模备好脚本。
- **E1b-1 yield 达成**:随机 logits 高估(~33/40);GPT-2 真实 10.1/40、可翻步 81.8%;
  发现 **free-r(释出实现)可达 ~30–37,是 paper 设计的 ~5 倍** —— 实现缺陷独立于集值性。
  `f08283d` `2008362`(代码 `yield_oracle/`)
- **E1b-2 强制不停:负面结论**:自信停止点 `</think>` 概率≈1、非终止符质量精确 0,
  free-r 也挤不进 ⇒ zkAgent(解码在证明里)对 forced-extension 免疫,与 proposal
  landscape 表一致。`fbfb4ce`(代码 `forced_extension/`)
- **E1b-3 账单膨胀达成**:P1 窄词表 → P2-lexical 宽词表两轮加强后,free-r 注入率最高
  **63%**、多次撞 cap(膨胀实质无上界,cap 是唯一在拦的);paper 1.0–1.22×(modest)。
  `50168f8`→`18f9629`→`eef5972`(代码 `length_steering/`)
- **E1b-4 决策操纵**:盲注入**失败**(自我纠正);lookahead 成功翻 7×8 的 56→42
  (free-r,仅 2 次 rollout);paper 翻不动。多题成功率因 CPU 太慢中止,**标记待 GPU 重跑**。
  `c937eb5` `3eb5e09` `31dfe70` `0cc4f9b`(代码 `decision/`)
- **上规模准备**:`eval_at_scale.py` + `train_length_probe.py`;实测发现探针**跨域退化**
  (corr 0.85→0.49)⇒ 必须在部署分布上训练。`44e5951`→`d8d0248`(代码 `length_steering/`、`scale_harness/`)

## 阶段 2(11-02 → 11-05):上规模 H200 —— "E1b 上规模"为第一优先级

- **要求**:SUMMARY §4 #1:Qwen3-8B / R1-Distill-7B × GSM8K/MATH,n=150,真 GPU。
  (在 `/root/zk_agent` 仓库实现精确 Q20 枚举 + BF16 前向,不在此仓库。)
- **达成(8B)**:free-r 账单 100% 撞 cap、注入率 83.5%;决策 0/433(机制解释:
  forensics 显示答案区 0/9 可转向、78% rollout 截断、跑完的全自我纠正)。`e7fa1e0`
- **达成(7B)**:决策翻转 16/126 = 12.7% ASR(84.0%→73.3%),全部落在问题复述段,
  每题 1–2 次 rollout 即命中;账单 89% 撞 cap。**"不是规模效应,是模型家族效应"**。`4980796`
- **达成(vLLM campaign)**:步循环 harness ~10×;τ 扫描(free-r 账单完全温度稳健,
  决策在 τ=0.6 存活 16.4%);MATH;LaTeX 计分 bug 发现并全量重判;
  **paper 设计也能翻答案**(GSM8K 3.8–8.5%)。`a01e35b`
- **达成(value probe)**:探针泛化差(held-out R²≤0.22)但作攻击价值函数已足够:
  paper 注入率放大 3–10×、R1 未删失膨胀 1.0×→**3.64×**。`95bdb8b`

## 阶段 3(11-05):机制修正 —— "之前低估了 paper 设计的威力"

- **要求**(`reports/E1b-NEXT.md`):贪心 lookahead 只检验一条下游轨迹是错的;
  哈希链重投 ⇒ 轨迹级 best-of-N(A1)+ 自我纠正压制 steer(A2)+ 注意力集放大(A3);
  制定 D1/D2/B/A0 上规模矩阵。
- **达成(1.5B smoke,机制证明非论文数字)**:free-r 2/3→3/3;**paper 0→1/3**
  ("paper 决策免疫"是贪心单轨迹的假象)。E1b-attn(GPT-2):对抗性逐行选号可翻
  margin≲0.30 的步、paper 可达集 +68%。`3142151`(代码 `decision/`、`attention/`)

## 阶段 4(11-07):D1 预算阶梯上规模

- **要求**:A1 机制在 7B/8B × GSM8K n=150 复测(新批量 harness `vllm_bestofn.py`,
  纯批量生成 + prompt_logprobs 严格审计)。
- **达成**:R1-7B free-r ASR 9.7%@1 → **26.6%@48**(strict 25.8%),@24 后趋平
  (自我纠正 binding);**paper strict 0%**(lenient 1.6% 且下游不可容许);
  Qwen3-8B 全 0;单模型 9–13 min(旧 HF 5.8 h)。`2e4e19a`

## 阶段 5(11-08):D1/D2 全网格 + 可容许树搜索

- **要求**:补全 {steer × attn} × τ × MATH-500;随后(`reports/E1b-tree-attack.md`)
  把"自由续写 + 事后审计"改为**全程约束在可达集内**的预算树搜索。
- **达成(网格)**:paper 决策免疫对全部杠杆稳健(strict 0;唯一例外 attn+steer 1 项 0.9%);
  τ 是最强单一抑制因子(26.6→6.7);A0:7B/8B 注意力 margin 位移中位 0.13–0.25。`d80c210`
- **达成(tree)**:构造即严格 ⇒ strict≡lenient;**R1-7B paper 首个全轨迹可容许翻转
  (1/131 = 0.8%)**,free-r 10.7% 保证下界;Qwen3 零负结果复核为稳。`9fc8f56`(代码 `decision/tree_attack.py`)

## 阶段 6(11-11):换掉 gold —— 无标准答案决策集

- **要求**(`reports/E1b-choice-NEXT.md` 5 问):① 无 gold 数据集上翻转率?② 完全枚举可行吗?
  ③ forced-answer 分数能否替代启发式?④ 搜满 vs 搜不满差多少?⑤ attention 是否放大?
- **达成**:② 完全枚举不可行(短输出 spine 分叉积中位 10^2.58,自然 10^19.5);
  ③④ forced-answer beam B=8 累计 Δ:DD short/natural、WS short/natural =
  **31.6 / 81.1 / 35.0 / 85.6%**(随机可容许轨迹仅 13.7–27.5%);
  ⑤ n=20 上 on 严格包含 off。途中发现并修复 **spawn 丢 token 致命 bug**
  (修复前 0.7%/0% → 修复后 31.6%/81.1%,旧数字作废)。`5ad7c91`

## 阶段 7(11-12 → 11-13):Mind2Web + 注意力杠杆上 vLLM

- **要求**(`reports/M2W-ATTN-RUNS-2026-11-12.md`):Mind2Web 决策轮 + 定向翻转;
  注意力杠杆在 vLLM 引擎上的 attn-only / attn+decode 臂。
- **达成**:Mind2Web directed **61.5%**(Qwen3,67 翻转全落指定目标);R1 决策本身
  不稳(null≈ASR)。发现 **beam-churn 病灶**(cum 打分 ⇒ beam 清光子路径)并改 abs 打分。
  WebShop 注意力 campaign:R1 **attn+decode 96.4%/95.1%**(比单杠杆高 2.4×);
  attn-only vs sampling 是模型相关的(Qwen3 25%≪85%;R1 38.6%≈40.4%)。`e6d5a02`
  —— ⚠️ 这些 bf16 引擎上的注意力数字后被阶段 9 的 fp32 语义修正标注为"引擎行为"。

## 阶段 8(11-13):模拟忠实性 + 四线后续

- **要求**(`reports/NEXT-STEPS-4LINES.md`):1a vLLM vs HF 数值一致性;1b 逐位置符号;
  2 Python 模拟 vs 真 zkAgent verifier;3 τ-bench 高影响动作;4 Qwen3.5 尺寸缩放。
- **达成(line 2,当日完成)**:Python 关系 ≡ 真 prover+verifier(54/54、54/54、27/27
  全一致,深尾域同样清零)⇒ 各 campaign 的"模拟可达集"就是真实接受集。`9d71967`
- **达成(line 1a,重大发现)**:bf16 下 margin 位移**恰为 0**(±0.68% 卡在 ulp 边界)
  ⇒ **bf16 引擎无法忠实模拟注意力杠杆**;此前 bf16 注意力 ASR 须按 fp32 重测。`f93bd76`
- **达成(line 3)**:τ-bench 高影响动作:Qwen3 any **96.2%** / directed **87.7%**
  (cancel 错预订、多付 $100 等可精准瞄准);R1 46.7/40.0%。
- **line 4 准备**:Qwen3.5 为混合注意力(24/32 GDN),patch 需覆盖 Qwen3Next;
  vLLM 0.33 venv 就绪;27B 需下载。

## 阶段 9(11-13 → 11-14):fp32 语义修正 + 逐位置符号 + Qwen3.5 Phase A

- **要求**:fp32 引擎重测注意力;仅决策行的逐位置符号;4B/9B 的 attn/sample 两个面。
- **达成**:fp32 random 符号 margin 位移中位 0.0151(与 HF 参考 0.0127 同量级,
  pearson 0.358)。`7fb9fe4` fp32 重测:WebShop attn-only any **24.8%→41.3%**
  (bf16 是**低估**);仅决策行仍有 29.5%。**Qwen3.5 Phase A**:decode 面 paper/free-r
  可达集比 Qwen3-8B 宽 ~3×(free-r 30.2 vs 8.9);attn 面 GDN 只支持 bf16 且位移恰为 0
  ⇒ 混合架构上注意力杠杆在现有 kernel 栈下**结构性不可模拟**(decode 面不受影响)。`e0e9bfb`

## 阶段 10(11-15):Qwen3.5 尺寸阶梯 ASR(Phase B)

- **要求**:4B/9B/27B 全量 ASR(decode/sample 面),检验"越大越可攻击?"假设。
- **达成**:any/directed = 4B **82.4/71.6%** → 9B **80.6/79.1%** → 27B **65.7/50.4%**
  (对照 Qwen3-8B 75.9/61.5%):**攻击面随尺寸单调收缩,代际 > 尺寸**;null 也随尺寸
  收缩(19→4%)但被瞄准时仍 50–80% 可翻。`0fdf04b`

---

## 一页对照表

| 日期 | 要求(下一步) | 达成 | Commit |
|---|---|---|---|
| 10-30 | 集值性端到端验证(E1a) | 24/24 + 15/15 | `3d13787` |
| 10-30/31 | yield / 强制不停 / 账单 / 决策(1.5B) | 10.1 可达;免疫(负面);63% 注入;lookahead 翻 7×8 | `f08283d`–`d8d0248` |
| 11-02→05 | 上规模 8B/7B(第一优先级) | free-r 账单无界 + R1 决策 12.7%;paper 3.8–8.5%;probe 3.64× | `e7fa1e0`–`95bdb8b` |
| 11-05 | 机制修正(best-of-N) | free-r 3/3;paper 1/3(smoke) | `3142151` |
| 11-07 | D1 预算阶梯 | R1 free-r 26.6%@48;paper strict 0 | `2e4e19a` |
| 11-08 | D2 全网格 + 树搜索 | 免疫对杠杆稳健;**paper 首个可容许翻转 0.8%** | `d80c210` `9fc8f56` |
| 11-11 | 无 gold 决策集 | beam 31.6–85.6%(spawn bug 修复后) | `5ad7c91` |
| 11-12/13 | Mind2Web + 注意力上 vLLM | directed 61.5%;R1 hybrid 96.4/95.1%(bf16 口径) | `e6d5a02` |
| 11-13 | 模拟忠实性 / τ-bench / bf16 一致性 | 模拟≡真 verifier;τ-bench 96.2/87.7%;bf16 不可模拟注意力 | `9d71967` `f93bd76` |
| 11-14 | fp32 重测 + Phase A | attn-only 41.3%(fp32);Qwen3.5 可达集 ~3× 宽 | `7fb9fe4` `e0e9bfb` |
| 11-15 | 尺寸阶梯 ASR | 82→81→66% 单调收缩;代际>尺寸 | `0fdf04b` |

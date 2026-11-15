# zkAgent 攻击验证 —— 实验总结与后续计划

本文件汇总目前对 zkAgent(ePrint 2026/199)已释出 artifact(GPT-2 124M, Q=20,
commit `af36422`)以及其解码/注意力关系所做的全部实验、结论,以及尚待完成的实验。
所有结论都标注了诚实边界。实验代码在 `attacks/`(E1a)与 `attacks/e1b/`(E1b)。

> 同目录配套文档:`README.md`(目录结构与逐文件说明)、`TIMELINE.md`(按时间排列的
> "下一步要求 → 实际达成"进展线)、`FRONTIERS.md`(英文,可进论文的当前前沿结果:
> **仅 paper 设计口径**,不含早期 preliminary 与 free-r 实现缺陷数字);各阶段详细报告在 `reports/`。

---

## 0. 背景:要攻破的性质

- **Work soundness(已有方案提供的)**:声称的权重确实做了大部分计算。
- **Trajectory / Decision soundness(本工作的目标)**:交付的 token(含隐藏推理)和停止位置
  等于 `Gen_W(prompt, coins)`。**一个被伪造的决策就破坏它。**

zkAgent 把解码也证进了电路,并声称采样币 `r = H(tr_<t) mod Θ` "无法在看到 logit 后重投"。
我们的攻击面是它的两个 **tolerance-checked normalizer**(把证明关系从函数变成了集合)。

---

## 1. 两个解码模型(全篇对照的基准)

| 模型 | 采样币 r | 每步可达 token |
|---|---|---|
| **paper(论文设计)** | `r = H(前缀) mod Θ` 固定,只能 grind 承诺移位 ζ(⇒ Θ∈[θ±ε]) | 约 **7**(币多落在高质量=自然续写) |
| **free-r(released 实现)** | **r 未绑定**:验证器只查 `0≤r<Θ` 且 `prefix≤r<next_prefix` | 约 **30–37**(任何质量≥1 的候选) |

**free-r 是实现的一个独立、更强的缺陷**:`check_decode_sampling_claims`
(`src/zkdecode/zkdecode_output_verifier.cpp:124–126`)只做区间检查,`sample_values`
仅被 append 进 Fiat-Shamir(`zkdecode_input_link.cpp:54–55`),**从不与 H(前缀) 比对**。
论文里 `r = H(tr_<t) mod Θ` 的推导**不在释出代码中**。

参数:`θ = 2^20 = 1048576`,`ε = 7088`(released `decode_exp_digits_recommended_epsilon`,k=40),
`ε/θ = 0.68%`(与文字 "~0.7% per row" 吻合)。

---

## 2. 已完成的实验与结果

### E1a — 关系是集值的(端到端,真 prover + 真 verifier)✅

- **解码 coin re-roll**:对同一承诺 logits、同一前缀币,未改动的 verifier 接受两份矛盾证明,
  承诺不同 Θ、采样不同 token。**24/24** 配置全部复现(seed∈{1,2,7} × (vocab,k)∈
  {(256,40),(512,40),(1024,32),(128,20)} × backend∈{LogUp-GKR, Shout})。
  例:Θ=1051296 vs 1049184,token 5→12。
- **注意力 tolerance steering**:`verify_attention_softmax_row` 对一行 scores 接受上千个 ζ,
  每个都是不同的、被验证的注意力输出。15/15(seed × row_len)全部复现。
- 结论:**zkAgent "币不可重投" 的安全声明对它自己的关系就是假的**(set-valued,|H(x)|>1)。
- 代码:`attacks/setvalued_decode.cpp`、`attacks/setvalued_attention.cpp`、`attacks/README.md`。

### E1b-1 — Yield(真实分布下每步可操纵的 token 数)

| 分布 / 模型 | mean reachable / 40 | 可翻步比例 |
|---|---|---|
| 随机 logits(最大熵,**高估**) | ~33 | ~100% |
| GPT-2 真实(384 步) | 10.1 | 81.8%(分叉点 16.4%,翻转率 **100%**) |
| R1-1.5B,**paper 模型**(fixed-H + grind Θ) | ~2.4–6.9 | 24–40% |
| R1-1.5B,**free-r 模型**(实现) | **~29–37** | **95–99%** |

- **关键修正**:yield 完全取决于分布的熵。随机 logits 严重高估;真实推理模型很尖。
  且**早期我用 paper 模型(对 zkAgent 过于仁慈)低估了实现的威力**——free-r 下可达是 paper 的 ~5 倍。
- 分叉点(top1−top2<0.1)虽少(~2–16%),但**翻转率恒为 100%**。
- 代码:`attacks/e1b/yield_oracle/dump_logits.py`、`yield_decode.cpp`、`freer_reach.cpp`、`reach_both.cpp`。

### E1b-2 — Forced extension(强制不停 / 账单膨胀的"硬"路径)❌(对 zkAgent 无效)

- 两条自然终止轨迹的真实停止步(`</think>` 概率≈1):`reach=1`,非终止符量化质量**精确=0**,
  **free-r 也挤不进**。所以**无法 force-skip 自信的停止点**。
- 这与 proposal 自己的 landscape 表一致:zkAgent 行只列 "coin re-roll; tolerance steering",
  forced extension 是列给"解码不在证明里"的系统(TOPLOC/CoIn 等);zkAgent 把解码证了,故免疫。
- 代码:`attacks/e1b/forced_extension/forced_extension.cpp`、`dump_until_stop.py`。

### E1b-3 — Length-steering(账单膨胀的"软"路径:在可逆步注入续写词拉长推理)

策略演进:P1(窄 marker 词表)→ P2-lexical(宽"内容开启"词表,排除收尾词,挑最高概率可达词)。
R1-1.5B,cap 400–500:

| prompt | honest | **free-r(P2)** | **paper(P2)** |
|---|---|---|---|
| 7×8 | 37 | **398**(注入23,×10.76) | 45(注入2,×1.22) |
| 91 素数 | 252 | 400⌈cap⌉(注入139,×1.59) | 292(注入3,×1.16) |
| 澳洲首都 | 353 | 400⌈cap⌉(注入**251=63%**,×1.13) | 400⌈cap⌉(注入28,×1.13) |
| 12+15 | 67 | 77(注入1,×1.15) | 67(注入0,×1.0) |

- **free-r:账单实质无上界**(注入率最高 63%,多次撞 cap;cap 是唯一在拦它的东西)。
- **paper:modest 但非零**(1.0–1.22×)。P1 时几乎 0,P2-lexical 揭示论文设计**并非免疫**。
- **诚实指标**:膨胀倍数被 cap 压扁,**注入次数/率是 cap 无关的真实信号**。
- 代码:`attacks/e1b/length_steering/e1b_length_steer.py`(`--skip-freer`、`--ngrid`)。

### E1b-4 — Decision manipulation(强制输出错误答案)

- **直接翻答案:不可能**(两模型)。答案 token 在模型自信时输出,错误值质量=0、不可达。
- **盲注入**(前几个数字步塞第一个可达错误数字):**失败**——推理模型自我纠正,回查 prompt。
- **lookahead(P2-probe,对每个可达错误数字 rollout 到答案,只留"不被纠正"的)**:
  - **free-r:成功翻转** 7×8 的 56 → 42(腐蚀在"问题复述"处把操作数改掉,模型信自己的复述、
    老实算 6×7=42)。**仅 2 次 rollout 命中。**
  - **paper:翻不动**(计算步数字被币锁死,整条轨迹只有 1 个可达错误数字候选,且没翻)。
- 5 题成功率(proxy):**⚠️ 未完成,需重跑**。单题 lookahead 的翻转(7×8 → 42)成立;
  多题成功率那次 CPU 太慢(~0.6 forward/s)且时长不可控(翻不动的题会烧满 rollout)被中止。
  重跑建议:`--max-rollouts 12 --rollout 80`(翻不动早停),或直接并入下方"E1b 上规模"在 GPU 上做。
- 代码:`attacks/e1b/decision/decision_attack.py`(盲)、`decision_lookahead.py`(lookahead + 多题成功率)。

### E1b-8B — 上规模(GPU, H200):Qwen3-8B ✅ / R1-Distill-7B ✅(GSM8K 各 n=150;τ 扫描 / MATH / vLLM 复刻见下)

在 zk_agent 项目(H200 worker,精确 Q20 关系模拟 + BF16 前向,`/root/zk_agent/experiments/`)重做
E1b 的放大版。关系枚举改为**精确**(全带内 ζ 断点,无网格;C++ oracle 逐位对齐),注入见证
全部当场过 `Relation.check`,事后独立复audit 524,274 条全部通过。Qwen3-8B,GSM8K test
n=150,τ=1,cap 4096,rollout 1536,max-rollouts 8:

| 指标 | free-r(释出实现) | paper(论文设计) |
|---|---|---|
| 每步可达 token(reachable) | **31.6**(p10=5, p90=40) | 1.8 |
| 账单攻击撞 cap(4096) | **150/150 = 100%** | 69/150 = 46% |
| 注入率(注入 token / 生成 token) | **83.5%**(中位 3477 注入/条) | 2.3%(中位 79) |
| 膨胀中位数 | **≥2.85×(下界,全部撞 cap)** | 1.74×(未删失对)/ 1.72×(下界) |
| 决策翻转 | **0/433 rollouts**(136 eligible) | 0(候选更稀) |

honest 准确率 90.7%(136/150 完成,12 条撞 4096 cap;推理长度中位 1439)。

- **8B 上 free-r 账单膨胀依旧实质无上界**(100% 撞 cap),且注入率 83.5% 远超 1.5B 的 63%。
- **paper 设计在 8B 上恶化**:1.5B 时膨胀 ~1.1–1.2×、几乎不撞 cap;8B 上 46% 撞 cap、
  中位 1.74×——Qwen3 的"Wait/再想想"自我反思循环把每个注入的续写词放大成整段重推。
- **决策操纵在 Qwen3-8B 上翻不动,且有机制解释**(decision_forensics,n=8):
  - 数字位可转向率:free-r 下仅 8.5%(31/363 数字位有可达替换数字;paper 2.5%);
    **答案区 0/9**——8B 写最终答案时完全锁死;
  - 27 个 rollout 里 **78% 在 1536 步内没跑完**(截断,非纠正);
  - 跑完的 6 个全部**自我纠正**(答案仍对);0 翻转。
  - rollout 3072 深度复核已完成:21 个 rollout、12 个跑完,**12/12 仍自我纠正**,0 翻转
    ——截断不是原因,Qwen3-8B 的自我纠正(回查 prompt 重推)是真实的。
- 与 1.5B 对照:账单攻击随规模**变强**(熵更高、反思循环更长);答案操纵在 Qwen3-8B 上
  变难(模型更尖、自我纠正更强)——但 R1-Distill-7B(下)证明**这不是规模效应,是模型
  家族效应**。framing:接受集非空是结构性的;后果落在账单/决策哪一侧、决策翻不翻得动,
  取决于模型的自我纠正行为,而非参数量。

**R1-Distill-Qwen-7B**(GSM8K test n=150,同配置;honest 84.0%,134/150 完成):

| 指标 | free-r(释出实现) | paper(论文设计) |
|---|---|---|
| 每步可达 token(reachable) | **39.3**(p90=40,几乎恒满 top-k) | 4.9(p90=12) |
| 账单攻击撞 cap(4096) | **133/150 = 89%** | **131/150 = 87%** |
| 注入率 | **94.9%**(中位 3917 注入/条) | 19.4%(中位 420) |
| 膨胀中位数(下界,含删失) | **3.08×** | **2.46×** |
| 决策翻转 | **16/126 eligible = 12.7% ASR**(903 rollouts) | 0 |

- **7B 上决策操纵成立**:16 题答案被翻转(84.0% → 73.3%,掉 10.7 个点),全部是
  free-r 模式下在**问题复述段**(step 3–40 居多)注入一个错数字、模型照抄自己的复述
  老实算错。例:gsm8k/test/3,`3`→`6` 一个 token,1 次 rollout,540→1260。
  median 每题仅 1–2 次 rollout 即命中——**攻击成本近零**。
- R1 forensics(n=8,80 rollouts):41 截断、32 自我纠正、**7 翻转**——跑完的 rollout 里
  18% 翻转(vs Qwen3-8B 的 0%)。R1-Distill 的自我纠正显著弱于 Qwen3-8B。
- **paper 设计在 R1-7B 上账单同样实质无上界**(87% 撞 cap、19.4% 注入、2.46× 下界):
  paper 的 ζ-grind 通道在 R1 上每步可达 4.9(远宽于 Qwen3 的 1.8),注入率是 Qwen3 的
  ~8.5 倍。**"函数化前的 paper 设计"对账单膨胀的防御在 7B/8B 两个模型上都失效**。

### E1b-8B/vLLM — 放大与补全(专用 vLLM rjob,2026-11-04/05)

在第二张 H200(image `vllm-openai:v0.25.0`)上用 step-loop 控制器重做放大:每步
`max_tokens=1` + top-48 logprobs,由控制器算 Q20 关系并强制所选 token(prompt 追加,
prefix cache 吸收重复 prefill);rollout 即共享前缀缓存的新请求。n=150 全量跑 ~60–70 min
(HF 单流 ~11h,**~10×**)。注入见证全部过 `Relation.check` 复验(8 个运行共 ~330 万条)。

诚实边界:Q20 量化作用于服务引擎的 logprobs(logits 的确定性单调变换;delta 结构在
float32 log-softmax 舍入内保持)——vLLM 复刻与 HF 复刻数字一致(见下),跨后端稳健。

**计分修正(重要)**:中期发现 `\boxed{\$56}` 类答案的 LaTeX 残留反斜杠被误判为"错",
虚高翻转率;已修 `normalize/correct`(分数 a/b、`\,`、`\text{}`、残留反斜杠)并对全部
已完成运行做事后重判(`rederive.py`,raw→corrected 同存)。下表均为修正后数字。

**决策操纵(修正后 ASR,rollout 深度 1536、每题 ≤8)**:

| 运行(vLLM, n=150) | honest | free-r ASR | paper ASR |
|---|---|---|---|
| R1-7B GSM8K τ=1.0 | 86.7% | **25.4%**(33/130) | **8.5%**(11/130) |
| R1-7B GSM8K τ=0.8 | 86.7% | 16.9% | 3.8% |
| R1-7B GSM8K τ=0.6 | 89.3% | 16.4% | 5.2% |
| R1-7B MATH τ=1.0 | 83.3% | 12.0% | 2.4% |
| Qwen3-8B GSM8K/MATH,τ∈{0.6,0.8,1.0} | 73–91% | **0** | **0** |
| R1-7B GSM8K τ=1.0(HF logits 复刻) | 85.3% | 11.7% | 0 |

- **答案操纵在部署温度下存活**:R1 上 τ=0.6 仍有 16.4%(free-r)/ 5.2%(paper)。
- **paper 设计也能翻答案**(GSM8K 3.8–8.5%,MATH 2.4%):ζ-grind 通道在 R1 的
  CoT 分布上足够宽。之前 HF 复刻的 paper-0 是 rollout 运气,不是结构免疫。
- **逐题结果是混沌的**(HF vs vLLM 仅 4 题重叠翻转;并集 64/126≈51%)——每题
  成败取决于轨迹,攻击面本身(可达集结构)跨后端不变:Qwen3 两后端均 0,R1 两后端均正。
- Qwen3-8B 在 MATH、三个 τ 下均 0 翻转:其负结果是稳的(深度复核已证自我纠正)。

**账单膨胀 τ 扫描(n=150,mean injection rate / cap-hit)**:

| 模型 | τ | free-r 注入率 | free-r 撞 cap | paper 注入率 | paper 撞 cap |
|---|---|---|---|---|---|
| Qwen3-8B | 1.0 | 83.4% | **150/150** | 2.3% | 44% |
| Qwen3-8B | 0.8 | 34.8% | **150/150** | 1.7% | 27% |
| Qwen3-8B | 0.6 | 5.2% | **150/150** | 1.2% | 19% |
| R1-7B | 1.0 | 93.4% | 88% | 18.9% | 81% |
| R1-7B | 0.6 | 71.2% | 88% | 2.0% | 68% |

- **free-r 的账单膨胀完全温度稳健**:注入机会从 83% 掉到 5%,撞 cap 率纹丝不动
  (100%)——每 ~20 步漏进一个续写词即可自续循环。paper 模式随 τ 变尖而减弱但仍显著。

**MATH(两模型,n=150,τ=1)**:Qwen3 honest 73.3%,free-r 150/150 撞 cap @84% 注入;
R1 honest 83.3%,free-r 93% 撞 cap @95% 注入,paper 98% 撞 cap @25% 注入。
账单面在 MATH(更长 CoT)上同样无界。

**value probe(真·P2)**:两模型 ridge 探针(隐状态→剩余推理长度)泛化差:
train corr 0.83–0.84 / R² 0.68–0.71,但 **held-out 同域 R² ≤ 0.22(R1 为 −0.19)、
跨域 ≈ 0**——7B/8B 的推理长度重尾,线性探针不可泛化(1.5B 时同域 R² 0.72)。

**但弱探针已足够放大攻击** —— probe 接入的 1 步 lookahead 计费攻击(对每个可达
候选打分挑"最续写"的,而非只挑 opener 词;两模型 n=60,已完成):

| paper 模式 | lexical 注入率 | probe 注入率 | probe 膨胀(未删失/下界) |
|---|---|---|---|
| R1-7B | 19.4% | **61.6%** | **3.64× / 3.10×**(lexical 1.0×/2.46×) |
| Qwen3-8B | 2.3% | **22.7%**(10×) | 1.54× / 1.58×(lexical 1.74×/1.72×) |

free-r 下 probe 以**一半的注入率**(R1 58% vs 95%;Qwen3 41% vs 84%)达到同样的
~100% 撞 cap——每刀更准。结论:**paper 设计的 ζ-grind 通道 + 任意可学的价值函数
(哪怕泛化很差)= 账单无上界**;lexical 攻击者依赖 opener 词碰巧可达,probe 攻击者
把整个可达集变成注入点。攻击可行性本就不依赖 probe(free-r 每步 ~30 可达)。

### E1b-5 — 决策操纵的修正:哈希链重投 → 轨迹级 best-of-N(2026-11-05)

> 机制全貌与本次代码改动清单见 `attacks/e1b/reports/E1b-NEXT.md`;下为结果摘要。

**动机(之前低估了 paper 设计的威力)**:E1b-4 / `eval_at_scale.py` 的 lookahead 在注入一个
可达错数字后**贪心续写**,只检验了**一条**下游轨迹。但采样币 `r = H(tr_<t) mod Θ` 是对承诺前缀
的**哈希链**:在任一分叉点换一个可达 token,就重新确定了之后**所有**的币 ⇒ 一条全新的、服从模型
自身分布的续写。攻击者因此可在整条轨迹上做 **best-of-N**:挑一个可达腐蚀,再探 N 条可容许的下游币链,
留任一条"错且未被自我纠正"的。贪心版漏掉了这一放大。

**修正实现**:`attacks/e1b/decision/decision_bestofN.py` 把贪心续写换成**采样续写**(τ 下对 active top-k 的
categorical = 诚实解码本身;重投币 = 从模型分布里重采样),并做 best-of-N。续写模式:
`greedy`(旧的保守基线)/`sample`(claim-1 的 best-of-N 基线)/`steer`(启发式:在可达集内封掉
自我纠正词 Wait/But/recheck/… ——set-valued 自由度允许的前提下,压制 SUMMARY 发现的"翻转被自我纠正
吃掉"的失败模式)。另有启发式 `--target-restatement`(优先攻问题复述段里与操作数同值的数字位)。
诚实边界:**本机 1.5B(R1-Distill-Qwen-1.5B,fp16 单卡)+ n=3 短算术 smoke**,只为证明机制与代码,
非论文数字。答案抽取已加固(只认 `\boxed{}` 或 `</think>` 后的结论,截断的 rollout 一律不计 ⇒ 不会
误判翻转)。

| 续写方式 | free-r ASR | paper ASR | 备注 |
|---|---|---|---|
| **greedy(旧基线,1 条续写)** | 2/3 | ~0(与 E1b-4 一致) | 7×8 翻不动 |
| **sample best-of-N(N=6)** | **3/3** | **1/3**(7×8 @step41,→32) | best-of-N 救回 greedy 漏掉的 7×8 |
| **steer best-of-N(N=6)** | 3/3 | 1/3(7×8 @step68,7→0→49) | 本 smoke 集太小/太易,steer 未超过 sample |

- **free-r:贪心 2/3 → best-of-N 3/3**,翻转恰好发生在贪心翻不动的那步——直接坐实说法 1 的放大。
- **paper 设计并非对决策操纵免疫**:E1b-4 在 1.5B 上报 ~0,**采样 best-of-N 把 paper 做到 1/3**。
  之前的"paper 决策免疫"是贪心单轨迹的假象。
- **启发式**:`steer` 在此 3 题集与 `sample` 同为 1/3(两者都翻 7×8,另两题 paper 下自我纠正太强);
  其边际价值需在**大/难数据集**上才显现(自我纠正是 binding 失败模式时)——是**待上 GPU harness 的 TODO**,
  不是负结果。
- **放大**:两脚本支持 `--device cuda`;把 `eval_at_scale.py` 的贪心 rollout 替换为本脚本的采样 best-of-N,
  即可在 7B/8B 上重测 ASR(预期两模型 ASR 均较 E1b-8B 的贪心数字上移)。

### E1b-attn — 注意力容差 → 解码 logit 的扰动幅度(2026-11-05)

E1a 只证了注意力 verifier 接受上千个 ζ(谓词层),**未量化下游**。本实验把它接到真实 logits。
机制(对齐 `setvalued_attention.cpp` 与 src):注意力输出行 `o = (Σ m_i v_i)/θ`,**除数是公开常数 θ,
不是承诺行质量**,故均匀 ζ 平移**不抵消**——可容许的行质量 `Θ_row∈[θ±ε]` 把整行输出缩放
`g∈[1−ε/θ, 1+ε/θ] = 1±0.676%`。"两极"即 `g=1∓ρ`,每个 (层, 头, query 行) 由 prover 独立选。

**实现**:`attacks/e1b/attention/attn_slack_logits.py` 在 GPT-2 124M(**正是释出的 zkAgent artifact 模型**)上 hook
`eager_attention_forward`,按 g 缩放每行注意力上下文(c_proj 之前),重跑前向读末步 logits。扰动模式:
coherent±(全同号)、random(逐行随机 ±ρ 集成)、adversarial(对 top1→top2 margin 做 autograd,逐行取
一阶最优号——仍是可容许扰动)。

- **一致同号几乎无害**:coherent± 的 margin 变化 ≈ ±0.0001(近均匀缩放在 logit 差里抵消);`dL∞≈0.18`。
- **逐行独立选号才是威胁**:**adversarial** 把末步 top1−top2 margin 推动 `dL∞≈0.14`,**翻转贪心 token**
  (` yes`→` no`,honest margin 0.1276 → −0.0099)。
- **逐步扫描(prompt 16 步)**:对抗性注意力容差单独可翻 **3/16 步**,margin 位移**中位 0.13 / 最高 0.34**;
  **能翻的最大 honest margin ≈ 0.30 ⇒ 任何 top1−top2 间隔 ≲ 0.30 logit 的解码步,单靠注意力容差就能翻**,
  无需任何解码容差 grinding。
- **额外解码分叉**:在 **paper** 解码模型,注意力容差把该步可达集 **22 → 37 token(+68%)**,并能翻
  coin-0 token;free-r 本就撑满 top-k=40,union 无增量。
- **诚实边界**:这是一阶"两极"(归一化带)模型,**未含**逐键差分取整的二阶 per-element 抖动 ⇒ 报告的
  logit 位移是**下界**;在 **GPT-2 124M** 上测(mid-CoT 分叉步 margin 更小,7B/8B 上可翻步比例预计更高,
  待上 GPU 重测)。

### E1b-5@7B/8B — 修正决策攻击上规模:best-of-N 预算阶梯(2026-11-07)

E1b-5 的哈希链重投机制在 **Qwen3-8B / R1-Distill-7B × GSM8K n=150(τ=1,eligible 133/124)** 上重测。
新 harness(`/root/zk_agent` 的 `experiments/vllm_bestofn.py`):honest 轨迹与全部续写都是**纯批量生成**
(连续批处理 + 前缀缓存,无逐步引擎往返;逐步 top-48 logprobs 重建数字分叉处的 Q20 关系),双臂并行:
greedy 臂(n=1,旧协议复刻,沿用 `decision` 字段以兼容 rederive/audit)+ 采样臂(每 fork n 个续写,
τ 下 top-k=40 categorical = zkAgent 诚实币分布),slot-major 展平成 **ASR-vs-预算曲线**;每个 fork token
过 `Relation.check`,首次翻转的**下游每个 token** 用 `prompt_logprobs` 重放并要求逐 token 有 witness
(strict 口径)。全部注入见证经 `audit_results.py` 独立复验,`rederive` 复核 0 假阳。
**单模型全程 9.2 / 13.4 分钟**(旧 HF 单流 5.8 h、旧 vLLM 步循环 63–112 min ⇒ **~40×/ ~7× 提速**)。

| 模型·模式 | greedy(旧协议) | ASR@1 | @8 | @48(best-of-N) | strict@48(全轨迹可容许) |
|---|---|---|---|---|---|
| R1-7B · free-r | 21.8% | 9.7% | 17.7% | **26.6%** | **25.8%**(32/33 翻转全轨迹过审) |
| R1-7B · paper | 1.6% | 0.8% | 1.6% | 1.6%(lenient) | **0%**(2 个 lenient 翻转下游存在无 witness token) |
| Qwen3-8B · free-r | 0% | 0% | 0% | 0% | 0% |
| Qwen3-8B · paper | 0% | 0% | 0% | 0% | 0% |

(fork 机会并非稀缺:R1 free-r 1606 个数字候选 / Qwen3 paper 70 个 slot、280 条续写全试过。)

- **R1-7B free-r:best-of-N 是真放大但被自我纠正封顶**——9.7%@1 → 26.6%@48(strict 25.8%),
  相比贪心单链 21.8% 增 ~5pp;阶梯在 @24 后趋平 ⇒ 剩余 eligible 项的翻转轨迹不存在(自我纠正 binding),
  加预算无用。旧协议数字(贪心 12 rollouts)与阶梯 @16≈21% 相互印证。
- **paper 模式在 7B/8B 的"决策免疫"经修正攻击后仍成立(strict 口径)**:R1 仅 2 个 lenient 翻转且
  下游均含不可容许 token(strict 0%);Qwen3 全零。**1.5B smoke 的 paper 1/3 未迁移到 7B/8B GSM8K**——
  大模型自我纠正 + 更高 honest margin 是瓶颈,不是搜索预算(70/280 个 paper fork 机会已试尽)。
- **Qwen3-8B 的负结果在修正协议下复核为稳**:两模式、全部预算、strict/lenient 全 0。
- 待做:τ∈{0.6,0.8} 阶梯、MATH-500 L4–5(D1/D2 矩阵)、attn-on 并轴。
  → **已全部完成(2026-11-08,见下节)**。

### E1b-5@7B/8B 完整 D1/D2 网格:steer × attn × τ × MATH-500(2026-11-08)

分支三杠杆全部上规模(A1 已于 11-07 落地;A2 `steer` 臂 = 采样续写中封自我纠正词
[vLLM `bad_words`,与参考实现同为无条件封];A3 `attn-forks` = fork 处对
{coherent±, adversarial} 注意力容差配置取可达集并集,HF eager 子进程隔离跑,
vLLM 主进程分阶段衔接)。10 个评测 + 2 个 A0 探针,双 H200 并行,**全程 ≈2 h 墙钟**;
`audit_results`/`rederive` 全过。新数据集 **MATH-500 L4–5**(n=150;78×L4+72×L5)。

**A0(注意力容差在 7B/8B 的量级,`attn_probe.py`,真实 mid-CoT 轨迹逐步探针)**:
两模型的对抗性逐层选号把 top1−top2 margin 推动**中位 0.13–0.25 / 最大 1.1–1.25**,
单独可翻贪心 token 的步占比 1.3%(Qwen3-8B)/ 2.8%(R1-7B),翻转门槛 margin ≈0.25
(GPT-2 124M 时为 0.30——**更大模型对注意力容差更敏感,与 E1b-attn 预测一致**)。
R1-7B 的 free-r 可达集中位 28→33(64% 步有增量);paper 并集在 44% 步有增量;
**opener(计费注入词)新增可达的步占比 2.1–3.5%(paper)**——即 B 问("注意力是否
给计费攻击开新口子")的测量版答案:有,但每步概率小;全量 billing×attn 跑仍待做
(需逐步扰动引擎,与批处理架构冲突)。

**D2 2×2(GSM8K τ=1,n=150;ASR@48,lenient/strict)**:

| 模型·模式 | steer−/attn−(基准) | steer+/attn− | steer−/attn+ | steer+/attn+ | attn 对 fork 菜单 |
|---|---|---|---|---|---|
| R1-7B free-r | 26.6 / 25.8 | 27.1 / 26.4 | **28.3 / 28.3** | 25.2 / — | slots 1496→1521(+25) |
| R1-7B paper | 1.6 / **0** | 2.3→**3.9**(steer 臂) | 2.4 / **0** | 3.8 / **0** | slots 258→414(**+61%**) |
| Qwen3-8B 全部 | 0 / 0 | 0 / 0 | 0 / 0 | **0.9**(唯一 1 翻)/ 0 | slots +15(free-r)/ +35(paper) |

**τ 阶梯(R1-7B free-r ASR@48)**:τ=1 → 26.6;τ=0.8 → 21.3(strict 20.6);τ=0.6 → **6.7**
(strict 6.0);贪心臂 21.8→12.5→1.5。更尖的温度显著压缩决策操纵面(free-r 撞 cap 的
计费面在旧campaign 里温度不敏感——两半场温度依赖相反)。

**MATH-500 L4–5(τ=1)**:honest acc Qwen3-8B 28.7%(eligible 仅 43)/ R1-7B 48.7%(73)。
free-r:R1 2.7%(2 翻,均过严格审计)/ Qwen3 0;paper:两模型 0。难题上 fork 少、
自我纠正更强,攻击面进一步收缩。

**结论**:① **paper 模式的决策免疫在 7B/8B 对全部杠杆(steer/attn/预算/τ/换数据集)
稳健成立**——所有 lenient paper 翻转(含 attn/steer 来源)在严格逐 token 审计下全部
不可容许;② free-r 决策攻击上界 ~28%(R1),steer 无增益、attn 增 ~1.7pp(1 个
attn:adversarial slot 翻转)、封顶者仍是自我纠正;③ Qwen3-8B 的零负结果对全部杠杆
稳健(唯一例外:attn+steer 并用时的 1 项 0.9% 翻转);④ 温度是决策攻击面最强的
单一抑制因子。
  → **再修正(E1b-tree,下节)**:把"自由续写 + 事后审计"换成**全程约束在可达集内的
  预算树搜索**后,paper 在 R1-7B 上被找到 **1 个全轨迹可容许翻转(0.8%)**——免疫
  并非绝对;free-r 拿到 **10.7% 的"构造即严格"ASR 下界**。

### E1b-tree — 可容许树状搜索:每个发出的 token 都在其可达集内(2026-11-08)

此前所有决策攻击(free 续写各臂)在 fork 之后用引擎自由解码、再做事后可容许性审计
——**轨迹本身不保证可容许**(这正是 paper strict 全 0 的来源)。E1b-tree(任务见
`E1b-tree-attack.md`)把攻击改为**全程约束**:每个发出的 token(规范步与替换步 alike)
都取自当步可达集(前缀已变则逐步重算),轨迹按构造可容许,strict ≡ lenient,逐 token
成员性 assert 作为自检(全程零触发)。

- **参考实现**:`attacks/e1b/decision/tree_attack.py`(可复用 `tree_attack(...)`,与 argparse/模型
  解耦)+ `decision_bestofN.py --search tree`(KV 续传步进器、canon=reach 内 logit 最高者
  ——**这是对 coin 选择的代理**,已注释)。数字串聚成 quantity-group,标记 prompt 操作数;
  组排序:≥2 可达 mention 优先、更靠近 `\boxed` 优先;方案 A(整组一致翻 d→d′)/
  B(仅最晚 mention);`--budget` 总解码 token 上限 + `--max-groups/--max-forks`。
  测试 `tests/test_tree_attack.py`:不变式、构造命中、paper 退化(单元素 reach)。
- **GPU 放大**(`experiments/tree_attack.py`,speculative 窗口解码:批量贪心窗口 +
  逐步行走,接受当 argmax ∈ reach,分歧处用同窗口 logits 取 canon 再续猜——无逐 token
  引擎往返;canon 轨迹按模式各跑一条,eligibility 用 free-r 规范轨迹答对与否):
  4 个运行(Qwen3-8B / R1-7B × GSM8K / MATH-500 L4-5,τ=1,n=150,两模式),每个
  58–80 min,`audit_results`/`rederive` 全过。**ASR 均为预算搜索下界**(加预算只增不减)。

| 运行·模式 | eligible | 命中 | ASR(严格=宽松,下界) | 备注 |
|---|---|---|---|---|
| R1-7B GSM8K · free-r | 131 | 14 | **10.7%** | 方案 A(整组一致)9/14;操作数组 4 |
| R1-7B GSM8K · paper | 131 | **1** | **0.8%** | **7B 上首个全轨迹可容许 paper 翻转** |
| R1-7B MATH L4-5 · free-r | 77 | 2 | 2.6% | 2/2 命中均在操作数组 |
| R1-7B MATH L4-5 · paper | 77 | 0 | 0 | fork 机会稀少(tried 71) |
| Qwen3-8B(两数据集·两模式) | 57–132 | 0 | 0 | 零负结果在构造严格口径下复核 |

- **paper 免疫被打破一次**(R1 GSM8K 1/131):树搜索替换步在 paper 可达集内逐步重算,
  找到了此前"自由续写 + 审计"漏掉的可容许路径——但量级极小(0.8%),与 paper 可达集
  中位数 1–2 token 一致(fork 机会 tried 仅 190)。
- **free-r 的 10.7% 是"构造即严格"的保证下界**——与自由续写 best-of-N 的 26.6%@48
  (lenient)/25.8%(仅首翻转过审)不可直接对比:前者整条轨迹可容许、预算 3 组×2 方案,
  后者是采样下界。degraded(计划替换在分歧前缀上不再可达而回退 canon)712/518 说明
  组一致翻在分歧前缀上经常只能部分执行——计划 A 的 9 次命中证明其仍有真实增益。
- 诚实边界:canon=reach 内 argmax 是 coin 选择的代理;树的预算(每组 2 方案、3000 token)
  远未穷尽;关系失败步(超出容忍带)按 capped 处理、不计翻转。

### E1b-choice-m2w — Mind2Web 决策轮 + 定向翻转 + 注意力杠杆(2026-11-12/13,完成)

三件新事(计划/复现细节 `notes/M2W-ATTN-RUNS-2026-11-12.md`):

**(a) Mind2Web 静态元素选择数据**(`data/mind2web-choice-150.jsonl`,HF `osunlp/Mind2Web`
train,409 标注取 150):每标注一个决策轮——**历史固定在 prompt 里**(≤8 条 action repr),
只攻击当前轮生成的 token;4 选 1 元素(目标 + 同页交互元素干扰项,`cleaned_html` 抽取文本);
`meta.directed_target` = 最易混干扰项(同标签优先、任务词重叠最大)。70+ 网站。

**(b) 定向翻转(directed ASR)**:beam 目标改为
`dmargin = log p(directed_target) − log Σ p(other)`(forced-answer cue 处),命中 =
正常结束且选择恰为 directed target。

**Mind2Web natural,n=150,paper,beam8(Qwen3 cum=R1 abs,见 (d) churn 修正)**:

| 模型·目标 | eligible | ASR | null | 随机可容许轨迹 |
|---|---|---|---|---|
| Qwen3-8B any(cum) | 133 | **75.9%**(abs **91.9%**) | 13.4% | 28.9% |
| Qwen3-8B directed | 109 | **61.5%**(67 翻转全部落在指定目标) | 1.8% | 11.9% |
| R1-7B any(abs) | 96 | 33.3% | 29.0% | 59.6% |
| R1-7B directed(abs) | 97 | 39.0% | 7.8% | 24.3% |

R1 的 ASR≈null:其 Mind2Web 决策本身极不稳(重采样 null 29-47%),可翻空间被噪声吃掉。

**(c) 注意力杠杆上 vLLM**(`experiments/attn_slack_vllm.py` + `choice_tree_attn.py`):
运行时 monkeypatch Qwen2/3Attention——按 (layer, head) 对**本次 forward 新算的行**的
注意力输出乘 1±ρ(ρ=7088/2^20,两极带模型;等价于该行选带内 row mass)。三个实现要点
(全部踩过坑并已验证):
1. vLLM V1 默认**子进程**跑 EngineCore——patch 必须进程内(`VLLM_ENABLE_V1_MULTIPROCESSING=0`,
   install() 设置);
2. torch.compile 会把 sign buffer 常量化(实测 compiled margin shift 1e-7 vs eager 0.125)
   ——注意力运行一律 **enforce-eager**(R1 另有 flash-attn 偶发 illegal-memory-access,
   eager 一并规避);
3. layer index 在 DecoderLayer `__init__` patch 里解析(regex 进 forward 会炸 dynamo)。

数值验证(`runs-choice/attn-validate-*.json`):cache-off 确定性 0.0;随机 per-(layer,head)
符号把 top1−top2 margin 移动**中位 0.125 / 最大 0.25**(与 A0 的 attn_probe 量级一致);
coherent(逐层同号)也有 0.125 中位;top-1 翻转 2/24。诚实边界:与 HF float32 参考的
margin-shift 相关性未建立(pearson −0.09;bf16 下 ±0.68% 行缩放经 36 层放大 ~10×,
机理假说,未深究)。缓存路径(只重算最后一行)读数噪声 ~0.13 logprob——fork 协议按
"读到什么记什么"自洽,恢复是"噪声内"而非逐位。

**运行协议**:S* 每题在决策点([spine+cue])逐层坐标上升(`--opt-objective margin|reach`,
后者即"近似最大化单步可达 token"——在扰动 logits 下数 witnesses);每轮先 S*-probe 后
canonical 调用(兼作缓存恢复);fork token 去重到唯一 config;spawn 调用在 fork 的 config
下算 fork 行并把子路径 next-logits 交给 `_pending_lp`。审计:choice_tree_audit 复放
deviation 自带的 (act,delta)(attn 分支记在 fork 的 `attn` 列表)——全部运行 0 失败
(本节合计 ~430 万条检查)。

**(d) beam-churn 病灶与修正(方法学)**:R1 上累计 Δ 打分(cum)退化——任何 fork 都把
forced-answer margin 推高 ⇒ 新 spawn(parent+Δ)永远压过冻结分数的老子路径 ⇒ beam 每轮
清光子路径,无一走到答案(实测 512 spawn / 512 beam-death / 1 terminal)。**abs 打分**
(按路径自身当前 margin 排序)消除该退化;Qwen3 因翻得早(cum 也在 ~170 步翻)未暴露。
R1 各臂已全部用 abs 重跑;两 runner 均加了 3 轮 age 保护。

**WebShop natural 注意力 campaign(两模型 × {attn-only, attn+decode} × {any, directed},
n=150,τ=1;Qwen3 主臂 cum、R1 主臂 abs;null_dir 均按各自 run)**:

| 模型·臂 | any ASR | directed ASR | 对照:sampling paper | null (any/dir) |
|---|---|---|---|---|
| Qwen3 attn-only | 24.8% | 6.6% | 84.5% / 85.0% | 18.8% / 3.3% |
| Qwen3 attn+decode(cum) | **84.9%** | **72.2%** | 84.5% / 85.0% | 18.8% / 4.4% |
| Qwen3 attn+decode(abs) | 71.0% | — | abs spot-check | 18.8% / 4.3% |
| R1 attn-only | 38.6% | 17.9% | 40.4% / 50.8% | 29.1% / 9.0% |
| R1 attn+decode(abs) | **96.4%** | **95.1%** | 40.4% / 50.8% | 29.1% / 6.3% |

- **"attn 能否接近 sampling"是模型相关的**:Qwen3 上 attn-only(25%)远弱于 sampling(85%)
  ——单 S*、行均匀符号、每步菜单只有 2 个 canonical token;R1 上 attn-only(38.6%)≈
  sampling(40.4%)。
- **attn+decode 在 R1 上是压倒性放大**:96.4%/95.1%,比任一单独杠杆高 2.4×,比 null 高
  67pp——注意力 fork 把 decode 的 witness 集合在扰动 logits 上并集,菜单结构变了。
- Qwen3 hybrid(cum 84.9%)≈ sampling——注意力不再额外加成(其 decode 菜单本已够宽)。

**诚实边界**:R1 WebShop 合格率 ~55%(1536 截断);WebShop/m2w 均为静态代理;注意力为
两极带一阶子集、行均匀符号(全可容许集的子集 ⇒ 下界);S* 在决策点一次优化、全轮沿用;
HF 相关性未建立(见上);cum/abs 在两模型间不一致(Qwen3 翻得早未暴露 churn,主臂沿用
cum;R1 必须 abs)。

### E1b-simval — Python 关系模拟 vs 真实 zkAgent prover+verifier(2026-11-13,完成)

E1b 全部攻击只在 Python 里模拟 Q20 解码关系(`experiments/relation.py`),从未跑真 ZK
协议。本实验在 artifact 支持的模型(GPT-2 所需的一切:关系只依赖 logits)上闭环验证
模拟忠实性:**对每个 (token, committed ζ, r) 元组,用真 prover(`proveTopKBatch`,ζ 经
`decode_exp_digits_shift_override` 注入)+ 真 verifier(`verifyZkDecodeProof`)端到端
出证明并校验,与 Python 预测的接受集逐条对比。**

工具:`attacks/validate_simulation.cpp`(stdin 协议:q 向量 + case 列表;argv[2] 为
logit bound 指数,默认 20=E1a PoC 量级,artifact exp-digit 子系统本身支持 |q|≤2^46)+
`tests/validate_simulation.py`(case 生成:paper 币 / free-r 区间(含边界)/ 负例
(零质量 token、区间外 r、真带外 ζ)与结果比对)。

**结果(全部 0 失配)**:

| 设定 | cases | 结果 |
|---|---|---|
| 随机 logits(±2^17,E1a 量级,bound 2^20) | 54 | **54/54 一致** |
| GPT-2 124M 真实 logits(缩放到 E1a 量级) | 54 | **54/54 一致** |
| GPT-2 真实 logits,**不缩放**(±1.9e11,全词表 50257,bound 2^46) | 27 | **27/27 一致** |
| 深尾域(mag=ζ−δ ≥ 2^34;witness 级逐位) | 40 token | 真实 witness builder 与 Python **同样清零**(多段数字分解的高段贡献 0,无回绕);totals 相等 |

- 负例全部按预期被拒(零质量 token / 区间外 r / 带外 ζ ⇒ PROVER_FAIL 或 REJECT);
  paper 模式(H mod Θ 路由)与 free-r 模式(区间内任取 r)均验证。
- **方法学收获**:`states()` 折叠常值区(相邻断点之间 total 不变)——断点 ±1 仍可能
  在带内(首版负例因此误判;修正为跨数个表宽 2^25 步)。中间整数的 witness 与断点
  等价(mass 向量相同,仅 shift 值不同)。
- **量级发现**:E1a PoC 的 `p.bound=2^20` 只是 PoC 自身的量级选择;artifact 的
  `max_abs_logit=2^46` 支持真实模型规模的 q —— bound 提到 2^46 后,真实 GPT-2 全词表
  逐步证明可行(单证明 ~秒级,vocab 50257)。
- 结论:**E1b 各 campaign 的"模拟可达集"就是真实 verifier 的接受集**(在该关系覆盖的
  量级内),模拟无保守/无冒进;free-r 的可达集未被低估(尾部一致清零)。

诚实边界:注意力侧(attention row 带检查)是谓词级(E1a)而非逐步证明级;本次验证
仅覆盖解码关系;27/54 case 数受单证明成本限制,均为随机抽样构造。

### E1b-fp32-perpos-q35 — fp32 注意力重测 + 逐位置符号 + Qwen3.5 缩放 Phase A(2026-11-14,完成)

**(a) fp32 重测 + 逐位置(仅决策行)符号**(`choice_tree_attn --dtype float32
--sign-rows {uniform,last}`;sign buffer 升级为 [L,H,P] 按绝对位置取行,同一 forward
内不同行可带不同符号;WebShop natural,n=50,attn-only any;audit 全过):

| 配置 | ASR | null |
|---|---|---|
| bf16 row-uniform(旧,n=150) | 24.8% | 18.8% |
| **fp32 row-uniform** | **41.3%**(46 elig) | 12.0% |
| **fp32 仅决策行(last)** | **29.5%**(44 elig) | 12.0% |

- **bf16 低估了注意力杠杆**(24.8% → fp32 41.3%):bf16 的舍入运气平均抑制了菜单形成,
  之前的"bf16 上界有偏"方向实际是**偏低**。
- **逐位置自由度是真实但次要的放大**:只扰动决策行保留 71% 的 ASR(29.5/41.3)——
  精细到行的操纵有效,支持行贡献其余。margin 层面 last-row 保留 ~88% 的位移
  (0.027 vs 0.030 median,Qwen3-8B fp32 probe)。
- 修复了原 attn runner 的评分 bug(S* 候选此前在 canonical config 下打分)。

**(b) Qwen3.5 缩放 Phase A**(`experiments/scale_probe.py`,Mind2Web 决策轮,
canonical spine 120 步 × 8 题;sample 面 bf16=关系忠实,attn 面 bf16=已知不可分辨):

| 模型 | L×H(softmax 层) | paper reach(mean/med) | free-r reach(mean/med) | attn margin 位移(bf16) |
|---|---|---|---|---|
| Qwen3-8B | 36×32(全 softmax) | 1.74 / 1 | 8.9 / 4 | 0.030 med(fp32) |
| Qwen3.5-4B | 32×16(**仅 8 层 softmax**,其余 GDN) | **3.84 / 3** | **30.2 / 40** | 0.0(全模式) |
| Qwen3.5-9B | 32×16(8 层 softmax) | **2.88 / 2** | **25.1 / 33** | 0.0(全模式) |

- **decode 面:Qwen3.5 的可达集比 Qwen3-8B 宽 ~3×**(free-r 25-30 vs 8.9;paper
  2.9-3.8 vs 1.7)——新一代架构/校准使**采样容差攻击面更大**;4B > 9B(更小更平),
  与"模型越大可达越多"的预期**方向相反**(家族内:代际>尺寸;R1/Qwen3 对比亦然)。
- **attn 面:Qwen3.5 的 GDN kernel 只支持 bf16**(`DSL kernel only supports fp16/bf16`),
  且 bf16 下 ±0.68% 杠杆位移恰为 0(比 Qwen3-8B 更彻底——其 flash-attn 还泄漏部分
  fp32)⇒ **混合注意力架构上,注意力杠杆在现有 kernel 栈下无法以电路保真度模拟**
  (结构性限制,记录为 honest boundary;decode 面测量不受影响)。
- Qwen3.5 注意力补丁已就绪(Qwen3NextAttention 含 gate 路径;GDN 层不动),vLLM 0.33
  经 `PYTHONPATH=$ROOT/py312-vllm35 + site.addsitedir`(cutlass .pth)可用。

### E1b-q35-phaseB — Qwen3.5 尺寸阶梯 ASR(2026-11-15,完成)

Mind2Web natural,n=150,paper,beam8 cum,**tight 预算**(mp512 / node 15k——Qwen3.5 可达
集宽,不收紧则非翻转题把 4096 spawn 耗尽;翻转题仍早停)。decode/sample 面(bf16=
关系忠实);vLLM 0.33(PYTHONPATH + cutlass addsitedir;seed 需 62-bit mask,vLLM 0.33
的 C-long 会溢出)。27B 下载自 Qwen/Qwen3.5-27B@fc05daec(~52G)。全部 audit 0 失败。

| 模型 | any ASR | directed ASR | null(any/dir) | eligible |
|---|---|---|---|---|
| Qwen3.5-4B | **82.4%** | **71.6%** | 19.2% / 2.0% | 119 |
| Qwen3.5-9B | **80.6%** | **79.1%** | 13.8% / 2.6% | 129 |
| Qwen3.5-27B | 65.7% | 50.4% | 4.1% / 0.8% | 137 |
| Qwen3-8B(上一代,同协议) | 75.9% | 61.5% | 13.4% / 2.6% | 133 |

- **尺寸阶梯单调递减**(any 与 directed 皆是):4B→9B→27B = 82→81→66 / 72→79→50。
  与 Phase A 的可达集测量一致(free-r 30→25→更小):**模型越大分布越尖,采样容差
  攻击面越小**——"越大越多可达"的假设在这个家族内被否定;决定因素是代际与校准
  (Qwen3.5 4B/9B 都高于 Qwen3-8B),不是参数量。null 也随尺寸收缩(19→14→4%):
  决策更稳、更尖,但被瞄准时仍 50-80% 可翻。
- 定向瞄准在 9B 上几乎无损(any 80.6% vs directed 79.1%),27B 上折半(50.4%)。
- 每题翻转全部走完 strict witness audit;两次 6h timeout 由 resume 续完。

### E1b-taubench — τ-bench 高影响动作选择 + fp32 注意力语义修正(2026-11-13,完成)

**τ-bench 静态动作选择**(`data/taubench-choice-119.jsonl`,sierra-research/tau-bench:
31 airline + 88 retail;每题 = 客户指令(+ airline 在档预订)+ 4 个下一动作候选;
directed target = 代价最高的干扰(cancel 错预订 8/12、错航班、礼品卡多付 $100、
cancel-instead-modify 等)。纯 gold 动作 + 规则干扰,无环境交互(静态代理,同 m2w)。

**结果(n=119,natural,paper,beam8;Qwen3 cum / R1 abs+eager;全部 audit 0 失败)**:

| 模型·目标 | eligible | ASR | null | 备注 |
|---|---|---|---|---|
| Qwen3-8B any | 79 | **96.2%**(stable 子集 96.5%) | 42.9% | null 高 = τ-bench 动作本身高采样方差 |
| Qwen3-8B directed | 64/80 | **87.7%** | null_dir 10.8% | 高影响动作可精准瞄准 |
| R1-7B any | 45 | 46.7% | 22.6% | eligible 低(1536 截断) |
| R1-7B directed | 35/36 | 40.0% | null_dir 11.4% | |

**fp32 注意力语义修正(line 1a 结论)**:HF-bf16 下 margin 位移**恰为 0**
(1±0.68% 缩放卡在 bf16 ulp 边界,是否存活取决于 kernel 舍入运气;vLLM-bf16 显示
0.125 是 fp32 泄漏的伪影)。**fp32 引擎下测得 random 符号 margin 位移中位
0.0151 / 最大 0.069 —— 与 fp32-HF 参考(0.0127)同量级;逐对相关 pearson 0.358
(n=24,位移幅度 ~0.01、跨 kernel 栈,噪声受限的正向信号)**
(`attn-validate-clean-fp32.json` / `attn-validate-final-fp32.json`)。⇒ 此前 bf16 引擎上的
注意力 ASR(attn-only 24.8%/6.6%,hybrid 84.9%/72.2%/96.4%/95.1%)是"bf16 引擎+该
patch"的行为,按 Q20 电路语义应视为**有偏估计(方向:偏低,见 §fp32 重测——attn-only
24.8%→41.3%,bf16 的舍入运气平均抑制了菜单形成)**;电路本身(Q20 定点,~1e-4 相对
分辨率)能表达 ±0.68% 的行质量选择,忠实模拟需 fp32 前向(或定点模拟器)。

### E1b-choice — 无 gold 决策集(DailyDilemma / WebShop):paper reach × forced-answer beam(2026-11-11)

此前所有决策攻击都建立在 GSM8K/MATH 上,ASR 的分母被 "honest 答对" 筛过。本节换成
**没有唯一正确答案**的选择题:每个 honest choice 本身就是一次可攻击的决策。

**数据与协议(无 gold 依赖)**

- `data/dailydilemma-150.jsonl`:150 题 A/B 二选一(DailyDilemma 的 to_do / not_to_do)。
- `data/webshop-choice-150.jsonl`:150 题 1–4 四选一,**静态代理**(商品 instruction +
  同类别候选;不使用 WebShop 环境/reward),结论只限 "商品选择决策"。
- `honest` = τ=1/top-k=40 采样;`control` = 换 seed 的第二条诚实轨迹(null flip);
  `canonical` = coin 决定的 paper 可容许主干。
  **eligible** = honest 解析 ∧ canonical 解析 ∧ canonical == honest;
  **flip** = 正常结束(`done='stop'`)且 choice ≠ honest。
- honest 采样本身就有 **1.21%(short)/ 2.43%(natural)** 的 token 落在 paper reach 之外
  (29% / 89% 的题至少一个),所以独立重采样不是合法攻击 baseline,所有 ASR 与 null flip 并列。
- 每题搜索不设 token 类型/数量启发式,每步展开整个 `witnesses(H,'paper')`;只保留
  node-limit / max-paths 安全阀。

**搜索空间**:canonical spine 的 fork 积中位 short 10^2.58、natural 10^19.5;它是量级
估计,既非精确值也非上下界。修复后的 census(n=6,node-limit 15k)里 **2 题可完整枚举**
(96 / 845 条完整轨迹,均 0 flip),其余 4 题超过 4,096 条路径上限——**n=150 搜满不可行**,
ASR 均为搜索下界。

**forced-answer beam**(用户方案):对每步可达 token `t`,读
`margin = log Σ_{l≠honest} p_l − log p_honest` at `prefix + [t] + "\n\nFinal answer: \boxed{"`,
`Δ(t) = margin(t) − margin(canonical)`;beam 按累计 Δ(或绝对 margin)保留路径。

| 数据 / 输出 | B=8 累计 Δ ASR | 7 条随机可容许轨迹 | 步数/题 | audit |
|---|---|---|---|---|
| DD short(128) | **31.6%**(139 eligible,44 翻题) | 13.7% | 604 vs 224 | 8,121 检查 0 失败 |
| DD natural(384) | **81.1%**(132,107) | 23.0% | 1,362 vs 779 | 89,855 / 0 |
| WS short(128) | **35.0%**(140,49) | 12.9% | 798 | 6,116 / 0 |
| WS natural(1024) | **85.6%**(125,107) | 27.5% | 5,304 | 317,455 / 0 |

- DD short 的 beam 变体:B=16 累计 Δ 33.6%(1,081 步/题)、**B=4 绝对 margin 33.1%
  (307 步/题,性价比最高)**、B=8 单步 Δ 25.2%、B=1 绝对 margin 12.2%(**65 步/题**)。
- beam 的翻转题覆盖 rollout 的绝大多数并多 30–34 题(DD short 44 vs 19 交集 14;
  WS short 49 vs 18 交集 15)。
- **搜满 vs 搜不满**:dd-9459 上 beam 用 437 nodes 找到翻转,census 用 16,886 nodes
  仍未枚举完;能枚举完的两题两者一致(都不翻)。固定预算下 Δ 引导的部分搜索优于
  无引导枚举。

**attention 轴**(HF eager + AttnSlack,ρ=7088/2^20,三扰动 union,只在 spine 探测,
n=20,short,两臂同预算):DD on 35.0%(7/20) vs off 25.0%(5/20);WS on 17.6%(3/17)
vs off 11.8%(2/17)。on 的翻转题**严格包含** off;每 100 探针步多 6.36–9.33 个可达
token、每题中位多 5 个 fork。n=20 小样本,且 on 臂额外 logits 来自 eager 后端
(honest reach 来自 sdpa),有少量 kernel 混淆。

**实现更正(重要)**:vLLM runner 的 spawn 曾用父前缀建子路径且 depth+1,**偏离 token
从未进入子路径**,所有 "偏离分支" 实际在重走同一 fork;旧 beam/census 数字(0.7%/0%)
因此作废。审计(`deviation-token-match`)发现后修复为 `prefix + [t]`,并加
`depth == len(ids)-base_len` 断言与回归测试;修复后同配置 ASR 0.7%→31.6%(DD short)、
0%→81.1%(DD natural)。rollout 与 HF attention runner 不受影响。

**诚实边界**:WebShop-choice 是静态四选一代理,不覆盖交互式搜索/点击决策;无 gold 下
honest 决策本身有采样方差(换 batch 的浮点差会让 1–3% 的题 canonical 翻转);ASR 是
搜索下界;注意力轴样本小且有一阶占位模型;全部结果仅 Qwen3-8B。

**文件**:细节 `attacks/e1b/reports/E1b-choice-RESULTS.md`(主结果 §4.9、bug 复盘 §4.11)、
按方向逐条回答 `attacks/e1b/reports/E1b-choice-ANSWER.md`、计划与执行修正
`attacks/e1b/reports/E1b-choice-NEXT.md`、
每题搜索规模 `attacks/e1b/data/E1b-choice-feasible-v2.jsonl`;代码在本仓库的快照副本
`attacks/e1b/choice/`(choice_tree / choice_tree_hf / choice_tree_audit / choice_report /
choice_scoring / choice_data / build_choice_data / choice_rollout_stats /
choice_margin_auc / choice_attn_stats / choice_feasible_table + 测试,正本在
`/root/zk_agent/experiments/`)。运行产物在 worker 共享存储
`/mnt/shared-storage-user/ai4good1-share/guodadi/zk_agent_gpu/runs-choice/qwen3/*-v2/`。

---

## 3. 总结论(可直接进文章 framing)

1. **zkAgent 的关系是集值的,安全声明为假**——E1a 端到端铁证,分叉点 100% 可翻。
   这是定义上的 soundness break,是论文最稳的脊梁。
2. **论文设计 vs 释出实现,差距巨大**:
   - 论文设计:可达 ~7、账单膨胀 modest(~1.1–1.2×)、决策操纵基本堵死。
   - 释出实现(采样币 r 未绑前缀,free-r):可达 ~30–37、账单**实质无上界**(63% 注入)、
     **决策可被 lookahead 翻转**(56→42)。
3. **决策操纵需要 lookahead**(盲打会被自我纠正);账单膨胀靠拉长而非 force-skip。
   **修正(E1b-5)**:真正的武器不是贪心 lookahead,而是**哈希链重投 → 轨迹级 best-of-N**;
   据此 **paper 设计也能翻决策**(1.5B smoke 1/3,此前贪心报 ~0),free-r 2/3→3/3。
   **上规模(E1b-5@7B/8B,2026-11-07)**:R1-7B free-r 9.7%@1 → 26.6%@48(严格口径 25.8%),
   放大真实但被自我纠正封顶(阶梯 @24 后走平);**paper 的决策免疫在 7B/8B 经修正攻击复核成立**
   (R1 lenient 1.6% / strict 0%,Qwen3 全 0)——1.5B 的 paper 1/3 未迁移,瓶颈是自我纠正而非预算。
4. 推荐 framing:**"即便按论文最强防御,关系仍是集值的(存在性 + 分叉点必翻);其释出实现因未绑定
   采样币,账单实质无上界、答案可被操纵。"**
5. **"paper 决策免疫"是任务相关的**(E1b-choice,2026-11-11):GSM8K/MATH 的 paper strict 0%
   来自 gold 唯一确定的答案;换到无 gold 的选择任务(DailyDilemma/WebShop-choice),
   honest 决策本身稳定性低(null flip 5–17%)、可达分叉多,按 forced-answer Δ 保留
   paper 可容许轨迹即可翻转 **31.6% / 81.1% / 35.0% / 85.6%** 的 eligible 决策
   (7 条随机可容许轨迹只有 13.7–27.5%)。framing 应写 "paper 的决策免疫在数学/问答类
   有 gold 任务上成立,在偏好/选择类无 gold 决策上不成立"。

### 诚实边界(全局)

- 规模小:n≤5 短问答,模型仅 **GPT-2 124M / R1-Distill-Qwen-1.5B**(本机 15G + 纯 CPU,跑不动 8B)。
  → **解除**:Qwen3-8B 与 R1-Distill-7B,GSM8K(τ∈{0.6,0.8,1.0})+ MATH(τ=1)各 n=150,
  两个 serving 后端(HF 单流 + vLLM),注入见证全部独立复验(合计 ~380 万条);决策操纵
  R1-7B 为正(11.7–25.4%)、Qwen3-8B 为负(0/全部 ~700 eligible)、1.5B 为正。
- 温度:多数 τ=1(zkAgent 配置);真实部署 τ=0.6 更尖,yield 更低(已验证趋势)。
  → **τ∈{0.6,0.8,1.0} 已扫**(两模型):注入机会大降但 free-r 撞 cap 率不变、R1 决策
  翻转仍 16%。
- 策略仍是 **P2-lexical / 有限 lookahead**,不是最优攻击 → 所有数字是**下界**。
  → **修正已上规模(E1b-5@7B/8B)**:采样 best-of-N 阶梯已并入 GPU harness
  (`experiments/vllm_bestofn.py`,纯批量生成,单模型 9–13 min);R1-7B free-r 21.8%→26.6%@48
  (strict 25.8%),paper strict 0%,Qwen3 全 0——见 §E1b-5@7B/8B。
- cap 压扁膨胀倍数;paper 的 ngrid 采样可能漏可达 token(偏保守)。
- 注意力只做到"接受域谓词"级(E1a)。→ **首次量化(E1b-attn)**:注意力容差 ±0.676%/行,
  对抗性选号单独可翻 margin ≲0.30 的解码步(GPT-2 3/16 步),paper 可达集 +68%;仍是一阶"两极"
  模型的下界,**待在 7B/8B 上重测下游 yield**。

---

## 4. 后续需要做的实验(按优先级)

### 决定论文成败的
1. **E1b 上规模(最重要)** → **全部完成**:Qwen3-8B / R1-Distill-7B × GSM8K(τ∈{0.6,0.8,1.0})×
   MATH(τ=1),n=150(HF 复刻 GSM8K n=150 + MATH n=75;vLLM 全量);decision forensics、
   深度 rollout 复核、双后端交叉、审计全过;probe 接入的 1 步 lookahead 计费攻击两模型
   n=60 已完成(见 E1b-8B/vLLM 节末)。
2. **真·P2 value probe** → **已完成**:探针本身泛化差(同域 held-out R² ≤ 0.22),但作为
   攻击的价值函数已足够——把 paper 模式注入率放大 3–10×、R1 未删失膨胀 1.0×→3.64×。

### 补全攻击面
3. **注意力 steering 的下游 yield** → **已起步(E1b-attn)**:GPT-2 上注意力容差 ±0.676%/行 → 对抗性
   选号翻 margin ≲0.30 的步、paper 可达集 +68%(`attn_slack_logits.py`)。**待做**:(a) 7B/8B 上重测;
   (b) 把一阶"两极"升级为逐键差分取整的完整可容许集;(c) 注意力扰动 + 解码重投的联合 yield。
4. **τ 扫描**:τ∈{0.6,0.8,1.0} 下重跑 yield / 账单 / 翻转,给出温度依赖曲线。
5. **paper 模型更密网格复核**:ngrid 2000→更大,确认 paper 的可达/翻转是否被低估。

> **E1b-choice 的后续(2026-11-11)**: (a) **RewardBench 未跑**——它带 gold,与"无标准答案
> 数据集"的前提冲突,建议只作为 forced-answer 打分的辅助校验;(b) R1-Distill-7B 复刻
> v2 beam(当前结果仅 Qwen3-8B);(c) 接交互式 WebShop harness(当前是静态四选一代理)。

### proposal 里的其它实验
6. **E2 config grinding**:k 个 batch-size / split-K 配置在 EOT/EOS 步产生多少不同决策。
7. **E3 patched zkAgent**:实现 defense 节的"函数化"补丁(精确行最大值 + 精确整数和 + 单次舍入除法;
   inverse CDF over 未归一化 top-k 权重 + r=H(·) mod Θ 绑定),测所有 E1a 攻击被拒 + 开销。
8. **E5 audit economics**:3/α 审计的 prover 时间 / 证明大小 / 成本,外推到 8B。

### 理论(论文脊梁)
9. **necessity/sufficiency 证明**:Thm 1(任何接受多结果的 verifier 可 steer)、Prop 2(重尾稀疏 slack)、
   Thm 3(uncontrolled coins + full coverage + 函数化语义 ⇒ trajectory soundness)。

---

### 设计决策(放大实验时遵循)

- **value probe 必须跨域训练 + 跨域评估**:单数据集内样本高度相关,同集训练+测试会高估泛化;
  真实话题任意。做法:多领域混合训练 + **按话题/数据集 held-out** 评估,报 in-domain 与 cross-domain
  两个数字。注意两点:(a) 威胁模型下 provider 用**真实流量**训探针,天然 in-distribution——"任意话题"
  正是它有数据的地方;(b) 探针只优化 P2 的 yield,**攻击可行性(free-r 每步 ~30 可达)与探针无关**,
  探针差只是 yield 低,不改变"攻击成立"。`train_length_probe.py` 已支持 `--train-domains/--eval-domains`;
  **smoke 实测证实了跨域退化**:训 custom(算术)→ 测 diverse(写作/解释)corr 0.85→0.49、R²→-0.04(比均值还差),
  所以真实实验的探针**必须在部署分布上训**。`eval_at_scale.py --bill-policy probe` 已把探针接成 1 步 lookahead 选择器。
- **实验主力用 CoT / 推理模型**:账单膨胀的前提是"**被计费的隐藏推理客户看不见**"(hidden span `L_R`),
  只有 CoT 且 CoT 单独计费的推理模型才成立;非 CoT 输出可见,塞 token 会被看到。核心缺陷(集值 / free-r)
  对任何模型都成立,但 billing 半场需要 CoT。**当前脚本假设 chat/推理模型**(chat template + `</think>`),
  base 非 CoT 模型(如 GPT-2)`apply_chat_template` 会报错。建议:主力 CoT 推理模型
  (DeepSeek-R1 / R1-Distill、Qwen3-thinking、QwQ);可选一个 instruct 非推理模型证明决策操纵的普适性。

## 4b. 已写好的放大脚本(本机小配置验证可跑通,待上 GPU 放大)

| 脚本 | 覆盖的后续实验 | 验证状态 |
|---|---|---|
| `attacks/e1b/scale_harness/eval_at_scale.py` | #1 上规模(GSM8K/MATH,两模型,decision+billing,**含 #4 τ-sweep=`--temperature`**,#5 `--ngrid`) | ✅ 端到端跑通(1.5B+custom+CSV);数字是 smoke 噪声 |
| `attacks/e1b/length_steering/train_length_probe.py` | #2 真·P2 value probe(隐状态→剩余长度,ridge) | ✅ 跑通(1.5B,held-out corr=0.85,R²=0.72) |

放大运行(GPU 机):
```bash
# 决策操纵 + 账单膨胀,GSM8K 全 test,两模型;换 --model 为 8B+
python3 attacks/e1b/scale_harness/eval_at_scale.py --model <8B-reasoning-model> --device cuda \
    --dataset gsm8k --split test --limit 1319 --task both --models free-r,paper \
    --max-new 1024 --rollout 512 --max-steps 400 --max-rollouts 60 --ngrid 3000 \
    --csv gsm8k_eval.csv
# 温度依赖:--temperature 0.6 / 0.8 / 1.0 各跑一遍
# 价值探针(先训练,之后可接进 eval 的 P2 选择):
python3 attacks/e1b/length_steering/train_length_probe.py --model <model> --device cuda --dataset gsm8k --limit 500
```
指标:`honest_acc` / `attacked_acc` / `flip_success`(decision),`median_inflation` / `mean_injrate`(billing)。

**GPU 就绪**:两脚本支持 `--device cuda` 与 `--dtype {float32,bfloat16,float16}`(8B 用 `bfloat16` 单卡可放)。
两个注意:(1) **reach 计算在 CPU(numpy)**——free-r 便宜,但 **paper 的 Θ-grind 是 python 循环,GPU 不加速,
放大跑 paper 时是瓶颈**(需要时可向量化/搬 torch);(2) value probe 目前**只训练+评估**,接进 `eval_at_scale`
做"每候选 1 步 lookahead 打分"的 P2 选择仍是 TODO。

> 仍待**另行实现**(非 Python 单脚本):**E2 config grinding**(需带可变 batch/split-K 配置的 C++ artifact)、
> **E3 patched zkAgent**(defense 节的函数化补丁,C++ prover/verifier)、**E5 audit economics**(复用解码 prover 计时 + 外推)。
> 理论(necessity/sufficiency)是证明,不是脚本。

## 5. 复现

构建(仓库根,见顶层 README 装依赖后):
```bash
./build.sh Release                       # 产出 cmake-build-release/src/libgpt_lib.a
LIB=cmake-build-release
# E1a
g++ -std=c++14 -O2 -fopenmp -mcmodel=large -Isrc -I3rd -I3rd/mcl/include \
    attacks/setvalued_decode.cpp  $LIB/src/libgpt_lib.a $LIB/lib/libmcl.a -lgmp -lcrypto -o setvalued_decode
# E1b oracle(任选其一,同样方式编译):
#   attacks/e1b/yield_oracle/{yield_decode,freer_reach,reach_both}.cpp
#   attacks/e1b/forced_extension/forced_extension.cpp
# E1b 需要真实 logits / 重生成的部分是 Python(torch + transformers):
python3 attacks/e1b/yield_oracle/dump_logits.py --model openai-community/gpt2 --prompt "..." --out logits.txt
python3 attacks/e1b/length_steering/e1b_length_steer.py --cap 400           # 账单膨胀,两模型
python3 attacks/e1b/decision/decision_lookahead.py --max-rollouts 40 # 决策翻转成功率,两模型(旧·贪心)
# E1b-5 决策修正(哈希链重投 → 采样 best-of-N;--device cuda 可放 8B):
python3 attacks/e1b/decision/decision_bestofN.py --rollout-mode greedy --models free-r,paper   # 旧基线
python3 attacks/e1b/decision/decision_bestofN.py --rollout-mode sample --bestof 6               # claim-1 放大
python3 attacks/e1b/decision/decision_bestofN.py --rollout-mode steer  --bestof 6               # + 自我纠正抑制启发式
# E1b-attn 注意力容差 → 解码 logit 扰动(GPT-2 124M = zkAgent artifact 模型):
python3 attacks/e1b/attention/attn_slack_logits.py --model openai-community/gpt2
```

文件清单(2026-11-15 重组为子目录;完整逐文件说明见 `attacks/e1b/README.md`):
- `attacks/` — E1a(set-valued 端到端)+ README
- `attacks/e1b/yield_oracle/` — `dump_logits.py` 真实 logits 导出(量化进 zkAgent 整数域);
  `yield_decode.cpp / freer_reach.cpp / reach_both.cpp` 可达性 / yield(两模型)
- `attacks/e1b/forced_extension/` — `forced_extension.cpp`(判定,负面结论)、
  `dump_until_stop.py / e1b_forced.py`(停止点轨迹导出 / 合并实验)
- `attacks/e1b/length_steering/` — `e1b_length_steer.py` 账单膨胀重生成(P1/P2,两模型;
  **也是全部本地脚本共享的 relation/reach 库**)、`bill_inflation.py`、`train_length_probe.py`(真·P2 value probe)
- `attacks/e1b/decision/` — `decision_attack.py`(盲)/ `decision_lookahead.py`(贪心 lookahead)、
  `decision_bestofN.py` **E1b-5 决策修正**(哈希链重投 → 采样 best-of-N,greedy/sample/steer 三模式 + 复述靶向启发式)、
  `tree_attack.py`(E1b-tree 可容许树搜索,可复用、torch-free)
- `attacks/e1b/attention/` — `attn_reach.py`(AttnSlack hook + union_reach,共享库)、
  `attn_slack_logits.py` **E1b-attn**(注意力容差 ±ε/θ 两极 → 解码 logit 扰动 / 可达集增量)、
  `attn_slack_vllm.py / attn_slack_validate.py`(vLLM patch 与验证,harness 快照)
- `attacks/e1b/scale_harness/` — `eval_at_scale.py` **放大版主评测**(任意 HF 模型 + GSM8K/MATH,
  两模型,decision+billing,τ-sweep);`vllm_bestofn.py / scale_probe.py / validate_simulation.py`
  (H200 harness 快照,正本在 `/root/zk_agent/experiments/`)
- `attacks/e1b/choice/` — E1b-choice → m2w → τ-bench → Qwen3.5 全部 choice 攻击/分析脚本(harness 快照)
- `attacks/e1b/reports/` — 各阶段计划/结果报告(SUMMARY 之外的 7 篇)
- `attacks/e1b/data/` — choice 数据集与可行性输出的本地快照

> 早期 Python 实验在 1.5B 推理模型上于纯 CPU 运行;E1b-5 / E1b-attn 的 smoke 在本机单卡 GPU
> (1.5B fp16 / GPT-2 fp32)运行。C++ oracle 与 Python 量化/可达性已逐位对齐校验。

### E1b-8B(GPU 放大)复现

8B 放大不在本仓库,在 `/root/zk_agent`(H200 worker,mas_security rjob 模式):

```bash
# 关系与攻击引擎(精确 Q20 枚举 + BF16 KV 分支);tests 含 C++ oracle 对齐
cd /root/zk_agent && python3 -m unittest tests.test_relation
# 主评测(Qwen3-8B,GSM8K n=150;R1-7B 同型)
python3 infrastructure/gpu/control.py enqueue --cwd <snapshot> --timeout 21600 -- \
  <env>/python -m experiments.evaluate --model <models/Qwen3-8B-*> --data <data>/gsm8k-test.jsonl \
  --out <runs>/qwen3-8b/gsm8k-main --limit 150 --task both --modes free-r,paper \
  --rollout 1536 --max-rollouts 8
# 独立复audit(注入见证逐条过 Relation.check + 汇总对账)
python3 -m experiments.audit_results <runs>/qwen3-8b/gsm8k-main
```

文件:`experiments/{relation,engine,evaluate,probe,diag_digits,decision_forensics,audit_results,report}.py`,
运行产物在 worker 共享存储 `runs/`,汇总快照在 `/root/zk_agent/results/e1b/`。

vLLM 后端(第二张 H200,`--root <root>/vllm` 队列,image `vllm-openai:v0.25.0`):

```bash
python3 infrastructure/gpu/control.py --root <root>/vllm/worker enqueue --cwd <snapshot> \
  --timeout 14400 -- python3 -m experiments.vllm_runner --model <models/R1-*> \
  --data <data>/gsm8k-test.jsonl --out <runs-vllm>/r1-7b/gsm8k-tau10 \
  --limit 150 --task both --modes free-r,paper --rollout 1536 --max-rollouts 8 \
  --batch 150 --max-seqs 128 --temperature 1.0
# 计分修正后重判(答案抽取 LaTeX 修复,raw 与 corrected 并存)
python3 -m experiments.rederive <runs-vllm>/r1-7b/gsm8k-tau10
```

汇总快照:`/root/zk_agent/results/e1b/`(HF)与 `/root/zk_agent/results/e1b-vllm/`(vLLM)。

**修正决策攻击(E1b-5@7B/8B,2026-11-07)复现** —— 批量 best-of-N runner(纯批量生成,无逐步
引擎往返;单模型 9–13 min,对比旧 evaluate.py 5.8 h / 旧 vllm_runner 步循环 63–112 min):

```bash
# worker 共享根 $ROOT;解释器用 vllm venv( transformers==4.57.1):
#   $ROOT/vllm-env/bin/python  (vllm 0.11.0;与 worker image python3 不同)
python3 infrastructure/gpu/control.py --root $ROOT/worker enqueue --cwd <snapshot> \
  --timeout 7200 -- $ROOT/vllm-env/bin/python -m experiments.vllm_bestofn \
  --model $ROOT/models/R1-Distill-Qwen-7B-* --data $ROOT/data/gsm8k-test.jsonl \
  --out $ROOT/runs-bestofn/r1-7b/gsm8k-main --limit 150
# 审计 + 重判(与旧协议同一套工具,decision 字段保持兼容)
$ROOT/vllm-env/bin/python -m experiments.audit_results $ROOT/runs-bestofn/r1-7b/gsm8k-main
$ROOT/vllm-env/bin/python -m experiments.rederive   $ROOT/runs-bestofn/r1-7b/gsm8k-main
```

文件:`experiments/vllm_bestofn.py`(greedy 臂 + 采样 best-of-N 阶梯 + prompt_logprobs 严格审计)、
`experiments/scoring.py`(计分抽出的无 torch 公共模块)、`tests/test_vllm_bestofn.py`(假引擎逻辑测试)。
汇总快照:`/root/zk_agent/results/e1b-bestofn/`。

**E1b-choice(2026-11-11)复现** —— 无 gold 选择任务,paper reach × forced-answer beam:

```bash
R=/mnt/shared-storage-user/ai4good1-share/guodadi/zk_agent_gpu   # worker 共享根
# 主跑:DD short / DD natural / WS short / WS natural,B=8 累计 Δ
$R/vllm-env/bin/python -m experiments.choice_tree \
  --model $R/models/Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218 \
  --data $R/data/dailydilemma-150.jsonl \
  --out $R/runs-choice/qwen3/dd-short-beam8-cum-v2 \
  --thinking off --short --search beam --beam 8 --beam-score cum \
  --max-new 128 --node-limit 40000 --batch 8 --record-forced --record-final
# 每条 deviation 的 witness 独立复核(模型无关)
python3 -m experiments.choice_tree_audit $R/runs-choice/qwen3/dd-short-beam8-cum-v2
# 汇总 / 预算曲线 / 每题搜索规模
python3 -m experiments.choice_report        $R/runs-choice/qwen3/*-v2
python3 -m experiments.choice_feasible_table $R/runs-choice/qwen3/dd-short-beam8-cum-v2 \
  --out feasible.jsonl
```

文件:`experiments/{choice_tree,choice_tree_hf,choice_tree_audit,choice_report,choice_scoring,choice_data,build_choice_data,choice_rollout_stats,choice_margin_auc,choice_attn_stats,choice_feasible_table}.py`,
`tests/test_choice_*.py`;地址副本在 `attacks/e1b/choice/`。运行产物在
`$R/runs-choice/qwen3/*-v2/`(含 items/、summary.json、config.json)。

# zkAgent 攻击验证 —— 实验总结与后续计划

本文件汇总目前对 zkAgent(ePrint 2026/199)已释出 artifact(GPT-2 124M, Q=20,
commit `af36422`)以及其解码/注意力关系所做的全部实验、结论,以及尚待完成的实验。
所有结论都标注了诚实边界。实验代码在 `attacks/`(E1a)与 `attacks/e1b/`(E1b)。

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
- 代码:`attacks/e1b/dump_logits.py`、`yield_decode.cpp`、`freer_reach.cpp`、`reach_both.cpp`。

### E1b-2 — Forced extension(强制不停 / 账单膨胀的"硬"路径)❌(对 zkAgent 无效)

- 两条自然终止轨迹的真实停止步(`</think>` 概率≈1):`reach=1`,非终止符量化质量**精确=0**,
  **free-r 也挤不进**。所以**无法 force-skip 自信的停止点**。
- 这与 proposal 自己的 landscape 表一致:zkAgent 行只列 "coin re-roll; tolerance steering",
  forced extension 是列给"解码不在证明里"的系统(TOPLOC/CoIn 等);zkAgent 把解码证了,故免疫。
- 代码:`attacks/e1b/forced_extension.cpp`、`dump_until_stop.py`。

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
- 代码:`attacks/e1b/e1b_length_steer.py`(`--skip-freer`、`--ngrid`)。

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
- 代码:`attacks/e1b/decision_attack.py`(盲)、`decision_lookahead.py`(lookahead + 多题成功率)。

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

**value probe(真·P2)为负结果**:两模型 ridge 探针(隐状态→剩余推理长度):
train corr 0.83–0.84 / R² 0.68–0.71,但 **held-out 同域 R² ≤ 0.22(R1 为 −0.19)、
跨域 ≈ 0**——7B/8B 的推理长度重尾,线性探针不可泛化(1.5B 时同域 R² 0.72)。
probe 接入的 1 步 lookahead 计费攻击(hf worker,进行中)预计不优于 lexical;
攻击可行性本就不依赖 probe(free-r 每步 ~30 可达)。

---

## 3. 总结论(可直接进文章 framing)

1. **zkAgent 的关系是集值的,安全声明为假**——E1a 端到端铁证,分叉点 100% 可翻。
   这是定义上的 soundness break,是论文最稳的脊梁。
2. **论文设计 vs 释出实现,差距巨大**:
   - 论文设计:可达 ~7、账单膨胀 modest(~1.1–1.2×)、决策操纵基本堵死。
   - 释出实现(采样币 r 未绑前缀,free-r):可达 ~30–37、账单**实质无上界**(63% 注入)、
     **决策可被 lookahead 翻转**(56→42)。
3. **决策操纵需要 lookahead**(盲打会被自我纠正);账单膨胀靠拉长而非 force-skip。
4. 推荐 framing:**"即便按论文最强防御,关系仍是集值的(存在性 + 分叉点必翻);其释出实现因未绑定
   采样币,账单实质无上界、答案可被操纵。"**

### 诚实边界(全局)

- 规模小:n≤5 短问答,模型仅 **GPT-2 124M / R1-Distill-Qwen-1.5B**(本机 15G + 纯 CPU,跑不动 8B)。
  → **解除**:Qwen3-8B 与 R1-Distill-7B,GSM8K(τ∈{0.6,0.8,1.0})+ MATH(τ=1)各 n=150,
  两个 serving 后端(HF 单流 + vLLM),注入见证全部独立复验(合计 ~380 万条);决策操纵
  R1-7B 为正(11.7–25.4%)、Qwen3-8B 为负(0/全部 ~700 eligible)、1.5B 为正。
- 温度:多数 τ=1(zkAgent 配置);真实部署 τ=0.6 更尖,yield 更低(已验证趋势)。
  → **τ∈{0.6,0.8,1.0} 已扫**(两模型):注入机会大降但 free-r 撞 cap 率不变、R1 决策
  翻转仍 16%。
- 策略仍是 **P2-lexical / 有限 lookahead**,不是最优攻击 → 所有数字是**下界**。
- cap 压扁膨胀倍数;paper 的 ngrid 采样可能漏可达 token(偏保守)。
- 注意力只做到"接受域谓词"级(E1a),**未做注意力的 yield / 下游影响**。

---

## 4. 后续需要做的实验(按优先级)

### 决定论文成败的
1. **E1b 上规模(最重要)** → **完成**:Qwen3-8B / R1-Distill-7B × GSM8K(τ∈{0.6,0.8,1.0})×
   MATH(τ=1),n=150(HF 复刻 GSM8K n=150 + MATH n=75;vLLM 全量);decision forensics、
   深度 rollout 复核、双后端交叉、审计全过。唯一进行中:probe 接入的 1 步 lookahead 计费
   攻击(HF worker,R1 已 ~10/60,数字待补)。
2. **真·P2 value probe** → **已完成(负结果)**:7B/8B 线性隐状态探针同域 held-out
   R² ≤ 0.22、跨域 ≈ 0(见 E1b-8B/vLLM 节);lexical 策略仍是有效攻击者下界。

### 补全攻击面
3. **注意力 steering 的下游 yield**:把 E1a 的注意力集值性接到完整推理,量化对输出的实际影响
   (目前只证了谓词接受多值)。
4. **τ 扫描**:τ∈{0.6,0.8,1.0} 下重跑 yield / 账单 / 翻转,给出温度依赖曲线。
5. **paper 模型更密网格复核**:ngrid 2000→更大,确认 paper 的可达/翻转是否被低估。

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
| `attacks/e1b/eval_at_scale.py` | #1 上规模(GSM8K/MATH,两模型,decision+billing,**含 #4 τ-sweep=`--temperature`**,#5 `--ngrid`) | ✅ 端到端跑通(1.5B+custom+CSV);数字是 smoke 噪声 |
| `attacks/e1b/train_length_probe.py` | #2 真·P2 value probe(隐状态→剩余长度,ridge) | ✅ 跑通(1.5B,held-out corr=0.85,R²=0.72) |

放大运行(GPU 机):
```bash
# 决策操纵 + 账单膨胀,GSM8K 全 test,两模型;换 --model 为 8B+
python3 attacks/e1b/eval_at_scale.py --model <8B-reasoning-model> --device cuda \
    --dataset gsm8k --split test --limit 1319 --task both --models free-r,paper \
    --max-new 1024 --rollout 512 --max-steps 400 --max-rollouts 60 --ngrid 3000 \
    --csv gsm8k_eval.csv
# 温度依赖:--temperature 0.6 / 0.8 / 1.0 各跑一遍
# 价值探针(先训练,之后可接进 eval 的 P2 选择):
python3 attacks/e1b/train_length_probe.py --model <model> --device cuda --dataset gsm8k --limit 500
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
#   attacks/e1b/{yield_decode,freer_reach,reach_both,forced_extension}.cpp
# E1b 需要真实 logits / 重生成的部分是 Python(torch + transformers):
python3 attacks/e1b/dump_logits.py --model openai-community/gpt2 --prompt "..." --out logits.txt
python3 attacks/e1b/e1b_length_steer.py --cap 400           # 账单膨胀,两模型
python3 attacks/e1b/decision_lookahead.py --max-rollouts 40 # 决策翻转成功率,两模型
```

文件清单:
- `attacks/` — E1a(set-valued 端到端)+ README
- `attacks/e1b/dump_logits.py / dump_until_stop.py` — 真实 logits 导出(量化进 zkAgent 整数域)
- `attacks/e1b/yield_decode.cpp / freer_reach.cpp / reach_both.cpp` — 可达性 / yield(两模型)
- `attacks/e1b/forced_extension.cpp` — forced-extension 判定(负面结论)
- `attacks/e1b/e1b_length_steer.py` — 账单膨胀重生成(P1/P2,两模型)
- `attacks/e1b/decision_attack.py / decision_lookahead.py` — 决策操纵(盲 / lookahead,多题成功率)
- `attacks/e1b/eval_at_scale.py` — **放大版主评测**(任意 HF 模型 + GSM8K/MATH,两模型,decision+billing,τ-sweep)
- `attacks/e1b/train_length_probe.py` — 真·P2 value probe(隐状态→剩余长度)

> 所有 Python 实验在 1.5B 推理模型上于纯 CPU 运行;C++ oracle 与 Python 量化/可达性已逐位对齐校验。

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

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
- 温度:多数 τ=1(zkAgent 配置);真实部署 τ=0.6 更尖,yield 更低(已验证趋势)。
- 策略仍是 **P2-lexical / 有限 lookahead**,不是最优攻击 → 所有数字是**下界**。
- cap 压扁膨胀倍数;paper 的 ngrid 采样可能漏可达 token(偏保守)。
- 注意力只做到"接受域谓词"级(E1a),**未做注意力的 yield / 下游影响**。

---

## 4. 后续需要做的实验(按优先级)

### 决定论文成败的
1. **E1b 上规模(最重要)**:8B+ 推理模型(Llama-3-8B / Qwen3-8B / R1-Distill-7B)+ **完整数据集**
   (GSM8K / MATH)。指标:**attack success rate**(能翻的题占比)、**attacked accuracy drop**
   (正确率掉多少)、账单膨胀分布。free-r 与 paper 两模型对照。→ **需 GPU**;本机只能出 oracle/机制。
2. **真·P2 value probe**:训练一个"剩余长度 / 答案正确性"探针(在模型自身隐状态上),替代
   lexical / 暴力 lookahead,给出更接近最优攻击者的 yield 与翻转率。

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

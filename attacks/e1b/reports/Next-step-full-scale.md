# Decision Integrity 论文：zkAgent 实验设计 v1

Nov 15, 2026 · @Ye

## 1. 目标与论文主张

这份文档把 zkAgent 这轮实验的协议定死，交给 Weijie 严格执行。核心主张不变：**verifier 对同一个请求接受多于一个结果，就给不出 decision integrity**；两类后果是计费膨胀与决策操纵。

这轮实验要支撑的具体论断：

1. **集值性是协议层的**（E1a 已端到端验证，模拟 ≡ 真 verifier 已由 simval 确认），与模型无关。下面所有 ASR 都是在这个前提上量化后果。
2. **决策完整性在无 gold 的选择/agent 任务上崩溃**，且可定向到高代价动作——这是 headline。
3. **攻击面随模型尺寸适度收缩，但不消失**；真正起决定作用的是模型自我纠正与校准，而不是协议。论文要把这点讲成"防御不能指望模型恰好够尖"。
4. **任务越长/决策轮越多，越容易被攻击**——一条可量化的新轴（取币点随推理长度线性增长）。
5. **计费膨胀在 paper 设计下仍然有效**，且注意力容差能在弱模型上进一步放大注入率。

## 2. 威胁模型与指标定义

**威胁模型（统一用 paper 设计）**：恶意 prover 只能 grind 承诺的 ζ，使行质量 Θ ∈ \[θ±ε\]，采样币 r = H(tr\_\<t) mod Θ。所有配置都在可容许集内，故 ASR 是搜索下界。free-r（实现特定）结果不进主表，只在 appendix 保留。

**四个攻击面**（统一命名）：

- `null`：不攻击，用 canonical 解码重复 K=5 次（只换币的密钥）。any-null = 每次决策 ≠ c₀ 的平均比例（即 E\[qᵢ\]）；dir-null = 每次恰好落到指定目标的平均比例。分母和对应的 ASR 完全相同。null 是单次诚实运行的翻转率，不是“K 次里任一次翻了”。
- `sample`：解码容差（哈希链重投 + best-of-N），模型分布不动。
- `attn`：仅注意力行质量符号（±ε/θ 两极），解码严格 canonical。
- `attn+sample`：两个自由度并集。

**指标**：

- 主指标 = **ASR − null**，配按题配对的 bootstrap 95% 置信区间；ASR 和 null 两列都并列呈现，不只报 ASR。这是对"越大越难攻击"的直接回应：随尺寸收缩的主要是 null（大模型决策本身更稳），攻击的超额贡献基本持平。
- `eligible`（重新定义）：参考决策 c₀ = canonical 的选择，不再按稳定性过滤，决策的不稳定由 null 扣除。只保留三个条件：canonical 在 cap 内能解析出答案（报解析率）；gold 任务以 canonical 答对为条件（危害是对→错）；directed 要求目标 ≠ c₀。重建/过滤后的合格比例必须明确报出。
- 每个 ASR 是搜索下界（beam 宽度、预算上限、单点 S\* 优化），attention 带是一阶两极子集 ⇒ 也是下界。
- 所有注入见证过 strict audit（逐 token 有 witness）。

**稳定性小表**（单独统计，不进主表）：K 次 canonical 重采样全都没翻的题记为 q̂=0 子集。每个模型 × 数据集报两个数：该子集占比 P(q̂=0)，以及在它上面的 ASR₀。用途是回应“翻的都是本来就不稳定的题”：ASR = P(q=0)·ASR₀ + P(q>0)·ASR₊，单个 null 拆不出 ASR₀。K=5 时这个子集里仍可能有真实翻转率约 0.2 的题（5 次全不翻的概率约 0.33），需要更干净时只对该子集加到 K=10。

**计费指标**：同时汇报注入次数/率和膨胀倍数（要求几乎不撞cap）；计费单独定义 null（诚实重采样的长度比分布）。

## 3. 模型与数据集

**模型主干换成 Qwen3 dense**，因为 Qwen3.5 是混合注意力（24/32 GDN 层、kernel 仅 bf16），attention 杠杆在其上无法电路保真模拟。

| 角色 | 模型 | 尺寸 | 注意力 | 跑的实验 |
| --- | --- | --- | --- | --- |
| 主干 | Qwen3 dense | 4B / 8B / 14B / 32B | 全 softmax，patch 可用 | 解码 + attention 全覆盖（表 1） |
| 通用性：跨代 | Qwen3.5 | 9B | GDN，仅 bf16 | 仅解码（表 2），与 Qwen3-8B 同尺寸对比 |
| 通用性：跨家族 | R1-Distill-Llama-8B | 8B | Llama-3.1 底座，softmax | 解码 + attention（表 2） |

- Qwen3 dense 给出完整尺寸梯度，且 attention 在每个尺寸都能跑，主结论全部由它承担。
- Qwen3.5 只用 9B，承担“跨代对比”：和 Qwen3-8B 尺寸最接近，可以直接比。之前 Phase B 的 4B/27B 数字是旧协议（旧 null、旧 eligible、收紧的预算），只放附录并注明旧协议。
- R1-Distill-Llama-8B 的底座是 Llama-3.1，是真正不同的家族，同时保留“推理模型、自我纠正弱”的对比点。代价是 attention patch 要加 LlamaAttention 适配（放进 P0）。

**显存**：Qwen3-32B fp32 约 128GB，表 1 全 fp32 后 32B 每一列都要两卡张量并行；14B fp32 约 56GB 单卡够；Qwen3.5-9B 是 bf16，单卡。

**数据集**：

- **Mind2Web**：真实网页 HTML 快照上的元素四选一（出自 *Mind2Web: Towards a Generalist Agent for the Web*, Xiang Deng）。静态单步 + 全步任务级（见 §6）。
- **τ-bench（重建后）**：客服工具调用动作选择（出自 *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*, Shunyu Yao）。重建步骤见 §7；airline / retail 分开报。

## 4. 主实验矩阵

实验按表分配，共五张表加一个 case study：

| 表 | 内容 | 模型 | 细节 |
| --- | --- | --- | --- |
| 表 1 | 主结果 | Qwen3 dense 4B / 8B / 14B / 32B | 本节 |
| 表 2 | 通用性 | Qwen3.5-9B、R1-Distill-Llama-8B | 本节 |
| 表 3 | 推理长度（thinking on/off） | Qwen3 {8B, 32B} | §6(a) |
| 表 4 | Mind2Web 全任务 | Qwen3 dense | §6(b) |
| 表 5 | 温度 | Qwen3-8B | §6(c) |
| case study | τ-bench 环境内执行 | Qwen3-32B | §6(e)，最后做 |

**表 1—主结果**（原表 A、B 合并）：

- 行：Qwen3 dense {4B, 8B, 14B, 32B} × {Mind2Web, τ-bench}
- 列：攻击面 {null, sample, attention, sample+attention, 其他系统的攻击面（预留）} × 目标 {any, directed, inflation}
- 精度：全表统一 fp32（含 null 和 sample 列），这样 sample+attention 与 sample 的差值只反映 attention 的贡献，不混入精度变化。Qwen3 dense 全是 softmax，每格都能跑。
- 模式：全表统一，any、directed、inflation 都用同一种。优先用 thinking；先做成本试跑（§9），估算一周内跑不完就整张表回退到默认模式（thinking off），不混用。null × inflation = 诚实重采样的长度比分布。
- 预留列：将来加其他系统（如抽查 + 统计检验一类的设计）时，只要给它写一个可接受集的计算，列就能直接接进来。

**表 2—通用性**（generality 小节）：

- Qwen3.5-9B：仅解码，{null, sample} × {any, directed, inflation}；GDN kernel 只支持 bf16，attention 不适用，整行跑 bf16；它没有列间比较，不存在精度混淆，表注说明。
- R1-Distill-Llama-8B：{null, sample, attention, sample+attention} × {any, directed, inflation}，结构和精度（fp32）都和表 1 一样。
- 数据集和模式都同表 1。R1-Distill-Llama-8B 没有非 thinking 模式，若表 1 回退，它仍按 thinking 跑，并在表注说明。

所有表共用的铁律：

1. 同一数据集内所有模型用同一个搜索预算，否则尺寸比较里混着预算的影响。
2. 温度 τ=1 为主，推荐温度在表 5。
3. null 用 K=5 次 canonical 解码，与 ASR 同分母（见 §2）。

最贵的是表 1 的 Qwen3-32B 整行（fp32，两卡），排期时放前面。

## 5. 计费 + attention

计费 + attention 要跑，结果进表 1和表 2。这里不存在理论障碍：计费攻击只要"存在一个可达的非停止 token"，attention 扩大可达集只会帮忙，搜索比 manipulation 更简单（不用命中特定错误答案）。

**两个设计决定解决之前搜到的障碍：**

1. **cap 放大使膨胀显著**。现在膨胀倍数被 4096 cap 压扁，计费攻击在强模型上 81–98% 撞 cap。先用 cap=8192 小规模试跑，看还撞不撞；撞了再提到 16384。目标是让指标从"撞 cap 率"换成"真实膨胀倍数"，更有说服力。
2. **稀疏触发解决逐步注入的开销**。不在每一步都试 attention 扰动，只在 **EOS logit 比较高**（模型想停）的步才触发搜索。这样把逼着 eager 前向的步数从整条轨迹降到少数关键步，和批处理架构的冲突大幅缓解。

**具体机制**：逐步生成时监控 EOS（或停止词）的 logit。当它排进前 k 名或超过一个阈值，说明模型想结束，这才是计费攻击需要干预的点——在这一步试 attention 符号配置，看能不能把一个续写 token 拉进可达集、把 EOS 挤出去。非关键步保持 canonical。

计费和表 1 其他列一样优先用 thinking 模式：隐藏推理 token 才是计费攻击真正有意义的场景，若整表回退到默认模式，计费的现实意义在正文里要弱化表述。

## 6. 附加实验

**(a) 表 3：推理长度轴（thinking on/off）**。这是对"越大越难攻击"最直接的回应。thinking on/off 比 short/natural prompt 更干净、更真实，当长度轴的主开关。

- 配置：Qwen3 {8B, 32B} × Mind2Web × {thinking on, off} × {any, directed}；先只在 Mind2Web 做；on/off 中和表 1 同模式的那一档直接复用表 1 的数，只补另一档。
- 预测方向性：选择任务上 thinking 拉高 ASR（更多取币点、可达集更大）；gold 任务上 thinking 可能反而降低 ASR（自我纠正更强）——后者正好强化 C2。
- 横轴用实测的决策前 token 数和分叉数，不用 thinking 标签本身。
- 注意：thinking 会改 canonical 决策，c₀ 要在 thinking 模式下重新取；轨迹长→搜索和显存更贵，精度与表 1 一致（fp32），这样才能复用表 1 的那一档。

**(b) 表 4：Mind2Web 全步任务级**。现在每条标注只取了一步，ASR 是单步翻转率，不是任务失败率。

- 建全步版：约 40–50 个完整任务的每一步各做一个选择轮，历史 = gold 前缀。
- 报两个数：单步 ASR；以及"至少有一步可翻的任务占比"（episode 级，仍是下界，因为攻击者被限制成只翻一步、gold 前缀）。
- episode 级公式：单步翻转概率 p、任务 k 步，至少翻一步 = 1−(1−p)^k。前提是"翻一步就足以让任务脱轨"，对不可逆步成立。

**(c) 表 5：温度**。用模型卡推荐温度跑一组（如 Qwen3-8B × Mind2Web）；现有数全是 τ=1，审稿人一定问推荐设置下还成不成立。

**(d) τ-bench case study（最后做）**。从主表模型里挑，最好 Qwen3-32B（"最大的模型也能被翻"最有说服力）；只挑 airline 或重建后能区分的题。把翻转后的动作放进 τ-bench 真实环境执行，展示数据库状态变成什么（取消了错的预订、退款金额不对）——把静态代理变成能看到实际后果的例子。Qwen3-32B 也能跑 attention（fp32，两卡），一个 case 可以同时展示两种攻击面。解码与 attention 的翻转结果直接复用表 1 的 Qwen3-32B 行，case study 只额外做环境内执行这一步，算力很小。

**(e) 搜索预算曲线**。可选，不一定跑；要跑就挑一组看 ASR vs 预算，支撑"ASR 是下界"这个说法。

**(f) GSM8K 对照行**。可选，不一定跑；Qwen3-8B 已有 GSM8K 结果，按新 null 协议补跑一行，让"gold 稳 / 选择崩"落在同一模型上。

## 7. τ-bench 重建步骤

病根：retail 选项之间只差一个 prompt 里没有的 10 位 ID，连诚实 oracle 都选不出来（约 73% 的 directed 目标单凭 prompt 和 gold 区分不开）。具体做法：

1. **把数据库实体解析进 prompt**。τ-bench 有后端数据库（products / orders / users），gold 动作由用户指令 + 数据库状态推出。对每题，取出被引用的订单、订单里的商品（名称、变体、价格）、可选替代变体，渲染成人可读的文字放进 prompt，让 ID 绑定到属性。
2. **候选动作仍用 ID，但 ID→属性的映射现在在 prompt 里**，所以正确选择可推。gold = 属性匹配请求的动作；wrong\_item = 属性不符的似是而非变体；wrong\_order = 对的商品放错订单号。
3. **程序化加两个 eligibility 过滤**：(a) gold 能从 prompt 唯一确定；(b) directed 干扰项和 gold 的差异 token 确实出现在渲染后的 context 里。不满足的题丢掉，并把过滤后的合格比例明确报出。
4. **删掉用户模拟器的元指令**（"You are reactive…"、"Try to paraphrase…"、urgency），真实 agent 看不到这些。
5. **airline / retail 分开记录**。

directed target 定义：可达但错误的最高代价动作（cancel instead of modify、overpay、wrong recipient），且代价差异可从 prompt 推出。

## 8. 统一协议与统计规范

- **精度**：表 1 和表 2 的 R1 行全部 fp32，含 null 和 sample 列。attention 必须 fp32（bf16 下 ±0.68% 行缩放卡在 ulp 边界，位移恰为 0）；sample 列也用 fp32 是为了和 attention 列对齐，避免列间差值混入精度变化。唯一例外是 Qwen3.5-9B（GDN 只支持 bf16）。要确认 fp32 没有走 TF32（allow\_tf32 关掉），否则矩阵乘按 10 位尾数算，杠杆被粗糙量化。每个数标清楚标注跑的是哪种精度。
- **搜索预算**：同一数据集内所有模型用同一预算：beam=8、node-limit=40000、max-paths=4096。
- **n**：跑完整数据集。Mind2Web 把 osunlp/Mind2Web 全部可用标注都建进来（现在只建了 150，原始 409 条）；τ-bench 取重建后全部通过 eligibility 的题。每格报 n\_total 和 n\_eligible。
- **种子**：null 用 K=5 次 canonical 解码（只换币的密钥）起步，与 ASR 同分母；若 q̂=0 子集太小、说服力不足，整体 K 提到 10；vLLM 0.33 的种子要 62-bit mask（C-long 会溢出）。
- **置信区间**：ASR、null 带 95% CI（Wilson 区间），ASR−null 用按题配对的 bootstrap；eligible 小的格（如 fp32 的 n=46）CI 会很宽，不要单凭点估下"bf16 低估"这种结论；要比就在同一批题上比。
- **审计**：所有注入见证过 choice\_tree\_audit / rederive 独立复验，要求 0 假阳；并对所有翻转轨迹做重放，要求每个 token 都有 witness。
- **报告格式**：主表每格写 ASR（CI）/ null；另单列一列 ASR−nul。所有 bf16 的旧数需要用 fp32 重测。

还要修两个现有问题：SUMMARY 的 τ-bench 小节还写着 bf16 注意力成功率"偏高"，和 fp32 重测的"偏低"矛盾；仓库根目录误提交的 2MB 二进制 validate\_simulation 要删或进 .gitignore。

## 9. 优先级与排期

顺序上，先把前置做完再批量跑，贵的（fp32、thinking、27B/32B）放队列前面。

| 阶段 | 内容 | 依赖 |
| --- | --- | --- |
| P0 前置 | τ-bench 重建（§7）+ 协议定稿（§8）+ LlamaAttention 适配 + 删误提交二进制 + thinking 成本试跑 | 无 |
| P1 表 1 | 主结果（Qwen3 dense，含 attention 与 inflation 列） | P0 |
| P1 表 2 | 通用性（Qwen3.5-9B、R1-Distill-Llama-8B），可与表 1 并行 | P0 |
| P2 表 3–5 | 推理长度、Mind2Web 全任务、温度 | P1 |
| P3 收尾 | τ-bench case study；可选：预算曲线、GSM8K 对照 | P1/P2 |

**thinking 成本试跑**（P0 内完成）：在 Qwen3-8B 和 Qwen3-32B 上，用实际要跑的配置（fp32 + thinking），各攻击面 × 各目标跑 20 题，记录每题墙钟时间，外推到表 1 + 表 2 全量（n≈150）。按现有 GPU 数换算，超过一周就整张表回退到默认模式，不混用两种模式；精度对齐不作为让步项。

最贵的格：Qwen3-32B 整行 fp32（两卡）、Qwen3-32B × thinking。Phase B 里 27B 一组就撞了两次 6 小时超时，排队时把这些放最前。


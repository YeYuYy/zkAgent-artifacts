# P0 前置执行记录（full-scale 设计 v1）

Nov 16, 2026 · 对应 `reports/Next-step-full-scale.md` §9 P0 行。执行人:Weijie(协议作者 @Ye)。

## 完成项

### 1. 删误提交二进制 ✅

`validate_simulation`(2.1MB ELF,f93bd76 误入)已 `git rm --cached` 并进 `.gitignore`
(commit `956d7bc`)。源码仍在 `scale_harness/validate_simulation.py`。

### 2. τ-bench 重建(§7)✅ — `build_taubench2.py`

**病根修复**:v1 的 retail 选项只差 prompt 里没有的 10 位 ID;v2 把客户数据库渲染进
prompt——被引用订单(逐 item 属性+价格)、每个相关产品的**完整变体表**(item_id ↔ 属性
↔ 价格 ↔ 可用性)、支付方式(礼品卡余额)、在档预订(航班/舱位/行李/保险)、按
`search_direct/onestop_flight` 语义逐舱位报价的航班表、以及管辖该动作族的 policy 段。
选项文本保持裸 ID(v1 harness 零改动),ID→属性映射在 context 里。用户模拟器元指令
("You are reactive…"、"Try to paraphrase…"、"It is urgent.")已删。

**程序化 eligibility(漏斗已报,`taubench2-funnel.json`)**:
- (a) gold-unique:新 item 用**否定感知**的属性匹配(否定短语里的值不作正向约束)
  要求唯一可用可容许变体;旧/退货物按名称(+同名时属性)精确匹配 gold 集;航班按
  指令声明的偏好链(cheapest/fastest/earliest/latest,逐方向)唯一幸存;行李/金额/
  ID 逐字校验;**多写任务先用迷你模拟器把先前写动作应用到 DB 副本再渲染**(决策点
  状态真实)。
- (b) directed 与 gold 的每个差异 token 必须出现在渲染 context 里。

**产出**:`taubench2-airline-choice.jsonl` **15** 题、`taubench2-retail-choice.jsonl`
**37** 题(共 52),airline/retail 分开记录。自动校验全过:4 个互异选项、directed≠gold、
选项中每个 ID 都在 context 中。丢弃均为真实歧义(偏好并列、指令未声明属性、金额由
对话协商产生)。funnel:airline 29 写任务→15;retail 99→37(v1 retail 88 里 ~73%
directed 不可区分,不可比)。

**诚实边界**:仍是静态决策代理(无环境交互);(a) 的属性匹配是逐字+规范化子串,
指令用比较级措辞("larger capacity")的题被保守丢弃;金额/名称匹配做了单复数与
`wi-fi`~`wifi` 归一。52 题的 canonical 解析率与 c₀ 分布要到 GPU 首跑后才知。

### 3. 协议定稿(§8)✅ — harness 快照同步

- `choice_tree.py`:默认预算统一 **beam=8 / abs / node-limit=40000 / max-paths=4096**;
  `--dtype float32` 默认 + 引擎初始化时 `allow_tf32=False`(cuda.matmul 与 cudnn);
  **新 null 协议 `--null-k 5`**:K 次 canonical 解码只换币 salt(62-bit),逐题记
  `null_any_frac`/`null_dir_frac`/`null_all_stable`,截断/未解析计为翻转;旧
  second-honest-sample 对照移到 `--legacy-control`(默认关);修复 check_honest 索引 bug。
- `choice_report.py`:Wilson 95% CI、ASR−null 按题配对 bootstrap(10k)、稳定性小表
  (P(q̂=0)、ASR₀);旧结果 JSON 向后兼容。
- Mind2Web 全量:`mind2web-choice-409.jsonl` = **354** 可用题(旧 150 为严格子集),
  已放共享存储。
- SUMMARY τ-bench 小节 bf16 措辞已修正(与 fp32 重测一致:**bf16 是低估方向的偏差**)。

### 4. LlamaAttention 适配 ✅(GPU 冒烟待做)

`attn_slack_vllm.py` 增加 `LlamaAttention`/`LlamaDecoderLayer`(两个在用 vLLM
0.33.0 / 0.11.0 均为 next 式 `forward(positions, hidden_states)`,逐类签名守卫);
attach() 为 Llama 模块预分配符号缓冲;clear() 完整还原;Qwen 路径不变。stub 单测
9/9(`tests/test_attn_slack_llama.py`)。真 Llama 前向 + attn 臂端到端待 R1-Distill-Llama-8B
下载完成后在 H200 冒烟。

### 5. thinking 成本试跑(§9)✅ 完成 —— 判定:thinking 模式保留(满足 <1 周)

实测(fp32 + thinking,beam8/abs/4096,null-k=5 含在内,20 题/组,M2W-409):

| 组 | 墙钟 | 每题 |
|---|---|---|
| Qwen3-8B any(1×H200) | 87 min | 4.4 min |
| Qwen3-8B directed(1×H200) | 85 min | 4.3 min |
| Qwen3-32B any(2×H200 TP) | 178 min | 8.9 min |

外推(全量 n:M2W 354、τ-bench v2 52;表 1+表 2 解码面):4B≈0.5×、8B=1×、
14B≈1.6×、32B=8.9min/题、R1-Llama-8B≈8B、Qwen3.5-9B≈8B ⇒ **解码面 ≈7 GPU-天,
加 attention 臂 ≈14 GPU-天;现有 3×H200(1+2)≈ 4.7 天墙钟 < 1 周** ⇒ 按设计
§9 判定**不回退,整表 thinking on**;若执行中出现 >30% 滑移(超时/排队),按预案
整表回退默认模式并重跑(已跑的 20 题试跑数据可复用为表 3 种子)。

**试跑顺带的首批结果(n=20,搜索下界,无 audit 结论,仅成本试跑口径)**:
- 8B any:ASR-c₀ 0.74 [0.51,0.88](eligible 19/20),null_any 0.19 [0.12,0.28],
  null_dir 0.017,ASR−null +0.55 [0.27,0.79];P(q̂=0)=0.53,ASR₀=0.70。
- 8B directed:ASR-c₀ 0.45,null_any 0.29,null_dir 0.092,ASR−null +0.16。
- 32B any(16 题中):ASR-eligible 0.77;null_any ≈0.26 —— thinking 下 32B 的
  null 并不比 8B 低(小样本,全量表定论)。
- 新 null-k 协议全链路在 GPU 验证通过(逐 token witness,0 失败);发现并修复
  null_dir 统计把 c₀==directed 的题计进 directed 的口径 bug(`9fe58c9`)。

## 排队中的后续(P1 起点)

判定已定,按"最贵的排前面"开跑表 1(worker2=2×H200:Qwen3-32B 整行;worker1=1×H200:
14B → 8B 续跑(试跑 20 题自动复用)→ 4B;全部 fp32 + thinking + null-k=5,预算
beam8/abs/40000/4096)。τ-bench v2 首跑 smoke(20 题 retail)进行中——通过后再上
全量;R1-Llama-8B attention patch 的 GPU 冒烟排在 smoke 之后。

## 模型下载(后台链)

Qwen3-32B(pinned `9216db57`)→ R1-Distill-Llama-8B(`6a6f4aa4`)→ Qwen3-4B → Qwen3-14B,
全部哈希校验(`download_model.py` 新增 pinned revisions)。**五个模型全部就位。**

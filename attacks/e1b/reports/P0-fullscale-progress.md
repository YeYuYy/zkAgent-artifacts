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

### 5. thinking 成本试跑(§9)⏳ 进行中

H200 worker 已起(`zk-agent-p0-trial`)。已排队:Qwen3-8B fp32+thinking × Mind2Web
409 版前 20 题 × {any, directed}(beam8/abs/4096 默认,max-new 2048),同跑即验证
新 null-k 协议。Qwen3-32B(下载中 ~45%)到位后补 32B 两卡试跑(`launch.py` 已加
`--gpus`,worker 单卡 assert 已放宽)。外推判定(>一周 ⇒ 整表回退默认模式)待计时。

## 模型下载(后台链)

Qwen3-32B(pinned `9216db57`)→ R1-Distill-Llama-8B(`6a6f4aa4`)→ Qwen3-4B → Qwen3-14B,
全部哈希校验(`download_model.py` 新增 pinned revisions)。

## 排队中的后续(P1 起点)

表 1(Qwen3 dense 4B/8B/14B/32B × M2W-409 + τ-bench v2,fp32,含 null-k 列)、
表 2(Qwen3.5-9B 已在 mas_security_h200;R1-Llama-8B 待下载)。τ-bench v2 的首跑
建议先 20 题 smoke 看 eligible/解析率,再上全量。

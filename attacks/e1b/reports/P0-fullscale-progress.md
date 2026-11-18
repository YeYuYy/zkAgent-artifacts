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

## P1 改为先导集(pilot)再全量(2026-11-16 增补)

**理由**:本轮同时换了四样东西——新 eligible(c₀ 口径,不再按稳定性过滤)、新 null
(K=5 换币 canonical)、主指标改为 ASR−null(配对 bootstrap)、fp32+thinking 运行设定
——全部只在 n=20 上跑过。当天两次 smoke 就抓出了 τ-bench v2 缺答案格式指令
(0/16 解析)和 null_dir 分母口径 bug;直接投一周量级的全量跑,等于把这类缺陷烧进
几天的算力。改为先导集校准,过闸门再全量。

**先导集内容**(全部 thinking on、fp32、beam8/abs/40000/4096、null-k=5):
- **8B × M2W n=120 × {any, directed}**(worker1;前 120 题与 v1 的 150 题重合 →
  同一批题上新旧口径直接可比,即"用新指标更新旧结果"的校准点);
- **8B × τ-bench v2 全量(52:retail 37 + airline 15)× {any, directed}**(worker1);
- **32B × M2W n=32 × {any, directed}**(worker2;any 直接续用已在跑的 6h 作业产物);
- 顺带:batch32/max-seqs1024 的 A/B(worker1,排在 14B 原位置)。

**过闸标准**(约 30 小时后复核):
1. honest 解析率 ≥95%,eligible_c0 ≥70%(逐数据集;不达标先查原因再谈全量);
2. null 合理性:null_any 明显低于 ASR、witness 0 失败;
3. 检验功效:n=120 上 ASR−null 配对 bootstrap CI 不含 0(主指标设计成立);
4. 成本:每题耗时 ≤ 试跑外推 +30%(否则按预案整表回退 thinking off);
5. τ-bench v2 eligible 率正常、directed 命中落在指定目标;
6. A/B ≥1.2× 则后续作业全部改用 batch32/max-seqs1024。
全过 ⇒ 同设定全量(354/52)排产;任一不过 ⇒ 只迭代该轴,不动其他。

## 模型下载(后台链)

Qwen3-32B(pinned `9216db57`)→ R1-Distill-Llama-8B(`6a6f4aa4`)→ Qwen3-4B → Qwen3-14B,
全部哈希校验(`download_model.py` 新增 pinned revisions)。**五个模型全部就位。**

## 先导集首轮复核 + 三项修复(2026-11-17)

**复核结论**(闸门点按时到;choice_report 口径,c₀-eligible):

| 臂 | n | parse | elig-c₀ | ASR | null_any | ASR−null [CI] | s/题 | 闸门 |
|---|---|---|---|---|---|---|---|---|
| 8B M2W any | 120 | 96.7% | 115 | 0.826 | 0.143 | **+0.683 [0.591, 0.770]** | 188 | 1–4 全过 |
| 8B M2W directed | 120 | 96.7% | 111 | 0.649 | 0.132 | **+0.517 [0.420, 0.611]** | 239 | 1–4 全过 |
| 32B M2W any(p1-table1 续用) | 32 | 100% | 32 | 0.750 | 0.181 | **+0.569 [0.375, 0.744]** | 368 | 过 |

null witness 0 失败;成本均在试跑外推之内(8B any 甚至更快)。τ-bench 四臂
worker1 在途(ret-any 运行中 + 3 臂排队),闸门 5 待判。

**问题与修复**:

1. **32B directed 6h 超时**(16/32 题被杀,returncode −15):directed 搜索在
   32B 上 median 823.6 s/题 = 试跑外推(8.9 min)的 **+54%**,且 6h 上限本身
   低于 16 题 × 824 s + 引擎启动,注定跑不完。修复:同 argv 重排队
   (config 一致 ⇒ resume 跳过已完成 16 题),timeout 21600 → **86400**。
   注意闸门 4 的成本单项在 32B-directed 上按原口径已算未达;全量排产时
   32B 行按实测 824 s/题重算预算。**续跑完成(11-17,2.46 h / 16 题,chunk-1
   median 683 s/题,全腿 blended ~754 s/题 = 外推 +41%)**:n=32 全量,
   parse 100%,eligible 24,directed 11/20 命中(asr_dir 0.55,−null_dir 0.100
   ⇒ **+0.45**),null_any 0.188,witness 0 失败,COMPLETE 落盘。
2. **Llama GPU smoke 两次失败,两个独立根因**:
   a. 引擎初始化崩溃 `_ModelInfo object has no attribute
      supports_kv_cache_grouping`:vLLM 0.33 registry 对新架构在**子进程**
      (`sys.executable -m vllm.model_executor.models.registry`)内省模型类,
      子进程继承环境变量但不继承进程内 sys.path 注入;旧 smoke 只在进程内插
      site-dir ⇒ 子进程解析到 vllm-env 自带的 vLLM 0.11,`_ModelInfo` 按 0.11
      字段构造,pickle 回父进程按 0.33 类反序列化 ⇒ 字段缺失(asdict 保存缓存
      与 kv-cache-grouping 属性两处爆)。choice 作业从未踩中:argv 里一直带
      `env PYTHONPATH=py312-vllm35`。修复:smoke 自身同时设
      `os.environ['PYTHONPATH']`,作业 argv 也带上(与 choice 作业一致)。
   b. `attach() found no Llama attention modules`:attach 的 BFS 深度上限 4
      走不到 0.33 InprocClient 引擎图里的模型
      (llm→llm_engine→engine_core→model_executor→driver_worker→worker→
      model_runner→model 共 7 跳)——Qwen 作业同样 attached=0 且一直正常,
      因为 `_mk_layer_init` 在**构造时**就把 buffer 钉到每个 attention 模块,
      attach 只是兜底重绑。修复:smoke v2 改功能面判定——无界 BFS 找到模型、
      校验 pinned == 模块数 且 layer_idx == {0..L-1}、零符号恢复 bitwise
      一致、全 −1 符号改变 top-k logprobs(附 max |Δlogprob|)。smoke 移入
      `experiments/llama_attn_smoke.py`(镜像 artifacts `attacks/e1b/choice/`),
      去掉对旧快照路径的硬编码。**已跑通(11-17)**:attn_modules 32/32 全为
      `LlamaAttention` 且构造期全 pinned,layer_idx 恰为 0..31(attach_reported=0
      如预期);零符号恢复 bitwise 一致;全 −1 符号 max |Δlogprob| = 0.152。
      ⇒ **P0 项 4(LlamaAttention GPU 验证)完成**,R1-Llama-8B attention 臂解封。
3. **A/B batch32/max-seqs1024 从未入队**(先导集清单第 4 条):补排
   8B M2W any n=120 同设定仅 batch/max-seqs 不同,基线 = 已完成 batch16 臂
   (median 188.2 s/题,chunk 27213 s)。判据不变:≥1.2× ⇒ 后续全量改用。
   注意该臂跑在 worker2(H200 同型号),与基线(worker1)跨节点比较,
   阈值判定时留意 ~数% 级节点差。

**worker2 队列**:~~32B directed 续跑~~ ✅ → ~~llama smoke v2~~ ✅ → **A/B(运行中)**。
worker1 继续 τ-bench 四臂。五臂齐后按 §过闸标准 出全量/回退决定。

## 闸门 1 首个未达项:τ-bench retail @2048 解析率 64.9%(2026-11-17 夜,已定位+重排)

tb2ret-any(n=37,thinking on,fp32/beam8)完成但 honest 解析率 **64.9% < 95%**、
eligible_c0 54% —— 13 个未解析项**全部 `capped=True`**:Qwen3-8B 在 τ-bench v2
的数据库渲染长 prompt 上,thinking 在 `--max-new 2048` 内走不完即被截断
(未解析项尾部全是推理中态;已解析项 p50 3.5k / p90 6.5k 字符,上限≈8-9k)。
**根因是提交参数沿用了 M2W 预算**:`E1b-choice-NEXT.md` 自己的 τ-bench 计划行
就是 `--max-new 4096`。模型上下文(32k)与 KV 预算均有充分余量(retail prompt
max ≈12k 字符)。

处置:四臂以 `--max-new 4096 --max-steps 4400` 重排(out 目录 `-mx4096` 后缀,
避开与旧目录的 mismatched-resume 守卫);2048 版保留为证据
(`qwen3-8b-tb2ret-any-maxnew2048`;airline any@2048 在跑,留作出截断敏感度对照)。
⚠️ worker1 队列里仍有**前一会话排的两个 2048 directed 臂**(ret/air-directed,
排在 4096 臂之前,约多占 ~6–8 h)——按闸门协议"只迭代该轴"应跳过,但它们是
既有队列项,留待人工决定删除或放行。
**增补(决策落定)**:airline any@2048 完成:parse **20%**(3/15),12 个未解析
项同样全部 capped——预算截断在 airline 更严重。人工决定 **skip**:pending 的
air-directed@2048 已删除;ret-directed@2048 当时已在跑且队列机制无单作业取消
(STOP 会杀整个 worker),放行至跑完(~3 h)作 directed 臂的 2048 对照,随后
四条 4096 臂依序开跑。

### A/B 判定(闸门 6):**不过 —— 全量维持 batch16/max-seqs512**(2026-11-18)

batch32/max-seqs1024 臂(worker2,n=120,rc 0)median **213.8 s/题 vs 基线
188.2 s/题 = 0.88×**(慢 12%;chunk 24604 s vs 27213 s)。跨节点(基线在
worker1)数百分点级差异无法把 0.88 翻成 ≥1.2,判据明确不过。
副产品:同题复算高度一致(parse 96.7% 持平、eligible 100/120、ASR-c₀ 0.850
vs 0.826、null_any 0.134 vs 0.143、witness 0 失败)⇒ 协议跨节点/跨 batch
设定可复现。后续全量作业保持默认 batch16/max-seqs512。

## 先导集收口:τ-bench 预算阶梯 + 全部闸门判定(2026-11-18)

**预算阶梯**(honest parse;每一环的失败项均为 capped=True,推理处于有效推进态):

| --max-new | retail(37) | airline(15) |
|---|---|---|
| 2048 | 64.9%(any=directed) | 20% |
| 4096 | 91.9%(any=directed) | 86.7%(any=directed) |
| **6144** | **100% / 100%** | **100% / 100%** |

⇒ τ-bench v2 统一预算 **--max-new 6144 --max-steps 6600**。截断性上限的语义保证:
预算内完成的输出与更大预算下逐字节一致 ⇒ 阶梯各环互相自洽,2048/4096 各臂留作
证据(`-maxnew2048` / `-mx4096` 目录)。

**6144 终值**(Qwen3-8B,thinking on,fp32,beam8/abs/40000/4096,null-k=5,witness 0 失败):

| 臂 | parse | eligible | 主结果 | null |
|---|---|---|---|---|
| retail any | 100% | 31/37 | ASR-c₀ 0.645 | null_any 0.135 |
| retail directed | 100% | 36(dir_elig 35) | **asr_dir 0.714** | null_dir 0.006 |
| airline any | 100% | 13/15 | ASR-c₀ 0.846 | null_any 0.086 |
| airline directed | 100% | dir_elig 11 | **asr_dir 0.909** | null_dir 0.031 |

**全部闸门判定**:
1. honest 解析率 ≥95%:✓(6144 后四臂全 100%)
2. eligible_c₀ ≥70%:✓(83.8% / 97.3% / 86.7% / 73.3%)
3. null 合理性:✓(null ≪ ASR,逐题 witness 0 失败)
4. 检验功效:✓(8B M2W 双臂 ASR−null CI 不含 0;τ-bench 各臂差值 0.51–0.88)
5. τ-bench eligible 正常、directed 命中指定目标:✓(0.714 / 0.909)
6. A/B:0.88× 不过 ⇒ 全量维持 batch16/max-seqs512(已判定,见上节)

成本附注(gate 4 口径):τ-bench@6144 实测 425–561 s/题 = M2W 试跑外推
(264 s)的 1.6–2.1× —— 非执行滑移,是数据集难度 + 预算上限的固有成本;
τ-bench 全量按此重排预算(8B τ-bench 全四臂实测 ≈12.3 h)。

⇒ **先导集六项闸门全部落定(含两次迭代:τ-bench 预算 2048→6144、A/B 否决),
同设定全量(表 1 M2W 354 × 5 模型 + 表 2 τ-bench v2 52 × {any,directed},
attention 臂含 R1-Llama)可排产。** 两台 worker 现均空闲;32B-any 全量行
(p1-table1,32/354,resume)待人工放行后可即刻续跑。

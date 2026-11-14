任务：把 attacks/e1b/decision/decision_bestofN.py 的决策操纵攻击从「注入一个错 token 后用 model.generate
自由续写 + 事后审计」改成「全程约束在每步可达集内的、有预算上限的树状搜索」。先读
attacks/e1b/SUMMARY.md 的 §E1b-5@7B/8B 和 attacks/e1b/reports/E1b-NEXT.md 的 A 节了解机制与动机。

== 背景（必须理解，否则会改错）==
- 采样币 r = H(tr_<\t) mod Θ 是对已提交前缀的哈希链（prefix_coin 给 H）。给定前缀，coin 确定，
  reach_paper(delta, shift, H) 在 in-band Θ 上 grind 出一个（paper 下通常很小的）可达 token 集。
- 攻击者的唯一自由度是「证明关系集值」：某步可达集 size>1 时，可替换成任一可达 token；一旦替换，
  前缀变化会重新确定其后所有 coin ⇒ 一条全新但仍确定的续写。不存在自由采样。
- 正确性不变式（本次改动的核心）：攻击轨迹里**每一个 emitted token 都必须 ∈ 该步的 reach_set**
  （free-r 用 reach_freeR，paper 用 reach_paper，均来自 e1b_length_steer.py）。满足它，整条轨迹按
  构造可容许，strict ≡ lenient，不再需要任何 prompt_logprobs 事后审计。当前用 model.generate 续写
  破坏了这条不变式，是 paper strict 掉 0 的真正来源。

== 要实现的算法：admissible tree search ==
新增一个可复用函数（之后要照搬到 GPU harness，所以不要和 argparse/模型加载耦合），签名类似：
    tree_attack(step_logits_fn, reach_fn, canon_fn, base_ids, correct, prompt, budget, cfg) -> dict
其中 reach_fn(logits, prefix_ids) 复用现有 reach_set 逻辑（free-r / paper 两种）。流程：

1) 规范轨迹：从 base 逐步解码。每步算 logits、reach=reach_fn(...)。canon_fn 取「honest/coin 选中的」
   token（harness 代理：取 reach 内 logit 最高者即可，但要在注释里标明这是对 coin 选择的代理）。
   发 canon，直到 term(EOS/</think>)/max-steps。记录所有 fork 候选：该步 reach 含 ≠canon 的数字
   token 的位置，连同其解码数字串、logit。规范轨迹答案应 == correct，否则该题跳过（同现有 honest 检查）。

2) 分组：把 fork 候选按「解码出的数字串」聚类成 quantity-group；标记数字 ∈ prompt 操作数
   （复用 operand_digits / target_restatement 逻辑）的组。组排序 heuristic（可配权重）：
     (a) 晚/流向答案：组内 mention 越靠后、越接近 \boxed 优先（下游自我纠正空间越小）；
     (b) 可一致翻：有 ≥2 个可达 mention 的组优先（能保持内部一致）。

3) 搜索（预算上限）：按排序遍历组，预算耗尽即停。对每组选一个错目标数字
   d' = 该组各 mention reach 内 logit 最高的错数字；生成候选轨迹：
     - 方案 A（整组一致）：在该组**每个可达 mention** 处把 d→d'，其余步发 canon；
     - 方案 B（仅最晚，fallback / 单 mention 组）：只在该组最后一个可达 mention 处 d→d'。
   生成时每步都重算 reach（前缀已变！），只允许发 reach 内 token，并对每个替换 assert ∈ reach。
   取终答：用 final_answer（保留现有「截断不计」逻辑）。若答案 concluded 且 != correct 即命中，记录
   (组, d', 方案, step, 答案)。可选：保留 steer 剪枝——某步要发 canon 但它是自我纠正词且 reach 有
   非纠正词替代时，改发替代（CORRECTION 列表已有）。

4) 预算：cfg 暴露 --budget（总解码 token 上限）与 --max-groups、--max-forks；best-first 按组排序消费。

== 其他改动 ==
- 新增 rollout/search 模式 `--search tree`，与现有 greedy/sample/steer 并存（后者保留做对比，别删）。
- 删除对「事后审计」作为可容许性来源的依赖；改为在 tree 路径上 assert 每个 emitted token ∈ reach，
  作为自检（永远应通过）。若文件里没有 prompt_logprobs 审计就无需改这部分。
- attn-on 分支（union_reach）保持可选，逻辑不变。
- 汇总输出：每题打印命中的 (组/值/方案/step/答案)，末尾打印 ASR = flipped/attackable；因为是有预算
  搜索，ASR 注释为**下界**（加预算只增不减）。

== 测试 ==
新增 tests/test_tree_attack.py（可用极小 fake logits/stub reach_fn，不必加载真模型）：
  (1) 不变式测试：对一条生成轨迹，断言每个 emitted token 都 ∈ 对应步 reach；
  (2) 一个构造用例：reach 在某数字步含错值 d'，断言 tree_attack 能找到并返回 ok=True；
  (3) paper 退化用例：所有步 reach 为单元素（无 fork）时，断言返回 ok=False 且未发任何 reach 外 token。

== 不要做 ==
- 不要在 fork 下游用 model.generate 自由采样/贪心生成后再审计——这正是要移除的 bug。
- 不要把 argmax 当 paper 模式的 canon（argmax 可能 ∉ reach_paper）；canon 必须取自 reach。
- 不要改 e1b_length_steer.py 里的 quant/active_and_shift/reach_* 数值逻辑。

提交信息末尾加：
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cccc9nVTDT9u2EUFqxEJgf

---
## Status (2026-11-08) — DONE

- Reference: `attacks/e1b/decision/tree_attack.py` (reusable `tree_attack(...)`, torch-free,
  invariant asserted per step) + `decision_bestofN.py --search tree` (KV-resuming
  stepper, canon = argmax-within-reach proxy); tests in `tests/test_tree_attack.py`
  (invariant / constructed flip / paper-degenerate singleton reach).
- GPU harness port: `experiments/tree_attack.py` — speculative-window canon decoding
  (batched greedy windows, per-step walk, divergence resolved from the same window's
  logits; no per-token engine calls). Runs: Qwen3-8B / R1-7B x GSM8K / MATH-500 L4-5,
  tau=1, n=150, both modes; 58-80 min each; audit_results + rederive pass.
- Headline (budgeted LOWER bounds, admissible by construction so strict == lenient):
  R1-7B GSM8K free-r 14/131 = 10.7% (9 via plan A); R1-7B GSM8K paper 1/131 = 0.8%
  (first fully-admissible paper flip at 7B); R1-7B MATH free-r 2/77 = 2.6% (both
  operand groups); Qwen3-8B zero everywhere. See SUMMARY.md sect. E1b-tree.

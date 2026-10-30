// Verification of the "dense attention slack" claim against the released
// attention softmax verifier (verify_attention_softmax_row).
//
// zkAgent commits a prover-chosen shift zeta per attention row and only checks
// that the row mass lies in [theta - eps, theta + eps]. Each admissible zeta
// rescales that row's attention output. This driver shows that the UNMODIFIED
// verifier accepts several distinct zeta for one score row, each yielding a
// different (verified) attention-row mass / output.

#undef NDEBUG
#include "model/attention_softmax.hpp"
#include "model/quant_plan.hpp"

#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <vector>

using std::vector;

static std::ostream &operator<<(std::ostream &os, __int128 v)
{
    if (v == 0) return os << '0';
    bool neg = v < 0; unsigned __int128 u = neg ? -(unsigned __int128)v : v;
    char buf[48]; int n = 0;
    while (u) { buf[n++] = char('0' + int(u % 10)); u /= 10; }
    if (neg) os << '-';
    while (n) os << buf[--n];
    return os;
}

int main(int argc, char **argv)
{
    int seed = 0, row_len = 8;
    for (int i = 1; i < argc; ++i)
    {
        std::string o = argv[i];
        if (o == "--seed") seed = std::atoi(argv[++i]);
        else if (o == "--row-len") row_len = std::atoi(argv[++i]);
    }
    // A representative GPT-2 attention plan (Q=20), mirroring quant_plan defaults.
    QuantAttentionPlan plan;
    plan.score_scale = 7.952219286828541e-10;
    plan.score_bound = static_cast<ll>(84380505604096LL);
    plan.theta = static_cast<ll>(1) << 20;
    plan.digit_bits = {10, 10, 3, 11, 10, 4};
    plan.shift_upper_bound = (static_cast<ll>(1) << 47); // < 2^(sum bits)=2^48
    plan.least_segments = 3;
    plan.nonlinear_segment = 3;
    plan.most_zero_segments = 2;
    // Row-mass tolerance per row length (index n-1). Use a uniform band.
    const ll eps = 7088;
    plan.epsilon_by_count.assign(64, eps);

    PreparedAttentionSoftmax prep(plan);

    // A score row bounded by score_bound. seed=0 keeps a fixed illustrative row;
    // other seeds randomize it for the robustness sweep.
    vector<ll> scores;
    if (seed == 0)
        scores = {120000, -35000, 80000, 15000, -90000, 42000, 5000, -12000};
    else
    {
        std::mt19937_64 rng((uint64_t)seed);
        std::uniform_int_distribution<int64_t> dist(-200000, 200000);
        scores.resize(row_len);
        for (auto &s : scores) s = dist(rng);
    }

    auto build_with_zeta = [&](ll override_zeta, bool honest) -> AttentionSoftmaxRowWitness
    {
        static ll g; g = override_zeta;
        attention_softmax_zeta_override = honest ? nullptr : +[](ll) { return g; };
        AttentionSoftmaxRowWitness w = prep.build(scores);
        attention_softmax_zeta_override = nullptr;
        return w;
    };

    AttentionSoftmaxRowWitness honest = build_with_zeta(0, true);
    if (!honest.success)
    {
        std::cout << "honest build failed: " << honest.message << "\n";
        return 1;
    }
    const ll zeta0 = honest.zeta;
    std::cout << "theta=" << plan.theta << " eps=" << eps
              << " band=[" << (plan.theta - eps) << "," << (plan.theta + eps) << "]\n";
    std::cout << "honest zeta0=" << zeta0 << " mass=" << honest.total_mass
              << " verifier " << (prep.verify(honest, scores) ? "ACCEPTS" : "REJECTS") << "\n\n";

    // Sweep committed zeta around the honest value; report each witness the
    // UNMODIFIED verifier accepts, with its (distinct) attention-row mass.
    std::set<ll> accepted_masses;
    int accepted = 0, shown = 0;
    for (ll d = -4000000; d <= 4000000; d += 1000)
    {
        AttentionSoftmaxRowWitness w = build_with_zeta(zeta0 + d, false);
        if (!w.success) continue;
        if (!prep.verify(w, scores)) continue; // the real acceptance predicate
        ++accepted;
        if (accepted_masses.insert(w.total_mass).second && shown < 8)
        {
            std::cout << "  ACCEPTED committed zeta=" << (zeta0 + d)
                      << "  row mass=" << w.total_mass
                      << "  (output scales with mass/theta)\n";
            ++shown;
        }
    }
    std::cout << "\naccepted distinct committed zeta sweep hits: " << accepted
              << " ; distinct verified row masses: " << accepted_masses.size() << "\n";
    std::cout << "RESULT: "
              << (accepted_masses.size() > 1
                      ? "DEMONSTRATED. verify_attention_softmax_row accepts many zeta for one row,\n"
                        "each a different verified attention output -> attention relation is set-valued."
                      : "not demonstrated.")
              << "\n";
    return 0;
}

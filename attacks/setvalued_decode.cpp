// Verification of the decision-integrity claim against the released decode
// verifier (zkDecode top-k / "tolerance-checked softmax normalizer").
//
// Question: for ONE committed logit vector and ONE fixed prefix, does the
// UNMODIFIED verifier accept more than one outcome?
//
// The paper's coin is r = H(tr_<t) mod Theta, where Theta is the committed
// total mass. The released verifier (check_decode_sampling_claims) only checks
// Theta in [theta - eps, theta + eps]. Theta is a deterministic function of the
// prover-committed shift zeta, so each admissible zeta yields a different Theta
// and therefore a different coin r for the SAME prefix. If two admissible Theta
// values route a fixed r to different tokens, decision integrity fails.
//
// This driver changes NOTHING in the verifier. It only varies the prover's
// committed shift (via decode_exp_digits_shift_override) and runs the real
// prover + real verifier end to end.

#undef NDEBUG

#include "common/bit_utils.hpp"
#include "common/field_integer.hpp"
#include "common/omp_threads.hpp"
#include "pcs/hyrax.hpp"
#include "pcs/srs.hpp"
#include "zkdecode/zkdecode_exp_digits.hpp"
#include "zkdecode/zkdecode_prover.hpp"
#include "zkdecode/zkdecode_verifier.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <map>
#include <set>
#include <vector>

#include <mcl/bn256.hpp>

using namespace mcl::bn;
using std::vector;

// ll is __int128; give it a stream inserter for the small values we print.
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

namespace
{
    constexpr ll theta = ZkDecodeProbabilityScale; // 2^20

    // Honest top-k selection, identical to the prover's buildTopKSelection.
    void select_topk(const vector<ll> &logits, int k, vector<ll> &deltas, vector<int> &active)
    {
        const int vocab = (int)logits.size();
        vector<int> order(vocab);
        std::iota(order.begin(), order.end(), 0);
        std::nth_element(order.begin(), order.begin() + k - 1, order.end(),
                         [&](int a, int b)
                         { return logits[a] != logits[b] ? logits[a] > logits[b] : a < b; });
        const ll separator = logits[order[k - 1]];
        deltas.resize(vocab);
        active.assign(vocab, 0);
        for (int i = 0; i < vocab; ++i)
            deltas[i] = logits[i] - separator;
        for (int r = 0; r < k; ++r)
            active[order[r]] = 1;
    }

    // Build the per-token mass vector for a given committed shift, using the
    // real witness builder. Returns empty if the witness is rejected.
    vector<ll> mass_for_shift(const vector<ll> &deltas, const vector<int> &active, ll shift)
    {
        decode_exp_digits_shift_override = nullptr;
        static ll g_shift;
        g_shift = shift;
        decode_exp_digits_shift_override = [](ll) { return g_shift; };
        DecodeExpDigitsWitness w = build_decode_exp_digits_witness(
            deltas, active, theta, DecodeExpDigitsConfig{});
        decode_exp_digits_shift_override = nullptr;
        vector<ll> mass;
        if (!w.success)
            return mass;
        mass = std::move(w.unmasked_outputs);
        for (size_t i = 0; i < mass.size(); ++i)
            if (!active[i])
                mass[i] = 0;
        return mass;
    }

    ll total(const vector<ll> &m) { return std::accumulate(m.begin(), m.end(), (ll)0); }

    // Token whose cumulative-probability interval contains the coin r.
    int token_for_coin(const vector<ll> &mass, ll r)
    {
        ll prefix = 0;
        for (size_t i = 0; i < mass.size(); ++i)
        {
            if (r < prefix + mass[i])
                return (int)i;
            prefix += mass[i];
        }
        return -1;
    }
}

int main(int argc, char **argv)
{
    int vocab = 256, k = 40, seed = 20260421, threads = 4;
    LookupBackend backend = LookupBackend::LogUpGkr;
    for (int i = 1; i < argc; ++i)
    {
        std::string o = argv[i];
        auto v = [&]() { return std::atoi(argv[++i]); };
        if (o == "--vocab") vocab = v();
        else if (o == "--k") k = v();
        else if (o == "--seed") seed = v();
        else if (o == "--threads") threads = v();
        else if (o == "--backend")
            backend = std::string(argv[++i]) == "shout" ? LookupBackend::Shout : LookupBackend::LogUpGkr;
    }

    const ScopedOmpThreads thread_scope(threads);
    initPairing(mcl::BN_SNARK1);

    const ll bound = (ll)1 << 20;
    std::mt19937_64 rng((uint64_t)seed);
    std::uniform_int_distribution<int64_t> dist(-(bound / 8), bound / 8);
    vector<ll> logits(vocab);
    for (auto &x : logits) x = dist(rng);

    vector<ll> deltas; vector<int> active;
    select_topk(logits, k, deltas, active);

    // Honest shift: the un-overridden witness reports it in w.shift.
    decode_exp_digits_shift_override = nullptr;
    DecodeExpDigitsWitness honest_w = build_decode_exp_digits_witness(
        deltas, active, theta, DecodeExpDigitsConfig{});
    assert(honest_w.success);
    const ll zeta0 = honest_w.shift;

    // The band the verifier enforces: Theta in [theta - eps, theta + eps].
    const ll eps = decode_exp_digits_recommended_epsilon(theta, k, DecodeExpDigitsConfig{});

    // A large fixed prefix hash H, as in r = H(tr_<t) mod Theta. Varying Theta
    // within the band changes the residue r, hence the sampled token.
    const ll H = ((__int128)0x5DEECE66D1234567LL << 40) ^ 0x9E3779B97F4A7C15LL;

    const ll masked0 = total(mass_for_shift(deltas, active, zeta0));
    std::cout << "theta=" << theta << " eps=" << eps
              << " band=[" << (theta - eps) << "," << (theta + eps) << "]\n";
    std::cout << "honest zeta0=" << zeta0 << " honest masked Theta0=" << masked0 << "\n";

    // Enumerate admissible committed shifts: the witness must build and Theta must
    // land in the verifier's band. Theta decreases by ~1 integer per ~1/(alpha)
    // units of zeta, so we step. Collect distinct (Theta, token-under-fixed-H).
    std::set<ll> distinct_theta;
    std::set<int> reachable_tokens;
    vector<std::pair<ll, ll>> admissible; // (zeta, Theta)
    std::map<int, ll> token_to_zeta;      // one witness per reachable token
    const ll step = 300;
    const ll span = 40000000; // wide enough to traverse the band
    for (ll d = -span; d <= span && distinct_theta.size() < 20000; d += step)
    {
        const vector<ll> m = mass_for_shift(deltas, active, zeta0 + d);
        if (m.empty()) continue;
        const ll Th = total(m);
        if (Th < theta - eps || Th > theta + eps || Th <= 0) continue;
        if (distinct_theta.insert(Th).second)
            admissible.emplace_back(zeta0 + d, Th);
        const ll r = ((H % Th) + Th) % Th;
        const int tok = token_for_coin(m, r);
        if (tok >= 0 && reachable_tokens.insert(tok).second)
            token_to_zeta[tok] = zeta0 + d;
    }

    std::cout << "admissible committed shifts found (sampled at step " << step
              << "): " << admissible.size()
              << " ; distinct in-band Theta values K>=" << distinct_theta.size() << "\n";
    std::cout << "for one fixed prefix hash H, distinct reachable tokens: "
              << reachable_tokens.size() << "\n";

    if (admissible.size() < 2)
    {
        std::cout << "RESULT: only one admissible outcome -> claim NOT demonstrated here.\n";
        return 0;
    }

    // End-to-end: run the REAL prover + REAL verifier for two different admissible
    // committed shifts and show the UNMODIFIED verifier accepts BOTH, with the
    // proofs committing to different Theta.
    const std::size_t qc = (std::size_t)vocab;
    const int qvars = MVKZG_SRS::min_vars_for_size(std::max<std::size_t>(qc, 1));
    const int col_vars = (10 + qvars + 1) / 2;
    SRSPlan plan; plan.mv_vars = std::max(15, col_vars);
    SRSBundle srs; srs.setup_cached(plan, "cache");

    auto run_one = [&](ll committed_zeta, ll &out_theta) -> bool
    {
        static ll g; g = committed_zeta;
        decode_exp_digits_shift_override = [](ll) { return g; };

        ZkDecodeProver prover(srs.mv.pk, srs.mv.vk, threads);
        prover.setLookupBackend(backend);
        prover.setSoftmaxEpsilon(eps);

        // Honest sampler would derive the coin from the committed Theta; here we
        // just need any in-range sample so the selector proof is well formed. Use
        // mass under the committed shift to pick a valid interval.
        const vector<ll> m = mass_for_shift(deltas, active, committed_zeta);
        out_theta = total(m);
        const ll sample = ((H % out_theta) + out_theta) % out_theta;
        const int out_tok = token_for_coin(m, sample);

        vector<vector<ll>> L = {logits};
        vector<int> outs = {out_tok};
        vector<ll> samples = {sample};

        ZkDecodeParameters p;
        p.kind = ZkDecodeProofKind::TopK;
        p.lookup_backend = backend;
        p.token_count = 1; p.vocab_size = vocab; p.k = k;
        p.bound = bound; p.theta = theta; p.epsilon_sm = eps;
        ZkDecodeVerifierKey key = prepareZkDecodeVerifierKey(p, srs.mv.pk, srs.mv.vk, threads);

        ZkDecodeStatement stmt;
        stmt.parameters = p;
        stmt.output_tokens = outs;
        stmt.sample_values = samples;
        // Public commitment to the logits, independent of the proof.
        {
            const int tb = zkagent_integer::ceil_log2(1);
            const int vb = zkagent_integer::ceil_log2(vocab);
            const int vars = tb + vb;
            const std::size_t width = (std::size_t)1 << vb;
            vector<ll> padded(((std::size_t)1 << tb) * width, 0);
            std::copy(logits.begin(), logits.end(), padded.begin());
            const std::size_t cols = (std::size_t)1 << (vars - vars / 2);
            vector<G1> gens(srs.mv.pk.srs_g1.begin(), srs.mv.pk.srs_g1.begin() + cols);
            G1 *c = hyrax::commit(padded.data(), gens.data(), vars, threads);
            stmt.logit_commitments.assign(c, c + ((std::size_t)1 << (vars / 2)));
            delete[] c;
        }

        ZkDecodeProveResult gen = prover.proveTopKBatch(L, outs, k, samples, bound);
        decode_exp_digits_shift_override = nullptr;
        if (!gen.stats.success)
        {
            std::cout << "  prover failed: " << gen.stats.message << "\n";
            return false;
        }
        ZkDecodeVerifyStats vs;
        bool ok = verifyZkDecodeProof(stmt, gen.proof, key, srs.mv.vk, &vs);
        std::cout << "  committed zeta=" << committed_zeta
                  << " Theta=" << out_theta
                  << " token=" << out_tok
                  << " -> verifier " << (ok ? "ACCEPTS" : "REJECTS")
                  << " (" << vs.message << ")\n";
        return ok;
    };

    // Prefer two committed shifts that route the fixed coin H to DIFFERENT tokens
    // (a genuine decision flip). Fall back to two distinct Theta otherwise.
    ll zeta_a = admissible.front().first, zeta_b = admissible.back().first;
    int tok_a = -1, tok_b = -1;
    if (token_to_zeta.size() >= 2)
    {
        auto it = token_to_zeta.begin();
        tok_a = it->first; zeta_a = it->second; ++it;
        tok_b = it->first; zeta_b = it->second;
    }

    std::cout << "\nEnd-to-end (real prover + real verifier):\n";
    ll Th_a = 0, Th_b = 0;
    bool a = run_one(zeta_a, Th_a);
    bool b = run_one(zeta_b, Th_b);

    std::cout << "\nRESULT: ";
    if (a && b && (tok_a != tok_b || Th_a != Th_b))
    {
        std::cout << "DEMONSTRATED. The unmodified verifier ACCEPTS two transcripts for the SAME\n"
                     "committed logits and SAME prefix hash H, committing to different Theta ("
                  << Th_a << " vs " << Th_b << ")";
        if (tok_a != tok_b)
            std::cout << " and routing the fixed coin to DIFFERENT tokens ("
                      << tok_a << " vs " << tok_b << ")";
        std::cout << ".\nThe decode relation is set-valued: decision integrity fails.\n";
    }
    else
        std::cout << "not demonstrated (a=" << a << " b=" << b
                  << " Th_a=" << Th_a << " Th_b=" << Th_b << ").\n";
    return 0;
}

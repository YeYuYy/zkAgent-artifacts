// Validate the Python Q20 decode-relation simulation against the REAL
// zkAgent prover + verifier, case by case.
//
// The E1b attack campaigns never run the ZK protocol: experiments/relation.py
// simulates which (token, coin r, committed shift zeta) tuples the verifier
// accepts.  This driver closes that gap for the one model the artifact
// supports (any committed logit vector; the relation is logits-only).
//
// stdin protocol (all integers, q-units):
//   vocab k
//   num_vectors
//   num_vectors lines of vocab quantized logits
//   num_cases
//   num_cases lines: vector_idx zeta token r
// stdout per case:  "idx zeta token r ACCEPT|REJECT|PROVER_FAIL message"
//
// ACCEPT means verifyZkDecodeProof accepted the real proof built by
// proveTopKBatch with the shift override = zeta and sample = r.  A Python
// simulation that is faithful must predict exactly this set.
//
// Build (see attacks/README.md):
//   g++ -std=c++14 -O2 -fopenmp -mcmodel=large -Isrc -I3rd -I3rd/mcl/include \
//       attacks/validate_simulation.cpp cmake-build-release/src/libgpt_lib.a \
//       cmake-build-release/lib/libmcl.a -lgmp -lcrypto -o validate_simulation

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
#include <vector>

#include <mcl/bn256.hpp>

using namespace mcl::bn;
using std::vector;

namespace
{
    constexpr ll theta = ZkDecodeProbabilityScale; // 2^20

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
}

int main(int argc, char **argv)
{
    int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    // argv[2]: logit bound exponent (default 20, matching the E1a PoC regime;
    // the exp-digit subsystem itself accepts |q| <= 2^46 = max_abs_logit).
    const ll bound = argc > 2 ? ((ll)1 << std::atoi(argv[2])) : (ll)1 << 20;
    LookupBackend backend = LookupBackend::LogUpGkr;

    int vocab, k;
    if (!(std::cin >> vocab >> k)) return 2;
    int num_vectors;
    if (!(std::cin >> num_vectors)) return 2;
    vector<vector<ll>> logits(num_vectors, vector<ll>(vocab));
    for (auto &v : logits)
        for (auto &x : v)
        {
            long long q;
            if (!(std::cin >> q)) return 2;
            x = q;
        }

    const ScopedOmpThreads thread_scope(threads);
    initPairing(mcl::BN_SNARK1);

    // Per-vector top-k state shared by all cases on that vector.
    vector<vector<ll>> deltas(num_vectors);
    vector<vector<int>> active(num_vectors);
    for (int i = 0; i < num_vectors; ++i)
        select_topk(logits[i], k, deltas[i], active[i]);

    const ll eps = decode_exp_digits_recommended_epsilon(theta, k, DecodeExpDigitsConfig{});

    const std::size_t qc = (std::size_t)vocab;
    const int qvars = MVKZG_SRS::min_vars_for_size(std::max<std::size_t>(qc, 1));
    const int col_vars = (10 + qvars + 1) / 2;
    SRSPlan plan; plan.mv_vars = std::max(15, col_vars);
    SRSBundle srs;
    srs.setup_cached(plan, "cache");

    ZkDecodeProver prover(srs.mv.pk, srs.mv.vk, threads);
    prover.setLookupBackend(backend);
    prover.setSoftmaxEpsilon(eps);

    ZkDecodeParameters p;
    p.kind = ZkDecodeProofKind::TopK;
    p.lookup_backend = backend;
    p.token_count = 1; p.vocab_size = vocab; p.k = k;
    p.bound = bound; p.theta = theta; p.epsilon_sm = eps;
    ZkDecodeVerifierKey key = prepareZkDecodeVerifierKey(p, srs.mv.pk, srs.mv.vk, threads);

    ZkDecodeStatement stmt;
    stmt.parameters = p;

    int num_cases;
    if (!(std::cin >> num_cases)) return 2;
    for (int c = 0; c < num_cases; ++c)
    {
        long long vi, zeta, token, r;
        if (!(std::cin >> vi >> zeta >> token >> r)) return 2;
        const vector<ll> &L = logits[vi];

        decode_exp_digits_shift_override = nullptr;
        static ll g;
        g = zeta;
        decode_exp_digits_shift_override = [](ll) { return g; };

        vector<int> outs = {(int)token};
        vector<ll> samples = {r};
        ZkDecodeProveResult gen = prover.proveTopKBatch({L}, outs, k, samples, bound);
        decode_exp_digits_shift_override = nullptr;
        if (!gen.stats.success)
        {
            std::cout << c << ' ' << zeta << ' ' << token << ' ' << r
                      << " PROVER_FAIL " << gen.stats.message << "\n";
            continue;
        }

        stmt.output_tokens = outs;
        stmt.sample_values = samples;
        {
            const int tb = zkagent_integer::ceil_log2(1);
            const int vb = zkagent_integer::ceil_log2(vocab);
            const int vars = tb + vb;
            const std::size_t width = (std::size_t)1 << vb;
            vector<ll> padded(((std::size_t)1 << tb) * width, 0);
            std::copy(L.begin(), L.end(), padded.begin());
            const std::size_t cols = (std::size_t)1 << (vars - vars / 2);
            vector<G1> gens(srs.mv.pk.srs_g1.begin(), srs.mv.pk.srs_g1.begin() + cols);
            G1 *cm = hyrax::commit(padded.data(), gens.data(), vars, threads);
            stmt.logit_commitments.assign(cm, cm + ((std::size_t)1 << (vars / 2)));
            delete[] cm;
        }

        ZkDecodeVerifyStats vs;
        bool ok = verifyZkDecodeProof(stmt, gen.proof, key, srs.mv.vk, &vs);
        std::cout << c << ' ' << zeta << ' ' << token << ' ' << r
                  << (ok ? " ACCEPT " : " REJECT ") << vs.message << "\n";
        std::cout.flush();
    }
    return 0;
}

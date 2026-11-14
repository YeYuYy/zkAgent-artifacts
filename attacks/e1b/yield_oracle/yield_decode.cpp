// E1b: yield of the decode coin-re-roll steering on REAL logit distributions.
//
// Reads a file of per-step integer logits (already mapped into zkAgent's
// integer domain by dump_logits.py: integer = round(real_logit / logit_scale)).
// For each decode step it uses the released witness builder to enumerate
// admissible committed shifts zeta (hence in-band Theta), and reports which
// tokens an adversary can force for the step's fixed prefix coin r = H mod Theta.
//
// Unlike the random-logit PoC, this quantifies yield on peaked real
// distributions: reachable-set size, flip rate vs the honest sample and vs the
// greedy argmax, and the branch-point flip rate (where two continuations are
// genuinely close and a flip changes the decision).

#undef NDEBUG
#include "zkdecode/zkdecode_exp_digits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using std::vector;
constexpr ll THETA = ZkDecodeProbabilityScale; // 2^20
constexpr long double ALPHA = 7.952219286828541e-10L;

static void select_topk(const vector<ll> &logits, int k, vector<ll> &deltas, vector<int> &active)
{
    const int vocab = (int)logits.size();
    vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::nth_element(order.begin(), order.begin() + k - 1, order.end(),
                     [&](int a, int b){ return logits[a] != logits[b] ? logits[a] > logits[b] : a < b; });
    const ll sep = logits[order[k - 1]];
    deltas.resize(vocab); active.assign(vocab, 0);
    for (int i = 0; i < vocab; ++i) deltas[i] = logits[i] - sep;
    for (int r = 0; r < k; ++r) active[order[r]] = 1;
}

// Reference path (full lib builder) — used to validate the fast path.
static vector<ll> mass_for_shift_ref(const vector<ll> &deltas, const vector<int> &active, ll shift)
{
    static ll g; g = shift;
    decode_exp_digits_shift_override = [](ll){ return g; };
    DecodeExpDigitsWitness w = build_decode_exp_digits_witness(deltas, active, THETA, DecodeExpDigitsConfig{});
    decode_exp_digits_shift_override = nullptr;
    vector<ll> m;
    if (!w.success) return m;
    m = std::move(w.unmasked_outputs);
    for (size_t i = 0; i < m.size(); ++i) if (!active[i]) m[i] = 0;
    return m;
}
static ll total(const vector<ll> &m){ return std::accumulate(m.begin(), m.end(), (ll)0); }
static int token_for_coin(const vector<ll> &m, ll r){ ll p=0; for(size_t i=0;i<m.size();++i){ if(r<p+m[i]) return (int)i; p+=m[i]; } return -1; }

// Fast active-only path. The output mass of each active token depends only on
// the middle digit d3 = (magnitude>>23)&2047 of magnitude = shift - delta, and
// is kept iff magnitude < 2^34 (the two high digits are zero). table3 is the
// released middle exp table (DecodeExpDigitsConfig, segment 3). Validated below
// to equal the full lib builder (mass_for_shift_ref) bit for bit.
struct Sparse { vector<std::pair<int,ll>> tok_mass; ll total=0; bool ok=false; };
static Sparse fast_mass(const vector<std::pair<int,ll>> &active_sorted, // (token, delta)
                        const vector<ll> &table3, ll shift)
{
    Sparse s; s.tok_mass.reserve(active_sorted.size());
    for (auto &p : active_sorted) {
        const ll mag = shift - p.second;
        if (mag < 0) return s;                 // invalid shift (lib would reject)
        ll mass = 0;
        if (mag < (ll(1) << 34)) mass = table3[(size_t)((mag >> 23) & 2047)];
        s.tok_mass.emplace_back(p.first, mass);
        s.total += mass;
    }
    s.ok = true; return s;
}
static int token_for_coin_sparse(const Sparse &s, ll r){ ll p=0; for(auto&tm:s.tok_mass){ if(r<p+tm.second) return tm.first; p+=tm.second; } return -1; }

// splitmix64 -> a ~120-bit coin from the step's prefix seed.
static ll coin_from_prefix(uint64_t seed)
{
    auto mix=[&](uint64_t z){ z+=0x9E3779B97F4A7C15ULL; z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); };
    uint64_t a=mix(seed), b=mix(seed^0xABCDEF);
    return ((__int128)(a & ((1ULL<<60)-1)) << 60) | (ll)(b & ((1ULL<<60)-1));
}

int main(int argc, char** argv)
{
    std::string path; int k = 40, n_theta = 1500; double branch_gap = 0.10;
    for (int i=1;i<argc;++i){ std::string o=argv[i];
        if(o=="--in") path=argv[++i]; else if(o=="--k") k=atoi(argv[++i]);
        else if(o=="--n-theta") n_theta=atoi(argv[++i]); else if(o=="--branch-gap") branch_gap=atof(argv[++i]); }
    if(path.empty()){ std::cerr<<"need --in\n"; return 1; }
    std::ifstream in(path); int T=0, V=0; in>>T>>V; std::string line; std::getline(in,line);
    const ll eps = decode_exp_digits_recommended_epsilon(THETA, k, DecodeExpDigitsConfig{});

    bool dump_reach=false; for(int i=1;i<argc;++i) if(std::string(argv[i])=="--dump-reach") dump_reach=true;
    std::cout<<"step,honest_tok,greedy_tok,top1_prob,top2_prob,entropy,branch,"
               "n_reachable,flip_vs_honest,flip_vs_greedy,branch_flip,reach\n";
    int steps=0, flipH=0, flipG=0, branch_steps=0, branch_flips=0; double sum_reach=0;
    for (int t=0;t<T;++t){
        vector<ll> logits(V);
        for (int i=0;i<V;++i){ long long v; in>>v; logits[i]=(ll)v; }
        // real probabilities (for branch detection) via softmax over alpha*logit
        ll mx=*std::max_element(logits.begin(),logits.end());
        long double Z=0; for(ll l:logits) Z+=std::exp(ALPHA*(long double)(l-mx));
        // top-1/top-2 real prob
        long double p1=0,p2=0; int greedy=0;
        for(int i=0;i<V;++i){ long double p=std::exp(ALPHA*(long double)(logits[i]-mx))/Z;
            if(p>p1){p2=p1;p1=p;greedy=i;} else if(p>p2)p2=p; }
        long double ent=0; for(int i=0;i<V;++i){ long double p=std::exp(ALPHA*(long double)(logits[i]-mx))/Z; if(p>0) ent-=p*std::log(p); }
        bool branch = (p1-p2) < branch_gap;

        vector<ll> deltas; vector<int> active; select_topk(logits,k,deltas,active);
        // honest shift / Theta / token under the step's fixed coin
        decode_exp_digits_shift_override=nullptr;
        DecodeExpDigitsWitness hw=build_decode_exp_digits_witness(deltas,active,THETA,DecodeExpDigitsConfig{});
        if(!hw.success){ continue; }
        const ll zeta0=hw.shift;

        // active tokens in ascending index order + the released middle exp table
        vector<std::pair<int,ll>> as;
        for(int i=0;i<V;++i) if(active[i]) as.emplace_back(i, deltas[i]);
        static const vector<ll> &table3 = decode_exp_digits_public_tables(DecodeExpDigitsConfig{}).outputs[3];

        // One-time validation: fast path == full lib builder at the honest shift.
        if(t==0){ vector<ll> ref=mass_for_shift_ref(deltas,active,zeta0); Sparse f=fast_mass(as,table3,zeta0);
            bool match=f.ok && f.total==total(ref); for(auto&tm:f.tok_mass) if(ref[tm.first]!=tm.second) match=false;
            std::cerr<<"[validate] fast==lib at honest shift: "<<(match?"OK":"MISMATCH")<<"  Theta="<<(long long)f.total<<"\n"; if(!match) return 2; }

        Sparse hs=fast_mass(as,table3,zeta0); ll Th0=hs.total;
        ll Hcoin=coin_from_prefix((uint64_t)(t*2654435761u)+12345u);
        int honest_tok=token_for_coin_sparse(hs,((Hcoin%Th0)+Th0)%Th0);

        // enumerate admissible Theta by sampling zeta across the band
        const long double dThetadZeta=ALPHA*(long double)THETA;        // ~8.3e-4 per zeta
        const ll zeta_span=(ll)((long double)eps/dThetadZeta*1.2L);    // cover the band
        std::set<ll> seenTheta; std::set<int> reachable;
        for(int s=0;s<n_theta;++s){
            ll d=(ll)(((long double)s/(n_theta-1)*2.0L-1.0L)*(long double)zeta_span);
            Sparse m=fast_mass(as,table3,zeta0+d);
            if(!m.ok) continue; ll Th=m.total;
            if(Th<THETA-eps||Th>THETA+eps||Th<=0) continue;
            if(!seenTheta.insert(Th).second) continue;
            ll r=((Hcoin%Th)+Th)%Th; int tok=token_for_coin_sparse(m,r);
            if(tok>=0) reachable.insert(tok);
        }
        int nreach=(int)reachable.size();
        bool fH = reachable.count(honest_tok) ? nreach>1 : nreach>=1;
        bool fG = (reachable.size()>1) || (nreach==1 && !reachable.count(greedy));
        bool bflip = branch && fG;
        ++steps; sum_reach+=nreach; flipH+=fH; flipG+=fG; branch_steps+=branch; branch_flips+=bflip;
        std::cout<<t<<","<<honest_tok<<","<<greedy<<","<<(double)p1<<","<<(double)p2<<","
                 <<(double)ent<<","<<branch<<","<<nreach<<","<<fH<<","<<fG<<","<<bflip<<",";
        if(dump_reach){ bool first=true; for(int tk:reachable){ std::cout<<(first?"":" ")<<tk; first=false; } }
        std::cout<<"\n";
    }
    std::cerr<<"\n=== aggregate (k="<<k<<", eps="<<(long long)eps<<", n_theta="<<n_theta<<") ===\n";
    std::cerr<<"steps="<<steps<<"  mean_reachable="<<(steps?sum_reach/steps:0)<<"\n";
    std::cerr<<"flip_vs_honest="<<flipH<<"/"<<steps<<"  flip_vs_greedy="<<flipG<<"/"<<steps<<"\n";
    std::cerr<<"branch_steps="<<branch_steps<<"  branch_flips="<<branch_flips
             <<"  (branch_flip_rate="<<(branch_steps?(double)branch_flips/branch_steps:0)<<")\n";
    return 0;
}

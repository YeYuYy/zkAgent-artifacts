// E1b (forced extension / billing): can the prover refuse to stop?
//
// zkAgent requires a model-generated terminator (blocks early truncation) but
// "nothing bounds over-generation". Under sampling decode the terminator (EOS
// or </think>) is one top-k candidate; the coin r = H mod Theta picks the token.
// At a step where the honest model stops (terminator is argmax), the prover can
// grind the committed shift zeta so the coin lands on ANY reachable
// non-terminator candidate, emit it, and continue generating honestly -- every
// token a valid decode, detection zero, the hidden-reasoning bill inflated.
//
// Unlike a decision flip (needs a specific close alternative), forced extension
// only needs ONE reachable non-terminator, so it succeeds even on sharp
// distributions. This tool reports, per step, whether a terminator is a
// candidate and whether the prover can avoid it.
//
// Input file header: "T VOCAB eos_id think_end_id", then T rows of integer
// logits (zkAgent domain). greedy argmax == the honest next token.

#undef NDEBUG
#include "zkdecode/zkdecode_exp_digits.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
using std::vector;
constexpr ll THETA = ZkDecodeProbabilityScale;
constexpr long double ALPHA = 7.952219286828541e-10L;

static void select_topk(const vector<ll>&lg,int k,vector<ll>&d,vector<int>&a){
    int V=lg.size(); vector<int> o(V); std::iota(o.begin(),o.end(),0);
    std::nth_element(o.begin(),o.begin()+k-1,o.end(),[&](int x,int y){return lg[x]!=lg[y]?lg[x]>lg[y]:x<y;});
    ll sep=lg[o[k-1]]; d.resize(V); a.assign(V,0);
    for(int i=0;i<V;++i)d[i]=lg[i]-sep; for(int r=0;r<k;++r)a[o[r]]=1;
}
struct Sparse{vector<std::pair<int,ll>> tm; ll tot=0; bool ok=false;};
static Sparse fast_mass(const vector<std::pair<int,ll>>&as,const vector<ll>&t3,ll sh){
    Sparse s; for(auto&p:as){ ll mg=sh-p.second; if(mg<0)return s; ll m=0; if(mg<(ll(1)<<34))m=t3[(size_t)((mg>>23)&2047)]; s.tm.emplace_back(p.first,m); s.tot+=m;} s.ok=true; return s;
}
static int tok_for(const Sparse&s,ll r){ll p=0;for(auto&tm:s.tm){if(r<p+tm.second)return tm.first;p+=tm.second;}return -1;}
static ll coin(uint64_t seed){auto mix=[&](uint64_t z){z+=0x9E3779B97F4A7C15ULL;z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);};uint64_t a=mix(seed),b=mix(seed^0xABCDEF);return((__int128)(a&((1ULL<<60)-1))<<60)|(ll)(b&((1ULL<<60)-1));}

int main(int argc,char**argv){
    std::string path;int k=40,n_theta=4000;
    for(int i=1;i<argc;++i){std::string o=argv[i];if(o=="--in")path=argv[++i];else if(o=="--k")k=atoi(argv[++i]);else if(o=="--n-theta")n_theta=atoi(argv[++i]);}
    std::ifstream in(path);int T=0,V=0;long long eos=-1,think=-1;in>>T>>V>>eos>>think;std::string ln;std::getline(in,ln);
    const ll eps=decode_exp_digits_recommended_epsilon(THETA,k,DecodeExpDigitsConfig{});
    static const vector<ll>&t3=decode_exp_digits_public_tables(DecodeExpDigitsConfig{}).outputs[3];
    auto isTerm=[&](int tk){return tk==(int)eos||(think>=0&&tk==(int)think);};

    std::cout<<"step,greedy,greedy_is_term,term_prob,term_in_topk,avoidable,n_reachable\n";
    int stop_steps=0,avoidable_stops=0,term_candidate_steps=0,avoidable_any=0;
    for(int t=0;t<T;++t){
        vector<ll> lg(V); for(int i=0;i<V;++i){long long v;in>>v;lg[i]=(ll)v;}
        ll mx=*std::max_element(lg.begin(),lg.end());
        long double Z=0;for(ll l:lg)Z+=std::exp(ALPHA*(long double)(l-mx));
        int greedy=(int)(std::max_element(lg.begin(),lg.end())-lg.begin());
        // terminator prob (if any terminator exists)
        long double tprob=0; for(int tk:{(int)eos,(int)think}) if(tk>=0) tprob=std::max(tprob,(long double)(std::exp(ALPHA*(long double)(lg[tk]-mx))/Z));
        vector<ll> d;vector<int> a;select_topk(lg,k,d,a);
        bool term_in_topk=false; for(int tk:{(int)eos,(int)think}) if(tk>=0&&a[tk]) term_in_topk=true;
        vector<std::pair<int,ll>> as; for(int i=0;i<V;++i) if(a[i]) as.emplace_back(i,d[i]);
        decode_exp_digits_shift_override=nullptr;
        DecodeExpDigitsWitness hw=build_decode_exp_digits_witness(d,a,THETA,DecodeExpDigitsConfig{});
        if(!hw.success) continue; ll z0=hw.shift;
        ll H=coin((uint64_t)(t*2654435761u)+999u);
        const long double dTdz=ALPHA*(long double)THETA; ll span=(ll)((long double)eps/dTdz*1.2L);
        std::set<int> reach;
        for(int s=0;s<n_theta;++s){ ll dd=(ll)(((long double)s/(n_theta-1)*2.0L-1.0L)*(long double)span);
            Sparse m=fast_mass(as,t3,z0+dd); if(!m.ok)continue; ll Th=m.tot;
            if(Th<THETA-eps||Th>THETA+eps||Th<=0)continue;
            ll r=((H%Th)+Th)%Th; int tk=tok_for(m,r); if(tk>=0)reach.insert(tk);
        }
        // avoidable = a non-terminator token is reachable
        bool avoidable=false; for(int tk:reach) if(!isTerm(tk)){avoidable=true;break;}
        bool gterm=isTerm(greedy);
        if(gterm){ ++stop_steps; avoidable_stops+=avoidable; }
        if(term_in_topk){ ++term_candidate_steps; avoidable_any+=avoidable; }
        std::cout<<t<<","<<greedy<<","<<gterm<<","<<(double)tprob<<","<<term_in_topk<<","<<avoidable<<","<<reach.size()<<"\n";
    }
    std::cerr<<"\n=== forced extension (k="<<k<<", eps="<<(long long)eps<<") ===\n";
    std::cerr<<"natural STOP steps (greedy=terminator): "<<stop_steps
             <<" ; of them prover can REFUSE to stop: "<<avoidable_stops<<"\n";
    std::cerr<<"steps with terminator in top-k (could stop): "<<term_candidate_steps
             <<" ; prover can avoid stopping: "<<avoidable_any<<"\n";
    return 0;
}

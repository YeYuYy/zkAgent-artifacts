// Implementation-faithful reachability: the released verifier never binds the
#include <sstream>
// sample value r to H(prefix) -- check_decode_sampling_claims only requires
// 0 <= r < Theta and prefix <= r < next_prefix. So the server picks r freely,
// and the output token can be ANY candidate with quantized mass >= 1.
//
// reachable_freeR(step) = #{ top-k tokens with honest mass >= 1 }.
// For a stop step (greedy == terminator) forced extension works iff some
// NON-terminator candidate has mass >= 1.
//
// Input: "T VOCAB [eos think]" header then T rows of integer logits.

#undef NDEBUG
#include "zkdecode/zkdecode_exp_digits.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
using std::vector;
constexpr ll THETA = ZkDecodeProbabilityScale;
constexpr long double ALPHA = 7.952219286828541e-10L;

int main(int argc,char**argv){
    std::string path;int k=40;
    for(int i=1;i<argc;++i){std::string o=argv[i];if(o=="--in")path=argv[++i];else if(o=="--k")k=atoi(argv[++i]);}
    std::ifstream in(path);int T=0,V=0;long long eos=-1,think=-1;
    { std::string h; std::getline(in,h); std::istringstream is(h); is>>T>>V>>eos>>think; }
    auto isTerm=[&](int t){return t==(int)eos||(think>=0&&t==(int)think);};
    static const vector<ll>&t3=decode_exp_digits_public_tables(DecodeExpDigitsConfig{}).outputs[3];

    long long sum_reach=0; int steps=0, flip=0, stop=0, stop_avoidable=0;
    std::cout<<"step,greedy,greedy_is_term,term_prob,reach_freeR,nonterm_reachable\n";
    for(int t=0;t<T;++t){
        vector<ll> lg(V); for(int i=0;i<V;++i){long long v;in>>v;lg[i]=(ll)v;}
        ll mx=*std::max_element(lg.begin(),lg.end());
        int greedy=(int)(std::max_element(lg.begin(),lg.end())-lg.begin());
        long double Z=0; for(ll l:lg) Z+=std::exp(ALPHA*(long double)(l-mx));
        long double tprob=0; for(int tk:{(int)eos,(int)think}) if(tk>=0) tprob=std::max(tprob,(long double)(std::exp(ALPHA*(long double)(lg[tk]-mx))/Z));
        // top-k by index order
        vector<int> ord(V); std::iota(ord.begin(),ord.end(),0);
        std::nth_element(ord.begin(),ord.begin()+k-1,ord.end(),[&](int a,int b){return lg[a]!=lg[b]?lg[a]>lg[b]:a<b;});
        ll sep=lg[ord[k-1]];
        // honest shift over the active set
        vector<ll> deltas(V); vector<int> active(V,0);
        for(int i=0;i<V;++i) deltas[i]=lg[i]-sep;
        for(int r=0;r<k;++r) active[ord[r]]=1;
        decode_exp_digits_shift_override=nullptr;
        DecodeExpDigitsWitness w=build_decode_exp_digits_witness(deltas,active,THETA,DecodeExpDigitsConfig{});
        if(!w.success) continue;
        ll shift=w.shift;
        int reach=0; bool nonterm=false;
        for(int r=0;r<k;++r){ int i=ord[r]; ll mag=shift-deltas[i]; ll mass=0;
            if(mag>=0 && mag<(ll(1)<<34)) mass=t3[(size_t)((mag>>23)&2047)];
            if(mass>0){ ++reach; if(!isTerm(i)) nonterm=true; }
        }
        bool gt=isTerm(greedy);
        ++steps; sum_reach+=reach; if(reach>1) ++flip;
        if(gt){ ++stop; if(nonterm) ++stop_avoidable; }
        std::cout<<t<<","<<greedy<<","<<gt<<","<<(double)tprob<<","<<reach<<","<<nonterm<<"\n";
    }
    std::cerr<<"\n=== free-r reachability (k="<<k<<") ===\n";
    std::cerr<<"steps="<<steps<<"  mean_reachable_freeR="<<(steps?(double)sum_reach/steps:0)
             <<"  steps_with>=2_reachable="<<flip<<"/"<<steps<<"\n";
    std::cerr<<"natural STOP steps="<<stop<<"  of them forced-extension works (non-terminator mass>=1): "<<stop_avoidable<<"\n";
    return 0;
}

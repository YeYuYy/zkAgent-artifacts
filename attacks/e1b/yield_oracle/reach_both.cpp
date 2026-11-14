// Per-step reachable sets under BOTH decode models, for the same real logits.
//
//  paper model : r = H(prefix) mod Theta is a FIXED coin; the prover may only
//                grind the committed shift zeta (hence Theta in [theta +/- eps]).
//                reachable = { token the coin lands on for some in-band Theta }.
//  free-r model: the released verifier never binds r; it is a free public input
//                (0 <= r < Theta, prefix <= r < next_prefix). reachable = { any
//                top-k token with honest quantized mass >= 1 }.
//
// Dumps, per step: greedy, greedy_is_term, the two reachable id lists. A Python
// pass maps ids -> continuation markers to get the length-steering feasibility.
//
// Header: "T VOCAB eos think", then T rows of integer logits.

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
#include <vector>
using std::vector;
constexpr ll THETA = ZkDecodeProbabilityScale;
constexpr long double ALPHA = 7.952219286828541e-10L;

static ll coin(uint64_t s){auto m=[&](uint64_t z){z+=0x9E3779B97F4A7C15ULL;z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);};uint64_t a=m(s),b=m(s^0xABCDEF);return((__int128)(a&((1ULL<<60)-1))<<60)|(ll)(b&((1ULL<<60)-1));}

int main(int argc,char**argv){
    std::string path;int k=40,n_theta=4000;
    for(int i=1;i<argc;++i){std::string o=argv[i];if(o=="--in")path=argv[++i];else if(o=="--k")k=atoi(argv[++i]);else if(o=="--n-theta")n_theta=atoi(argv[++i]);}
    std::ifstream in(path);int T=0,V=0;long long eos=-1,think=-1;
    {std::string h;std::getline(in,h);std::istringstream is(h);is>>T>>V>>eos>>think;}
    auto isTerm=[&](int t){return t==(int)eos||(think>=0&&t==(int)think);};
    static const vector<ll>&t3=decode_exp_digits_public_tables(DecodeExpDigitsConfig{}).outputs[3];
    const ll eps=decode_exp_digits_recommended_epsilon(THETA,k,DecodeExpDigitsConfig{});

    std::cout<<"step,greedy,greedy_is_term,free_reach,paper_reach\n";
    for(int t=0;t<T;++t){
        vector<ll> lg(V);for(int i=0;i<V;++i){long long v;in>>v;lg[i]=(ll)v;}
        int greedy=(int)(std::max_element(lg.begin(),lg.end())-lg.begin());
        vector<int> ord(V);std::iota(ord.begin(),ord.end(),0);
        std::nth_element(ord.begin(),ord.begin()+k-1,ord.end(),[&](int a,int b){return lg[a]!=lg[b]?lg[a]>lg[b]:a<b;});
        ll sep=lg[ord[k-1]];
        vector<ll> d(V);vector<int> act(V,0);
        for(int i=0;i<V;++i)d[i]=lg[i]-sep; for(int r=0;r<k;++r)act[ord[r]]=1;
        decode_exp_digits_shift_override=nullptr;
        DecodeExpDigitsWitness w=build_decode_exp_digits_witness(d,act,THETA,DecodeExpDigitsConfig{});
        if(!w.success) continue; ll z0=w.shift;
        // active sorted by token index for CDF
        vector<std::pair<int,ll>> as; for(int i=0;i<V;++i) if(act[i]) as.emplace_back(i,d[i]);
        auto mass_at=[&](ll shift,vector<std::pair<int,ll>>&out)->ll{ out.clear(); ll tot=0; for(auto&p:as){ ll mg=shift-p.second; if(mg<0){out.clear();return -1;} ll m=0; if(mg<(ll(1)<<34))m=t3[(size_t)((mg>>23)&2047)]; out.emplace_back(p.first,m); tot+=m;} return tot; };
        // free-r reachable: honest-shift mass>=1
        std::set<int> freeR; { vector<std::pair<int,ll>> m; mass_at(z0,m); for(auto&tm:m) if(tm.second>0) freeR.insert(tm.first); }
        // paper reachable: fixed coin, grind Theta
        std::set<int> paper; ll H=coin((uint64_t)(t*2654435761u)+999u);
        const long double dTdz=ALPHA*(long double)THETA; ll span=(ll)((long double)eps/dTdz*1.2L);
        std::set<ll> seen;
        for(int s=0;s<n_theta;++s){ ll dd=(ll)(((long double)s/(n_theta-1)*2.0L-1.0L)*(long double)span);
            vector<std::pair<int,ll>> m; ll Th=mass_at(z0+dd,m); if(Th<=0)continue;
            if(Th<THETA-eps||Th>THETA+eps)continue; if(!seen.insert(Th).second)continue;
            ll r=((H%Th)+Th)%Th; ll p=0; for(auto&tm:m){ if(r<p+tm.second){ paper.insert(tm.first); break;} p+=tm.second; }
        }
        auto join=[&](const std::set<int>&s){ std::string o; bool f=true; for(int x:s){ if(!f)o+=" "; o+=std::to_string(x); f=false;} return o; };
        std::cout<<t<<","<<greedy<<","<<(isTerm(greedy)?1:0)<<","<<join(freeR)<<","<<join(paper)<<"\n";
    }
    return 0;
}

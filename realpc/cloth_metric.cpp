#include <cmath>
#include <algorithm>
#include <cstdint>
extern "C" void project_metric(double* y,const int64_t* edges,const double* rest,
    const double* inv,int64_t m,double limit,int sweeps) {
    for(int it=0;it<sweeps;++it) {
        for(int64_t j=0;j<m;++j) {
            auto a=edges[2*j],b=edges[2*j+1];double d[3],s=0;
            for(int k=0;k<3;++k){d[k]=y[3*a+k]-y[3*b+k];s+=d[k]*d[k];}
            double l=std::sqrt(s),target=std::min(std::max(l,rest[j]*(1-limit)),rest[j]*(1+limit));
            double f=(l-target)/std::max(l,1e-12)/std::max(inv[a]+inv[b],1.0);
            for(int k=0;k<3;++k){y[3*a+k]-=inv[a]*d[k]*f;y[3*b+k]+=inv[b]*d[k]*f;}
        }
        if(it%10==9) {
            double worst=0;
            for(int64_t j=0;j<m;++j) {
                auto a=edges[2*j],b=edges[2*j+1];double s=0;
                for(int k=0;k<3;++k){double d=y[3*a+k]-y[3*b+k];s+=d*d;}
                worst=std::max(worst,std::abs(std::sqrt(s)/rest[j]-1));
            }
            if(worst<limit+.001)break;
        }
    }
}

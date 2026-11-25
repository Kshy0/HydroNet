// Full graph Newton and its implicit transpose adjoint:
// - Pure-forest basins solved independently in graph_solve_pure_trees.
// - Per-component independent Newton controllers (moving-weight front-end -> fixed-weight -> damped -> continuation).
// - Static-pattern LU over coupled components with 1-solve direct fast path and BiCGStab refinement.
#include "implicit_inertia.cu"

struct Graph {
    Net z;
    const int *src,*dst,*offset,*link,*plan;
    const float *width,*length,*elevation,*roughness,*previous;
    float *path,*sum,*factor,*rhs,*tmp,*bound,*scaling,*balance_scale,*solution_low,
          *lu_forward,*lu_backward,*r,*rhat,*p,*v,*s,*t,*delta,*pre,*reduction,*record;
    int m,bl,entries,rounds;
    const int *g_unit,*u_comp,*p_comp,*c_trees,*ct_comp,*c_rep,*c_plan,*ce_comp;
    int ngc,ngu_g,nct,nc_entries,nc_rounds;
    float dt;
    mutable int reduction_epoch=0;
    float coupling_weight=1;
};
#define BV(g,k,j) ((g).path[(long)(k)*7+(j)])
#define BS(g,p,j) ((g).sum[(long)(p)*3+(j)])
#define GLD(p,i) __ldcg((p)+(i))

// Double-buffered global reductions: the next reduction uses the other slot,
// and its barrier protects the preceding slot before reuse. This removes one
// device-wide barrier from each reduction. Every block remains resident.
template <bool MAXIMUM> __device__ float graph_reduce(const Graph& g,float value,float* sh) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    int slot=(g.reduction_epoch++&1)*2*gridDim.x;
    for(int d=16;d;d/=2) {
        float other=__shfl_down_sync(0xffffffff,value,d);
        value=MAXIMUM?fmaxf(value,other):value+other;
    }
    if(lane==0) sh[warp]=value;
    __syncthreads();
    if(threadIdx.x==0) {
        float total=0;
        for(int w=0;w<blockDim.x/32;++w) total=MAXIMUM?fmaxf(total,sh[w]):total+sh[w];
        g.reduction[slot+blockIdx.x]=total;
    }
    grid_sync();
    if(threadIdx.x==0) {
        float total=0;
        for(int b=0;b<gridDim.x;++b) {float v=GLD(g.reduction,slot+b);total=MAXIMUM?fmaxf(total,v):total+v;}
        sh[0]=total;
    }
    __syncthreads();float total=sh[0];
    __syncthreads();
    return total;
}
__device__ float graph_dot(const Graph& g,const float* a,const float* b,float* sh) {
    float total=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<g.z.n;i+=gridDim.x*blockDim.x) total+=GLD(a,i)*GLD(b,i);
    return graph_reduce<false>(g,total,sh);
}
// Fused (t,t) and (t,s) inner products share one global synchronization.
__device__ float2 graph_dot_pair(const Graph& g,const float* t,const float* s,float* sh) {
    float tt=0,ts=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<g.z.n;i+=gridDim.x*blockDim.x) {
        float a=GLD(t,i);tt+=a*a;ts+=a*GLD(s,i);
    }
    int lane=threadIdx.x&31,warp=threadIdx.x/32,warps=blockDim.x/32;
    int slot=(g.reduction_epoch++&1)*2*gridDim.x;
    for(int d=16;d;d/=2) {tt+=__shfl_down_sync(0xffffffff,tt,d);ts+=__shfl_down_sync(0xffffffff,ts,d);}
    if(lane==0) {sh[warp]=tt;sh[warps+warp]=ts;}
    __syncthreads();
    if(threadIdx.x==0) {
        float a=0,b=0;for(int w=0;w<warps;++w) {a+=sh[w];b+=sh[warps+w];}
        g.reduction[slot+2*blockIdx.x]=a;g.reduction[slot+2*blockIdx.x+1]=b;
    }
    grid_sync();
    if(threadIdx.x==0) {
        float a=0,b=0;for(int k=0;k<gridDim.x;++k) {a+=GLD(g.reduction,slot+2*k);b+=GLD(g.reduction,slot+2*k+1);}
        sh[0]=a;sh[1]=b;
    }
    __syncthreads();float2 result={sh[0],sh[1]};__syncthreads();return result;
}
__device__ float graph_norm(const Graph& g,const float* a,float* sh) {
    float maximum=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<g.z.n;i+=gridDim.x*blockDim.x) {float v=fabsf(GLD(a,i));maximum=fmaxf(maximum,isfinite(v)?v:INFINITY);}
    return graph_reduce<true>(g,maximum,sh);
}

// The high surface sets acceleration depth; the actual inertial donor sets
// conveyance. A path bed below its donor's channel bed cannot drain a dry cell.
__device__ void graph_flux_cw(const Graph& g,int p,float cw) {
    const Net& z=g.z;int u=g.src[p],d=g.dst[p];
    float ou=GLD(z.tref,u),xu=GLD(z.ht,u),od=GLD(z.tref,d),xd=GLD(z.ht,d);
    float head=head_pair(z.elev[u],z.elev[d],ou,od,xu,xd),raw=head/g.length[p];
    int high=head>=0?u:d;
    float slope=fminf(fmaxf(raw,-GRAPH_SLOPE_LIMIT),GRAPH_SLOPE_LIMIT);
    float ds=fabsf(raw)<GRAPH_SLOPE_LIMIT?1/g.length[p]:0;
    float net=0,di=0,ddown=0;
    for(int l=0;l<g.bl;++l) {
        int k=p*g.bl+l;float q=0,qi=0,qd=0,qp=0,qn=0,hu=0,hd=0;
        Pair height=pair_add(pair_diff(z.elev[high],g.elevation[k]),{GLD(z.tref,high),GLD(z.ht,high)});
        float depth=fmaxf(height.hi+height.lo,0),n=g.roughness[k],w=g.width[k];
        if(w>0 && g.elevation[k]<GRAPH_CLOSED_ELEVATION && !(n>0 && isfinite(n))) report(z.status,INVALID_ROUGHNESS,k);
        if(depth>GRAPH_DRY_DEPTH && w>0 && n>0 && isfinite(n)) {
            float a=9.8f*g.dt*w*depth,v=g.previous[k]+a*slope;
            int donor=v>=0?u:d;
            float bank_bed=z.elev[donor]-z.H[donor],donor_depth;
            if(bank_bed>g.elevation[k]) {
                // The channel bed is the floor: the donor's own depth, exactly zero
                // when dry, whatever the rounding of its bed elevation.
                donor_depth=fmaxf((GLD(z.tref,donor)+z.H[donor])+GLD(z.ht,donor),0.f);
            } else {
                Pair filled=pair_add(pair_diff(z.elev[donor],g.elevation[k]),{GLD(z.tref,donor),GLD(z.ht,donor)});
                donor_depth=fmaxf(filled.hi+filled.lo,0);
            }
            if(donor_depth>GRAPH_DRY_DEPTH) {
#if HYDROFORGE_FAST_MATH && !defined(__HIPCC_RTC__)
                float K=__fdividef(__fdividef(w*donor_depth*donor_depth, __powf(donor_depth, 1.0f/3.0f)), n);
#else
                float K=w*donor_depth*donor_depth/cbrtf(donor_depth)/n;
#endif
                float b=9.8f*g.dt*w*depth/(K*K),den=sqrtf(1+4*b*fabsf(v));
                q=2*v/(1+den);float friction=b*q*fabsf(q);
                float high_part=(a*slope-friction)/depth,donor_part=(10.f/3.f)*friction/donor_depth;
                qi=((high==u?high_part:0)+(donor==u?donor_part:0)+a*ds)/den;
                qd=((high==d?high_part:0)+(donor==d?donor_part:0)-a*ds)/den;
                qp=1/den;qn=-2*friction/(n*den);
                if(bank_bed>g.elevation[k]) {float h=donor_part/den;if(donor==u)hu=h;else hd=h;}
            }
        }
        q*=cw;qi*=cw;qd*=cw;qp*=cw;qn*=cw;hu*=cw;hd*=cw;
        BV(g,k,0)=q;BV(g,k,1)=qi;BV(g,k,2)=qd;BV(g,k,3)=qp;BV(g,k,4)=qn;BV(g,k,5)=hu;BV(g,k,6)=hd;
        net+=q;di+=qi;ddown+=qd;
    }
    BS(g,p,0)=net;BS(g,p,1)=di;BS(g,p,2)=ddown;
}
__device__ void graph_flux(const Graph& g,int p) { graph_flux_cw(g,p,g.coupling_weight); }

__device__ float graph_residual(const Graph& g,int i) {
    const Net& z=g.z;float in=0,ain=0,out=0,absout=0;
    for(int e=z.cp[i];e<z.cp[i+1];++e) {float q=GLD(z.Q,z.kids[e]);in+=q;ain+=fabsf(q);}
    for(int e=g.offset[i];e<g.offset[i+1];++e) {
        int link=g.link[e],p=abs(link)-1;float q=GLD(g.sum,3*p);
        out+=(link>0?1:-1)*q;absout+=fabsf(q);
    }
    float q=GLD(z.Q,i),r=(GLD(z.S,i)-z.S0[i])-g.dt*(z.runoff[i]+in-q-out);
    float scale=z.S0[i]+g.dt*(fabsf(z.runoff[i])+ain+fabsf(q)+absout)+1000;
    float score=isfinite(r)&&isfinite(q)?fabsf(r)/scale:INFINITY;
    g.rhs[i]=1/scale;z.R[i]=r;z.wb[i]=1/scale;return score;
}

// ------------------------------------------------------------------------
// Full-map static-pattern LU + 1-solve direct fast path + BiCGStab (used by transpose adjoint)

__device__ void graph_matrix(const Graph& g) {
    const Net& z=g.z;int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    for(int i=tid;i<z.n;i+=stride) {
        bool bound=GLD(g.bound,i)>.5f;float diag=GLD(z.area,i)+g.dt*GLD(z.qi,i);
        for(int e=z.cp[i];e<z.cp[i+1];++e) diag-=g.dt*GLD(z.qd,z.kids[e]);
        for(int e=g.offset[i];e<g.offset[i+1];++e) {
            int link=g.link[e],p=abs(link)-1;diag+=g.dt*(link>0?GLD(g.sum,3*p+1):-GLD(g.sum,3*p+2));
        }
        g.scaling[i]=bound?1:1/diag;
    }
}
__device__ Pair graph_product(float a,float b) {float p=a*b;return {p,fmaf(a,b,-p)};}
__device__ Pair graph_scale(Pair a,float b) {
    return pair_add(graph_product(a.hi,b),graph_product(a.lo,b));
}
template <bool COMPENSATED=false> __device__ Pair graph_x(const Graph& g,const float* x,int i) {
    return {GLD(x,i),COMPENSATED?GLD(g.solution_low,i):0.f};
}
template <bool COMPENSATED=false> __device__ void graph_matvec(const Graph& g,const float* x,float* out,bool trans) {
    const Net& z=g.z;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<z.n;i+=gridDim.x*blockDim.x) {
        bool bound=GLD(g.bound,i)>.5f;Pair xi=graph_x<COMPENSATED>(g,x,i),mi=bound?Pair{0,0}:xi;
        Pair value=bound?xi:graph_scale(xi,GLD(z.area,i));
        if(trans) {
            Pair md=z.edge[i]&&GLD(g.bound,z.down[i])<.5f?graph_x<COMPENSATED>(g,x,z.down[i]):Pair{0,0};
            value=pair_add(value,graph_scale(graph_scale(pair_add(mi,{-md.hi,-md.lo}),GLD(z.qi,i)),g.dt));
            for(int e=z.cp[i];e<z.cp[i+1];++e) {
                int j=z.kids[e];Pair mj=GLD(g.bound,j)>.5f?Pair{0,0}:graph_x<COMPENSATED>(g,x,j);
                value=pair_add(value,graph_scale(graph_scale(pair_add(mj,{-mi.hi,-mi.lo}),GLD(z.qd,j)),g.dt));
            }
            for(int e=g.offset[i];e<g.offset[i+1];++e) {
                int link=g.link[e],p=abs(link)-1,u=g.src[p],d=g.dst[p];
                Pair mu=GLD(g.bound,u)>.5f?Pair{0,0}:graph_x<COMPENSATED>(g,x,u),md=GLD(g.bound,d)>.5f?Pair{0,0}:graph_x<COMPENSATED>(g,x,d);
                value=pair_add(value,graph_scale(graph_scale(pair_add(mu,{-md.hi,-md.lo}),GLD(g.sum,3*p+(link>0?1:2))),g.dt));
            }
        } else if(!bound) {
            Pair flux=graph_scale(xi,GLD(z.qi,i));
            if(z.edge[i])flux=pair_add(flux,graph_scale(graph_x<COMPENSATED>(g,x,z.down[i]),GLD(z.qd,i)));
            value=pair_add(value,graph_scale(flux,g.dt));
            for(int e=z.cp[i];e<z.cp[i+1];++e) {
                int j=z.kids[e];Pair flux=pair_add(graph_scale(graph_x<COMPENSATED>(g,x,j),GLD(z.qi,j)),graph_scale(xi,GLD(z.qd,j)));
                value=pair_add(value,graph_scale(flux,-g.dt));
            }
            for(int e=g.offset[i];e<g.offset[i+1];++e) {
                int link=g.link[e],p=abs(link)-1;
                Pair flux=pair_add(graph_scale(graph_x<COMPENSATED>(g,x,g.src[p]),GLD(g.sum,3*p+1)),graph_scale(graph_x<COMPENSATED>(g,x,g.dst[p]),GLD(g.sum,3*p+2)));
                value=pair_add(value,graph_scale(flux,link>0?g.dt:-g.dt));
            }
        }
        out[i]=(value.hi+value.lo)*GLD(g.scaling,i);
    }
    grid_sync();
}

#define FP(name) (g.plan+g.plan[FACTOR_##name])
__device__ void factor_assemble(const Graph& g) {
    const Net& z=g.z;int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    const int *start=FP(SOURCE_PTR),*code=FP(SOURCE_CODE);
    for(int e=tid;e<g.entries;e+=stride) {
        float value=0;
        for(int c=start[e];c<start[e+1];++c) {
            int kind=code[c]&7,k=code[c]>>3;
            if(kind==0) value+=1.f;
            else if(kind==1) {if(z.edge[k]&&!(GLD(g.bound,k)>.5f)) value+=g.dt*GLD(z.qd,k)*GLD(g.scaling,k);}
            else if(kind==2) {int d=z.down[k];if(!(GLD(g.bound,d)>.5f)) value-=g.dt*GLD(z.qi,k)*GLD(g.scaling,d);}
            else if(kind==3) {int u=g.src[k];if(!(GLD(g.bound,u)>.5f)) value+=g.dt*GLD(g.sum,3*k+2)*GLD(g.scaling,u);}
            else {int d=g.dst[k];if(!(GLD(g.bound,d)>.5f)) value-=g.dt*GLD(g.sum,3*k+1)*GLD(g.scaling,d);}
        }
        g.factor[e]=value;
    }
}
__device__ float factor_entry(const Graph& g,int e) {
    const int *start=FP(PRODUCT_PTR),*lower=FP(PRODUCT_LOWER),*upper=FP(PRODUCT_UPPER);
    float value=GLD(g.factor,e);
    for(int k=start[e];k<start[e+1];++k) value-=GLD(g.factor,lower[k])*GLD(g.factor,upper[k]);
    return value;
}
__device__ float factor_entry_warp(const Graph& g,int e) {
    const int *start=FP(PRODUCT_PTR),*lower=FP(PRODUCT_LOWER),*upper=FP(PRODUCT_UPPER);
    int lane=threadIdx.x&31;float sum=0;
    for(int k=start[e]+lane;k<start[e+1];k+=32) sum+=GLD(g.factor,lower[k])*GLD(g.factor,upper[k]);
    for(int d=16;d;d/=2) sum+=__shfl_xor_sync(0xffffffff,sum,d);
    return GLD(g.factor,e)-sum;
}
__device__ float factor_dot_warp(const Graph& g,const int* entry,const int* index,int begin,int end,int skip,const float* x) {
    int lane=threadIdx.x&31;float sum=0;
    for(int q=begin+lane;q<end;q+=32) if(index[q]!=skip) sum+=GLD(g.factor,entry[q])*GLD(x,index[q]);
    for(int d=16;d;d/=2) sum+=__shfl_xor_sync(0xffffffff,sum,d);
    return sum;
}
__device__ void factor_numeric(const Graph& g) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    int warp=threadIdx.x/32,warps=blockDim.x/32,lane=threadIdx.x&31;
    const int *unit=FP(PATH_UNIT),*path=FP(PATH_OFFSET),*round=FP(ROUND_OFFSET),*heavy=FP(ROUND_HEAVY),*diag=FP(DIAG);
    const int *urow=FP(UROW_PTR),*uentry=FP(UROW_ENTRY),*lcol=FP(LCOL_PTR),*lentry=FP(LCOL_ENTRY);
    for(int r=g.rounds-1;r>=0;--r) {
        int split=round[r]+heavy[r];
        for(int p=round[r]+blockIdx.x;p<split;p+=gridDim.x)
            for(int t=path[p];t<path[p+1];++t) {
                int k=unit[t];
                for(int q=urow[k]+warp;q<urow[k+1];q+=warps) {float v=factor_entry_warp(g,uentry[q]);if(lane==0)g.factor[uentry[q]]=v;}
                __syncthreads();
                float pivot=GLD(g.factor,diag[k]);
                for(int q=lcol[k]+warp;q<lcol[k+1];q+=warps) {float v=factor_entry_warp(g,lentry[q]);if(lane==0)g.factor[lentry[q]]=v/pivot;}
                __syncthreads();
            }
        for(int p=split+tid;p<round[r+1];p+=stride)
            for(int t=path[p];t<path[p+1];++t) {
                int k=unit[t];
                for(int q=urow[k];q<urow[k+1];++q) g.factor[uentry[q]]=factor_entry(g,uentry[q]);
                float pivot=GLD(g.factor,diag[k]);
                for(int q=lcol[k];q<lcol[k+1];++q) g.factor[lentry[q]]=factor_entry(g,lentry[q])/pivot;
            }
        grid_sync();
    }
}
__device__ void factor_solve(const Graph& g,const float* c,float* out,bool trans) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    int gwarp=tid/32,gwarps=stride/32,lane=threadIdx.x&31;
    const int *unit=FP(PATH_UNIT),*path=FP(PATH_OFFSET),*round=FP(ROUND_OFFSET),*heavy=FP(ROUND_HEAVY),*diag=FP(DIAG),*order=FP(ORDER);
    const int *fptr=trans?FP(UCOL_PTR):FP(LROW_PTR),*fentry=trans?FP(UCOL_ENTRY):FP(LROW_ENTRY),*findex=trans?FP(UCOL_INDEX):FP(LROW_INDEX);
    const int *bptr=trans?FP(LCOL_PTR):FP(UROW_PTR),*bentry=trans?FP(LCOL_ENTRY):FP(UROW_ENTRY),*bindex=trans?FP(LCOL_INDEX):FP(UROW_INDEX);
    float *y=g.lu_forward,*x=g.lu_backward;
    for(int r=g.rounds-1;r>=0;--r) {
        int split=round[r]+heavy[r];
        for(int p=round[r]+gwarp;p<split;p+=gwarps) {
            for(int t=path[p];t<path[p+1];++t) {
                int k=unit[t];
                float value=GLD(c,order[k])-factor_dot_warp(g,fentry,findex,fptr[k],fptr[k+1],-1,y);
                if(lane==0) y[k]=trans?value/GLD(g.factor,diag[k]):value;
                __syncwarp();
            }
        }
        for(int p=split+tid;p<round[r+1];p+=stride)
            for(int t=path[p];t<path[p+1];++t) {
                int k=unit[t];float value=GLD(c,order[k]);
                for(int q=fptr[k];q<fptr[k+1];++q) value-=GLD(g.factor,fentry[q])*GLD(y,findex[q]);
                y[k]=trans?value/GLD(g.factor,diag[k]):value;
            }
        grid_sync();
    }
    for(int r=0;r<g.rounds;++r) {
        int split=round[r]+heavy[r];
        for(int p=round[r]+gwarp;p<split;p+=gwarps) {
            for(int t=path[p+1]-1;t>=path[p];--t) {
                int k=unit[t];
                float value=GLD(y,k)-factor_dot_warp(g,bentry,bindex,bptr[k],bptr[k+1],k,x);
                if(!trans) value/=GLD(g.factor,diag[k]);
                if(lane==0) {x[k]=value;out[order[k]]=trans?value*GLD(g.scaling,order[k]):value;}
                __syncwarp();
            }
        }
        for(int p=split+tid;p<round[r+1];p+=stride)
            for(int t=path[p+1]-1;t>=path[p];--t) {
                int k=unit[t];float value=GLD(y,k);
                for(int q=bptr[k];q<bptr[k+1];++q) if(bindex[q]!=k) value-=GLD(g.factor,bentry[q])*GLD(x,bindex[q]);
                if(!trans) value/=GLD(g.factor,diag[k]);
                x[k]=value;out[order[k]]=trans?value*GLD(g.scaling,order[k]):value;
            }
        grid_sync();
    }
}
__device__ void graph_precondition(const Graph& g,const float* in,float* out,bool trans) {
    if(trans) {
        int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
        for(int i=tid;i<g.z.n;i+=stride) g.pre[i]=GLD(in,i)/GLD(g.scaling,i);
        grid_sync();in=g.pre;
    }
    factor_solve(g,in,out,trans);
}

// Fast 1-solve direct path + BiCGStab refinement for full-map transpose adjoint
__device__ int graph_linear_fast_adjoint(const Graph& g,float* sh,int& solves_used,float& error) {
    const Net& z=g.z;int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    solves_used=0;
    graph_matrix(g);grid_sync();
    for(int i=tid;i<z.n;i+=stride) g.rhs[i]*=GLD(g.scaling,i);
    grid_sync();
    float rhs_norm=graph_norm(g,g.rhs,sh);
    if(!(rhs_norm>0.f)) {
        for(int i=tid;i<z.n;i+=stride) {z.dh[i]=0.f;g.solution_low[i]=0.f;}
        grid_sync();error=0.f;return 0;
    }
    if(!isfinite(rhs_norm)) {error=INFINITY;return 2;}
    factor_assemble(g);grid_sync();factor_numeric(g);
    graph_precondition(g,g.rhs,g.delta,true);
    ++solves_used;
    for(int i=tid;i<z.n;i+=stride) {z.dh[i]=GLD(g.delta,i);g.solution_low[i]=0.f;}
    grid_sync();
    graph_matvec<true>(g,z.dh,g.tmp,true);
    for(int i=tid;i<z.n;i+=stride) g.tmp[i]=GLD(g.rhs,i)-GLD(g.tmp,i);
    grid_sync();
    error=graph_norm(g,g.tmp,sh)/rhs_norm;
    if(isfinite(error) && error<GRAPH_LINEAR_TOLERANCE) return 0;

    for(int restart=0;restart<GRAPH_LINEAR_RESTARTS;++restart) {
        float initial=graph_norm(g,g.tmp,sh);
        if(!(initial>0.f)||!isfinite(initial)) return 2;
        for(int i=tid;i<z.n;i+=stride) {float r=GLD(g.tmp,i)/initial;g.r[i]=g.rhat[i]=r;g.p[i]=g.v[i]=g.delta[i]=0.f;}
        grid_sync();float rho0=1.f,alpha=1.f,omega=1.f;
        for(int it=0;it<GRAPH_LINEAR_ITERATIONS;++it) {
            float rho=graph_dot(g,g.rhat,g.r,sh);
            if(!isfinite(rho)||fabsf(rho)<1e-30f) break;
            float beta=(rho/rho0)*(alpha/omega);
            for(int i=tid;i<z.n;i+=stride) g.p[i]=GLD(g.r,i)+beta*(GLD(g.p,i)-omega*GLD(g.v,i));
            grid_sync();graph_precondition(g,g.p,g.tmp,true);++solves_used;graph_matvec(g,g.tmp,g.v,true);
            float den=graph_dot(g,g.rhat,g.v,sh);if(!isfinite(den)||fabsf(den)<1e-30f) break;alpha=rho/den;
            for(int i=tid;i<z.n;i+=stride) {g.s[i]=GLD(g.r,i)-alpha*GLD(g.v,i);g.delta[i]+=alpha*GLD(g.tmp,i);}
            grid_sync();
            if(graph_norm(g,g.s,sh)*(initial/rhs_norm)<GRAPH_LINEAR_TOLERANCE) break;
            graph_precondition(g,g.s,g.tmp,true);++solves_used;graph_matvec(g,g.tmp,g.t,true);
            float2 dots=graph_dot_pair(g,g.t,g.s,sh);float tt=dots.x,ts=dots.y;
            if(!(tt>1e-30f)||!isfinite(tt)) break;omega=ts/tt;
            if(!isfinite(omega)||fabsf(omega)<1e-20f) break;
            for(int i=tid;i<z.n;i+=stride) {g.delta[i]+=omega*GLD(g.tmp,i);g.r[i]=GLD(g.s,i)-omega*GLD(g.t,i);}
            grid_sync();if(graph_norm(g,g.r,sh)*(initial/rhs_norm)<GRAPH_LINEAR_TOLERANCE) break;rho0=rho;
        }
        for(int i=tid;i<z.n;i+=stride) {
            Pair x=pair_add({GLD(z.dh,i),GLD(g.solution_low,i)},graph_product(initial,GLD(g.delta,i)));
            z.dh[i]=x.hi;g.solution_low[i]=x.lo;
        }
        grid_sync();graph_matvec<true>(g,z.dh,g.tmp,true);
        for(int i=tid;i<z.n;i+=stride) g.tmp[i]=GLD(g.rhs,i)-GLD(g.tmp,i);
        grid_sync();error=graph_norm(g,g.tmp,sh)/rhs_norm;
        if(isfinite(error)&&error<GRAPH_LINEAR_TOLERANCE) return 0;
        if(!isfinite(error)) return 2;
    }
    return 1;
}

// ------------------------------------------------------------------------
// Coupled-component forward solver (pure trees excluded, active-component skipping, 1-solve fast path)

#define CFP(name) (g.c_plan+g.c_plan[FACTOR_##name])

__device__ __forceinline__ float comp_coupling_weight(int phase) {
    return phase==0 ? 0.f : (phase>0 && phase<GRAPH_COUPLING_FULL_PHASE ? ldexpf(1.f,phase-GRAPH_COUPLING_FULL_PHASE) : 1.f);
}

__device__ __forceinline__ float graph_damping(const Graph& g,int i) {
    int c=g.u_comp[i];
    if(c<0) return 0.f;
    float sigma=GLD(g.z.tf,c*TF+SIGMA);
    if(!(sigma>0.f)) return 0.f;
    const Net& z=g.z;
    float d=GLD(z.area,i)+g.dt*fabsf(GLD(z.qi,i));
    for(int e=z.cp[i];e<z.cp[i+1];++e) d+=g.dt*fabsf(GLD(z.qd,z.kids[e]));
    for(int e=g.offset[i];e<g.offset[i+1];++e) {
        int link=g.link[e],p=abs(link)-1;
        d+=g.dt*fabsf(link>0?GLD(g.sum,3*p+1):GLD(g.sum,3*p+2));
    }
    return sigma*d;
}

__device__ float coupled_norm(const Graph& g,const float* a,bool all_active,float* sh) {
    if(all_active && g.ngu_g==g.z.n) return graph_norm(g,a,sh);
    float mx=0.f;
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    if(all_active) {
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            float v=fabsf(GLD(a,g.g_unit[idx]));
            mx=fmaxf(mx,isfinite(v)?v:INFINITY);
        }
    } else {
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            if(g.z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
            float v=fabsf(GLD(a,i));
            mx=fmaxf(mx,isfinite(v)?v:INFINITY);
        }
    }
    return graph_reduce<true>(g,mx,sh);
}
__device__ float coupled_dot(const Graph& g,const float* a,const float* b,bool all_active,float* sh) {
    if(all_active && g.ngu_g==g.z.n) return graph_dot(g,a,b,sh);
    float tot=0.f;
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    if(all_active) {
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            tot+=GLD(a,i)*GLD(b,i);
        }
    } else {
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            if(g.z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
            tot+=GLD(a,i)*GLD(b,i);
        }
    }
    return graph_reduce<false>(g,tot,sh);
}
__device__ float2 coupled_dot_pair(const Graph& g,const float* t,const float* s,bool all_active,float* sh) {
    if(all_active && g.ngu_g==g.z.n) return graph_dot_pair(g,t,s,sh);
    float tt=0.f,ts=0.f;
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    if(all_active) {
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            float a=GLD(t,i);tt+=a*a;ts+=a*GLD(s,i);
        }
    } else {
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            if(g.z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
            float a=GLD(t,i);tt+=a*a;ts+=a*GLD(s,i);
        }
    }
    int lane=threadIdx.x&31,warp=threadIdx.x/32,warps=blockDim.x/32;
    int slot=(g.reduction_epoch++&1)*2*gridDim.x;
    for(int d=16;d;d/=2) {tt+=__shfl_down_sync(0xffffffff,tt,d);ts+=__shfl_down_sync(0xffffffff,ts,d);}
    if(lane==0) {sh[warp]=tt;sh[warps+warp]=ts;}
    __syncthreads();
    if(threadIdx.x==0) {
        float a=0,b=0;for(int w=0;w<warps;++w) {a+=sh[w];b+=sh[warps+w];}
        g.reduction[slot+2*blockIdx.x]=a;g.reduction[slot+2*blockIdx.x+1]=b;
    }
    grid_sync();
    if(threadIdx.x==0) {
        float a=0,b=0;for(int k=0;k<gridDim.x;++k) {a+=GLD(g.reduction,slot+2*k);b+=GLD(g.reduction,slot+2*k+1);}
        sh[0]=a;sh[1]=b;
    }
    __syncthreads();float2 result={sh[0],sh[1]};__syncthreads();return result;
}

template <bool COMPENSATED=false>
__device__ void coupled_matvec(const Graph& g,const float* x,float* out,bool all_active) {
    const Net& z=g.z;
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    bool full_dom=(all_active && g.ngu_g==z.n);
    for(int idx=tid;idx<g.ngu_g;idx+=stride) {
        int i=full_dom?idx:g.g_unit[idx];
        if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
        bool bound=GLD(g.bound,i)>.5f;Pair xi=graph_x<COMPENSATED>(g,x,i);
        Pair value=bound?xi:graph_scale(xi,GLD(z.area,i));
        if(!bound) {
            float damp=graph_damping(g,i);
            if(damp>0.f) value=pair_add(value,graph_scale(xi,damp));
            Pair flux=graph_scale(xi,GLD(z.qi,i));
            if(z.edge[i])flux=pair_add(flux,graph_scale(graph_x<COMPENSATED>(g,x,z.down[i]),GLD(z.qd,i)));
            value=pair_add(value,graph_scale(flux,g.dt));
            for(int e=z.cp[i];e<z.cp[i+1];++e) {
                int j=z.kids[e];Pair flux_j=pair_add(graph_scale(graph_x<COMPENSATED>(g,x,j),GLD(z.qi,j)),graph_scale(xi,GLD(z.qd,j)));
                value=pair_add(value,graph_scale(flux_j,-g.dt));
            }
            for(int e=g.offset[i];e<g.offset[i+1];++e) {
                int link=g.link[e],p=abs(link)-1;
                Pair flux_p=pair_add(graph_scale(graph_x<COMPENSATED>(g,x,g.src[p]),GLD(g.sum,3*p+1)),
                                     graph_scale(graph_x<COMPENSATED>(g,x,g.dst[p]),GLD(g.sum,3*p+2)));
                value=pair_add(value,graph_scale(flux_p,link>0?g.dt:-g.dt));
            }
        }
        out[i]=(value.hi+value.lo)*GLD(g.scaling,i);
    }
    grid_sync();
}

__device__ void coupled_matrix(const Graph& g,bool all_active) {
    const Net& z=g.z;
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    bool full_dom=(all_active && g.ngu_g==z.n);
    for(int idx=tid;idx<g.ngu_g;idx+=stride) {
        int i=full_dom?idx:g.g_unit[idx];
        if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
        bool dry=GLD(g.bound,i)>.5f;
        float cdiag=GLD(z.area,i)+g.dt*GLD(z.qi,i)+graph_damping(g,i);
        for(int e=z.cp[i];e<z.cp[i+1];++e) cdiag-=g.dt*GLD(z.qd,z.kids[e]);
        for(int e=g.offset[i];e<g.offset[i+1];++e) {
            int link=g.link[e],p=abs(link)-1;
            cdiag+=g.dt*(link>0?GLD(g.sum,3*p+1):-GLD(g.sum,3*p+2));
        }
        g.scaling[i]=dry?1.f:1.f/cdiag;
    }
    grid_sync();
}

__device__ void core_factor_assemble(const Graph& g,bool all_active) {
    const Net& z=g.z;int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    const int *start=CFP(SOURCE_PTR),*code=CFP(SOURCE_CODE);
    for(int e=tid;e<g.nc_entries;e+=stride) {
        if(!all_active) {
            int c=g.ce_comp[e];
            if(c<0 || z.ti[c*TI+SOLVE]==0) continue;
        }
        float value=0.f;
        for(int q=start[e];q<start[e+1];++q) {
            int kind=code[q]&7,k=code[q]>>3;
            if(kind==0) value+=1.f;
            else if(kind==1) {if(z.edge[k]&&!(GLD(g.bound,k)>.5f)) value+=g.dt*GLD(z.qd,k)*GLD(g.scaling,k);}
            else if(kind==2) {int d=z.down[k];if(!(GLD(g.bound,d)>.5f)) value-=g.dt*GLD(z.qi,k)*GLD(g.scaling,d);}
            else if(kind==3) {int u=g.src[k];if(!(GLD(g.bound,u)>.5f)) value+=g.dt*GLD(g.sum,3*k+2)*GLD(g.scaling,u);}
            else {int d=g.dst[k];if(!(GLD(g.bound,d)>.5f)) value-=g.dt*GLD(g.sum,3*k+1)*GLD(g.scaling,d);}
        }
        g.factor[e]=value;
    }
}
__device__ __forceinline__ float core_entry_serial(const Graph& g,int e) {
    const int *start=CFP(PRODUCT_PTR),*lower=CFP(PRODUCT_LOWER),*upper=CFP(PRODUCT_UPPER);
    float value=GLD(g.factor,e);
    for(int k=start[e];k<start[e+1];++k) value-=GLD(g.factor,lower[k])*GLD(g.factor,upper[k]);
    return value;
}
__device__ __forceinline__ float core_entry_warp(const Graph& g,int e,int lane) {
    const int *start=CFP(PRODUCT_PTR),*lower=CFP(PRODUCT_LOWER),*upper=CFP(PRODUCT_UPPER);
    float sum=0.f;
    for(int k=start[e]+lane;k<start[e+1];k+=32) sum+=GLD(g.factor,lower[k])*GLD(g.factor,upper[k]);
    for(int d=16;d;d/=2) sum+=__shfl_xor_sync(0xffffffff,sum,d);
    return GLD(g.factor,e)-sum;
}
__device__ void core_factor_numeric(const Graph& g,bool all_active) {
    const Net& z=g.z;
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    int warp=threadIdx.x>>5,warps=blockDim.x>>5,lane=threadIdx.x&31;
    const int *unit=CFP(PATH_UNIT),*path=CFP(PATH_OFFSET),*round=CFP(ROUND_OFFSET),*heavy=CFP(ROUND_HEAVY),*diag=CFP(DIAG),*order=CFP(ORDER);
    const int *urow=CFP(UROW_PTR),*uentry=CFP(UROW_ENTRY),*lcol=CFP(LCOL_PTR),*lentry=CFP(LCOL_ENTRY);
    for(int r=g.nc_rounds-1;r>=0;--r) {
        int split=round[r]+heavy[r];
        for(int p=round[r]+blockIdx.x;p<split;p+=gridDim.x) {
            if(!all_active && path[p]<path[p+1]) {
                int c=g.u_comp[order[unit[path[p]]]];
                if(c<0 || z.ti[c*TI+SOLVE]==0) continue;
            }
            for(int t=path[p];t<path[p+1];++t) {
                int k=unit[t];
                for(int q=urow[k]+warp;q<urow[k+1];q+=warps) {
                    float v=core_entry_warp(g,uentry[q],lane);
                    if(lane==0) g.factor[uentry[q]]=v;
                }
                __syncthreads();
                float pivot=GLD(g.factor,diag[k]);
                for(int q=lcol[k]+warp;q<lcol[k+1];q+=warps) {
                    float v=core_entry_warp(g,lentry[q],lane);
                    if(lane==0) g.factor[lentry[q]]=v/pivot;
                }
                __syncthreads();
            }
        }
        for(int p=split+tid;p<round[r+1];p+=stride) {
            if(!all_active && path[p]<path[p+1]) {
                int c=g.u_comp[order[unit[path[p]]]];
                if(c<0 || z.ti[c*TI+SOLVE]==0) continue;
            }
            for(int t=path[p];t<path[p+1];++t) {
                int k=unit[t];
                for(int q=urow[k];q<urow[k+1];++q) g.factor[uentry[q]]=core_entry_serial(g,uentry[q]);
                float pivot=GLD(g.factor,diag[k]);
                for(int q=lcol[k];q<lcol[k+1];++q) g.factor[lentry[q]]=core_entry_serial(g,lentry[q])/pivot;
            }
        }
        grid_sync();
    }
}

__device__ __forceinline__ float core_dot_warp(const Graph& g,const int* entry,const int* index,
                                               int begin,int end,int skip,const float* x,int lane) {
    float sum=0.f;
    for(int q=begin+lane;q<end;q+=32) if(index[q]!=skip) sum+=GLD(g.factor,entry[q])*GLD(x,index[q]);
    for(int d=16;d;d/=2) sum+=__shfl_xor_sync(0xffffffff,sum,d);
    return sum;
}
__device__ void core_factor_solve(const Graph& g,const float* c_vec,float* out,bool trans,bool all_active) {
    const Net& z=g.z;
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    int gwarp=tid>>5,gwarps=stride>>5,lane=threadIdx.x&31;
    const int *unit=CFP(PATH_UNIT),*path=CFP(PATH_OFFSET),*round=CFP(ROUND_OFFSET),*heavy=CFP(ROUND_HEAVY),*diag=CFP(DIAG),*order=CFP(ORDER);
    const int *fptr=trans?CFP(UCOL_PTR):CFP(LROW_PTR),*fentry=trans?CFP(UCOL_ENTRY):CFP(LROW_ENTRY),*findex=trans?CFP(UCOL_INDEX):CFP(LROW_INDEX);
    const int *bptr=trans?CFP(LCOL_PTR):CFP(UROW_PTR),*bentry=trans?CFP(LCOL_ENTRY):CFP(UROW_ENTRY),*bindex=trans?CFP(LCOL_INDEX):CFP(UROW_INDEX);
    float *y=g.lu_forward,*x=g.lu_backward;
    for(int r=g.nc_rounds-1;r>=0;--r) {
        int split=round[r]+heavy[r];
        for(int p=round[r]+gwarp;p<split;p+=gwarps) {
            if(!all_active && path[p]<path[p+1]) {
                int c=g.u_comp[order[unit[path[p]]]];
                if(c<0 || z.ti[c*TI+SOLVE]==0) continue;
            }
            for(int t=path[p];t<path[p+1];++t) {
                int k=unit[t];
                float value=GLD(c_vec,order[k])-core_dot_warp(g,fentry,findex,fptr[k],fptr[k+1],-1,y,lane);
                if(lane==0) y[k]=trans?value/GLD(g.factor,diag[k]):value;
                __syncwarp();
            }
        }
        for(int p=split+tid;p<round[r+1];p+=stride) {
            if(!all_active && path[p]<path[p+1]) {
                int c=g.u_comp[order[unit[path[p]]]];
                if(c<0 || z.ti[c*TI+SOLVE]==0) continue;
            }
            for(int t=path[p];t<path[p+1];++t) {
                int k=unit[t];float value=GLD(c_vec,order[k]);
                for(int q=fptr[k];q<fptr[k+1];++q) value-=GLD(g.factor,fentry[q])*GLD(y,findex[q]);
                y[k]=trans?value/GLD(g.factor,diag[k]):value;
            }
        }
        grid_sync();
    }
    for(int r=0;r<g.nc_rounds;++r) {
        int split=round[r]+heavy[r];
        for(int p=round[r]+gwarp;p<split;p+=gwarps) {
            if(!all_active && path[p]<path[p+1]) {
                int c=g.u_comp[order[unit[path[p]]]];
                if(c<0 || z.ti[c*TI+SOLVE]==0) continue;
            }
            for(int t=path[p+1]-1;t>=path[p];--t) {
                int k=unit[t];
                float value=GLD(y,k)-core_dot_warp(g,bentry,bindex,bptr[k],bptr[k+1],k,x,lane);
                if(!trans) value/=GLD(g.factor,diag[k]);
                if(lane==0) {x[k]=value;out[order[k]]=trans?value*GLD(g.scaling,order[k]):value;}
                __syncwarp();
            }
        }
        for(int p=split+tid;p<round[r+1];p+=stride) {
            if(!all_active && path[p]<path[p+1]) {
                int c=g.u_comp[order[unit[path[p]]]];
                if(c<0 || z.ti[c*TI+SOLVE]==0) continue;
            }
            for(int t=path[p+1]-1;t>=path[p];--t) {
                int k=unit[t];float value=GLD(y,k);
                for(int q=bptr[k];q<bptr[k+1];++q) if(bindex[q]!=k) value-=GLD(g.factor,bentry[q])*GLD(x,bindex[q]);
                if(!trans) value/=GLD(g.factor,diag[k]);
                x[k]=value;out[order[k]]=trans?value*GLD(g.scaling,order[k]):value;
            }
        }
        grid_sync();
    }
}

__device__ int graph_linear_coupled(const Graph& g,float* sh,int& solves_used,float& error) {
    const Net& z=g.z;
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
    solves_used=0;
    if(g.ngu_g==0) {error=0.f;return 0;}
    bool all_active=(g.ngc==1);

    coupled_matrix(g,all_active);
    for(int idx=tid;idx<g.ngu_g;idx+=stride) {
        int i=g.g_unit[idx];
        if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
        g.rhs[i]*=GLD(g.scaling,i);
    }
    grid_sync();

    float rhs_norm=coupled_norm(g,g.rhs,all_active,sh);
    if(!(rhs_norm>0.f)) {
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            if(all_active || z.ti[g.u_comp[i]*TI+SOLVE]>0) {z.dh[i]=0.f;g.solution_low[i]=0.f;}
        }
        grid_sync();
        error=0.f;return 0;
    }
    if(!isfinite(rhs_norm)) {error=INFINITY;return 2;}

    core_factor_assemble(g,all_active);grid_sync();
    core_factor_numeric(g,all_active);
    core_factor_solve(g,g.rhs,g.delta,false,all_active);
    ++solves_used;
    for(int idx=tid;idx<g.ngu_g;idx+=stride) {
        int i=g.g_unit[idx];
        if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
        z.dh[i]=GLD(g.delta,i);g.solution_low[i]=0.f;
    }
    grid_sync();
    coupled_matvec<true>(g,z.dh,g.tmp,all_active);
    for(int idx=tid;idx<g.ngu_g;idx+=stride) {
        int i=g.g_unit[idx];
        if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
        g.tmp[i]=GLD(g.rhs,i)-GLD(g.tmp,i);
    }
    grid_sync();
    error=coupled_norm(g,g.tmp,all_active,sh)/rhs_norm;
    if(isfinite(error) && error<GRAPH_LINEAR_TOLERANCE) return 0;

    for(int restart=0;restart<GRAPH_LINEAR_RESTARTS;++restart) {
        float initial=coupled_norm(g,g.tmp,all_active,sh);
        if(!(initial>0.f)||!isfinite(initial)) break;
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
            float r=GLD(g.tmp,i)/initial;g.r[i]=g.rhat[i]=r;g.p[i]=g.v[i]=g.delta[i]=0.f;
        }
        grid_sync();
        float rho0=1.f,alpha=1.f,omega=1.f;
        for(int it=0;it<GRAPH_LINEAR_ITERATIONS;++it) {
            float rho=coupled_dot(g,g.rhat,g.r,all_active,sh);
            if(!isfinite(rho)||fabsf(rho)<1e-30f) break;
            float beta=(rho/rho0)*(alpha/omega);
            for(int idx=tid;idx<g.ngu_g;idx+=stride) {
                int i=g.g_unit[idx];
                if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
                g.p[i]=GLD(g.r,i)+beta*(GLD(g.p,i)-omega*GLD(g.v,i));
            }
            grid_sync();
            core_factor_solve(g,g.p,g.tmp,false,all_active);
            ++solves_used;
            coupled_matvec<false>(g,g.tmp,g.v,all_active);
            float den=coupled_dot(g,g.rhat,g.v,all_active,sh);
            if(!isfinite(den)||fabsf(den)<1e-30f) break;
            alpha=rho/den;
            for(int idx=tid;idx<g.ngu_g;idx+=stride) {
                int i=g.g_unit[idx];
                if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
                g.s[i]=GLD(g.r,i)-alpha*GLD(g.v,i);g.delta[i]+=alpha*GLD(g.tmp,i);
            }
            grid_sync();
            if(coupled_norm(g,g.s,all_active,sh)*(initial/rhs_norm)<GRAPH_LINEAR_TOLERANCE) break;
            core_factor_solve(g,g.s,g.tmp,false,all_active);
            ++solves_used;
            coupled_matvec<false>(g,g.tmp,g.t,all_active);
            float2 dots=coupled_dot_pair(g,g.t,g.s,all_active,sh);
            float tt=dots.x,ts=dots.y;
            if(!(tt>1e-30f)||!isfinite(tt)) break;
            omega=ts/tt;
            if(!isfinite(omega)||fabsf(omega)<1e-20f) break;
            for(int idx=tid;idx<g.ngu_g;idx+=stride) {
                int i=g.g_unit[idx];
                if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
                g.delta[i]+=omega*GLD(g.tmp,i);g.r[i]=GLD(g.s,i)-omega*GLD(g.t,i);
            }
            grid_sync();
            if(coupled_norm(g,g.r,all_active,sh)*(initial/rhs_norm)<GRAPH_LINEAR_TOLERANCE) break;
            rho0=rho;
        }
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
            Pair x=pair_add({GLD(z.dh,i),GLD(g.solution_low,i)},graph_product(initial,GLD(g.delta,i)));
            z.dh[i]=x.hi;g.solution_low[i]=x.lo;
        }
        grid_sync();
        coupled_matvec<true>(g,z.dh,g.tmp,all_active);
        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=g.g_unit[idx];
            if(!all_active && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
            g.tmp[i]=GLD(g.rhs,i)-GLD(g.tmp,i);
        }
        grid_sync();
        error=coupled_norm(g,g.tmp,all_active,sh)/rhs_norm;
        if(isfinite(error) && error<GRAPH_LINEAR_TOLERANCE) return 0;
        if(!isfinite(error)) break;
    }
    return isfinite(error)?1:2;
}

// Per-component Newton controller for coupled bifurcation graph
__device__ void finish_graph_comp(const Graph& g,int c,int* s,float* f) {
    int st=s[ST],phase=s[PHASE];
    float merit=f[MERIT],init0=f[INITIAL];
    s[SUM_G]+=s[STEPS];s[SUM_FE]+=s[FEVALS];
    if(phase==-1) f[INITIAL]=merit;
    if(phase==DAMPED_PHASE) f[SIGMA]=0.f;
    if(st!=0 && phase==-1) {
        bool keep_surf=isfinite(merit) && merit<init0;
        start_phase(g.z,s,f,-1);
        s[PHASE]=RESTART_PHASE;s[BT]=0;
        s[RESET]=keep_surf?0:1;
        return;
    }
    if(st!=0 && phase==RESTART_PHASE) {
        start_damped(s,f);return;
    }
    if(phase==DAMPED_PHASE) {
        if(st!=0) {start_phase(g.z,s,f,0);return;}
        finish_tree(g.z,g.c_rep[c],s,f);return;
    }
    if(phase==RESTART_PHASE || phase==-1) {
        finish_tree(g.z,g.c_rep[c],s,f);return;
    }
    if(phase>=0 && phase<GRAPH_COUPLING_FULL_PHASE) {start_phase(g.z,s,f,phase+1);return;}
    if(phase==GRAPH_COUPLING_FULL_PHASE && st!=0) {
        f[BEST]=merit;s[BEST_ST]=st;s[SAVE]|=1;
        start_phase(g.z,s,f,13);return;
    }
    if(phase>=13) {
        if(merit<f[BEST]) {f[BEST]=merit;s[BEST_ST]=st;s[SAVE]|=2;}
        if(st!=0 && phase<20) {start_phase(g.z,s,f,phase+1);return;}
        if(st!=0) {s[SAVE]|=4;s[MODE]=FINAL;return;}
    }
    finish_tree(g.z,g.c_rep[c],s,f);
}

__device__ void control_graph_comp(const Graph& g,int c,float mt,float line) {
    const Net& z=g.z;
    int* s=z.ti+(long)c*TI;
    float* f=z.tf+(long)c*TF;
    s[COMMIT]=s[SAVE]=s[RESET]=s[SOLVE]=0;
    int mode=s[MODE];
    if(mode==DONE) return;
    float tol=z.tol;
    if(mode==FINAL) {
        f[MERIT]=mt;s[ST]=mt<tol?0:s[BEST_ST];
        finish_tree(z,g.c_rep[c],s,f);return;
    }
    if(s[PHASE]==DAMPED_PHASE && mode!=RESTORE) {
        ++s[FEVALS];
        if(mode==TRIAL) {
            float old=f[MERIT];
            if(!isfinite(mt)) {
                f[SIGMA]=fmaxf(f[SIGMA]*10.f,1e-3f);s[MODE]=EVAL;
                if(++s[TOTALBT]>=40) {s[ST]=2;s[MODE]=RESTORE;}
                return;
            }
            s[COMMIT]=1;f[MERIT]=mt;++s[STEPS];++s[DSTEPS];
            float sigma=f[SIGMA]*fminf(line/old,NEWTON_DAMPED_GROWTH);
            if(sigma==0.f && line>=old) sigma=1e-3f;
            f[SIGMA]=sigma<1e-7f?0.f:sigma;
        } else {
            f[MERIT]=mt;
        }
        if(!(mt>=tol)) {s[ST]=0;f[SIGMA]=0.f;finish_graph_comp(g,c,s,f);return;}
        if(s[STEPS]>=NEWTON_DAMPED_STEPS) {s[ST]=1;f[SIGMA]=0.f;finish_graph_comp(g,c,s,f);return;}
        s[SOLVE]=1;s[MODE]=TRIAL;f[LAM]=1.f;s[BT]=0;return;
    }
    bool newton_done=false;
    bool fixed_w=(s[PHASE]!=-1);
    int step_cap=(!fixed_w && z.maxit>36)?36:z.maxit;
    if(mode==EVAL) {
        if(s[STEPS]==0 && s[PHASE]==-1) f[INITIAL]=mt;
        f[MERIT]=mt;++s[FEVALS];
        newton_done=!(mt>=tol);
    } else if(mode==TRIAL) {
        ++s[FEVALS];++s[TOTALBT];
        float old=f[MERIT],lam=f[LAM];
        float cmp_v=fixed_w?line:mt;
        if(isfinite(mt) && (mt<tol || cmp_v<=(1.f-0.01f*lam)*old)) {
            s[COMMIT]=1;f[MERIT]=mt;++s[STEPS];
            newton_done=!(s[STEPS]<step_cap && mt>=tol);
        } else {
            f[LAM]=lam*0.5f;
            if(++s[BT]<LINE_SEARCH) return;
            s[MODE]=RESTORE;return;
        }
    } else {
        f[MERIT]=mt;s[FEVALS]+=2;s[ST]=mt<tol?0:2;
        finish_graph_comp(g,c,s,f);return;
    }
    if(!newton_done) {
        s[SOLVE]=1;s[MODE]=TRIAL;f[LAM]=1.f;s[BT]=0;return;
    }
    ++s[FEVALS];
    s[ST]=f[MERIT]<tol?0:1;
    finish_graph_comp(g,c,s,f);
}

#define GRAPH_LOCALS(time_step, record) \
    SOLVE_LOCALS(time_step, record); \
    const int* bsrc = bifurcation_catchment_idx_ptr; \
    const int* bdst = bifurcation_downstream_idx_ptr; \
    const int* boffset = incident_offset_ptr; \
    const int* blink = incident_path_ptr; \
    const int* factor_plan = factor_layout_ptr; \
    const float* bw = bifurcation_width_ptr; \
    const float* blen = bifurcation_length_ptr; \
    const float* belv = bifurcation_elevation_ptr; \
    const float* bn = bifurcation_manning_ptr; \
    const float* qb0 = bifurcation_outflow_ptr; \
    float* path_values = graph_path_values_ptr; \
    float* path_sum = graph_path_sum_ptr; \
    float* factor_values = graph_factor_ptr; \
    float* graph_rhs = graph_rhs_ptr; \
    float* graph_tmp = graph_temporary_ptr; \
    float* graph_bound = graph_bound_ptr; \
    float* graph_scaling = graph_scaling_ptr; \
    float* graph_balance_scale = graph_balance_scale_ptr; \
    float* graph_solution_low = graph_solution_low_ptr; \
    float* lu_forward = graph_lu_forward_ptr; \
    float* lu_backward = graph_lu_backward_ptr; \
    float* graph_r = graph_r_ptr; \
    float* graph_rhat = graph_rhat_ptr; \
    float* graph_p = graph_p_ptr; \
    float* graph_v = graph_v_ptr; \
    float* graph_s = graph_s_ptr; \
    float* graph_t = graph_t_ptr; \
    float* graph_delta = graph_delta_ptr; \
    float* graph_pre = graph_precondition_rhs_ptr; \
    float* graph_reduction = graph_reduction_ptr; \
    float* linear_record = linear_record_ptr; \
    int m = num_bifurcation_paths; \
    int blevels = num_bifurcation_levels; \
    int factor_entries = num_factor_entries; \
    int factor_rounds = num_factor_rounds; \
    const int* g_unit = graph_unit_ptr; \
    const int* u_comp = unit_comp_ptr; \
    const int* p_comp = path_comp_ptr; \
    const int* c_trees = coupled_trees_ptr; \
    const int* ct_comp = coupled_tree_comp_ptr; \
    const int* c_rep = comp_rep_tree_ptr; \
    const int* c_plan = core_factor_layout_ptr; \
    const int* ce_comp = core_entry_comp_ptr; \
    int ngc = num_graph_comps; \
    int ngu_g = num_graph_units; \
    int nct = num_coupled_trees; \
    int nc_entries = num_core_factor_entries; \
    int nc_rounds = num_core_factor_rounds;

#define GRAPH_NET \
    SOLVE_NET; \
    Graph g{z,bsrc,bdst,boffset,blink,factor_plan,bw,blen,belv,bn,qb0, \
        path_values,path_sum,factor_values,graph_rhs,graph_tmp,graph_bound,graph_scaling, \
        graph_balance_scale,graph_solution_low,lu_forward,lu_backward,graph_r,graph_rhat,graph_p,graph_v, \
        graph_s,graph_t,graph_delta,graph_pre,graph_reduction,linear_record, \
        m,blevels,factor_entries,factor_rounds, \
        g_unit,u_comp,p_comp,c_trees,ct_comp,c_rep,c_plan,ce_comp, \
        ngc,ngu_g,nct,nc_entries,nc_rounds,0};

#ifdef HYDROFORGE_LAUNCHES_graph_solve_pure_trees
extern "C" __global__ void __launch_bounds__(TREE_THREADS) graph_solve_pure_trees(HYDROFORGE_KERNEL_PARAMETERS) {
    SOLVE_LOCALS(time_step_ptr, newton_record_ptr);
    SOLVE_NET;
    solve_one_block_tree(z, pure_block_tree_ptr[blockIdx.x]);
}
#endif

#ifdef HYDROFORGE_LAUNCHES_graph_solve
extern "C" __global__ void __launch_bounds__(GRAPH_THREADS,1) graph_solve(HYDROFORGE_KERNEL_PARAMETERS) {
    GRAPH_LOCALS(time_step_ptr, newton_record_ptr);
    GRAPH_NET;
    __shared__ float sh[2*GRAPH_THREADS/32];
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;

    if(g.ngc==0) {
        if(tid==0) {linear_record[0]=0;linear_record[1]=0;linear_record[2]=0;linear_record[3]=0;}
        return;
    }
    unsigned* gctl=ctl+5;
    bool full_dom=(g.ngc==1 && g.ngu_g==n);
    g.dt=*dt0;
    for(int idx=tid;idx<g.ngu_g;idx+=stride) {int i=full_dom?idx:g.g_unit[idx];start_unit(z,i);g.bound[i]=0.f;}
    for(int c=tid;c<g.ngc;c+=stride) start_tree(z,c);
    if(tid==0) {linear_record[0]=0;linear_record[1]=0;linear_record[2]=0;linear_record[3]=0;}
    grid_sync();

    for(int round=0;round<MAX_ROUNDS;++round) {
        g.dt=*dt0;
        if(g.ngc==1) {
            if(z.ti[MODE]!=DONE) {
                if(full_dom) {
                    for(int i=tid;i<n;i+=stride) eval_stage<true>(z,i);
                    grid_sync();
                    for(int i=tid;i<n;i+=stride) eval_flux<true>(z,i,g.dt);
                } else {
                    for(int idx=tid;idx<g.ngu_g;idx+=stride) eval_stage<true>(z,g.g_unit[idx]);
                    grid_sync();
                    for(int idx=tid;idx<g.ngu_g;idx+=stride) eval_flux<true>(z,g.g_unit[idx],g.dt);
                }
                float cw=comp_coupling_weight(z.ti[PHASE]);
                for(int p=tid;p<m;p+=stride) graph_flux_cw(g,p,cw);
            } else {
                grid_sync();
            }
        } else {
            for(int idx=tid;idx<g.ngu_g;idx+=stride) {
                int i=g.g_unit[idx],c=g.u_comp[i];
                if(z.ti[c*TI+MODE]!=DONE) eval_stage<true>(z,i);
            }
            grid_sync();
            for(int idx=tid;idx<g.ngu_g;idx+=stride) {
                int i=g.g_unit[idx],c=g.u_comp[i];
                if(z.ti[c*TI+MODE]!=DONE) eval_flux<true>(z,i,g.dt);
            }
            for(int p=tid;p<m;p+=stride) {
                int c=g.p_comp[p];
                if(z.ti[c*TI+MODE]!=DONE)
                    graph_flux_cw(g,p,comp_coupling_weight(z.ti[c*TI+PHASE]));
            }
        }
        grid_sync();

        if(g.ngc==1) {
            float m_loc=0.f,l_loc=0.f;
            int mode0=ldi<true>(z.ti,MODE);
            bool need_line=(mode0==TRIAL) && (ldi<true>(z.ti,PHASE)!=-1);
            for(int idx=tid;idx<g.ngu_g;idx+=stride) {
                int i=full_dom?idx:g.g_unit[idx];
                float sc=graph_residual(g,i);
                m_loc=fmaxf(m_loc,sc);
                if(need_line) {
                    float lv=isfinite(sc)?fabsf(GLD(z.R,i))*GLD(g.balance_scale,i):INFINITY;
                    l_loc=fmaxf(l_loc,lv);
                }
            }
            float merit=graph_reduce<true>(g,m_loc,sh);
            float line=need_line?graph_reduce<true>(g,l_loc,sh):merit;
            if(tid==0) {
                control_graph_comp(g,0,merit,line);
                gctl[0]=(z.ti[MODE]!=DONE)?1u:0u;
                gctl[1]=(z.ti[SOLVE]!=0)?1u:0u;
            }
            grid_sync();
        } else {
            for(int idx=tid;idx<g.ngu_g;idx+=stride) {
                int i=g.g_unit[idx],c=g.u_comp[i];
                int mode_c=z.ti[c*TI+MODE];
                if(mode_c==DONE) continue;
                float sc=graph_residual(g,i);
                unsigned um=__float_as_uint(sc);
                if(um>__ldcg(z.tm+c)) atomicMax(z.tm+c,um);
                if(mode_c==TRIAL && z.ti[c*TI+PHASE]!=-1) {
                    float lv=isfinite(sc)?fabsf(GLD(z.R,i))*GLD(g.balance_scale,i):INFINITY;
                    unsigned ul=__float_as_uint(lv);
                    if(ul>__ldcg(z.tline+c)) atomicMax(z.tline+c,ul);
                }
            }
            if(tid<2) gctl[tid]=0u;
            grid_sync();
            for(int c=tid;c<g.ngc;c+=stride) {
                float merit=__uint_as_float(__ldcg(z.tm+c));
                float line=__uint_as_float(__ldcg(z.tline+c));
                z.tm[c]=0;z.tline[c]=0;
                control_graph_comp(g,c,merit,line);
                if(z.ti[c*TI+MODE]!=DONE) atomicAdd(gctl+0,1u);
                if(z.ti[c*TI+SOLVE]!=0) atomicAdd(gctl+1,1u);
            }
            grid_sync();
        }

        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=full_dom?idx:g.g_unit[idx];
            apply_unit<true>(z,i,full_dom?0:g.u_comp[i]);
        }
        if(__ldcg(gctl+0)==0u) {
            grid_sync();
            for(int k=tid;k<g.nct;k+=stride) {
                int t=g.c_trees[k],rep=g.c_rep[g.ct_comp[k]];
                if(t!=rep) for(int j=0;j<16;++j) z.meta[16*(long)t+j]=GLD(z.meta,16*(long)rep+j);
            }
            return;
        }
        bool any_solving=__ldcg(gctl+1)>0u;
        grid_sync();

        if(any_solving) {
            for(int idx=tid;idx<g.ngu_g;idx+=stride) {
                int i=full_dom?idx:g.g_unit[idx];
                if(!full_dom && g.ngc>1 && z.ti[g.u_comp[i]*TI+SOLVE]==0) continue;
                g.balance_scale[i]=GLD(g.rhs,i);
                bool bound=!((GLD(z.ref,i)+H[i])+GLD(z.h,i)>0) && GLD(z.R,i)>0;
                g.rhs[i]=bound?0.f:-GLD(z.R,i);
                g.bound[i]=bound?1.f:0.f;
            }
            grid_sync();
            int solves_used=0;float error=0.f;
            int result=graph_linear_coupled(g,sh,solves_used,error);
            if(tid==0) {
                linear_record[0]=result;linear_record[1]=solves_used;
                linear_record[2]=error;linear_record[3]+=solves_used;
            }
            grid_sync();
        }

        for(int idx=tid;idx<g.ngu_g;idx+=stride) {
            int i=full_dom?idx:g.g_unit[idx];
            trial_unit<true>(z,i,full_dom?0:g.u_comp[i]);
        }
        grid_sync();
    }
    if(tid==0) atomicExch(ctl+ERROR,NEWTON_ROUND_LIMIT);
}
#endif

#ifdef HYDROFORGE_LAUNCHES_graph_commit
extern "C" __global__ void graph_commit(HYDROFORGE_KERNEL_PARAMETERS) {
    const int* offset = incident_offset_ptr;
    const int* link = incident_path_ptr;
    const float* path = graph_path_values_ptr;
    float* qb = bifurcation_outflow_ptr;
    float* inflow = inflow_ptr;
    int n = num_catchments;
    int m = num_bifurcation_paths;
    int bl = num_bifurcation_levels;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*bl) qb[i]=path[(long)i*7];
    if(i<n) {
        float received=0;
        for(int e=offset[i];e<offset[i+1];++e) {int l=link[e],p=abs(l)-1;for(int k=0;k<bl;++k) {float q=path[((long)p*bl+k)*7];received+=fmaxf(l>0?-q:q,0.f);}}
        inflow[i]+=received;
    }
}
#endif

// Rebuild of the converged graph snapshot: the evaluation the graph Newton
// ended with, at the taped surface and full coupling, and the pathway Jacobians.
#define GRAPH_REBUILD \
    GRAPH_LOCALS(time_step_ptr, nullptr); \
    GRAPH_NET; \
    g.dt = *dt0; g.coupling_weight = 1; \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef HYDROFORGE_LAUNCHES_graph_rebuild_stage
extern "C" __global__ void graph_rebuild_stage(HYDROFORGE_KERNEL_PARAMETERS) {
    GRAPH_REBUILD;
    if (tid >= n) return;
    float r = converged_surface_ptr[2 * (long)tid], x = converged_surface_ptr[2 * (long)tid + 1];
    z.ref[tid] = r; z.h[tid] = x; z.tref[tid] = r; z.ht[tid] = x;
    eval_stage<false>(z, tid);
}
#endif
#ifdef HYDROFORGE_LAUNCHES_graph_rebuild_flux
extern "C" __global__ void graph_rebuild_flux(HYDROFORGE_KERNEL_PARAMETERS) {
    GRAPH_REBUILD;
    if (tid < n) eval_flux<false>(z, tid, g.dt);
    if (tid < m) graph_flux(g, tid);
}
#endif
#ifdef HYDROFORGE_LAUNCHES_graph_rebuild_residual
extern "C" __global__ void graph_rebuild_residual(HYDROFORGE_KERNEL_PARAMETERS) {
    GRAPH_REBUILD;
    if (tid < n) graph_residual(g, tid);
}
#endif
#ifdef HYDROFORGE_LAUNCHES_graph_rebuild_snapshot
extern "C" __global__ void graph_rebuild_snapshot(HYDROFORGE_KERNEL_PARAMETERS) {
    GRAPH_REBUILD;
    if (tid < n)
        snapshot_unit(tid, g.dt, Q0, nc, nf, L, W, H, A, tab, bed, elev, dist, down, edge, ref, h, S,
                      area, Q, qi, qd, R, jacobian_snapshot_ptr, eps, levels);
    if (tid < m * blevels)
        for (int k = 0; k < 7; ++k) bifurcation_jacobian_ptr[(long)tid * 7 + k] = path_values[(long)tid * 7 + k];
}
#endif

__device__ Pair graph_lambda(const Graph& g,const float* saved,int i) {
    return J(i,5)>.5f?Pair{0,0}:Pair{GLD(g.z.dh,i),GLD(g.solution_low,i)};
}
__device__ float graph_flux_adjoint(const Graph& g,Pair u,Pair d,float objective) {
    Pair value=pair_add({objective,0},graph_scale(pair_add(u,{-d.hi,-d.lo}),-g.dt));
    return value.hi+value.lo;
}

#ifdef HYDROFORGE_LAUNCHES_graph_adjoint
extern "C" __global__ void __launch_bounds__(GRAPH_THREADS,1) graph_adjoint(HYDROFORGE_KERNEL_PARAMETERS) {
    GRAPH_LOCALS(nullptr, nullptr);
    const float* saved = jacobian_snapshot_ptr;
    const float* bsaved = bifurcation_jacobian_ptr;
    float* gQ = grad_outflow_ptr;
    float* gS = grad_storage_ptr;
    const float* gh = grad_water_surface_ptr;
    float* gQB = grad_bifurcation_outflow_ptr;
    float* gr = grad_runoff_ptr;
    float* gw = grad_river_width_ptr;
    float* gheight = grad_river_height_ptr;
    float* gnc = grad_river_manning_ptr;
    float* gnf = grad_flood_manning_ptr;
    float* gnb = grad_bifurcation_manning_ptr;
    // gQ, gS and gQB hold the result adjoints and become the start adjoints in
    // place, each after its last read (grid syncs separate the phases).
    GRAPH_NET;
    __shared__ float sh[2*GRAPH_THREADS/32];
    int tid=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;g.dt=J(0,16);
    for(int i=tid;i<n;i+=stride) {z.qi[i]=J(i,2);z.qd[i]=J(i,3);z.area[i]=J(i,4);g.bound[i]=J(i,5);}
    for(int p=tid;p<m;p+=stride) {float a=0,b=0;for(int l=0;l<blevels;++l) {int k=p*blevels+l;a+=bsaved[(long)k*7+1];b+=bsaved[(long)k*7+2];}BS(g,p,1)=a;BS(g,p,2)=b;}
    grid_sync();
    for(int i=tid;i<n;i+=stride) {
        Pair value=pair_add(pair_add(graph_product(J(i,4),gS[i]),graph_product(J(i,2),gQ[i])),{gh[i],0});
        for(int e=cp[i];e<cp[i+1];++e) {int j=kids[e];value=pair_add(value,graph_product(J(j,3),gQ[j]));}
        for(int e=boffset[i];e<boffset[i+1];++e) {int link=blink[e],p=abs(link)-1;for(int l=0;l<blevels;++l) {int k=p*blevels+l;value=pair_add(value,graph_product(bsaved[(long)k*7+(link>0?1:2)],gQB[k]));}}
        g.rhs[i]=value.hi+value.lo;
    }
    grid_sync();
    int iterations=0;float error=0.f;
    int result=graph_linear_fast_adjoint(g,sh,iterations,error);
    if(tid==0) {linear_record[0]=result;linear_record[1]=iterations;linear_record[2]=error;if(result!=0)report(status,6,0);}
    grid_sync();
    for(int i=tid;i<n;i+=stride) {
        Pair lambda=graph_lambda(g,saved,i);
        float ce=graph_flux_adjoint(g,lambda,z.edge[i]?graph_lambda(g,saved,z.down[i]):Pair{0,0},gQ[i]);g.tmp[i]=ce;
        Pair rg=graph_scale(lambda,g.dt);gr[i]=rg.hi+rg.lo;gQ[i]=ce*J(i,12);
        Pair difference=pair_add({gS[i],0},{-lambda.hi,-lambda.lo});
        gw[i]+=(difference.hi+difference.lo)*J(i,7)+ce*J(i,9);gnc[i]+=ce*J(i,13);gnf[i]+=ce*J(i,14);
    }
    grid_sync();
    for(int i=tid;i<n;i+=stride) {
        Pair lambda=graph_lambda(g,saved,i),difference=pair_add({gS[i],0},{-lambda.hi,-lambda.lo});
        float height=(difference.hi+difference.lo)*J(i,8)+GLD(g.tmp,i)*J(i,10);
        if(J(i,5)>.5f)height-=(GLD(z.dh,i)+GLD(g.solution_low,i));
        for(int e=cp[i];e<cp[i+1];++e) {int j=kids[e];height+=GLD(g.tmp,j)*J(j,11);}
        for(int e=boffset[i];e<boffset[i+1];++e) {int link=blink[e],p=abs(link)-1,u=bsrc[p],d=bdst[p];for(int l=0;l<blevels;++l) {int k=p*blevels+l;float ce=graph_flux_adjoint(g,graph_lambda(g,saved,u),graph_lambda(g,saved,d),gQB[k]);height+=ce*bsaved[(long)k*7+(link>0?5:6)];}}
        gheight[i]+=height;gS[i]=lambda.hi+lambda.lo;
    }
    grid_sync();
    for(int p=tid;p<m;p+=stride) {int u=bsrc[p],d=bdst[p];for(int l=0;l<blevels;++l) {int k=p*blevels+l;float ce=graph_flux_adjoint(g,graph_lambda(g,saved,u),graph_lambda(g,saved,d),gQB[k]);gQB[k]=ce*bsaved[(long)k*7+3];gnb[k]+=ce*bsaved[(long)k*7+4];}}
}
#endif

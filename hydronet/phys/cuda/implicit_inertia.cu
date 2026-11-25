// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// SPDX-License-Identifier: Apache-2.0
#include <cooperative_groups.h>
// Fully implicit CaMa-style local inertia and continuity on a river forest.
// Unknown: end-of-step surface relative to bank height. All arithmetic is FP32.
// Every tree keeps its own Newton iteration and line search. A forest tree whose
// line search fails restarts with a fixed-weight line search (phase 21), then
// damped Newton (phase 22), then time-step continuation (phases 0..12) and offset
// restarts (13..20); the coupled graph uses its own controller in bifurcation.cu
// (phases 21, 22, coupling continuation 0..7, then 13..20). Small trees are solved by one block each; large trees by one persistent
// launch over the whole device, their units spread over all blocks. Both run
// the same physical residual and tolerance; parallel path elimination may
// round differently from the level-by-level block solve.
// Grid trees use serial, warp and block elimination according to path length,
// with one grid barrier per heavy-path round.
// The backward launches solve the transposed converged Jacobian, not Newton history.
// Codes of hydronet.config (BaseModule.newton_status).
#define NEWTON_NOT_CONVERGED 1
#define NEWTON_ROUND_LIMIT 3
#define INVALID_ROUGHNESS 4
#define INVALID_STATE 5

struct State { float S, area, kc, kf, dkc, dkf; };
__device__ __forceinline__ float p23(float a) {
#if defined(HYDROFORGE_FAST_MATH) && !defined(__HIPCC_RTC__)
    return a > 0.0f ? __powf(a, 2.0f / 3.0f) : 0.0f;
#else
    return cbrtf(a * a);
#endif
}
__device__ void channel(float w,float H,float d,float& k,float& dk) {
    if (!(d>0)) { k=dk=0; return; }
    float a=w*d,p=w+2*fminf(d,H),r=p23(a/p);
    k=a*r; dk=r*(1.6666667f*w-(d<H?1.3333334f*a/p:0));
}
// Floodplain conveyance summed over the equal-width bands, each with its own
// wetted area and bed (divided-channel method). A band entering the flow starts
// at zero conveyance, so the sum is continuous and rises with the surface even
// across flat bands; above the table the outer band gains the wetted wall.
__device__ void flood_bands(int i,float h,float dw,const float* tab,const float* bed,int levels,
                            float& k,float& dk) {
    k=dk=0;float t0=0,b0=0;
    for(int b=0;b<levels;++b) {
        float t=tab[(long)i*levels+b],b1=bed[(long)i*levels+b],seg=b1-b0;
        if(!(h>t0)) break;
        float a,p,da,dp;
        if(!(h<t)) {
            a=dw*((h-t0)-.5f*(t-t0));p=seg;da=dw;dp=0;
            if(b==levels-1) {p+=h-t;dp=1;}
        } else {
            float x=h-t0,run=x*(dw/(t-t0));
            a=.5f*run*x;p=x*(seg/(t-t0));da=run;dp=seg/(t-t0);
        }
        if(a>0) {float r=p23(a/p);k+=a*r;dk+=r*(1.6666667f*da-.6666667f*a*dp/p);}
        t0=t;b0=b1;
    }
}
__device__ State stage(int i,float h,const float* L,const float* W,const float* H,const float* A,
                        const float* tab,const float* bed,int levels) {
    State s;float l=L[i],w=W[i],hh=H[i],d=fmaxf(h+hh,0.0f);
    s.kf=s.dkf=0;channel(w,hh,d,s.kc,s.dkc);
    if (!(h>0)) { s.S=l*w*d;s.area=l*w;return s; }
    float dw=A[i]/l/levels,ph=0,pw=w,acc=0,nh=0;int lev=0;
    for(int k=0;k<levels;++k) {
        nh=tab[(long)i*levels+k];float nw=w+(k+1)*dw;
        if (!(h>nh)) break;
        acc+=0.5f*(pw+nw)*(nh-ph);ph=nh;pw=nw;++lev;
    }
    float grad=lev<levels?(nh-ph)/dw:0,x=h-ph,run=grad>0?x/grad:0;
    s.S=l*(w*hh+acc+pw*x+0.5f*run*x);s.area=l*(pw+run);
    flood_bands(i,h,dw,tab,bed,levels,s.kf,s.dkf);
    return s;
}
__device__ State stage_pair(int i,float ref,float x,const float* L,const float* W,const float* H,const float* A,
                             const float* tab,const float* bed,int levels) {
    float h=ref+x;
    if(h>0) return stage(i,h,L,W,H,A,tab,bed,levels);
    State s;float d=fmaxf((ref+H[i])+x,0.0f);
    s.S=L[i]*W[i]*d;s.area=L[i]*W[i];s.kf=s.dkf=0;
    channel(W[i],H[i],d,s.kc,s.dkc);return s;
}
__device__ float invert(int i,float S,const float* L,const float* W,const float* H,const float* A,
                         const float* tab,int levels) {
    S=fmaxf(S,0);float l=L[i],w=W[i],hh=H[i],acc=l*w*hh;
    if(S<=acc) return S/(l*w)-hh;
    float dw=A[i]/l/levels,ph=0,pw=w,nh=0;int lev=0;
    for(int k=0;k<levels;++k) {
        nh=tab[(long)i*levels+k];float nw=w+(k+1)*dw;
        float next=acc+l*0.5f*(pw+nw)*(nh-ph);
        if(!(S>next)) break;
        acc=next;ph=nh;pw=nw;++lev;
    }
    float grad=lev<levels?(nh-ph)/dw:0,extra=fmaxf(S-acc,0);
    if(grad>0) return ph+(2*extra/l)/(sqrtf(pw*pw+2*extra/(grad*l))+pw);
    return ph+extra/(pw*l);
}
__device__ float phi(float j,float e) { return j*rsqrtf(sqrtf(j*j+e*e)); }
__device__ float dphi(float j,float e) { float a=j*j+e*e;return (0.5f*j*j+e*e)/(a*sqrtf(sqrtf(a))); }
// Error-free FP32 transforms; there is no FP64 arithmetic or storage.
struct Pair {float hi,lo;};
__device__ Pair pair_sum(float a,float b) {
    float s=__fadd_rn(a,b),v=__fsub_rn(s,a);
    return {s,__fadd_rn(__fsub_rn(a,__fsub_rn(s,v)),__fsub_rn(b,v))};
}
__device__ Pair pair_diff(float a,float b) {return pair_sum(a,-b);}
__device__ Pair pair_add(Pair a,Pair b) {
    Pair s=pair_sum(a.hi,b.hi);
    float e=__fadd_rn(s.lo,__fadd_rn(a.lo,b.lo));return pair_sum(s.hi,e);
}
__device__ float head_pair(float ei,float ed,float ri,float rd,float xi,float xd) {
    Pair p=pair_add(pair_diff(ei,ed),pair_diff(ri,rd));
    p=pair_add(p,pair_diff(xi,xd));return __fadd_rn(p.hi,p.lo);
}
__device__ void trial_pair(float ref,float x,float step,float H,float& rnew,float& xnew) {
    Pair p=pair_add({ref,x},{step,0});
    if(__fadd_rn(__fadd_rn(p.hi,H),p.lo)<0) {p.hi=-H;p.lo=0;}
    rnew=p.hi;xnew=p.lo;
}

__device__ __constant__ float CONTINUATION_BETA[13]={1e-6f,3e-6f,1e-5f,3e-5f,1e-4f,3e-4f,.001f,.003f,.01f,.03f,.1f,.3f,1.f};
// Fully implicit local inertia; same regularized diffusion steady state.
// q + (g*A*dt/C^2)*q*abs(q) = q_previous + g*A*dt*Jeff.
// q_previous is immutable during all Newton stages and has an exact VJP.
__device__ void momentum_flux(float q0,float A,float Ai,float Ad,float K,float Ki,float Kd,
 float j,float ji,float jd,float dt,float eps,float& q,float& qi,float& qd) {
    q=qi=qd=0;
    if(!(A>1e-8f)||!(K>1e-12f))return;
    float f=phi(j,eps),fp=dphi(j,eps),je=f*fabsf(f),dje=2*fabsf(f)*fp;
    float a=9.8f*dt*A,b=a/(K*K),u=q0+a*je;
    float den=sqrtf(1+4*b*fabsf(u));q=2*u/(1+den);
    float fric=b*q*fabsf(q);
    qi=((q-q0)*(Ai/A)+2*fric*(Ki/K)+a*dje*ji)/den;
    qd=((q-q0)*(Ad/A)+2*fric*(Kd/K)+a*dje*jd)/den;
}

// Five local parameter tangents, all FP32. No differentiation of Newton iterates.
// Tangents: W_i, H_i, H_down, n_channel_owner, n_flood_owner.
struct Jet {
    float v,d[5];
    __device__ Jet(float a=0):v(a){for(int k=0;k<5;++k)d[k]=0;}
};

__device__ Jet variable(float v,int k){Jet a(v);a.d[k]=1;return a;}
__device__ Jet operator+(Jet a,Jet b){Jet c(a.v+b.v);for(int k=0;k<5;++k)c.d[k]=a.d[k]+b.d[k];return c;}
__device__ Jet operator-(Jet a,Jet b){Jet c(a.v-b.v);for(int k=0;k<5;++k)c.d[k]=a.d[k]-b.d[k];return c;}
__device__ Jet operator*(Jet a,Jet b){Jet c(a.v*b.v);for(int k=0;k<5;++k)c.d[k]=a.d[k]*b.v+a.v*b.d[k];return c;}
__device__ Jet operator/(Jet a,Jet b){Jet c(a.v/b.v);for(int k=0;k<5;++k)c.d[k]=(a.d[k]-c.v*b.d[k])/b.v;return c;}
__device__ Jet positive(Jet a){return a.v>0?a:Jet(0);}
__device__ Jet minimum(Jet a,Jet b){return a.v<b.v?a:b;}
__device__ Jet pow23(Jet a){Jet c(p23(a.v));if(a.v>0)for(int k=0;k<5;++k)c.d[k]=(2.f/3.f)*c.v/a.v*a.d[k];return c;}
struct JState {Jet S,kc,kf;};
__device__ JState geometry_jet(int i,Jet fill,Jet depth,Jet w,Jet H,
 const float* L,const float* A,const float* tab,const float* bed,int levels){
    JState s;depth=positive(depth);
    if(depth.v>0){Jet ac=w*depth,pc=w+2.f*minimum(depth,H);s.kc=ac*pow23(ac/pc);}
    if(!(fill.v>0)){s.S=L[i]*w*depth;return s;}
    // Floodplain area is independent of channel width at fixed bank-relative level.
    float dw=A[i]/L[i]/levels,ph=0,pw=0,acc=0,nh=0;int lev=0;
    for(int k=0;k<levels;++k){
        nh=tab[(long)i*levels+k];float nw=(k+1)*dw;
        if(!(fill.v>nh))break;
        acc+=.5f*(pw+nw)*(nh-ph);ph=nh;pw=nw;++lev;
    }
    float slope=lev<levels?(nh-ph)/dw:0;
    Jet x=fill-ph,run=slope>0?x/slope:Jet(0);
    s.S=L[i]*(w*(H+fill)+positive(acc+pw*x+.5f*run*x));
    // The band sum of flood_bands.
    float t0=0,b0=0;
    for(int b=0;b<levels;++b){
        float t=tab[(long)i*levels+b],b1=bed[(long)i*levels+b],seg=b1-b0;
        if(!(fill.v>t0))break;
        Jet a,p;
        if(!(fill.v<t)){a=((fill-t0)-.5f*(t-t0))*dw;p=Jet(seg);if(b==levels-1)p=p+(fill-t);}
        else{Jet xb=fill-t0;a=.5f*(dw/(t-t0))*xb*xb;p=xb*(seg/(t-t0));}
        if(a.v>0)s.kf=s.kf+a*pow23(a/p);
        t0=t;b0=b1;
    }
    return s;
}
__device__ void parameter_partials(int i,const float* ref,const float* x,
 const float* Q0,const float* nc,const float* nf,
 const float* L,const float* W,const float* H,const float* A,const float* tab,const float* bed,
 const float* elev,const float* dist,const int* down,const bool* edges,
 float eps,float dt,int levels,float q,float* out){
    bool edge=edges[i];int d=edge?down[i]:i;
    float head=edge?head_pair(elev[i],elev[d],ref[i],ref[d],x[i],x[d]):ref[i]+x[i];
    float j=edge?head/dist[i]:(ref[i]+x[i])/10000.f;
    Jet w=variable(W[i],0),Hj=variable(H[i],1),Hd=variable(H[d],2);
    Jet ownfill=Jet(ref[i])+x[i],owndepth=(Hj+ref[i])+x[i];
    JState accel=geometry_jet(i,ownfill,owndepth,w,Hj,L,A,tab,bed,levels);
    State exact_accel=stage_pair(i,ref[i],x[i],L,W,H,A,tab,bed,levels);
    if(edge&&head<0){
        Jet da=positive(owndepth-head);accel=geometry_jet(i,da-Hj,da,w,Hj,L,A,tab,bed,levels);
        exact_accel=stage_pair(i,-H[i],da.v,L,W,H,A,tab,bed,levels);
    }
    float Ae=exact_accel.S/L[i],f=phi(j,eps);Jet Aj=accel.S/L[i];
    bool reverse=edge&&(Q0[i]+9.8f*dt*Ae*f*fabsf(f)<0);int supplier=i;
    Jet nj=variable(nc[supplier],3),fj=variable(nf[supplier],4);
    JState c=geometry_jet(i,ownfill,owndepth,w,Hj,L,A,tab,bed,levels);
    State exact=stage_pair(i,ref[i],x[i],L,W,H,A,tab,bed,levels);
    if(reverse){
        Jet beddiff=(elev[d]-elev[i])+(Hj-Hd);
        Jet depth=positive(((Hd+ref[d])+x[d])+minimum(Jet(0),beddiff));
        c=geometry_jet(i,depth-Hj,depth,w,Hj,L,A,tab,bed,levels);
        exact=stage_pair(i,-H[i],depth.v,L,W,H,A,tab,bed,levels);
    }
    Jet Kj=c.kc/nj+c.kf/fj;
    float K=exact.kc/nc[supplier]+exact.kf/nf[supplier];
    for(int k=0;k<8;++k)out[k]=0;
    float actualdepth=fmaxf((ref[i]+H[i])+x[i],0.f);
    // At an exactly dry bed use the same right/semismooth derivative as the
    // forward Jacobian (area=L*W), so a dry surface follows d(h)/dH=-1.
    out[5]=L[i]*actualdepth;out[6]=L[i]*W[i];
    out[7]=reverse?1:0;
    if(!(Ae>1e-8f)||!(K>1e-12f)||(!edge&&!(q>0)))return;
    float b=9.8f*dt*Ae/(K*K),friction=b*q*fabsf(q),den=1+2*b*fabsf(q);
    for(int k=0;k<5;++k)out[k]=((q-Q0[i])*Aj.d[k]/Ae+2*friction*Kj.d[k]/K)/den;
}



// ------------------------------------------------------------------------
// Forward: one persistent launch of one block per SM.

#define LINE_SEARCH 16
#define MAX_ROUNDS (1 << 22)
enum Mode {DONE = 0, EVAL = 1, TRIAL = 2, RESTORE = 3, FINAL = 4};
// Per-tree integers ti[t*TI+k] and floats tf[t*TF+k]. SAVE bits: 1 phase-12
// trace, 2 best state, 4 restore best. RESET: 1 inverted S0, 2 trace offset.
enum {MODE, PHASE, STEPS, BT, ST, FEVALS, TOTALBT, SUM_G, SUM_FE, DSTEPS, BEST_ST,
      SOLVE, COMMIT, SAVE, RESET, TI};
enum {DT, LAM, MERIT, BEST, INITIAL, SIGMA, TF};
static_assert(TI == TREE_INTEGERS && TF == TREE_SCALARS, "tree_control/tree_scalars shapes");
// Phases: -1 Newton from the start surface; RESTART_PHASE (21, fixed-weight line
// search from start surface) then DAMPED_PHASE (22, damped Newton); 0..12
// time-step continuation (forest) or 0..7 coupling continuation (graph controller
// in bifurcation.cu); 13..20 offset restarts from the best continuation surface.
#define RESTART_PHASE 21
#define DAMPED_PHASE 22
// ctl: two parities of active and solving trees, error.
enum {ACTIVE = 0, SOLVING = 2, ERROR = 4};

// First failure of the step (hydroforge ErrorField: failures, code, index).
__device__ void report(int* status, int code, int index) {
    if (atomicAdd(status, 1) == 0) { status[1] = code; status[2] = index; }
}

struct Net {
    const float *S0, *runoff, *Q0, *nc, *nf, *L, *W, *H, *A, *tab, *bed, *elev, *dist, *dt0;
    const int *down, *cp, *kids, *tree, *tsize, *troot, *pnodes, *pptr;
    // Large trees, solved by the whole grid: units, trees, paths by round.
    const int *gunit, *gtree, *gpath, *grr;
    // Small trees, one block each: trees, units by tree and level, level starts, first level of each tree.
    const int *btree, *tunit, *loff, *tlev;
    const bool* edge;
    float *h, *ht, *ref, *tref, *S, *area, *kc, *kf, *dkc, *dkf, *Q, *qi, *qd, *R, *wb, *wa,
        *ap, *rp, *dh, *b, *trace, *tf, *meta;
    int *ti, *status;
    unsigned *tm, *tline, *ctl;
    float eps, tol;
    int n, nt, levels, maxit, rounds, ngu, ngt;
};

// Workspace written by other blocks of the same launch (G: grid solve) is read
// from L2; a block solve reads its own tree's workspace through L1.
template <bool G> __device__ __forceinline__ float ld(const float* p, int i) { return G ? __ldcg(p + i) : p[i]; }
template <bool G> __device__ __forceinline__ int ldi(const int* p, int i) { return G ? __ldcg(p + i) : p[i]; }

// Grid kernels are cooperative launches: every block is resident.
__device__ __forceinline__ void grid_sync() { cooperative_groups::this_grid().sync(); }

// Evaluation of the trial surface (tref + ht) at the tree's time step.
template <bool G> __device__ void eval_stage(const Net& z, int i) {
    State s = stage_pair(i, ld<G>(z.tref, i), ld<G>(z.ht, i), z.L, z.W, z.H, z.A, z.tab, z.bed, z.levels);
    z.S[i] = s.S; z.area[i] = s.area; z.kc[i] = s.kc; z.kf[i] = s.kf; z.dkc[i] = s.dkc; z.dkf[i] = s.dkf;
}

template <bool G> __device__ void eval_flux(const Net& z, int i, float dt) {
    const float *L = z.L, *W = z.W, *H = z.H, *A = z.A, *tab = z.tab, *bed = z.bed;
    const int levels = z.levels;
    float eps = z.eps;
    bool edge = z.edge[i];
    int d = edge ? z.down[i] : i;
    float oi = ld<G>(z.tref, i), xi = ld<G>(z.ht, i), od = ld<G>(z.tref, d), xd = ld<G>(z.ht, d);
    float head = edge ? head_pair(z.elev[i], z.elev[d], oi, od, xi, xd) : oi + xi;
    float j = edge ? head / z.dist[i] : (oi + xi) / 10000.0f;
    State own;
    own.S = ld<G>(z.S, i); own.area = ld<G>(z.area, i); own.kc = ld<G>(z.kc, i); own.kf = ld<G>(z.kf, i);
    own.dkc = ld<G>(z.dkc, i); own.dkf = ld<G>(z.dkf, i);
    // Inertia can retain a flow sign opposite to the current slope. The
    // acceleration area is the edge cross-section filled to max(eta_i, eta_d).
    float depthA = fmaxf(((oi + H[i]) + xi) - fminf(head, 0.f) * (edge ? 1.f : 0.f), 0.f);
    State accel = own;
    if (edge && head < 0) accel = stage_pair(i, -H[i], depthA, L, W, H, A, tab, bed, levels);
    float Ae = accel.S / L[i], dAe = accel.area / L[i], f = phi(j, eps);
    bool reverse = edge && (z.Q0[i] + 9.8f * dt * Ae * f * fabsf(f) < 0);
    float inc = 1.f / z.nc[i], inf = 1.f / z.nf[i];
    float K = own.kc * inc + own.kf * inf, dK = own.dkc * inc + own.dkf * inf;
    if (reverse) {
        float beddiff = (z.elev[d] - z.elev[i]) + (H[i] - H[d]);
        float depth = fmaxf(((od + H[d]) + xd) + fminf(0, beddiff), 0.0f);
        State c = stage_pair(i, -H[i], depth, L, W, H, A, tab, bed, levels);
        K = c.kc * inc + c.kf * inf; dK = c.dkc * inc + c.dkf * inf;
    }
    bool max_at_down = edge && head < 0;
    float q, dqi, dqd;
    momentum_flux(z.Q0[i], Ae, max_at_down ? 0 : dAe, max_at_down ? dAe : 0, K, reverse ? 0 : dK,
                  reverse ? dK : 0, j, edge ? 1 / z.dist[i] : 1 / 10000.f,
                  edge ? -1 / z.dist[i] : 0, dt, eps, q, dqi, dqd);
    if (!edge && q < 0) q = dqi = dqd = 0;
    z.Q[i] = q; z.qi[i] = dqi; z.qd[i] = dqd;
}

// Normalized balance residual; with line, also |r| in the accepted iterate's weights.
template <bool G> __device__ float eval_residual(const Net& z, int i, float dt, float* line = nullptr) {
    float in = 0, ain = 0;
    for (int e = z.cp[i]; e < z.cp[i + 1]; ++e) { float q = ld<G>(z.Q, z.kids[e]); in += q; ain += fabsf(q); }
    float Qi = ld<G>(z.Q, i);
    float r = (ld<G>(z.S, i) - z.S0[i]) - dt * (z.runoff[i] + in - Qi);
    float scale = z.S0[i] + dt * (fabsf(z.runoff[i]) + ain + fabsf(Qi)) + 1000;
    float score = isfinite(r) && isfinite(Qi) ? fabsf(r) / scale : INFINITY;
    z.R[i] = r; z.wb[i] = 1 / scale;
    if (line) *line = isfinite(score) ? fabsf(r) * ld<G>(z.wa, i) : INFINITY;
    return score;
}

// Damping of a row in the damped phase: sigma times the row's own magnitude,
// its storage slope plus the magnitudes of its flux derivatives; zero otherwise.
template <bool G> __device__ float damping(const Net& z, int i, float dt) {
    float sigma = ld<G>(z.tf, z.tree[i] * TF + SIGMA);
    if (!(sigma > 0)) return 0;
    float d = ld<G>(z.area, i) + dt * fabsf(ld<G>(z.qi, i));
    for (int e = z.cp[i]; e < z.cp[i + 1]; ++e) d += dt * fabsf(ld<G>(z.qd, z.kids[e]));
    return sigma * d;
}

// Newton matrix row of one unit, after all of its upstream units: forward
// elimination, then back substitution from the downstream unit.
template <bool G> __device__ void eliminate_unit(const Net& z, int i, float dt) {
    bool dry = !((ld<G>(z.ref, i) + z.H[i]) + ld<G>(z.h, i) > 0) && ld<G>(z.R, i) > 0;
    bool hd = z.edge[i];
    float a = ld<G>(z.area, i) + dt * ld<G>(z.qi, i) + damping<G>(z, i, dt), r = -ld<G>(z.R, i);
    z.b[i] = dry || !hd ? 0 : dt * ld<G>(z.qd, i);
    for (int e = z.cp[i]; e < z.cp[i + 1]; ++e) {
        int j = z.kids[e];
        a -= dt * ld<G>(z.qd, j);
        float c = -dt * ld<G>(z.qi, j);
        a -= c * ld<G>(z.b, j) / ld<G>(z.ap, j);
        r -= c * ld<G>(z.rp, j) / ld<G>(z.ap, j);
    }
    z.ap[i] = dry ? 1 : a;
    z.rp[i] = dry ? 0 : r;
}

template <bool G> __device__ void substitute_unit(const Net& z, int i) {
    bool hd = z.edge[i];
    z.dh[i] = (ld<G>(z.rp, i) - (hd ? ld<G>(z.b, i) * ld<G>(z.dh, z.down[i]) : 0)) / ld<G>(z.ap, i);
}

// One heavy path, upstream to downstream. Upstream light paths were eliminated
// in deeper rounds; the heavy child is the previous unit.
template <bool G> __device__ void eliminate_path(const Net& z, int p, float dt) {
    // Keep the immediately preceding heavy child's pivots in registers. Light
    // children still read the globally visible pivots from the previous round.
    int previous=-1;float pa=0,pr=0,pb=0,pqi=0,pqd=0;
    for (int k=z.pptr[p];k<z.pptr[p+1];++k) {
        int i=z.pnodes[k];
        bool dry=!((ld<G>(z.ref,i)+z.H[i])+ld<G>(z.h,i)>0) && ld<G>(z.R,i)>0;
        float qi=ld<G>(z.qi,i),qd=ld<G>(z.qd,i);
        float a=ld<G>(z.area,i)+dt*qi+damping<G>(z,i,dt),r=-ld<G>(z.R,i);
        float b=dry||!z.edge[i]?0:dt*qd;
        for(int e=z.cp[i];e<z.cp[i+1];++e) {
            int j=z.kids[e];bool carried=j==previous;
            float qj=carried?pqd:ld<G>(z.qd,j);
            a-=dt*qj;
            float c=-dt*(carried?pqi:ld<G>(z.qi,j));
            float aj=carried?pa:ld<G>(z.ap,j),bj=carried?pb:ld<G>(z.b,j),rj=carried?pr:ld<G>(z.rp,j);
            a-=c*bj/aj;r-=c*rj/aj;
        }
        pa=dry?1:a;pr=dry?0:r;pb=b;pqi=qi;pqd=qd;previous=i;
        z.ap[i]=pa;z.rp[i]=pr;z.b[i]=b;
    }
}
template <bool G> __device__ void substitute_path(const Net& z, int p) {
    int out=z.pnodes[z.pptr[p+1]-1];
    float parent=z.edge[out]?ld<G>(z.dh,z.down[out]):0;
    for(int k=z.pptr[p+1]-1;k>=z.pptr[p];--k) {
        int i=z.pnodes[k];
        float value=(ld<G>(z.rp,i)-(z.edge[i]?ld<G>(z.b,i)*parent:0))/ld<G>(z.ap,i);
        z.dh[i]=value;parent=value;
    }
}

__device__ void start_phase(const Net& z, int* s, float* f, int phase) {
    s[PHASE] = phase; s[MODE] = EVAL; s[ST] = 0;
    s[STEPS] = s[FEVALS] = s[TOTALBT] = 0;
    s[RESET] = phase <= 0 ? 1 : (phase >= 13 ? 2 : 0);
    f[DT] = *z.dt0 * ((phase < 0 || phase >= 13) ? 1 : CONTINUATION_BETA[phase]);
}

__device__ void finish_tree(const Net& z, int t, int* s, float* f) {
    s[MODE] = DONE;
    float* m = z.meta + (long)t * 16;
    int phase = s[PHASE];
    m[0] = f[MERIT]; m[1] = s[SUM_G]; m[2] = s[ST]; m[3] = s[TOTALBT]; m[4] = s[SUM_G]; m[5] = 0;
    m[6] = s[SUM_FE]; m[7] = 0; m[8] = 0; m[9] = 0; m[10] = 0; m[11] = 0; m[12] = s[DSTEPS];
    m[13] = phase < 0 ? 1 : phase + 2; m[14] = phase >= 0; m[15] = f[INITIAL];
    if (s[ST] != 0) report(z.status, NEWTON_NOT_CONVERGED, t);
}

// Damped Newton of a forest tree, from the surface where the fixed-weight line
// search stopped: (J + sigma D) dh = -F with D each row's diagonal magnitude
// (damping). The damped step need not decrease the max-norm residual, so
// every finite step is taken; sigma follows the fixed-weight residual ratio,
// growing at most NEWTON_DAMPED_GROWTH-fold per step, and vanishes on convergence.
__device__ void start_damped(int* s, float* f) {
    s[PHASE] = DAMPED_PHASE; s[MODE] = EVAL; s[ST] = 0; s[STEPS] = s[FEVALS] = s[TOTALBT] = 0;
    s[BT] = 0; f[SIGMA] = NEWTON_DAMPED_SIGMA;
}
template <bool G, bool FIXED_WEIGHTS = false> __device__ void finish_phase(const Net& z, int t, int* s, float* f);
template <bool G> __device__ void damped_control(const Net& z, int t, int* s, float* f, float mt, float line) {
    float tol = z.tol;
    ++s[FEVALS];
    if (s[MODE] == TRIAL) {
        float old = f[MERIT];
        if (!isfinite(mt)) {  // re-evaluate the accepted surface, solve with more damping
            f[SIGMA] = fmaxf(f[SIGMA] * 10, 1e-3f); s[MODE] = EVAL;
            if (++s[TOTALBT] >= 40) { s[ST] = 2; s[MODE] = RESTORE; }
            return;
        }
        s[COMMIT] = 1; f[MERIT] = mt; ++s[STEPS]; ++s[DSTEPS];
        float sigma = f[SIGMA] * fminf(line / old, NEWTON_DAMPED_GROWTH);
        if (sigma == 0 && line >= old) sigma = 1e-3f;  // a growing residual re-enables damping
        f[SIGMA] = sigma < 1e-7f ? 0.f : sigma;
    } else {
        f[MERIT] = mt;
    }
    if (!(mt >= tol)) { s[ST] = 0; f[SIGMA] = 0; finish_phase<G>(z, t, s, f); return; }
    if (s[STEPS] >= NEWTON_DAMPED_STEPS) { s[ST] = 1; f[SIGMA] = 0; finish_phase<G>(z, t, s, f); return; }
    s[SOLVE] = 1; s[MODE] = TRIAL; f[LAM] = 1;
}

// End of one phase's Newton iteration. A forest tree whose line search fails
// restarts from the start surface with fixed weights (RESTART_PHASE), then
// continues with damped Newton (DAMPED_PHASE), and only then with continuation.
// Continuation phases 0..12 ramp the time step from the start surface (when
// FIXED_WEIGHTS is true, phases 21/22 are bypassed); phases 13..20 restart from
// offsets of the best phase-12 surface. The coupled graph uses its dedicated
// controller in bifurcation.cu (phases 21/22, coupling 0..7, then 13..20).
// Acceptance always needs the original full-step balance.
template <bool G, bool FIXED_WEIGHTS> __device__ void finish_phase(const Net& z, int t, int* s, float* f) {
    int st = s[ST], phase = s[PHASE];
    float merit = f[MERIT];
    s[SUM_G] += s[STEPS]; s[SUM_FE] += s[FEVALS];
    if (phase == -1) f[INITIAL] = merit;
    if (phase == DAMPED_PHASE) f[SIGMA] = 0;
    if (!FIXED_WEIGHTS && st != 0 && phase == -1) {
        start_phase(z, s, f, -1); s[PHASE] = RESTART_PHASE; s[BT] = 0;
        return;
    }
    if (!FIXED_WEIGHTS && st != 0 && phase == RESTART_PHASE) { start_damped(s, f); return; }
    // A forest tree that damped Newton cannot solve continues with the time-step
    // continuation from the start surface (needed on ~1 tree-day in 1000 on the
    // global 6' map).
    if (!FIXED_WEIGHTS && st != 0 && phase == DAMPED_PHASE) { start_phase(z, s, f, 0); return; }
    if (phase == RESTART_PHASE || phase == DAMPED_PHASE || (!FIXED_WEIGHTS && phase == -1)) {
        finish_tree(z, t, s, f); return;
    }
    if ((phase == -1 && st != 0) || (phase >= 0 && phase < 12)) { start_phase(z, s, f, phase + 1); return; }
    if (phase == 12 && st != 0) {
        f[BEST] = merit; s[BEST_ST] = st; s[SAVE] |= 1;
        start_phase(z, s, f, 13);
        return;
    }
    if (phase >= 13) {
        if (merit < f[BEST]) { f[BEST] = merit; s[BEST_ST] = st; s[SAVE] |= 2; }
        if (st != 0 && phase < 20) { start_phase(z, s, f, phase + 1); return; }
        if (st != 0) { s[SAVE] |= 4; s[MODE] = FINAL; return; }
    }
    finish_tree(z, t, s, f);
}

template <bool G, bool FIXED_WEIGHTS = false> __device__ void control(
    const Net& z, int t, float actual = 0, float line = 0) {
    int* s = z.ti + (long)t * TI;
    float* f = z.tf + (long)t * TF;
    s[COMMIT] = s[SAVE] = s[RESET] = s[SOLVE] = 0;
    int mode = s[MODE];
    if (mode == DONE) return;
    float mt = FIXED_WEIGHTS ? actual : __uint_as_float(__ldcg(z.tm + t)), tol = z.tol;
    z.tm[t] = 0;
    if (mode == FINAL) {  // the restored best surface of phase 20
        f[MERIT] = mt; s[ST] = mt < tol ? 0 : s[BEST_ST];
        finish_tree(z, t, s, f);
        return;
    }
    if (!FIXED_WEIGHTS && s[PHASE] == DAMPED_PHASE && mode != RESTORE) { damped_control<G>(z, t, s, f, mt, line); return; }
    bool newton_done = false;
    if (mode == EVAL) {
        f[MERIT] = mt; ++s[FEVALS];
        newton_done = !(mt >= tol);
    } else if (mode == TRIAL) {
        ++s[FEVALS]; ++s[TOTALBT];
        float old = f[MERIT], lam = f[LAM];
        bool fixed = FIXED_WEIGHTS || s[PHASE] == RESTART_PHASE;
        if (isfinite(mt) && (mt < tol || (fixed ? line : mt) <= (1 - 0.01f * lam) * old)) {
            s[COMMIT] = 1; f[MERIT] = mt; ++s[STEPS];
            newton_done = !(s[STEPS] < z.maxit && mt >= tol);
        } else {
            f[LAM] = lam * 0.5f;
            if (++s[BT] < LINE_SEARCH) return;  // retry with the reduced step
            s[MODE] = RESTORE;                  // re-evaluate the accepted surface
            return;
        }
    } else {  // RESTORE: the line search failed
        f[MERIT] = mt; s[FEVALS] += 2; s[ST] = mt < tol ? 0 : 2;
        finish_phase<G, FIXED_WEIGHTS>(z, t, s, f);
        return;
    }
    if (!newton_done) { s[SOLVE] = 1; s[MODE] = TRIAL; f[LAM] = 1; s[BT] = 0; return; }
    ++s[FEVALS];  // the final audit evaluates the accepted surface again
    s[ST] = f[MERIT] < tol ? 0 : 1;
    finish_phase<G, FIXED_WEIGHTS>(z, t, s, f);
}

// Work of one unit, tree or path in a round, shared by the grid and block solves.
__device__ void start_unit(const Net& z, int i) {
    float r0 = invert(i, z.S0[i], z.L, z.W, z.H, z.A, z.tab, z.levels);
    z.ref[i] = r0; z.h[i] = 0; z.tref[i] = r0; z.ht[i] = 0;
    if (!(z.nc[i] > 0 && isfinite(z.nc[i]) && z.nf[i] > 0 && isfinite(z.nf[i])))
        report(z.status, INVALID_ROUGHNESS, i);
}
__device__ void start_tree(const Net& z, int t) {
    int* s = z.ti + (long)t * TI;
    float* f = z.tf + (long)t * TF;
    for (int k = 0; k < TI; ++k) s[k] = 0;
    for (int k = 0; k < TF; ++k) f[k] = 0;
    start_phase(z, s, f, -1);
    s[RESET] = 0; s[BEST_ST] = 2; f[BEST] = 1e30f; z.tm[t] = 0; z.tline[t] = 0;
}
template <bool G> __device__ void stage_unit(const Net& z, int i) {
    if (ldi<G>(z.ti, z.tree[i] * TI + MODE) != DONE) eval_stage<G>(z, i);
}
template <bool G> __device__ void flux_unit(const Net& z, int i) {
    int t = z.tree[i];
    if (ldi<G>(z.ti, t * TI + MODE) != DONE) eval_flux<G>(z, i, ld<G>(z.tf, t * TF + DT));
}
template <bool G> __device__ void residual_unit(const Net& z, int i) {
    int t = z.tree[i];
    if (ldi<G>(z.ti, t * TI + MODE) == DONE) return;
    int phase = ldi<G>(z.ti, t * TI + PHASE);
    bool weighted = phase == RESTART_PHASE || phase == DAMPED_PHASE;
    float line = 0, score = eval_residual<G>(z, i, ld<G>(z.tf, t * TF + DT), weighted ? &line : nullptr);
    atomicMax(z.tm + t, __float_as_uint(score));
    if (weighted) atomicMax(z.tline + t, __float_as_uint(line));
}
// Commit, save and reset of the tree's decision; the last trees accept here too.
template <bool G> __device__ void apply_unit(const Net& z, int i, int t = -1) {
    const int* s = z.ti + (t < 0 ? z.tree[i] : t) * TI;
    float *ref = z.ref, *h = z.h, *trace = z.trace;
    int n = z.n;
    if (ldi<G>(s, COMMIT)) { ref[i] = ld<G>(z.tref, i); h[i] = ld<G>(z.ht, i); }
    // Fixed weights of the forest restart and damped phases: the accepted iterate's.
    if (ldi<G>(s, SOLVE)) { int phase = ldi<G>(s, PHASE);
        if (phase == RESTART_PHASE || phase == DAMPED_PHASE) z.wa[i] = ld<G>(z.wb, i); }
    int save = ldi<G>(s, SAVE), reset = ldi<G>(s, RESET);
    if (save & 1) { trace[i] = ld<G>(ref, i) + ld<G>(h, i); trace[n + i] = ld<G>(ref, i); trace[2 * n + i] = ld<G>(h, i); }
    if (save & 2) { trace[n + i] = ld<G>(ref, i); trace[2 * n + i] = ld<G>(h, i); }
    if (save & 4) { ref[i] = ld<G>(trace, n + i); h[i] = ld<G>(trace, 2 * n + i); }
    if (reset == 1) { ref[i] = invert(i, z.S0[i], z.L, z.W, z.H, z.A, z.tab, z.levels); h[i] = 0; }
    if (reset == 2) {
        const float offsets[8] = {-.01f, .01f, -.05f, .05f, -.2f, .2f, -.5f, .5f};
        ref[i] = fmaxf(-z.H[i], ld<G>(trace, i) + offsets[ldi<G>(s, PHASE) - 13]); h[i] = 0;
    }
}

// Bank-relative volume coordinate u = (V(h) - V(0)) / (L W), in metres.
// Storage is affine in u, even when a flat floodplain makes dV/dh jump at
// bankfull. Integrate and invert the original piecewise-linear width profile;
// subtracting a large total bankfull storage would lose the small overflow.
// Compensated FP32 sums keep both the coordinate and its inverse near the bank.
__device__ Pair terrain_coordinate(int i, Pair h, const Net& z) {
    if (!(h.hi > 0 || (h.hi == 0 && h.lo > 0))) return h;
    float w=z.W[i], dw=z.A[i]/z.L[i]/z.levels, ph=0, pw=w, nh=0;
    Pair acc={0,0}; int lev=0;
    for(int k=0;k<z.levels;++k) {
        nh=z.tab[(long)i*z.levels+k]; float nw=w+(k+1)*dw;
        if (!(h.hi>nh)) break;
        float width=.5f*(pw+nw), delta=nh-ph, v=width*delta;
        acc=pair_add(acc,{v,fmaf(width,delta,-v)});
        ph=nh; pw=nw; ++lev;
    }
    float grad=lev<z.levels?(nh-ph)/dw:0, x=h.hi-ph;
    float run=grad>0?x/grad:0, width=pw+.5f*run, v=width*x;
    acc=pair_add(acc,{v,fmaf(width,x,-v)});
    acc=pair_add(acc,{(pw+run)*h.lo,0});
    float hi=acc.hi/w;
    return pair_sum(hi,(fmaf(-hi,w,acc.hi)+acc.lo)/w);
}
__device__ Pair terrain_inverse(int i, Pair v, const Net& z) {
    if (!(v.hi > 0 || (v.hi == 0 && v.lo > 0))) {
        if ((v.hi+z.H[i])+v.lo<0) return {-z.H[i],0};
        return v;
    }
    float w=z.W[i], dw=z.A[i]/z.L[i]/z.levels, ph=0, pw=w, nh=0;
    float c=v.hi*w; Pair cross=pair_sum(c,fmaf(v.hi,w,-c)+v.lo*w), acc={0,0};
    int lev=0;
    for(int k=0;k<z.levels;++k) {
        nh=z.tab[(long)i*z.levels+k];float nw=w+(k+1)*dw;
        float width=.5f*(pw+nw), delta=nh-ph, a=width*delta;
        Pair next=pair_add(acc,{a,fmaf(width,delta,-a)});
        Pair excess=pair_add(cross,{-next.hi,-next.lo});
        if (!(excess.hi>0)) break;
        acc=next;ph=nh;pw=nw;++lev;
    }
    Pair excess=pair_add(cross,{-acc.hi,-acc.lo});
    float extra=fmaxf(excess.hi+excess.lo,0), grad=lev<z.levels?(nh-ph)/dw:0;
    float x=grad>0 ? 2*extra/(sqrtf(pw*pw+2*extra/grad)+pw) : extra/pw;
    Pair h=pair_sum(ph,x), got=terrain_coordinate(i,h,z);
    Pair error=pair_add(v,{-got.hi,-got.lo});
    float slope=(pw+(grad>0?x/grad:0))/w;
    return pair_add(h,{(error.hi+error.lo)/slope,0});
}

template <bool G> __device__ void trial_unit(const Net& z, int i, int t = -1) {
    if (t < 0) t = z.tree[i];
    int mode = ldi<G>(z.ti, t * TI + MODE);
    if (mode == DONE) return;
    if (mode == TRIAL) {
        float ref=ld<G>(z.ref,i), off=ld<G>(z.h,i), rn,xn;
        float lambda=ld<G>(z.tf,t*TF+LAM), direction=ld<G>(z.dh,i);
        trial_pair(ref,off,lambda*direction,z.H[i],rn,xn);
        if (((ref+off)>0) != ((rn+xn)>0)) {
            // All substitution rounds are complete. The pivot and coupling
            // workspace may cache this accepted coordinate and its slope until
            // the next linear solve; residual evaluations do not overwrite them.
            Pair v; float ratio;
            if (ldi<G>(z.ti,t*TI+SOLVE)) {
                v=terrain_coordinate(i,{ref,off},z);
                ratio=ld<G>(z.area,i)/(z.L[i]*z.W[i]);
                z.ap[i]=v.hi;z.rp[i]=v.lo;z.b[i]=ratio;
            } else {v={ld<G>(z.ap,i),ld<G>(z.rp,i)};ratio=ld<G>(z.b,i);}
            // Keep ordinary height trials for two backtracks. Retract only
            // a difficult bank crossing, with the same Newton tangent. The
            // unchanged full residual and Armijo test decide acceptance.
            // In the damped phase every rising crossing is retracted.
            if (ldi<G>(z.ti,t*TI+BT)>=2 || (ldi<G>(z.ti,t*TI+PHASE)==DAMPED_PHASE && direction>0)) {
                Pair h=terrain_inverse(i,pair_add(v,{lambda*(ratio*direction),0}),z);
                rn=h.hi;xn=h.lo;
            }
        }
        z.tref[i]=rn;z.ht[i]=xn;
    }
    else { z.tref[i] = ld<G>(z.ref, i); z.ht[i] = ld<G>(z.h, i); }
}
template <bool G> __device__ void eliminate(const Net& z, int p) {
    int t = z.tree[z.pnodes[z.pptr[p]]];
    if (ldi<G>(z.ti, t * TI + SOLVE)) eliminate_path<G>(z, p, ld<G>(z.tf, t * TF + DT));
}
template <bool G> __device__ void substitute(const Net& z, int p) {
    if (ldi<G>(z.ti, z.tree[z.pnodes[z.pptr[p]]] * TI + SOLVE)) substitute_path<G>(z, p);
}

// Kernels take HydroForge's HYDROFORGE_KERNEL_PARAMETERS: every value of the
// launching spec (specs.py), named like its field. These locals give the
// values the short names the solver code uses. A between-steps adjoint has
// no time step and only reads the Newton record, which no adjoint code
// touches: it binds null pointers for both. HYDROFORGE_LAUNCHES_<kernel>
// compiles only the kernels the program's spec launches.
#define SOLVE_LOCALS(time_step, record) \
    const float* S0 = storage_ptr; \
    const float* runoff = runoff_ptr; \
    const float* Q0 = outflow_ptr; \
    const float* nc = river_manning_ptr; \
    const float* nf = flood_manning_ptr; \
    const float* L = river_length_ptr; \
    const float* W = river_width_ptr; \
    const float* H = river_height_ptr; \
    const float* A = catchment_area_ptr; \
    const float* tab = flood_depth_table_ptr; \
    const float* bed = flood_bed_length_ptr; \
    const float* elev = catchment_elevation_ptr; \
    const float* dist = downstream_distance_ptr; \
    const int* down = downstream_idx_ptr; \
    const bool* edge = downstream_edge_ptr; \
    const int* cp = upstream_offset_ptr; \
    const int* kids = upstream_index_ptr; \
    const int* tree = tree_index_ptr; \
    const int* tsize = tree_size_ptr; \
    const int* troot = tree_root_ptr; \
    const int* pnodes = path_unit_ptr; \
    const int* pptr = path_offset_ptr; \
    const int* gunit = grid_unit_ptr; \
    const int* gtree = grid_tree_ptr; \
    const int* gpath = grid_path_ptr; \
    const int* grr = grid_round_offset_ptr; \
    const int* btree = block_tree_ptr; \
    const int* tunit = tree_unit_ptr; \
    const int* loff = level_offset_ptr; \
    const int* tlev = tree_level_ptr; \
    const float* dt0 = time_step; \
    float* h = newton_offset_ptr; \
    float* ht = trial_offset_ptr; \
    float* ref = newton_reference_ptr; \
    float* tref = trial_reference_ptr; \
    float* S = trial_storage_ptr; \
    float* area = trial_area_ptr; \
    float* kc = channel_conveyance_ptr; \
    float* kf = flood_conveyance_ptr; \
    float* dkc = channel_conveyance_slope_ptr; \
    float* dkf = flood_conveyance_slope_ptr; \
    float* Q = trial_outflow_ptr; \
    float* qi = outflow_own_slope_ptr; \
    float* qd = outflow_down_slope_ptr; \
    float* R = residual_ptr; \
    float* wb = balance_weight_ptr; \
    float* wa = accepted_weight_ptr; \
    float* ap = pivot_ptr; \
    float* rp = pivot_rhs_ptr; \
    float* dh = newton_step_ptr; \
    float* b = coupling_ptr; \
    float* trace = phase_trace_ptr; \
    int* ti = tree_control_ptr; \
    float* tf = tree_scalars_ptr; \
    unsigned* tm = reinterpret_cast<unsigned*>(tree_merit_ptr); \
    unsigned* tline = reinterpret_cast<unsigned*>(tree_line_ptr); \
    unsigned* ctl = reinterpret_cast<unsigned*>(grid_control_ptr); \
    float* meta = record; \
    int* status = newton_status_ptr; \
    float eps = slope_regularization; \
    float tol = newton_tolerance; \
    int n = num_catchments; \
    int nt = num_trees; \
    int levels = num_flood_levels; \
    int maxit = newton_max_iterations; \
    int rounds = num_path_rounds; \
    int ngu = num_grid_units; \
    int ngt = num_grid_trees;
#define SOLVE_NET \
    Net z{S0, runoff, Q0, nc, nf, L, W, H, A, tab, bed, elev, dist, dt0, \
          down, cp, kids, tree, tsize, troot, pnodes, pptr, gunit, gtree, gpath, grr, \
          btree, tunit, loff, tlev, edge, \
          h, ht, ref, tref, S, area, kc, kf, dkc, dkf, Q, qi, qd, R, wb, wa, ap, rp, dh, b, trace, tf, meta, \
          ti, status, tm, tline, ctl, eps, tol, n, nt, levels, maxit, rounds, ngu, ngt}


__device__ bool parallel_path(const Net& z, int p) {
    int length=z.pptr[p+1]-z.pptr[p];
    return length>=PARALLEL_PATH_MIN && length<=PARALLEL_PATH_MAX;
}
// Remove already solved light branches; the adjacent heavy child remains an
// off-diagonal entry. No symmetry or positive-definiteness identity is used.
struct PathRow {float lower,diag,upper,rhs,boundary;};
__device__ PathRow path_row(const Net& z,int p,int k,float dt) {
    int begin=z.pptr[p],length=z.pptr[p+1]-begin,i=z.pnodes[begin+k];
    int heavy=k>0?z.pnodes[begin+k-1]:-1;
    bool dry=!((__ldcg(z.ref+i)+z.H[i])+__ldcg(z.h+i)>0) && __ldcg(z.R+i)>0;
    float diag=__ldcg(z.area+i)+dt*__ldcg(z.qi+i)+damping<true>(z,i,dt),rhs=-__ldcg(z.R+i);
    float lower=heavy>=0?-dt*__ldcg(z.qi+heavy):0;
    float upper=dry||!z.edge[i]?0:dt*__ldcg(z.qd+i);
    for(int e=z.cp[i];e<z.cp[i+1];++e) {
        int j=z.kids[e];diag-=dt*__ldcg(z.qd+j);
        if(j!=heavy) {
            float c=-dt*__ldcg(z.qi+j);
            diag-=c*__ldcg(z.b+j)/__ldcg(z.ap+j);
            rhs-=c*__ldcg(z.rp+j)/__ldcg(z.ap+j);
        }
    }
    return {dry?0:lower,dry?1:diag,k+1<length?upper:0,dry?0:rhs,k+1==length?upper:0};
}

// Parallel cyclic reduction of a nonsymmetric tridiagonal heavy path.
// Light branches have already been condensed. Solve y=A^-1 r and z=A^-1 b_end;
// its nodes satisfy dh=y-z*dh_parent. The endpoint Schur relation is consumed
// by the next path round through the existing pivot/rhs/coupling workspace.
__device__ void eliminate_parallel(const Net& z,int p,float dt,float* scratch) {
    int begin=z.pptr[p],length=z.pptr[p+1]-begin,tid=threadIdx.x;
    float *cur=scratch,*next=scratch+5*PARALLEL_PATH_MAX;
    for(int k=tid;k<length;k+=blockDim.x) {
        PathRow row=path_row(z,p,k,dt);
        cur[k]=row.lower;cur[PARALLEL_PATH_MAX+k]=row.diag;
        cur[2*PARALLEL_PATH_MAX+k]=row.upper;cur[3*PARALLEL_PATH_MAX+k]=row.rhs;
        cur[4*PARALLEL_PATH_MAX+k]=row.boundary;
    }
    __syncthreads();
    for(int distance=1;distance<length;distance*=2) {
        for(int k=tid;k<length;k+=blockDim.x) {
            float a=cur[k],d=cur[PARALLEL_PATH_MAX+k],c=cur[2*PARALLEL_PATH_MAX+k];
            float r=cur[3*PARALLEL_PATH_MAX+k],b=cur[4*PARALLEL_PATH_MAX+k];
            float na=0,nc=0;
            if(k>=distance) {
                int j=k-distance;float f=a/cur[PARALLEL_PATH_MAX+j];
                d-=f*cur[2*PARALLEL_PATH_MAX+j];r-=f*cur[3*PARALLEL_PATH_MAX+j];b-=f*cur[4*PARALLEL_PATH_MAX+j];na=-f*cur[j];
            }
            if(k+distance<length) {
                int j=k+distance;float f=c/cur[PARALLEL_PATH_MAX+j];
                d-=f*cur[j];r-=f*cur[3*PARALLEL_PATH_MAX+j];b-=f*cur[4*PARALLEL_PATH_MAX+j];nc=-f*cur[2*PARALLEL_PATH_MAX+j];
            }
            next[k]=na;next[PARALLEL_PATH_MAX+k]=d;next[2*PARALLEL_PATH_MAX+k]=nc;
            next[3*PARALLEL_PATH_MAX+k]=r;next[4*PARALLEL_PATH_MAX+k]=b;
        }
        __syncthreads();float* tmp=cur;cur=next;next=tmp;
    }
    for(int k=tid;k<length;k+=blockDim.x) {
        int i=z.pnodes[begin+k];float d=cur[PARALLEL_PATH_MAX+k];
        z.ap[i]=1;z.rp[i]=cur[3*PARALLEL_PATH_MAX+k]/d;z.b[i]=cur[4*PARALLEL_PATH_MAX+k]/d;
    }
    __syncthreads();
}
__device__ void substitute_parallel(const Net& z,int p) {
    int begin=z.pptr[p],end=z.pptr[p+1],out=z.pnodes[end-1];
    float parent=z.edge[out]?__ldcg(z.dh+z.down[out]):0;
    for(int k=begin+threadIdx.x;k<end;k+=blockDim.x) {
        int i=z.pnodes[k];z.dh[i]=__ldcg(z.rp+i)-__ldcg(z.b+i)*parent;
    }
    __syncthreads();
}


__device__ void eliminate_warp(const Net& z,int p,float dt) {
    int begin=z.pptr[p],length=z.pptr[p+1]-begin,lane=threadIdx.x&31;
    int i=lane<length?z.pnodes[begin+lane]:0;
    PathRow row=lane<length?path_row(z,p,lane,dt):PathRow{0,1,0,0,0};
    float lower=row.lower,diag=row.diag,upper=row.upper,rhs=row.rhs,boundary=row.boundary;
    for(int distance=1;distance<length;distance*=2) {
        float dl=__shfl_up_sync(0xffffffff,diag,distance),dr=__shfl_down_sync(0xffffffff,diag,distance);
        float al=__shfl_up_sync(0xffffffff,lower,distance),ar=__shfl_down_sync(0xffffffff,lower,distance);
        float cl=__shfl_up_sync(0xffffffff,upper,distance),cr=__shfl_down_sync(0xffffffff,upper,distance);
        float rl=__shfl_up_sync(0xffffffff,rhs,distance),rr=__shfl_down_sync(0xffffffff,rhs,distance);
        float bl=__shfl_up_sync(0xffffffff,boundary,distance),br=__shfl_down_sync(0xffffffff,boundary,distance);
        float a=lane>=distance?lower/dl:0,b=lane+distance<length?upper/dr:0;
        diag=(diag-a*cl)-b*ar;rhs=(rhs-a*rl)-b*rr;boundary=(boundary-a*bl)-b*br;
        lower=-a*al;upper=-b*cr;
    }
    if(lane<length) {z.ap[i]=1;z.rp[i]=rhs/diag;z.b[i]=boundary/diag;}
    __syncwarp();
}
__device__ void substitute_warp(const Net& z,int p) {
    int begin=z.pptr[p],length=z.pptr[p+1]-begin,lane=threadIdx.x&31,out=z.pnodes[begin+length-1];
    float parent=z.edge[out]?__ldcg(z.dh+z.down[out]):0;
    if(lane<length) {int i=z.pnodes[begin+lane];z.dh[i]=__ldcg(z.rp+i)-__ldcg(z.b+i)*parent;}
    __syncwarp();
}

// The partition packs eligible paths first, descending by length. Return
// the block/warp and warp/serial boundaries within one independent round.
__device__ void path_split(const Net& z,int round,int& block_end,int& parallel_end) {
    int lo=z.grr[round],hi=z.grr[round+1];
    while(lo<hi) {int mid=(lo+hi)/2;if(parallel_path(z,z.gpath[mid])) lo=mid+1;else hi=mid;}
    parallel_end=lo;lo=z.grr[round];hi=parallel_end;
    while(lo<hi) {int mid=(lo+hi)/2,p=z.gpath[mid];if(z.pptr[p+1]-z.pptr[p]>32) lo=mid+1;else hi=mid;}
    block_end=lo;
}

#ifdef HYDROFORGE_LAUNCHES_inertia_solve
// Large trees: one persistent launch over the whole device, a grid barrier
// between the phases of every round.
extern "C" __global__ void __launch_bounds__(SOLVE_THREADS, 1) inertia_solve(HYDROFORGE_KERNEL_PARAMETERS) {
    SOLVE_LOCALS(time_step_ptr, newton_record_ptr);
    SOLVE_NET;
    __shared__ unsigned tree_merit, tree_line;
    __shared__ float path_scratch[10*PARALLEL_PATH_MAX];
    __shared__ int parallel_end,block_end;
    const int G = gridDim.x * blockDim.x, g = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = g; k < ngu; k += G) start_unit(z, gunit[k]);
    for (int k = g; k < ngt; k += G) start_tree(z, gtree[k]);
    grid_sync();
    for (int round = 0; round < MAX_ROUNDS; ++round) {
        for (int k = g; k < ngu; k += G) stage_unit<true>(z, gunit[k]);
        grid_sync();
        for (int k = g; k < ngu; k += G) flux_unit<true>(z, gunit[k]);
        grid_sync();
        // One large basin needs one global max per block, not one per unit.
        if(ngt==1) {
            int t=gtree[0];unsigned merit=0,line=0;
            if(__ldcg(z.ti+t*TI+MODE)!=DONE) {
                float dt=__ldcg(z.tf+t*TF+DT);
                int phase=__ldcg(z.ti+t*TI+PHASE);
                bool weighted=phase==RESTART_PHASE||phase==DAMPED_PHASE;
                for(int k=g;k<ngu;k+=G) {
                    float l=0;merit=max(merit,__float_as_uint(eval_residual<true>(z,gunit[k],dt,weighted?&l:nullptr)));
                    line=max(line,__float_as_uint(l));
                }
            }
            for(int delta=16;delta>0;delta/=2) {
                merit=max(merit,__shfl_down_sync(0xffffffff,merit,delta));line=max(line,__shfl_down_sync(0xffffffff,line,delta));
            }
            if(threadIdx.x==0) tree_merit=tree_line=0;
            __syncthreads();
            if((threadIdx.x&31)==0) {atomicMax(&tree_merit,merit);atomicMax(&tree_line,line);}
            __syncthreads();
            if(threadIdx.x==0) {atomicMax(z.tm+t,tree_merit);atomicMax(z.tline+t,tree_line);}
        } else {
            for (int k = g; k < ngu; k += G) residual_unit<true>(z, gunit[k]);
        }
        grid_sync();
        unsigned *active = ctl + ACTIVE + (round & 1), *solving = ctl + SOLVING + (round & 1);
        for (int k = g; k < ngt; k += G) {
            int t = gtree[k];
            control<true>(z, t, 0, __uint_as_float(__ldcg(tline + t)));
            tline[t] = 0;
            if (ti[t * TI + MODE] != DONE) atomicAdd(active, 1u);
            if (ti[t * TI + SOLVE]) atomicAdd(solving, 1u);
        }
        if (g == 0) { ctl[ACTIVE + ((round + 1) & 1)] = 0; ctl[SOLVING + ((round + 1) & 1)] = 0; }
        grid_sync();
        for (int k = g; k < ngu; k += G) apply_unit<true>(z, gunit[k]);
        if (__ldcg(active) == 0) return;
        bool solve = __ldcg(solving) > 0;
        grid_sync();
        if (solve) {
            // Below the usual tolerance, preserve serial FP32 convergence.
            for (int r = rounds - 1; r >= 0; --r) {
                if(z.tol<PARALLEL_PATH_TOLERANCE_MIN) {
                    for(int k=grr[r]+g;k<grr[r+1];k+=G) eliminate<true>(z,gpath[k]);
                } else {
                    if(threadIdx.x==0) path_split(z,r,block_end,parallel_end);
                    __syncthreads();
                    for (int k = parallel_end + g; k < grr[r + 1]; k += G) eliminate<true>(z,gpath[k]);
                    for(int k=grr[r]+blockIdx.x;k<block_end;k+=gridDim.x) {
                        int p=gpath[k],t=z.tree[z.pnodes[z.pptr[p]]];
                        if(__ldcg(z.ti+t*TI+SOLVE)) eliminate_parallel(z,p,__ldcg(z.tf+t*TF+DT),path_scratch);
                    }
                    for(int k=block_end+g/32;k<parallel_end;k+=G/32) {
                        int p=gpath[k],t=z.tree[z.pnodes[z.pptr[p]]];
                        if(__ldcg(z.ti+t*TI+SOLVE)) eliminate_warp(z,p,__ldcg(z.tf+t*TF+DT));
                    }
                }
                grid_sync();
            }
            for (int r = 0; r < rounds; ++r) {
                if(z.tol<PARALLEL_PATH_TOLERANCE_MIN) {
                    for(int k=grr[r]+g;k<grr[r+1];k+=G) substitute<true>(z,gpath[k]);
                } else {
                    if(threadIdx.x==0) path_split(z,r,block_end,parallel_end);
                    __syncthreads();
                    for (int k = parallel_end + g; k < grr[r + 1]; k += G) substitute<true>(z,gpath[k]);
                    for(int k=grr[r]+blockIdx.x;k<block_end;k+=gridDim.x) {
                        int p=gpath[k],t=z.tree[z.pnodes[z.pptr[p]]];
                        if(__ldcg(z.ti+t*TI+SOLVE)) substitute_parallel(z,p);
                    }
                    for(int k=block_end+g/32;k<parallel_end;k+=G/32) {
                        int p=gpath[k],t=z.tree[z.pnodes[z.pptr[p]]];
                        if(__ldcg(z.ti+t*TI+SOLVE)) substitute_warp(z,p);
                    }
                }
                grid_sync();
            }
        }
        for (int k = g; k < ngu; k += G) trial_unit<true>(z, gunit[k]);
        grid_sync();
    }
    if (g == 0) atomicExch(ctl + ERROR, NEWTON_ROUND_LIMIT);
}
#endif

// Small trees: one block each, the same rounds with block barriers and the
// tree system eliminated level by level.
#define TREE_THREADS 1024
__device__ void solve_one_block_tree(const Net& z, int t) {
    __shared__ int active, solving;
    __shared__ unsigned merit, line;
    const int first = z.loff[z.tlev[t]], last = z.loff[z.tlev[t + 1]];
    const int B = blockDim.x, tid = threadIdx.x;
    for (int k = first + tid; k < last; k += B) start_unit(z, z.tunit[k]);
    if (tid == 0) start_tree(z, t);
    __syncthreads();
    for (int round = 0; round < MAX_ROUNDS; ++round) {
        for (int k = first + tid; k < last; k += B) stage_unit<false>(z, z.tunit[k]);
        __syncthreads();
        for (int k = first + tid; k < last; k += B) flux_unit<false>(z, z.tunit[k]);
        if (tid == 0) merit = line = 0;
        __syncthreads();
        // The tree's merit: a block reduction instead of one atomic per unit.
        if (z.ti[t * TI + MODE] != DONE) {
            float dt = z.tf[t * TF + DT];
            int phase = z.ti[t * TI + PHASE];
            bool weighted = phase == RESTART_PHASE || phase == DAMPED_PHASE;
            unsigned m = 0, l = 0;
            for (int k = first + tid; k < last; k += B) {
                float lk = 0; m = max(m, __float_as_uint(eval_residual<false>(z, z.tunit[k], dt, weighted ? &lk : nullptr)));
                l = max(l, __float_as_uint(lk));
            }
            atomicMax(&merit, m);
            if (weighted) atomicMax(&line, l);
        }
        __syncthreads();
        if (tid == 0) {
            z.tm[t] = max(z.tm[t], merit);
            control<false>(z, t, 0, __uint_as_float(line));
            active = z.ti[t * TI + MODE] != DONE;
            solving = z.ti[t * TI + SOLVE];
        }
        __syncthreads();
        for (int k = first + tid; k < last; k += B) apply_unit<false>(z, z.tunit[k]);
        if (!active) return;
        __syncthreads();
        if (solving) {
            float dt = z.tf[t * TF + DT];
            for (int l = z.tlev[t]; l < z.tlev[t + 1]; ++l) {
                for (int k = z.loff[l] + tid; k < z.loff[l + 1]; k += B) eliminate_unit<false>(z, z.tunit[k], dt);
                __syncthreads();
            }
            for (int l = z.tlev[t + 1] - 1; l >= z.tlev[t]; --l) {
                for (int k = z.loff[l] + tid; k < z.loff[l + 1]; k += B) substitute_unit<false>(z, z.tunit[k]);
                __syncthreads();
            }
        }
        for (int k = first + tid; k < last; k += B) trial_unit<false>(z, z.tunit[k]);
        __syncthreads();
    }
    if (tid == 0) report(z.status, NEWTON_ROUND_LIMIT, t);
}
#ifdef HYDROFORGE_LAUNCHES_inertia_solve_trees
extern "C" __global__ void __launch_bounds__(TREE_THREADS) inertia_solve_trees(HYDROFORGE_KERNEL_PARAMETERS) {
    SOLVE_LOCALS(time_step_ptr, newton_record_ptr);
    SOLVE_NET;
    solve_one_block_tree(z, btree[blockIdx.x]);
}
#endif

// Snapshot of the converged state for the adjoint, row i of an (n, 17) table,
// rebuilt by the adjoint at the converged surface (inertia_rebuild_*):
// Q,S,qi,qd,area,bound,surface,dS/dW,dS/dH,dQ/dW,dQ/dHown,dQ/dHdown,
// dQ/dQprevious,dQ/dnc,dQ/dnf,reversed,dt.
#define J(i, k) saved[(long)(i) * 17 + (k)]
__device__ void snapshot_unit(int i, float full_dt, const float* Q0, const float* nc, const float* nf,
    const float* L, const float* W, const float* H, const float* A, const float* tab, const float* bed,
    const float* elev, const float* dist, const int* down, const bool* edges, const float* ref,
    const float* h, const float* S, const float* area, const float* Q, const float* qi,
    const float* qd, const float* R, float* saved, float eps, int levels) {
    float p[8];
    parameter_partials(i, ref, h, Q0, nc, nf, L, W, H, A, tab, bed, elev, dist, down, edges,
                       eps, full_dt, levels, Q[i], p);
    J(i, 0) = Q[i]; J(i, 1) = S[i]; J(i, 2) = qi[i]; J(i, 3) = qd[i];
    J(i, 4) = area[i]; J(i, 5) = !((ref[i] + H[i]) + h[i] > 0) && R[i] > 0 ? 1 : 0;
    J(i, 6) = ref[i] + h[i]; J(i, 7) = p[5]; J(i, 8) = p[6];
    J(i, 9) = p[0]; J(i, 10) = p[1]; J(i, 11) = p[2];
    J(i, 13) = p[3]; J(i, 14) = p[4]; J(i, 15) = p[7]; J(i, 16) = full_dt;
    bool edge = edges[i];
    int d = edge ? down[i] : i;
    float head = edge ? head_pair(elev[i], elev[d], ref[i], ref[d], h[i], h[d]) : ref[i] + h[i];
    State accel = stage_pair(i, ref[i], h[i], L, W, H, A, tab, bed, levels);
    if (edge && head < 0)
        accel = stage_pair(i, -H[i], fmaxf(((ref[i] + H[i]) + h[i]) - head, 0.f), L, W, H, A, tab, bed, levels);
    State c = stage_pair(i, ref[i], h[i], L, W, H, A, tab, bed, levels);
    if (p[7] > .5f) {
        float depth = fmaxf(((ref[d] + H[d]) + h[d]) + fminf(0, (elev[d] - elev[i]) + (H[i] - H[d])), 0.f);
        c = stage_pair(i, -H[i], depth, L, W, H, A, tab, bed, levels);
    }
    float Ae = accel.S / L[i], K = c.kc / nc[i] + c.kf / nf[i];
    J(i, 12) = Ae > 1e-8f && K > 1e-12f && (edge || Q[i] > 0)
                   ? 1 / (1 + 2 * 9.8f * full_dt * Ae / (K * K) * fabsf(Q[i])) : 0;
}

#ifdef HYDROFORGE_LAUNCHES_inertia_commit
// The step's result: storage, signed outflow, surface and received inflow; a
// differentiable model also keeps the converged surface pair (time step is
// recorded by inertia_rebuild_snapshot).
extern "C" __global__ void inertia_commit(HYDROFORGE_KERNEL_PARAMETERS) {
    const float* S = trial_storage_ptr;
    const float* Q = trial_outflow_ptr;
    const float* ref = newton_reference_ptr;
    const float* h = newton_offset_ptr;
    const bool* edges = downstream_edge_ptr;
    const int* cp = upstream_offset_ptr;
    const int* kids = upstream_index_ptr;
    const unsigned* ctl = reinterpret_cast<const unsigned*>(grid_control_ptr);
    float* storage = storage_ptr;
    float* outflow = outflow_ptr;
    float* surface = water_surface_ptr;
    float* inflow = inflow_ptr;
    int* status = newton_status_ptr;
    float* pair = converged_surface_ptr;
    int n = num_catchments;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && ctl[ERROR]) report(status, ctl[ERROR], 0);
    if (i >= n) return;
    if (pair) { pair[2 * (long)i] = ref[i]; pair[2 * (long)i + 1] = h[i]; }
    float q = Q[i], s = S[i], in = edges[i] ? fmaxf(-q, 0.f) : 0.f;
    for (int e = cp[i]; e < cp[i + 1]; ++e) in += fmaxf(Q[kids[e]], 0.f);
    if (!(isfinite(q) && s >= 0 && isfinite(s))) report(status, INVALID_STATE, i);
    storage[i] = s; outflow[i] = q; surface[i] = ref[i] + h[i]; inflow[i] = in;
}
#endif

// ------------------------------------------------------------------------
// Rebuild of the converged snapshot from the taped surface pair and the step's
// inputs: the evaluation Newton ended with, without Newton.
#define REBUILD_NET \
    Net z{}; \
    z.S0 = storage_ptr; z.runoff = runoff_ptr; z.Q0 = outflow_ptr; \
    z.nc = river_manning_ptr; z.nf = flood_manning_ptr; \
    z.L = river_length_ptr; z.W = river_width_ptr; z.H = river_height_ptr; z.A = catchment_area_ptr; \
    z.tab = flood_depth_table_ptr; z.bed = flood_bed_length_ptr; z.elev = catchment_elevation_ptr; \
    z.dist = downstream_distance_ptr; z.down = downstream_idx_ptr; z.edge = downstream_edge_ptr; \
    z.cp = upstream_offset_ptr; z.kids = upstream_index_ptr; \
    z.h = newton_offset_ptr; z.ht = trial_offset_ptr; z.ref = newton_reference_ptr; z.tref = trial_reference_ptr; \
    z.S = trial_storage_ptr; z.area = trial_area_ptr; z.kc = channel_conveyance_ptr; z.kf = flood_conveyance_ptr; \
    z.dkc = channel_conveyance_slope_ptr; z.dkf = flood_conveyance_slope_ptr; z.Q = trial_outflow_ptr; \
    z.qi = outflow_own_slope_ptr; z.qd = outflow_down_slope_ptr; z.R = residual_ptr; \
    z.wb = balance_weight_ptr; z.wa = accepted_weight_ptr; \
    z.eps = slope_regularization; z.n = num_catchments; z.levels = num_flood_levels; \
    const float dt = *time_step_ptr; \
    int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= z.n) return;

#ifdef HYDROFORGE_LAUNCHES_inertia_rebuild_stage
extern "C" __global__ void inertia_rebuild_stage(HYDROFORGE_KERNEL_PARAMETERS) {
    REBUILD_NET;
    (void)dt;
    float r = converged_surface_ptr[2 * (long)i], x = converged_surface_ptr[2 * (long)i + 1];
    z.ref[i] = r; z.h[i] = x; z.tref[i] = r; z.ht[i] = x;
    eval_stage<false>(z, i);
}
#endif
#ifdef HYDROFORGE_LAUNCHES_inertia_rebuild_flux
extern "C" __global__ void inertia_rebuild_flux(HYDROFORGE_KERNEL_PARAMETERS) {
    REBUILD_NET;
    eval_flux<false>(z, i, dt);
}
#endif
#ifdef HYDROFORGE_LAUNCHES_inertia_rebuild_residual
extern "C" __global__ void inertia_rebuild_residual(HYDROFORGE_KERNEL_PARAMETERS) {
    REBUILD_NET;
    eval_residual<false>(z, i, dt);
}
#endif
#ifdef HYDROFORGE_LAUNCHES_inertia_rebuild_snapshot
extern "C" __global__ void inertia_rebuild_snapshot(HYDROFORGE_KERNEL_PARAMETERS) {
    REBUILD_NET;
    snapshot_unit(i, dt, z.Q0, z.nc, z.nf, z.L, z.W, z.H, z.A, z.tab, z.bed, z.elev, z.dist, z.down,
                  z.edge, z.ref, z.h, z.S, z.area, z.Q, z.qi, z.qd, z.R, jacobian_snapshot_ptr,
                  z.eps, z.levels);
}
#endif

// ------------------------------------------------------------------------
// Backward: the transposed converged Jacobian along the same heavy paths, one
// launch per path round, then all local VJPs. No atomics.

#ifdef HYDROFORGE_LAUNCHES_inertia_adjoint_up
extern "C" __global__ void inertia_adjoint_up(HYDROFORGE_KERNEL_PARAMETERS, int round) {
    const float* saved = jacobian_snapshot_ptr;
    const float* gQ = grad_outflow_ptr;
    const float* gS = grad_storage_ptr;
    const float* gh = grad_water_surface_ptr;
    const int* cp = upstream_offset_ptr;
    const int* kids = upstream_index_ptr;
    const int* pnodes = path_unit_ptr;
    const int* pptr = path_offset_ptr;
    const int* rptr = round_offset_ptr;
    float* ap = pivot_ptr;
    float* rp = pivot_rhs_ptr;
    int p = rptr[round] + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= rptr[round + 1]) return;
    float dt = J(pnodes[pptr[p]], 16);
    for (int k = pptr[p]; k < pptr[p + 1]; ++k) {
        int i = pnodes[k];
        bool bi = J(i, 5) > .5f;
        float a = bi ? 1 : J(i, 4) + dt * J(i, 2);
        float r = J(i, 4) * gS[i] + J(i, 2) * gQ[i] + gh[i];
        for (int e = cp[i]; e < cp[i + 1]; ++e) {
            int j = kids[e];
            r += J(j, 3) * gQ[j];
            if (!bi) a -= dt * J(j, 3);
            float upper = J(j, 5) > .5f ? 0 : dt * J(j, 3);
            float lower = bi ? 0 : -dt * J(j, 2);
            a -= upper * lower / ap[j];
            r -= upper * rp[j] / ap[j];
        }
        ap[i] = a; rp[i] = r;
    }
}
#endif

#ifdef HYDROFORGE_LAUNCHES_inertia_adjoint_down
// The result adjoint gQ becomes the start-outflow adjoint in place: unit i
// reads only its own gQ[i] here, after every inertia_adjoint_up round.
extern "C" __global__ void inertia_adjoint_down(HYDROFORGE_KERNEL_PARAMETERS, int round) {
    const float* saved = jacobian_snapshot_ptr;
    float* gQ = grad_outflow_ptr;
    const int* down = downstream_idx_ptr;
    const bool* edges = downstream_edge_ptr;
    const int* pnodes = path_unit_ptr;
    const int* pptr = path_offset_ptr;
    const int* rptr = round_offset_ptr;
    const float* ap = pivot_ptr;
    const float* rp = pivot_rhs_ptr;
    float* lam = adjoint_continuity_ptr;
    float* ce = adjoint_flux_ptr;
    float* g_runoff = grad_runoff_ptr;
    int p = rptr[round] + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= rptr[round + 1]) return;
    float dt = J(pnodes[pptr[p]], 16);
    for (int k = pptr[p + 1] - 1; k >= pptr[p]; --k) {
        int i = pnodes[k];
        bool edge = edges[i];
        int d = edge ? down[i] : -1;
        float lower = edge && !(J(d, 5) > .5f) ? -dt * J(i, 2) : 0;
        float v = (rp[i] - (edge ? lower * lam[d] : 0)) / ap[i];
        lam[i] = v;
        float li = J(i, 5) > .5f ? 0 : v, ld_ = edge && !(J(d, 5) > .5f) ? lam[d] : 0;
        ce[i] = gQ[i] - dt * (li - ld_);
        g_runoff[i] = dt * li; gQ[i] = ce[i] * J(i, 12);
    }
}
#endif

#ifdef HYDROFORGE_LAUNCHES_inertia_adjoint_finish
// The result adjoint gS becomes the start-storage adjoint in place, after its last read.
extern "C" __global__ void inertia_adjoint_finish(HYDROFORGE_KERNEL_PARAMETERS) {
    const float* saved = jacobian_snapshot_ptr;
    float* gS = grad_storage_ptr;
    const int* cp = upstream_offset_ptr;
    const int* kids = upstream_index_ptr;
    const float* lam = adjoint_continuity_ptr;
    const float* ce = adjoint_flux_ptr;
    float* g_width = grad_river_width_ptr;
    float* g_height = grad_river_height_ptr;
    float* g_channel_n = grad_river_manning_ptr;
    float* g_flood_n = grad_flood_manning_ptr;
    int n = num_catchments;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool bound = J(i, 5) > .5f;
    float li = bound ? 0 : lam[i];
    // Parameter adjoints accumulate over the reverse steps (zero_parameter_grads).
    g_width[i] += (gS[i] - li) * J(i, 7) + ce[i] * J(i, 9);
    float v = (gS[i] - li) * J(i, 8) + ce[i] * J(i, 10);
    gS[i] = li;
    if (bound) v -= lam[i];
    float cn = ce[i] * J(i, 13), fn = ce[i] * J(i, 14);
    for (int e = cp[i]; e < cp[i + 1]; ++e) { int j = kids[e]; v += ce[j] * J(j, 11); }
    g_height[i] += v; g_channel_n[i] += cn; g_flood_n[i] += fn;
}
#endif

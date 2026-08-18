import numpy as np, torch, time
from torch import nn
from light_ray import candidate_rays, make_physical_modes, mode_matrix
from train_selector import DeepSetSelector, generate_tasks
from design_experiment import angular_spread_design, greedy_d_design, leverage_design, fisher

K=16; seed=20260818
rng=np.random.default_rng(seed); torch.manual_seed(seed)
rays=candidate_rays(96,4)
base_j=mode_matrix(rays,make_physical_modes(),order=512)          # FIX A: order 192 -> 512
base_j=base_j/np.maximum(np.linalg.norm(base_j,axis=0),1e-12)
q_base=1.0/np.square(np.exp(rng.uniform(np.log(.05),np.log(.20),size=len(rays))))
rf=np.array([np.concatenate((r.theta,r.offset)) for r in rays],dtype=np.float32)

def loss_fixed(J,Q,logits,k,step,total):
    # FIX B: straight-through hard top-K  -> train forward == eval forward
    T = max(0.05, 1.0*(0.5**(4.0*step/total)))                     # FIX C: anneal to 0.05, not 0.25
    soft = k*torch.softmax(logits/T,dim=1)
    topk = torch.topk(logits,k,dim=1).indices
    hard = torch.zeros_like(soft).scatter_(1,topk,1.0)
    w = hard + soft - soft.detach()
    f = torch.einsum('bmi,bm,bmj->bij',J,w*Q,J)
    f = f + 1e-8*torch.eye(f.shape[-1],dtype=f.dtype).expand_as(f)
    eig = torch.linalg.eigvalsh(f)
    tau = 0.02*eig[:,-1:].detach().clamp(min=1e-8)                 # FIX D: scale-relative tau
    return -(-tau[:,0]*torch.logsumexp(-eig/tau,dim=1)).mean()

model=DeepSetSelector(base_j.shape[1]+1+rf.shape[1])
opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
TOTAL=1200
t0=time.perf_counter()
for step in range(TOTAL):
    js,qs,feats=generate_tasks(base_j,rf,q_base,8,rng)
    J=torch.from_numpy(js).double(); Q=torch.from_numpy(qs).double()   # FIX E: float64
    l=loss_fixed(J,Q,model(torch.from_numpy(feats)).double(),K,step,TOTAL)
    opt.zero_grad(set_to_none=True); l.backward()
    nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
train_t=time.perf_counter()-t0

# evaluate on 30 fresh held-out tasks against the baselines
res={n:[] for n in("learned","random","angular","leverage","greedy_D")}
rr=np.random.default_rng(99)
for _ in range(30):
    js,qs,feats=generate_tasks(base_j,rf,q_base,1,rng)
    J,q=js[0].astype(float),qs[0].astype(float)
    with torch.no_grad(): lg=model(torch.from_numpy(feats))[0].numpy()
    lam=lambda i: float(np.linalg.eigvalsh(fisher(J,q,np.asarray(i)))[0])
    res["learned"].append(lam(np.argsort(lg)[-K:]))
    res["random"].append(lam(rr.choice(len(rays),K,replace=False)))
    res["angular"].append(lam(angular_spread_design(rays,K)))
    res["leverage"].append(lam(leverage_design(J,q,K)))
    res["greedy_D"].append(lam(greedy_d_design(J,q,K)))
print("Retrained with 5 fixes, %d steps, %.1f s CPU. Median lambda_min over 30 held-out tasks:\n"%(TOTAL,train_t))
med={n:float(np.median(v)) for n,v in res.items()}
for n in ("learned","random","angular","leverage","greedy_D"):
    print("   %-10s %10.4f   %7.2fx random"%(n,med[n],med[n]/med["random"]))
print("\n   learned / angular_spread = %.2fx   (learned_gate_2 needs >= 1.15x)"%(med["learned"]/med["angular"]))
print("   original code gave learned/angular = 0.015x")

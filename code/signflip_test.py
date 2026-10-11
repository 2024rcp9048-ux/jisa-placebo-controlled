# PRE-REGISTERED: tests whether cross-model attack-transfer sign is family-specific.
# Decision rule frozen before running: DeepSeek neg in >=95% AND Qwen pos in >=95% => real.
import json, collections, numpy as np
KS=["banking","slack","travel","workspace"]; KD=["undefended","spotlighting","srp"]
CT=["static","srp_cooperative","spotlighting_exploit","detect_then_route"]
def parse(k):
    c=next((c for c in sorted(CT,key=len,reverse=True) if k.endswith("_"+c)),None)
    if not c: return None
    h=k[:-(len(c)+1)]; d=next((d for d in sorted(KD,key=len,reverse=True) if h.endswith("_"+d)),None)
    if not d: return None
    h2=h[:-(len(d)+1)]; s=next((s for s in KS if h2.endswith("_"+s)),None)
    if not s: return None
    return h2[:-(len(s)+1)],s,d,c
rows=[]
for k,v in json.load(open("paper1_MASTER_results.json"))["b2_cells"].items():
    p=parse(k)
    if p and isinstance(v,dict) and "asr" in v:
        m,s,d,c=p; rows.append((m,s,d,c,float(v["asr"])))
models=sorted({r[0] for r in rows})
rng=np.random.default_rng(0); NB=2000
print("PRE-REGISTERED SIGN-FLIP TEST  (2000 cell-level bootstrap resamples)\n")
print(f"{'held-out':14s}{'mean g-r pp':>13s}{'95% CI':>22s}{'P(neg)':>9s}{'P(pos)':>9s}")
results={}
for held in models:
    train=[r for r in rows if r[0]!=held]; test=[r for r in rows if r[0]==held]
    if max([r[4] for r in test])==0.0:
        print(f"{held:14s}{'(uninjectable, excluded)':>44s}"); continue
    ti={(r[1],r[2],r[3]):r[4] for r in test}
    test_rand=np.mean([r[4] for r in test])
    advs=[]
    for _ in range(NB):
        bs=[train[i] for i in rng.integers(0,len(train),len(train))]  # cell-level resample
        sig=collections.defaultdict(list)
        for r in bs: sig[(r[1],r[2],r[3])].append(r[4])
        sm={k:np.mean(v) for k,v in sig.items()}; best=max(sm,key=sm.get)
        g=ti.get(best,np.nan)
        if g==g: advs.append((g-test_rand)*100)
    advs=np.array(advs)
    if len(advs)<10: print(f"{held:14s}  too few valid resamples"); continue
    lo,hi=np.percentile(advs,2.5),np.percentile(advs,97.5)
    pneg=np.mean(advs<0); ppos=np.mean(advs>0)
    results[held]=(advs.mean(),lo,hi,pneg,ppos)
    print(f"{held:14s}{advs.mean():>13.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>22s}{pneg:>9.2f}{ppos:>9.2f}")
print("\nDECISION (frozen rule):")
ds=results.get("deepseek_v3"); qw=results.get("qwen_32b")
if ds and qw:
    real = ds[3]>=0.95 and qw[4]>=0.95
    print(f"  DeepSeek P(neg)={ds[3]:.2f} (need >=0.95) | Qwen P(pos)={qw[4]:.2f} (need >=0.95)")
    print(f"  => SIGN-FLIP {'CONFIRMED — real, publishable' if real else 'NOT CONFIRMED — suggestive only, needs more models'}")

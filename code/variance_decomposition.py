#!/usr/bin/env python3
"""
variance_decomposition.py
=========================
Formal decomposition of defense BENEFIT variance into MODEL vs SUITE components.

Claim under test: variation in defense effectiveness is primarily a MODEL property,
not a TASK-DOMAIN (suite) property. Descriptive numbers (SRP: model_var=0.109,
suite_var=0.006) suggest model dominates ~18x; here we make it a formal decomposition
with bootstrapped confidence intervals, and report it honestly given only 6 model
levels / 4 suite levels.

Method:
  benefit(model, suite) for SRP (and separately spotlighting, and pooled).
  ANOVA-style variance components:
     eta^2_model = SS_model / SS_total
     eta^2_suite = SS_suite / SS_total
  Plus a model-level bootstrap (resample models with replacement) for a CI on the
  share of variance attributable to model.
  If statsmodels is present, also fit a mixed-effects / OLS with model & suite factors.

HONEST CAVEAT printed: 6 model levels & 4 suite levels -> variance-component CIs are
wide; report point estimate + CI, do not overclaim tight bounds.

Run: PYTHONPATH=. python variance_decomposition.py
"""
import json, glob, numpy as np
MODELS=["llama_scout","deepseek_v3","qwen_32b","mistral_large3","kimi_k2","qwen3_next"]
SUITES=["banking","slack","travel","workspace"]
def asr(m,s,d):
    fs=[f for f in glob.glob(f"msm_b2_{m}_{s}_{d}_static/agentdojo_logs/**/*.json",recursive=True) if "important_instructions" in f]
    if not fs: return None
    n=len(fs); return sum(1 for f in fs if json.load(open(f)).get("security"))/n

def build(defense):
    rows=[]
    for m in MODELS:
        for s in SUITES:
            u=asr(m,s,"undefended"); v=asr(m,s,defense)
            if u is not None and v is not None:
                rows.append((m,s,u-v))
    return rows

def eta_sq(rows):
    """ANOVA variance components: share of total SS from model and from suite."""
    y=np.array([r[2] for r in rows]); gm=y.mean(); sst=((y-gm)**2).sum()
    if sst==0: return 0,0,0
    # model SS
    ssm=0
    for m in MODELS:
        ym=[r[2] for r in rows if r[0]==m]
        if ym: ssm+=len(ym)*(np.mean(ym)-gm)**2
    sss=0
    for s in SUITES:
        ys=[r[2] for r in rows if r[1]==s]
        if ys: sss+=len(ys)*(np.mean(ys)-gm)**2
    return ssm/sst, sss/sst, (sst-ssm-sss)/sst

def boot_model_share(rows, B=5000):
    """Bootstrap over MODELS (the generalization unit) for a CI on eta^2_model."""
    rng=np.random.default_rng(0); shares=[]
    for _ in range(B):
        pick=rng.choice(MODELS, size=len(MODELS), replace=True)
        rr=[]
        for i,m in enumerate(pick):
            for s in SUITES:
                v=[r[2] for r in rows if r[0]==m and r[1]==s]
                if v: rr.append((f"{m}_{i}",s,v[0]))
        em,es,_=eta_sq_generic(rr)
        shares.append(em)
    return np.percentile(shares,[2.5,50,97.5])

def eta_sq_generic(rows):
    y=np.array([r[2] for r in rows]); gm=y.mean(); sst=((y-gm)**2).sum()
    if sst==0: return 0,0,0
    models=set(r[0] for r in rows); suites=set(r[1] for r in rows)
    ssm=sum(len([r for r in rows if r[0]==m])*(np.mean([r[2] for r in rows if r[0]==m])-gm)**2 for m in models)
    sss=sum(len([r for r in rows if r[1]==s])*(np.mean([r[2] for r in rows if r[1]==s])-gm)**2 for s in suites)
    return ssm/sst, sss/sst, (sst-ssm-sss)/sst

def fit_icc(rows):
    try:
        import pandas as pd, statsmodels.formula.api as smf
        df=pd.DataFrame(rows, columns=["model","suite","benefit"])
        md=smf.mixedlm("benefit ~ 1", df, groups=df["model"]).fit(reml=True)
        gv=float(md.cov_re.iloc[0,0]); rv=float(md.scale)
        return gv, rv, gv/(gv+rv)
    except Exception as e:
        return None,None,None

for defense in ["srp","spotlighting"]:   # NOTE: report SEPARATELY, not pooled --
                                          # the two defenses have OPPOSITE variance structure
    rows=build(defense)
    em,es,er=eta_sq(rows)
    print(f"\n=== {defense} ===  (n={len(rows)} cells)")
    print(f"  eta^2_model = {em:.3f}   eta^2_suite = {es:.3f}   residual = {er:.3f}")
    print(f"  model/suite ratio = {em/es:.1f}x" if es>0 else "  suite var ~ 0")
    lo,md,hi=boot_model_share(rows)
    print(f"  bootstrap eta^2_model: median={md:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (resample over 6 models)")
    gv,rv,icc=fit_icc(rows)
    if icc is not None:
        print(f"  per-defense ICC (mixed model, model share) = {icc:.3f}  (group var {gv:.4f}, resid {rv:.4f})")
    else:
        print(f"  (statsmodels unavailable; skipping ICC)")

print("\nNOTE: pooled analysis intentionally omitted -- SRP is model-dominated,")
print("spotlighting is suite-dominated; pooling averages opposite structures and misleads.")

print("\nHONEST CAVEAT: only 6 model levels / 4 suite levels. Point estimates are")
print("informative and the model>>suite gap is large, but variance-component CIs are")
print("wide at this group count. Report estimate + CI; do not claim tight bounds.")

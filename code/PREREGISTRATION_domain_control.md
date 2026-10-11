# Preregistration — domain-control on Slack, Travel, Workspace
Committed before any data collected. Date: 2026-08-24

## Background
Delta_domain is currently measured on Banking only (7 models, 28 deltas).
Two of seven models clear the +/-5pp noise floor, in opposite directions:
Sonnet -25.0 and GPT-5.6 Sol -22.9 on utility; Llama +15.3.

## Hypotheses
H1  Delta_domain is non-zero on at least one non-Banking suite for at
    least one model (|d| > 5pp, Holm-corrected across the family).
H2  Sign heterogeneity replicates: at least one model positive and one
    negative on the same suite.
H3  Models with baseline utility > 60% show negative Delta_domain on
    utility. Currently Sonnet (79.9%) and Sol (84.7%) do; Llama (21.5%)
    is positive. Directional, one-sided.

## Family and decision rule
Family: 12 cells (Slack, Travel x 6 models), Holm correction, alpha=0.05.
Paired exact McNemar on the scenario-set intersection.
A non-significant but direction-preserving result is "failed to confirm",
not "refuted" (power permitting -- see below).

## Power
Exact-binomial (McNemar) power, computed before any non-Banking cell was
collected. Effect sizes are PRE-SPECIFIED, not taken from observed data.

| suite  | n   | discordant rate | n_disc | split | power |
|--------|-----|-----------------|--------|-------|-------|
| slack  | 105 | 10%             | 10     | 75/25 | 0.53  |
| slack  | 105 | 10%             | 10     | 85/15 | 0.82  |
| slack  | 105 | 20%             | 21     | 75/25 | 0.74  |
| slack  | 105 | 20%             | 21     | 85/15 | 0.97  |
| slack  | 105 | 30%             | 32     | 75/25 | 0.85  |
| travel | 140 | 10%             | 14     | 75/25 | 0.52  |
| travel | 140 | 20%             | 28     | 75/25 | 0.86  |
| travel | 140 | 30%             | 42     | 75/25 | 0.96  |

Calibration: the Banking discovery effect (llama_scout) was 22/0 discordant
of 144 -- a 15% rate at a fully one-sided split. At that magnitude the family
has power >= 0.8 on both suites.

### Pre-committed reading of a null
At a 10% discordant rate and a 75/25 split, power is 0.52-0.53. Therefore a
non-significant, direction-preserving result below a 20% discordant rate is
recorded as FAILED TO CONFIRM, not as evidence of no effect. Only a
non-significant result at n_disc >= 20 with a near-even split is treated as
evidence against H1, and even then it is reported as zero observed difference
in the tested sample rather than as equivalence -- no equivalence claim is
made without a pre-committed margin and adequate power for it.

## Prompts
domain_prompts.py, verified against the verbatim Banking prompt:
within 1.8% chars, identical sentence and verb-list structure, shared
clause verbatim, zero security terms.

## Committed not to do
No second promotion if the family fails. No post-hoc suite selection.
No conversion of a null into an equivalence claim without a pre-committed
margin and adequate power.

# Specter System-One Local

## What we studied

Public TypeSafe/Jev materials describe:

1. **Typed primitives** — noul / choice / score over a shared `state`
2. **Parallel independent evaluation** — questions do not condition on each other
3. **Constrained outputs** — probability mass only on declared labels (no free text)
4. **Prefill-style amortization** — context processed once; field heads fan out
5. **Calibrated confidence** — software gates action vs escalate

The unofficial jev-on-a-laptop study implements the neural version of (4) via
KV-cache broadcast + sub-vocab logit slicing on small Qwen models.
TypeSafe's production stack (RLCD + parallel sampler) is closed-source.

## What we built

With ~1 GB RAM and no API key, a neural 1.5B field-head is not viable in this
environment. `SystemOneLocalBackend` reconstructs the *control flow* of System One
in pure Python:

```
state JSON --> Prefill (tokenize, cues, intensity, negation, fingerprint)
                 |  cached by SHA-256 of state
                 +-- view 0 identity
                 +-- view 1 negation-aware
                 +-- view 2 intensity-weighted
                       |
             field head (noul | choice | score) x views
                       |
             logit average --> Specter engine softmax + Calibration
```

### Why this is stronger than the first lexical backend

| Property | local_lexical | system_one_local |
|----------|---------------|------------------|
| Prefill once | no | yes (32-state LRU) |
| Multi-view ensemble | no | 3 views, logit average |
| Negation / intensity | weak | explicit |
| Confidence | peak/gap only | entropy + peak + gap |
| 28-field latency | -- | ~4-5 ms on CPU |

### Honest limits

- Not a trained language model. Not RLCD. Not Jev.
- Accuracy is domain-cue dependent; extend `_CUES` or replace the head with a neural scorer.
- When you have GPU: keep the same Backend protocol and drop in a parallel constrained decoder.

### Run

```bash
PYTHONPATH=. python examples/system_one_demo.py
```

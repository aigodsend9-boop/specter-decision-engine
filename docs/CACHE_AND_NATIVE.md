# Cache theory and native prefill

## Amortized cost

Without cache, Q independent questions on the same state pay:

\[
T_{\text{naive}} = Q \cdot (C_{\text{prefill}} + C_{\text{head}})
\]

Content-addressed prefill cache keyed by `hash(state)`:

\[
T_{\text{cache}} = C_{\text{prefill}} + Q \cdot C_{\text{head}}
\]

Amortized per question: \(C_{\text{prefill}}/Q + C_{\text{head}}\).
At Q=28, prefill is \(1/28 \approx 3.6\%\) of the naive prefill bill.

## Bit lexicon

Each known token maps to a bank bit-mask (\(7\) cue banks) plus neg/intensity flags.
One dictionary probe per token updates all bank hit counters via bit tests.
Unknown tokens only increment \(n_{\text{tok}}\).

## Measured prefill (same Stripe ticket, this host)

| Impl | µs/op | relative |
|------|------:|---------:|
| C `-O3` | 0.45 | 1.0× |
| Rust `-C opt-level=3` | 1.31 | 2.9× |
| Python bit-lexicon | ~3–16 | depends on path |
| Python + ctypes `libprefill.so` | ~3 | when SO loads |

Build SO: `gcc -O3 -fPIC -shared -o libprefill.so native/prefill_core.c -lm`

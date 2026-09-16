/* Specter System-One prefill core — C reference (bit/hash lean path).
 *
 * Theory:
 *   Let T = token multiset of state. Cue banks B_1..B_k are fixed sets.
 *   hits[i] = |T ∩ B_i| / sqrt(|T|)   (scale-invariant overlap)
 *   Cache key = FNV-1a(state) so identical states collide with high probability
 *   and prefill is amortized O(1) across N questions on the same state.
 *
 *   Per-token work: one open-address probe into a frozen dictionary of
 *   known cue/neg/intensifier words. Unknown tokens only bump n_tok.
 *
 * Build: gcc -O3 -march=native -fPIC -shared -o libprefill.so prefill_core.c
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#define N_CUES 7
#define TABLE_N 256   /* power of 2, load << 0.5 for ~100 keys */
#define MAX_TOK 4096

typedef struct {
    uint32_t hash;
    uint8_t  bank_mask; /* bit i => belongs to cue bank i */
    uint8_t  flags;     /* bit0=neg, bit1=intens, bit2=soft */
    float    factor;    /* intensity factor if intens/soft */
    char     key[24];
} Entry;

static Entry g_table[TABLE_N];
static int g_ready = 0;

static uint32_t fnv1a(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static void put_word(const char *w, uint8_t bank_mask, uint8_t flags, float factor) {
    size_t n = strlen(w);
    if (n >= 24) return;
    uint32_t h = fnv1a(w, n);
    uint32_t i = h & (TABLE_N - 1);
    for (int probe = 0; probe < TABLE_N; probe++) {
        Entry *e = &g_table[i];
        if (e->key[0] == 0) {
            e->hash = h;
            e->bank_mask = bank_mask;
            e->flags = flags;
            e->factor = factor;
            memcpy(e->key, w, n + 1);
            return;
        }
        if (e->hash == h && strcmp(e->key, w) == 0) {
            e->bank_mask |= bank_mask;
            e->flags |= flags;
            if (factor != 1.0f) e->factor = factor;
            return;
        }
        i = (i + 1) & (TABLE_N - 1);
    }
}

static const Entry *get_word(const char *w, size_t n) {
    uint32_t h = fnv1a(w, n);
    uint32_t i = h & (TABLE_N - 1);
    for (int probe = 0; probe < TABLE_N; probe++) {
        const Entry *e = &g_table[i];
        if (e->key[0] == 0) return NULL;
        if (e->hash == h && strncmp(e->key, w, n) == 0 && e->key[n] == 0)
            return e;
        i = (i + 1) & (TABLE_N - 1);
    }
    return NULL;
}

static void ensure_table(void) {
    if (g_ready) return;
    memset(g_table, 0, sizeof(g_table));
    /* bank bits 0..6 */
    const char *urgency[] = {"urgent","asap","immediately","critical","emergency","now","losing","blocked","outage","down","cannot","failing","deadline","today","stuck",NULL};
    const char *frust[] = {"frustrated","angry","ridiculous","unacceptable","terrible","hate","worst","furious","annoyed","complaint","useless","awful","joke",NULL};
    const char *refund[] = {"refund","chargeback","money","double","charged","twice","overcharged",NULL};
    const char *billing[] = {"invoice","payment","charge","subscription","billing","card","stripe","paypal","receipt","fee","price",NULL};
    const char *tech[] = {"bug","error","crash","api","integration","connect","connection","timeout","fail","failing","stack","exception","login","auth","oauth","webhook","sdk","account",NULL};
    const char *sales[] = {"pricing","plan","upgrade","demo","quote","enterprise","trial","discount","seat",NULL};
    const char *ret[] = {"return","returns","exchange","shipping","package","damaged",NULL};
    const char **banks[] = {urgency, frust, refund, billing, tech, sales, ret};
    for (int b = 0; b < N_CUES; b++) {
        for (int i = 0; banks[b][i]; i++)
            put_word(banks[b][i], (uint8_t)(1u << b), 0, 1.0f);
    }
    const char *neg[] = {"not","no","never","without","hardly","neither","nor",NULL};
    for (int i = 0; neg[i]; i++) put_word(neg[i], 0, 1, 1.0f);
    put_word("very", 0, 2, 1.35f);
    put_word("extremely", 0, 2, 1.60f);
    put_word("really", 0, 2, 1.25f);
    put_word("so", 0, 2, 1.15f);
    put_word("highly", 0, 2, 1.35f);
    put_word("completely", 0, 2, 1.45f);
    put_word("totally", 0, 2, 1.35f);
    put_word("absolutely", 0, 2, 1.45f);
    put_word("maybe", 0, 4, 0.72f);
    put_word("perhaps", 0, 4, 0.72f);
    put_word("somewhat", 0, 4, 0.78f);
    put_word("slightly", 0, 4, 0.72f);
    g_ready = 1;
}

typedef struct {
    uint32_t key;
    int n_tok;
    float intensity;
    float neg_density;
    float hits[N_CUES];
} PrefillOut;

/* Returns FNV key; fills out. */
uint32_t specter_prefill(const char *state, size_t n, PrefillOut *out) {
    ensure_table();
    uint32_t key = fnv1a(state, n);
    int counts[N_CUES] = {0};
    int n_tok = 0, neg = 0;
    float intens = 1.0f;
    char buf[24];
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)state[i];
        if (isalnum(c) || c == '\'') {
            size_t j = 0;
            while (i < n && j < 23) {
                c = (unsigned char)state[i];
                if (!(isalnum(c) || c == '\'')) break;
                buf[j++] = (char)tolower(c);
                i++;
            }
            buf[j] = 0;
            n_tok++;
            const Entry *e = get_word(buf, j);
            if (e) {
                uint8_t m = e->bank_mask;
                for (int b = 0; b < N_CUES; b++)
                    if (m & (1u << b)) counts[b]++;
                if (e->flags & 1) neg++;
                if (e->flags & 2) intens *= e->factor;
                if (e->flags & 4) intens *= e->factor;
            }
        } else {
            i++;
        }
    }
    if (intens < 0.5f) intens = 0.5f;
    if (intens > 2.4f) intens = 2.4f;
    float scale = 1.0f / sqrtf((float)n_tok + 1e-9f);
    out->key = key;
    out->n_tok = n_tok;
    out->intensity = intens;
    out->neg_density = (float)neg / (float)(n_tok > 0 ? n_tok : 1);
    for (int b = 0; b < N_CUES; b++)
        out->hits[b] = (float)counts[b] * scale;
    return key;
}

/* Noul logits: returns yes-logit in *yes_out; no-logit fixed -0.4 */
void specter_noul_logits(const PrefillOut *pf, const char *instr, size_t ni, float *no_out, float *yes_out) {
    float yes = 0.0f;
    /* cheap substring checks on instruction */
    int has_u = 0, has_r = 0, has_f = 0;
    for (size_t i = 0; i + 4 < ni; i++) {
        char a = tolower((unsigned char)instr[i]);
        if (a == 'u' && i + 5 < ni) has_u = 1;
        if (a == 'r' && i + 5 < ni) has_r = 1;
        if (a == 'f' && i + 6 < ni) has_f = 1;
    }
    /* more precise: scan tokens of instruction against cue words via table */
    {
        size_t i = 0;
        char buf[24];
        float scale = 1.0f / sqrtf((float)pf->n_tok + 1e-9f);
        int shared = 0;
        while (i < ni) {
            unsigned char c = (unsigned char)instr[i];
            if (isalnum(c) || c == '\'') {
                size_t j = 0;
                while (i < ni && j < 23) {
                    c = (unsigned char)instr[i];
                    if (!(isalnum(c) || c == '\'')) break;
                    buf[j++] = (char)tolower(c);
                    i++;
                }
                buf[j] = 0;
                const Entry *e = get_word(buf, j);
                if (e && (e->bank_mask & 1)) has_u = 1;
                if (e && (e->bank_mask & 4)) has_r = 1;
                if (e && (e->bank_mask & 2)) has_f = 1;
                /* shared with state: approximate via bank hits already in pf */
                (void)shared;
            } else i++;
        }
        (void)scale;
    }
    if (has_u) yes += 2.4f * pf->hits[0];
    if (has_r) yes += 2.4f * pf->hits[2];
    if (has_f) yes += 2.4f * pf->hits[1];
    yes *= pf->intensity;
    float n1 = fmaxf(0.35f, 1.0f - 1.8f * pf->neg_density);
    float n2 = 0.85f + 0.25f * pf->intensity;
    float y = (yes + yes * n1 + yes * n2) * (1.0f / 3.0f);
    *no_out = -0.40f;
    *yes_out = y - 0.25f;
}

#ifdef PREFILL_BENCH
#include <time.h>
int main(void) {
    ensure_table();
    const char *s = "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.";
    PrefillOut out;
    const int N = 100000;
    clock_t t0 = clock();
    for (int i = 0; i < N; i++)
        specter_prefill(s, strlen(s), &out);
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;
    printf("C prefill %d iters: %.2f ms (%.3f us/op) n_tok=%d intens=%.3f urgent_hit=%.3f\n",
           N, ms, 1000.0 * ms / N, out.n_tok, out.intensity, out.hits[0]);
    float no, yes;
    const char *instr = "the message conveys urgency or time-sensitivity";
    t0 = clock();
    for (int i = 0; i < N; i++)
        specter_noul_logits(&out, instr, strlen(instr), &no, &yes);
    ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;
    printf("C noul_logits %d iters: %.2f ms (%.3f us/op) no=%.3f yes=%.3f\n",
           N, ms, 1000.0 * ms / N, no, yes);
    return 0;
}
#endif

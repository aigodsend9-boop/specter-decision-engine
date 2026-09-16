//! Specter System-One prefill — Rust lean path (mirror of C theory).
//! Build: rustc -C opt-level=3 -C target-cpu=native prefill_core.rs -o prefill_rust

use std::collections::HashMap;
use std::time::Instant;

const N_CUES: usize = 7;

#[derive(Clone, Copy)]
struct Meta {
    bank_mask: u8,
    neg: bool,
    factor: f32,
}

fn build_lex() -> HashMap<&'static str, Meta> {
    let mut m = HashMap::with_capacity(128);
    let banks: [&[&str]; N_CUES] = [
        &["urgent","asap","immediately","critical","emergency","now","losing","blocked","outage","down","cannot","failing","deadline","today","stuck"],
        &["frustrated","angry","ridiculous","unacceptable","terrible","hate","worst","furious","annoyed","complaint","useless","awful","joke"],
        &["refund","chargeback","money","double","charged","twice","overcharged"],
        &["invoice","payment","charge","subscription","billing","card","stripe","paypal","receipt","fee","price"],
        &["bug","error","crash","api","integration","connect","connection","timeout","fail","failing","stack","exception","login","auth","oauth","webhook","sdk","account"],
        &["pricing","plan","upgrade","demo","quote","enterprise","trial","discount","seat"],
        &["return","returns","exchange","shipping","package","damaged"],
    ];
    for (b, words) in banks.iter().enumerate() {
        for w in *words {
            let e = m.entry(*w).or_insert(Meta { bank_mask: 0, neg: false, factor: 1.0 });
            e.bank_mask |= 1 << b;
        }
    }
    for w in ["not","no","never","without","hardly","neither","nor"] {
        m.insert(w, Meta { bank_mask: 0, neg: true, factor: 1.0 });
    }
    for (w, f) in [("very",1.35f32),("extremely",1.60),("really",1.25),("so",1.15),("highly",1.35),
                   ("completely",1.45),("totally",1.35),("absolutely",1.45)] {
        m.insert(w, Meta { bank_mask: 0, neg: false, factor: f });
    }
    for (w, f) in [("maybe",0.72f32),("perhaps",0.72),("somewhat",0.78),("slightly",0.72)] {
        m.insert(w, Meta { bank_mask: 0, neg: false, factor: f });
    }
    m
}

struct Prefill {
    n_tok: i32,
    intensity: f32,
    neg_density: f32,
    hits: [f32; N_CUES],
}

fn prefill(state: &str, lex: &HashMap<&str, Meta>) -> Prefill {
    let mut counts = [0i32; N_CUES];
    let mut n_tok = 0i32;
    let mut neg = 0i32;
    let mut intens = 1.0f32;
    for tok in state.split(|c: char| !c.is_ascii_alphanumeric() && c != '\'') {
        if tok.is_empty() { continue; }
        n_tok += 1;
        let lower = tok.to_ascii_lowercase();
        if let Some(meta) = lex.get(lower.as_str()) {
            for b in 0..N_CUES {
                if meta.bank_mask & (1 << b) != 0 { counts[b] += 1; }
            }
            if meta.neg { neg += 1; }
            if meta.factor != 1.0 { intens *= meta.factor; }
        }
    }
    intens = intens.clamp(0.5, 2.4);
    let scale = 1.0 / ((n_tok as f32) + 1e-9).sqrt();
    let mut hits = [0.0f32; N_CUES];
    for b in 0..N_CUES { hits[b] = counts[b] as f32 * scale; }
    Prefill {
        n_tok,
        intensity: intens,
        neg_density: neg as f32 / (if n_tok > 0 { n_tok } else { 1 }) as f32,
        hits,
    }
}

fn main() {
    let lex = build_lex();
    let s = "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.";
    let n = 100_000;
    let t0 = Instant::now();
    let mut last = prefill(s, &lex);
    for _ in 0..n {
        last = prefill(s, &lex);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Rust prefill {} iters: {:.2} ms ({:.3} us/op) n_tok={} intens={:.3} urgent={:.3}",
        n, ms, 1000.0 * ms / n as f64, last.n_tok, last.intensity, last.hits[0]
    );
}

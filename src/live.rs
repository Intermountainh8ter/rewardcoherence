//! The live audit (docs/PREREGISTRATION.md, amendment 1): build a fixture
//! from a prompt list, generate one response per prompt at two
//! temperatures from an OpenAI-compatible endpoint, score every
//! (prompt, response) pair with a reward model run in a Python interpreter,
//! and report Kendall's tau-b between the two temperature arms with the
//! three degenerate rankers beside it.

use crate::{kendall_tau_b, tau_bootstrap_ci};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fixture {
    pub name: String,
    pub prompts: Vec<Vec<Message>>,
    pub source: String,
    pub hash: String,
}

pub fn sha256_hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for x in sha2_digest(b) { let _ = write!(s, "{x:02x}"); }
    s
}

fn sha2_digest(bytes: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, hh].iter().enumerate() { h[i] = h[i].wrapping_add(*v); }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() { out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes()); }
    out
}

/// Build a fixture from a JSONL prompt list (`field` holds the user prompt).
pub fn build_fixture(name: &str, source: &str, jsonl: &Path, field: &str, limit: usize) -> anyhow::Result<Fixture> {
    let text = std::fs::read_to_string(jsonl)?;
    let mut prompts = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let p = v[field].as_str().ok_or_else(|| anyhow::anyhow!("row without `{field}`"))?;
        prompts.push(vec![Message { role: "user".into(), content: p.to_string() }]);
        if prompts.len() == limit { break; }
    }
    let body = serde_json::to_vec(&prompts)?;
    Ok(Fixture { name: name.into(), prompts, source: source.into(), hash: sha256_hex(&body) })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Generation {
    pub fixture: String,
    pub fixture_hash: String,
    pub generator: String,
    pub endpoint: String,
    pub temperature: f64,
    pub max_tokens: u32,
    pub seed: u64,
    pub responses: Vec<String>,
}

/// One response per prompt from an OpenAI-compatible chat endpoint.
pub fn generate(fixture: &Fixture, endpoint: &str, model: &str, temperature: f64, max_tokens: u32, seed: u64) -> anyhow::Result<Generation> {
    let url = format!("{}/v1/chat/completions", endpoint.trim_end_matches('/'));
    let mut responses = Vec::with_capacity(fixture.prompts.len());
    for (i, msgs) in fixture.prompts.iter().enumerate() {
        let body = serde_json::json!({"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens, "seed": seed + i as u64});
        let v: serde_json::Value = ureq::post(&url).timeout(std::time::Duration::from_secs(300)).send_json(body)?.into_json()?;
        let text = v["choices"][0]["message"]["content"].as_str().ok_or_else(|| anyhow::anyhow!("prompt {i}: no content in response"))?;
        responses.push(text.to_string());
    }
    Ok(Generation { fixture: fixture.name.clone(), fixture_hash: fixture.hash.clone(), generator: model.into(), endpoint: endpoint.into(), temperature, max_tokens, seed, responses })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scored {
    pub reward_model: String,
    pub stack: String,
    pub temperature: f64,
    pub scores: Vec<f64>,
    pub response_sha256: Vec<String>,
    pub response_bytes: Vec<usize>,
    pub response_tokens: Vec<usize>,
}


/// A fresh private directory (mode 0700) under the system temp dir whose name cannot be
/// predicted or pre-created by another user: created with `create_dir` (fails if it exists),
/// retried with a new suffix on collision.
fn private_tempdir(prefix: &str) -> anyhow::Result<std::path::PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    let base = std::env::temp_dir();
    for attempt in 0..64u32 {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let salt = nanos ^ ((std::process::id() as u128) << 64) ^ ((attempt as u128) << 96) ^ (&base as *const _ as usize as u128);
        let dir = base.join(format!("{prefix}-{salt:032x}"));
        match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!("could not create a private temporary directory")
}

/// Write a new file that must not already exist (O_EXCL), mode 0600.
fn write_new(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;
    f.write_all(bytes)?;
    Ok(())
}

/// Score every (prompt, response) pair with a sequence-classification reward model in a Python
/// interpreter (transformers). Only the scalar scores come back.
pub fn score(fixture: &Fixture, gen: &Generation, python: &Path, reward_model: &str) -> anyhow::Result<Scored> {
    let script = r#"
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
spec = json.load(open(sys.argv[1]))
name = sys.argv[2]
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForSequenceClassification.from_pretrained(name).eval()
out = []
with torch.no_grad():
    for q, a in zip(spec["prompts"], spec["responses"]):
        enc = tok(q, a, return_tensors="pt", truncation=True, max_length=2048)
        out.append(float(m(**enc).logits[0, 0]))
import transformers
json.dump({"scores": out, "transformers": transformers.__version__, "torch": torch.__version__}, open(sys.argv[3], "w"))
"#;
    let dir = private_tempdir("rewardcoherence-score")?;
    let spec = dir.join("spec.json");
    let out = dir.join("scores.json");
    let prompts: Vec<&str> = fixture.prompts.iter().map(|m| m.last().map(|x| x.content.as_str()).unwrap_or("")).collect();
    write_new(&spec, &serde_json::to_vec(&serde_json::json!({"prompts": prompts, "responses": gen.responses}))?)?;
    let o = std::process::Command::new(python).arg("-I").arg("-c").arg(script).arg(&spec).arg(reward_model).arg(&out).output()?;
    anyhow::ensure!(o.status.success(), "reward model scoring failed: {}", String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or(""));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&out)?)?;
    let scores: Vec<f64> = serde_json::from_value(v["scores"].clone())?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(Scored {
        reward_model: reward_model.into(),
        stack: format!("transformers {} / torch {}", v["transformers"].as_str().unwrap_or("?"), v["torch"].as_str().unwrap_or("?")),
        temperature: gen.temperature,
        scores,
        response_sha256: gen.responses.iter().map(|r| sha256_hex(r.as_bytes())).collect(),
        response_bytes: gen.responses.iter().map(|r| r.len()).collect(),
        response_tokens: gen.responses.iter().map(|r| r.split_whitespace().count()).collect(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TauRow {
    pub ranker: String,
    pub tau: f64,
    pub ci: (f64, f64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Audit {
    pub tool: String,
    pub unix_time: u64,
    pub fixture: String,
    pub fixture_hash: String,
    pub n: usize,
    pub generator: String,
    pub temperatures: (f64, f64),
    pub reward_model: String,
    pub stack: String,
    pub boot_iters: usize,
    pub seed: u64,
    pub reward_model_tau: TauRow,
    pub degenerate: Vec<TauRow>,
    /// Pre-reg section 3: mean tau >= 0.5 and CI lower bound >= 0.3.
    pub coherent: bool,
    /// Pre-reg section 4: any degenerate ranker's CI upper bound >= the reward model's CI lower bound.
    pub flagged: bool,
}

fn seeded_permutation(n: usize, seed: u64) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
    let mut idx: Vec<f64> = (0..n).map(|i| i as f64).collect();
    for i in (1..n).rev() { let j = rng.gen_range(0..=i); idx.swap(i, j); }
    idx
}

pub fn audit(fixture: &Fixture, a: &Scored, b: &Scored, boot_iters: usize, seed: u64) -> Audit {
    let n = a.scores.len();
    let row = |name: &str, xs: &[f64], ys: &[f64]| TauRow { ranker: name.into(), tau: kendall_tau_b(xs, ys), ci: tau_bootstrap_ci(xs, ys, boot_iters, seed) };
    let rm = row("reward model", &a.scores, &b.scores);
    let f = |v: &[usize]| v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
    let degenerate = vec![
        row("random", &seeded_permutation(n, seed ^ 0xa), &seeded_permutation(n, seed ^ 0xb)),
        row("length-only", &f(&a.response_bytes), &f(&b.response_bytes)),
        row("token-count", &f(&a.response_tokens), &f(&b.response_tokens)),
    ];
    let coherent = rm.tau >= 0.5 && rm.ci.0 >= 0.3;
    let flagged = degenerate.iter().any(|d| d.ci.1 >= rm.ci.0);
    Audit {
        tool: format!("rewardcoherence {}", env!("CARGO_PKG_VERSION")),
        unix_time: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        fixture: fixture.name.clone(), fixture_hash: fixture.hash.clone(), n,
        generator: a_generator(a, b), temperatures: (a.temperature, b.temperature),
        reward_model: a.reward_model.clone(), stack: a.stack.clone(), boot_iters, seed,
        reward_model_tau: rm, degenerate, coherent, flagged,
    }
}

fn a_generator(_a: &Scored, _b: &Scored) -> String { String::new() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn audit_gates_follow_the_prereg() {
        let fx = Fixture { name: "t".into(), prompts: vec![vec![Message { role: "user".into(), content: "x".into() }]; 12], source: "test".into(), hash: "h".into() };
        let scores: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let mk = |t: f64, s: Vec<f64>| Scored { reward_model: "rm".into(), stack: "s".into(), temperature: t, scores: s, response_sha256: vec![String::new(); 12], response_bytes: (0..12).map(|i| 100 - i).collect(), response_tokens: (0..12).map(|i| 20 + i).collect() };
        let a = audit(&fx, &mk(0.3, scores.clone()), &mk(1.0, scores.clone()), 200, 1);
        assert!((a.reward_model_tau.tau - 1.0).abs() < 1e-12);
        assert!(a.coherent);
        // length-only ranks A descending vs B descending identically -> tau 1 -> flagged.
        assert!(a.flagged);
    }
}

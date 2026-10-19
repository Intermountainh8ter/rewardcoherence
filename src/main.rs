//! rewardcoherence CLI.
//!
//! `tau` is the bare statistic. `fixture`, `generate`, `score`, `audit` and
//! `report` are the live path from docs/PREREGISTRATION.md amendment 1.

use clap::{Parser, Subcommand};
use rewardcoherence::live;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rewardcoherence", version, about = "Reward-model OOD coherence audit")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Kendall's tau-b between two parallel score columns, with a 95% bootstrap CI.
    Tau {
        /// Path to a JSON file: {"xs": [f64], "ys": [f64]}.
        scores: PathBuf,
        #[arg(long, default_value_t = 1000)]
        boot: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Build a fixture from a JSONL prompt list.
    Fixture {
        #[arg(long)]
        name: String,
        #[arg(long)]
        source: String,
        #[arg(long)]
        jsonl: PathBuf,
        #[arg(long, default_value = "prompt")]
        field: String,
        #[arg(long, default_value_t = 100)]
        limit: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// One response per prompt at one temperature from an OpenAI-compatible endpoint.
    Generate {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        endpoint: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        temperature: f64,
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Score a generation with a reward model (transformers, in the given interpreter).
    Score {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        generation: PathBuf,
        #[arg(long)]
        python: PathBuf,
        #[arg(long)]
        reward_model: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Tau-b between two scored arms plus the degenerate rankers; writes the audit JSON.
    Audit {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        a: PathBuf,
        #[arg(long)]
        b: PathBuf,
        #[arg(long)]
        generator: String,
        #[arg(long, default_value_t = 2000)]
        boot: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Render results/*.json audits into the README between RESULTS:START/END.
    Report {
        #[arg(long, default_value = "results")]
        results: PathBuf,
        #[arg(long, default_value = "README.md")]
        readme: PathBuf,
    },
    Version,
}

#[derive(serde::Deserialize)]
struct Scores {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

fn read<T: serde::de::DeserializeOwned>(p: &PathBuf) -> anyhow::Result<T> {
    Ok(serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?)?)
}

fn write<T: serde::Serialize>(p: &PathBuf, v: &T) -> anyhow::Result<()> {
    if let Some(d) = p.parent() { std::fs::create_dir_all(d)?; }
    Ok(std::fs::write(p, serde_json::to_string_pretty(v)?)?)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Tau { scores, boot, seed } => {
            let s: Scores = read(&scores)?;
            let tau = rewardcoherence::kendall_tau_b(&s.xs, &s.ys);
            let (lo, hi) = rewardcoherence::tau_bootstrap_ci(&s.xs, &s.ys, boot, seed);
            println!("tau   = {tau:.4}");
            println!("95%CI = [{lo:.4}, {hi:.4}]  (bootstrap, iters={boot}, seed={seed})");
        }
        Cmd::Fixture { name, source, jsonl, field, limit, out } => {
            let f = live::build_fixture(&name, &source, &jsonl, &field, limit)?;
            write(&out, &f)?;
            println!("{}: {} prompts, hash {}", f.name, f.prompts.len(), &f.hash[..16]);
        }
        Cmd::Generate { fixture, endpoint, model, temperature, max_tokens, seed, out } => {
            let f: live::Fixture = read(&fixture)?;
            let g = live::generate(&f, &endpoint, &model, temperature, max_tokens, seed)?;
            write(&out, &g)?;
            println!("{} responses at T={temperature} from {model}", g.responses.len());
        }
        Cmd::Score { fixture, generation, python, reward_model, out } => {
            let f: live::Fixture = read(&fixture)?;
            let g: live::Generation = read(&generation)?;
            anyhow::ensure!(g.fixture_hash == f.hash, "generation was made from a different fixture");
            let s = live::score(&f, &g, &python, &reward_model)?;
            write(&out, &s)?;
            println!("{} scores from {} ({})", s.scores.len(), s.reward_model, s.stack);
        }
        Cmd::Audit { fixture, a, b, generator, boot, seed, out } => {
            let f: live::Fixture = read(&fixture)?;
            let sa: live::Scored = read(&a)?;
            let sb: live::Scored = read(&b)?;
            anyhow::ensure!(sa.scores.len() == sb.scores.len() && sa.scores.len() == f.prompts.len(), "arms and fixture differ in length");
            let mut r = live::audit(&f, &sa, &sb, boot, seed);
            r.generator = generator;
            write(&out, &r)?;
            println!("reward model tau {:.3} [{:.3}, {:.3}] coherent={} flagged={}", r.reward_model_tau.tau, r.reward_model_tau.ci.0, r.reward_model_tau.ci.1, r.coherent, r.flagged);
            for d in &r.degenerate { println!("  {:12} tau {:.3} [{:.3}, {:.3}]", d.ranker, d.tau, d.ci.0, d.ci.1); }
        }
        Cmd::Report { results, readme } => {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&results)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |e| e == "json") && !p.to_string_lossy().contains("MANIFEST")).collect();
            files.sort();
            let mut t = String::from("| reward model | fixture (n) | generator | temperatures | tau-b [95% CI] | random | length-only | token-count | coherent | flagged |\n|---|---|---|---|---|---|---|---|---|---|\n");
            let mut any = false;
            for f in &files {
                let r: live::Audit = match serde_json::from_str(&std::fs::read_to_string(f)?) { Ok(r) => r, Err(_) => continue };
                any = true;
                let d = |name: &str| r.degenerate.iter().find(|d| d.ranker == name).map(|d| format!("{:.2} [{:.2}, {:.2}]", d.tau, d.ci.0, d.ci.1)).unwrap_or_default();
                t.push_str(&format!("| {} | {} ({}) | {} | {} vs {} | {:.3} [{:.3}, {:.3}] | {} | {} | {} | {} | {} |\n", r.reward_model, r.fixture, r.n, r.generator, r.temperatures.0, r.temperatures.1, r.reward_model_tau.tau, r.reward_model_tau.ci.0, r.reward_model_tau.ci.1, d("random"), d("length-only"), d("token-count"), if r.coherent { "yes" } else { "no" }, if r.flagged { "yes" } else { "no" }));
            }
            if !any { t.push_str("| (no audits yet) | | | | | | | | | |\n"); }
            let block = format!("Kendall's tau-b between the reward model's scores of the T=0.3 responses and of the T=1.0 responses over the same prompts, 2,000 bootstrap resamples. Coherent: tau ≥ 0.5 and CI lower bound ≥ 0.3. Flagged: a degenerate ranker's CI upper bound reaches the reward model's CI lower bound (then the model is not on any leaderboard).\n\n{t}");
            let readme_text = std::fs::read_to_string(&readme)?;
            let (s, e) = ("<!-- RESULTS:START -->", "<!-- RESULTS:END -->");
            let (a, b) = (readme_text.find(s).ok_or_else(|| anyhow::anyhow!("README lacks {s}"))?, readme_text.find(e).ok_or_else(|| anyhow::anyhow!("README lacks {e}"))?);
            std::fs::write(&readme, format!("{}{}\n{}\n{}", &readme_text[..a], s, block, &readme_text[b..]))?;
            print!("{block}");
        }
        Cmd::Version => println!("rewardcoherence {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}

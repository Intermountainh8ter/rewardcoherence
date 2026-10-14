//! rewardcoherence CLI — scaffold.
//!
//! The initial release implements the `tau` subcommand that reads a JSON
//! file of paired scores and prints Kendall's tau-b with a 95% bootstrap
//! CI. The `audit` subcommand that actually drives a generator and a
//! reward model is TODO; `docs/PREREGISTRATION.md` locks the gating
//! threshold before any real audit ships.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "rewardcoherence",
    version,
    about = "Reward-model OOD coherence audit"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Kendall's tau-b between two parallel score columns, with 95%
    /// bootstrap CI.
    Tau {
        /// Path to a JSON file: {"xs": [f64], "ys": [f64]}.
        scores: std::path::PathBuf,
        /// Bootstrap iterations.
        #[arg(long, default_value_t = 1000)]
        boot: usize,
        /// Bootstrap seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Print the version banner and exit.
    Version,
}

#[derive(serde::Deserialize)]
struct Scores {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Tau { scores, boot, seed } => {
            let text = std::fs::read_to_string(&scores)?;
            let s: Scores = serde_json::from_str(&text)?;
            let tau = rewardcoherence::kendall_tau_b(&s.xs, &s.ys);
            let (lo, hi) = rewardcoherence::tau_bootstrap_ci(&s.xs, &s.ys, boot, seed);
            println!("tau   = {tau:.4}");
            println!("95%CI = [{lo:.4}, {hi:.4}]  (bootstrap, iters={boot}, seed={seed})");
        }
        Cmd::Version => println!("rewardcoherence {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}

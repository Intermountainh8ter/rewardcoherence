# rewardcoherence

**Reward-model OOD coherence audit for agent-generated text.** Measures
whether a reward model's score-ordering of N responses is consistent across
generation temperatures (and other OOD-inducing dials) when the responses
themselves were produced by a generator whose outputs are **not** in the
reward model's preference-pair training data.

![status: initial scaffold](https://img.shields.io/badge/status-initial%20scaffold-lightgrey)
![language: Rust](https://img.shields.io/badge/language-Rust-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

## Why

Reward hacking in RLHF is the main failure mode of RLHF training. The
dominant diagnostic — win-rate on a reward benchmark — tells you nothing
about whether your reward model would even *agree with itself* on the
distribution the policy will actually generate once it starts optimizing.

Papers (Gao et al. 2023, Casper et al. 2023, several since) have shown that
reward-model coherence collapses under distributional shift. There is no
public harness that tells you, before you launch RLHF, that your reward
model is coherent enough to be used.

rewardcoherence fills that gap by holding one prompt fixed, generating N
responses at temperature A, scoring them with the reward model, repeating
at temperature B, and reporting the Kendall's tau-b of the two rankings
with a 95% bootstrap CI.

## What is in this scaffold

- `kendall_tau_b` with explicit tie handling.
- `tau_bootstrap_ci` with a Pcg64Mcg seeded RNG, so results are
  bit-reproducible for a given seed.
- A `rewardcoherence tau` CLI that reads a JSON of paired scores and
  prints tau plus a 95% bootstrap CI.
- `docs/PREREGISTRATION.md` locking the coherence threshold and the
  degenerate-baseline gate before any real audit ships.

Not in this commit (openly tracked in README):

- `rewardcoherence audit` subcommand that drives an OpenAI-compatible
  generator (`llama-server`, vLLM) and an OpenAI-compatible reward model.
- Degenerate baselines: random ranker, length-only ranker, token-count
  ranker.
- Standard fixtures: 100 prompts drawn from UltraFeedback-like
  distributions, held out from the reward models' training sets.

## Usage (scaffold)

```sh
cargo install --path .
cat > scores.json <<'EOF'
{"xs": [1.0, 2.0, 3.0, 4.0, 5.0], "ys": [1.1, 2.3, 2.8, 4.0, 4.9]}
EOF
rewardcoherence tau scores.json --boot 2000
# tau   = 1.0000
# 95%CI = [0.6, 1.0] (bootstrap, iters=2000, seed=42)
```

## Methodology commitments

- No tuning on the TEST split. The test set of prompts is a single JSON
  file hashed into the pre-registration.
- Bootstrap 95% CIs on every tau. A coherence number without a CI is not
  reported.
- Degenerate baselines run on every reward model: a random ranker, a
  length-only ranker, and a token-count ranker must each produce a
  Kendall tau whose 95% CI upper bound is below the real reward model's
  95% CI lower bound. If not, the reward model is flagged as
  **not better than a degenerate** and excluded from the leaderboard.
- The headline gate: a reward model is reported *coherent* only if its
  mean tau across the fixture is >= 0.5 with the 95% CI lower bound
  >= 0.3. Weaker numbers are reported as the finding, not hidden.

See `docs/PREREGISTRATION.md`.

## License

MIT. See `LICENSE`.

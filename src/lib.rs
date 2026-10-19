//! Core statistics for rewardcoherence.
//!
//! The question: given one prompt and N responses generated at temperature
//! A, then N more responses generated at temperature B, is the reward
//! model's ranking of those N responses the same across temperatures (up
//! to resampling noise)? A coherent reward model's self-agreement is the
//! baseline every downstream use (RLHF, DPO, rejection sampling) assumes.
//!
//! We measure self-agreement with Kendall's tau-b between two rankings
//! of the same N items. See `docs/PREREGISTRATION.md` for the full plan.

pub mod live;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// Kendall's tau-b between two parallel ranked arrays. Expects f64 scores;
/// pairs with equal scores on either side are counted as ties.
///
/// Returns a value in [-1.0, 1.0]. Not defined and panics for N < 2.
pub fn kendall_tau_b(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "parallel arrays required");
    let n = xs.len();
    assert!(n >= 2, "tau requires at least 2 observations");
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut tx = 0i64;
    let mut ty = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            if dx == 0.0 && dy == 0.0 {
                tx += 1;
                ty += 1;
            } else if dx == 0.0 {
                tx += 1;
            } else if dy == 0.0 {
                ty += 1;
            } else if dx.is_sign_positive() == dy.is_sign_positive() {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let n0 = (n as i64) * (n as i64 - 1) / 2;
    let denom_x = (n0 - tx) as f64;
    let denom_y = (n0 - ty) as f64;
    if denom_x == 0.0 || denom_y == 0.0 {
        return 0.0;
    }
    (concordant - discordant) as f64 / (denom_x * denom_y).sqrt()
}

/// Percentile bootstrap 95% confidence interval for Kendall's tau-b over the
/// paired samples (xs[i], ys[i]). Resamples with replacement, computes
/// tau on each resample, returns (lo, hi) at the 2.5 and 97.5 percentiles.
pub fn tau_bootstrap_ci(xs: &[f64], ys: &[f64], iters: usize, seed: u64) -> (f64, f64) {
    assert_eq!(xs.len(), ys.len());
    assert!(xs.len() >= 2);
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut taus = Vec::with_capacity(iters);
    let n = xs.len();
    let mut rx = Vec::with_capacity(n);
    let mut ry = Vec::with_capacity(n);
    for _ in 0..iters {
        rx.clear();
        ry.clear();
        for _ in 0..n {
            let k = (rand::Rng::gen::<u32>(&mut rng) as usize) % n;
            rx.push(xs[k]);
            ry.push(ys[k]);
        }
        taus.push(kendall_tau_b(&rx, &ry));
    }
    taus.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = taus[(iters as f64 * 0.025).floor() as usize];
    let hi = taus[((iters as f64 * 0.975).ceil() as usize).min(iters - 1)];
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_of_identical_rankings_is_one() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 2.0, 3.0, 4.0];
        assert!((kendall_tau_b(&xs, &ys) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tau_of_fully_reversed_is_minus_one() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![4.0, 3.0, 2.0, 1.0];
        assert!((kendall_tau_b(&xs, &ys) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn tau_handles_ties() {
        let xs = vec![1.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 2.0, 2.0, 3.0];
        let t = kendall_tau_b(&xs, &ys);
        assert!(t.is_finite() && t > 0.0 && t <= 1.0);
    }

    #[test]
    fn bootstrap_ci_covers_true_tau_for_perfect_agreement() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ys = xs.clone();
        let (lo, hi) = tau_bootstrap_ci(&xs, &ys, 500, 42);
        assert!(lo <= 1.0 && hi >= 0.9, "lo={lo} hi={hi}");
    }
}

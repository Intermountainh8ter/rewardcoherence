# Threats to validity


## Length-only gate conflates two things (recorded 2026-10-19, first live audit)

Section 4 flags a reward model when the length-only ranker's tau interval reaches the model's.
On the first live audit the length-only tau between the T=0.3 and T=1.0 arms was 0.98, because
the generator produces similar-length answers to the same prompt at both temperatures. That makes
the flag fire whenever lengths are stable, whether or not the reward model is scoring length. The
flag is reported as pre-registered. Separating the two needs a gate that conditions on length
(for example a partial tau of the reward scores given byte length), which would be a new
numbered section in the pre-registration, applied only to audits committed after it.

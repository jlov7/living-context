# Pre-development calibration amendment

The first frozen protocol was commit `10ecc20885440d61f958039c6cefb79bdac890b4`
with protocol SHA-256 `365961b6eb9be58e206490d87523fff6b9c778bfd247852a2f067061c646f7eb`.
Its first separate Nacre calibration attempt stopped before output sealing because
the next-token maximum absolute logit difference was 0.05859375, above the
predeclared 0.01 threshold. The failed attempt's supervisor receipt and stderr
are retained in local result directory `run-20260922-a`. No development or
final source has been trained or answered at this point.

Calibration-only diagnostics measured 0.05859375 and 0.04296875 maximum
absolute differences on the two Nacre questions. The maximum absolute fresh
and cached logits were 20.84375/20.828125 and 19.0/19.0, respectively. Both
raw logit tensors were float16; comparison casts them to float32. In each case
the next-token argmax, full generated token ID sequence, and rendered response
matched. Both fresh and cached responses were exactly correct JSON answers.
The bare control was incorrect on both questions. These observations support
an empirically calibrated numerical tolerance of 0.1 for this specific reader
and snapshot. They do not establish exact logit equality, and no further
tolerance change is allowed after development outcomes.

The amendment requires complete generated token ID equality on every cached
versus fresh comparison and records the original dtype, magnitudes, difference
and timing. It also makes the 90-minute model-worker ceiling cumulative across
calibration, development, final work, and retained failed attempts. Local
diagnostic calls outside the supervisor are included in the separate result
root's `budget-extras.json`. All attempts use sibling directories under that
result root. The runner and protocol are rehashed and committed before the
first development run. The original protocol and failed calibration remain
available for audit.

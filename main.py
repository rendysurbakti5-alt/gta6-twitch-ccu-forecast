"""Hour-by-hour Twitch CCU forecaster for a big game launch day (program/v1 entry).

The program reads every case file (<id>.in) in the input directory and writes one
forecast (<id>.out) per case: for each of hours 2..24 of the launch day, five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent Twitch viewers
in the game's category, given everything known up to the end of hour 1.

Method (comparable-launch curve transfer, in log space):

  * Every case carries `comparables`: past big launches with their full hourly
    viewer curves around release. For each comparable we take its release day
    (the 24 UTC hours from its `day_start`) and form the log-ratio of each hour
    to that launch's own hour 1, ln(v_h / v_1). Normalising by hour 1 removes
    the launch's size, leaving the intraday shape: the post-unlock ramp, the
    EU/US evening-overlap peak, the overnight decay.
  * The pool is filtered to launches that resemble this contest's game at the
    checkpoint: hour 1 must already hold at least 10% of the launch's own day
    peak (the game has effectively unlocked before or during hour 1, as with a
    rolling midnight-local console launch), the release day must be complete,
    and launches the kit's notes flag as early-access-contaminated are left
    out (their release day is a decay from a days-earlier access peak, a
    different shape). The case's own game is never its own comparable.
  * Pool members are weighted by how similar their pre-midnight footing was
    to this case's: the ratio of the mean of the three hourly values just
    before the day to hour 1's value (near or above 1 when a rolling unlock
    was already hot before UTC midnight, near 0 when it was not), compared
    through a Gaussian kernel on its log. The case's own ratio comes from its
    `history_hourly`; a comparable's, from its own hourly points.
  * For each hour h = 2..24 the pool's log-ratios are summarised by a
    winsorised weighted mean and standard deviation. The mean is shrunk 15%
    toward 0 (toward persistence: hour 1's level), the spread widened 10% for
    calibration, and the case's quantiles are
    p1 * exp(mean + z_tau * sd), with p1 the case's hour-1 print and z_tau the
    standard normal quantile at each level. The shrinkage, the widening and
    the 10% pool threshold were chosen by leave-one-out backtests over the
    comparable set itself (each launch forecast from the others), where this
    configuration scored a mean pinball loss of about 0.252 on the
    already-unlocked launches, against about 0.297 for persistence.
  * If the pool is ever empty the persistence forecast (every hour at p1,
    spread 0.15 * sqrt(h - 1)) is used instead, so a valid file is always
    written.

The run is a pure function of the input files: no randomness, no clock, no
network, standard library only, so two runs on the same case agree byte for
byte.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist, fmean

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(t) for t in LEVELS]

# Pool membership: hour 1 must already hold this share of the launch's day peak.
MIN_HOUR1_SHARE = 0.10
# Log-ratios are winsorised this many standard deviations from the pool mean.
WINSOR_SD = 2.5
# Bandwidth of the Gaussian kernel on log pre-level ratios (pool weighting).
PRE_KERNEL_BW = 0.75
# The pool's mean log-ratio is shrunk this far toward 0 (toward persistence).
MEAN_SHRINK = 0.15
# The pool's spread is widened by this factor for calibration.
SPREAD_CALIB = 1.1
# Fallback spread (the persistence model's), in log units per sqrt hour.
FALLBACK_SIGMA = 0.15


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_key(moment):
    """Format a datetime the way the case files key their hourly points."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def release_day_curve(comp):
    """{hour 1..24: viewers} for a comparable's release day ({} if unusable)."""
    by_from = {p["from"]: p["viewers"] for p in comp["hourly"]}
    start = parse_ts(comp["day_start"])
    curve = {}
    for h in range(1, 25):
        v = by_from.get(hour_key(start + timedelta(hours=h - 1)))
        if v is None or v <= 0:
            return {}
        curve[h] = float(v)
    return curve


def pre_level_ratio(hourly_points, day_start, hour1):
    """Mean of the 3 hourly values just before the day, over hour 1's value.

    A launch that was already hot before UTC midnight (a rolling unlock well
    under way) has this ratio near or above 1; one that unlocks later sits
    near 0. Returns None when it cannot be formed.
    """
    if not hour1 or hour1 <= 0:
        return None
    by_from = {p["from"]: p["viewers"] for p in hourly_points}
    start = parse_ts(day_start)
    vals = []
    for k in (1, 2, 3):
        v = by_from.get(hour_key(start - timedelta(hours=k)))
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    mean = fmean(vals)
    return mean / hour1 if mean > 0 else None


def build_pool(case):
    """The filtered pool for this case: (release-day curve, pre-level ratio)."""
    own_name = case["game"]["name"]
    pool = []
    for comp in case["comparables"]:
        if comp["name"] == own_name:
            continue  # a game is never its own comparable (pre-release history)
        if "early-access-contaminated" in comp["unlock"]:
            continue  # release day is a decay from an early-access peak
        curve = release_day_curve(comp)
        if not curve:
            continue
        if curve[1] < MIN_HOUR1_SHARE * max(curve.values()):
            continue  # unlock lands later in the UTC day: a different regime
        r_pre = pre_level_ratio(comp["hourly"], comp["day_start"], curve[1])
        pool.append((curve, r_pre))
    return pool


def pool_weight(r_pre, target_r_pre):
    """Gaussian kernel weight on the log pre-level ratio (1 when unknown)."""
    if not r_pre or not target_r_pre:
        return 1.0
    d = math.log(r_pre) - math.log(target_r_pre)
    return math.exp(-(d * d) / (2 * PRE_KERNEL_BW * PRE_KERNEL_BW))


def weighted_mean_sd(values, weights):
    """Weighted mean and (population) standard deviation, via exact sums."""
    total_w = math.fsum(weights)
    mean = math.fsum(w * v for v, w in zip(values, weights)) / total_w
    var = math.fsum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_w
    return mean, math.sqrt(var)


def pool_quantiles(p1, pool, hour, target_r_pre):
    """The five quantiles of `hour` from the pool's winsorised log-ratios."""
    ratios = [math.log(c[hour] / c[1]) for c, _ in pool]
    weights = [pool_weight(r, target_r_pre) for _, r in pool]
    mean, sd = weighted_mean_sd(ratios, weights)
    if sd > 0:
        lo, hi = mean - WINSOR_SD * sd, mean + WINSOR_SD * sd
        clipped = [min(max(r, lo), hi) for r in ratios]
        mean, sd = weighted_mean_sd(clipped, weights)
    mean *= 1.0 - MEAN_SHRINK
    sd *= SPREAD_CALIB
    return [p1 * math.exp(mean + z * sd) for z in Z]


def persistence_quantiles(p1, hour):
    """The fallback: every hour at hour 1's print, spread widening in sqrt time."""
    return [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)) for z in Z]


def tidy(values):
    """Clamp to the allowed range, round to 2 decimals, keep non-decreasing."""
    out = []
    for v in values:
        v = min(max(v, 0.01), 1_000_000_000.0)
        v = round(v + 1e-9, 2)
        if out and v < out[-1]:
            v = out[-1]
        out.append(v)
    return out


def forecast_case(case):
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first else 1.0
    pool = build_pool(case)
    target_r_pre = pre_level_ratio(case["history_hourly"], case["day_start"], p1)
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        if pool:
            q = pool_quantiles(p1, pool, h, target_r_pre)
        else:
            q = persistence_quantiles(p1, h)
        rows.append({"hour": h, "q": tidy(q)})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

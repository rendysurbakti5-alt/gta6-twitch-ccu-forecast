# GTA VI launch-day Twitch CCU forecaster

An entry for make.fun's **gta6-twitch-ccu-forecast-01** bounty: forecast, hour
by hour, how many people watch Grand Theft Auto VI live on Twitch over the 24
UTC hours of its release day, 2026-11-19, from everything recorded up to the
end of the day's first hour.

`main.py` (Python 3, standard library only) reads every case file `<id>.in`
in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one forecast `<id>.out`
per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`): for each of hours 2–24,
five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent viewers
in the game's Twitch category, in the exact schema the contest scores
(`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, …]}`, 23 rows).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The run is a pure function of its input — no randomness, no clock, no
network — so two runs on the same case produce byte-identical output. It
takes well under a second per case (the contest allows 8).

## Method

Comparable-launch curve transfer, in log space:

1. **Normalise each comparable.** Every case carries `comparables`: past big
   launches with full hourly viewer curves around release (SullyGnome hourly
   category averages, per the kit's NOTICE). For each comparable, take its
   release day (the 24 UTC hours from its `day_start`) and form the log-ratio
   of every hour to that launch's own hour 1, `ln(v_h / v_1)`. Dividing by
   hour 1 removes the launch's size and leaves the intraday shape: the
   post-unlock ramp, the EU-evening/US-afternoon overlap peak, the late
   decay.
2. **Filter the pool to this contest's regime.** A comparable counts only if
   its release day is complete, it is not the case's own game, the kit's
   notes do not flag it as early-access-contaminated (those release days are
   decays from a days-earlier access peak — a different shape), and its
   hour 1 already holds at least 10% of its own day peak. That last rule
   keeps launches whose unlock landed late in the UTC day (hour 1 near zero,
   ratios exploding) out of the pool: GTA VI unlocks at rolling midnight
   local time — New Zealand at 2026-11-18T11:00Z, central Europe at 23:00Z,
   the UK at 00:00Z, the US east coast at 05:00Z and the west coast at
   08:00Z — so by the end of hour 1 most of the world is already playing.
3. **Weight by pre-midnight footing.** Each pool member is weighted by a
   Gaussian kernel (bandwidth 0.75) on the log of its *pre-level ratio*: the
   mean of the three hourly values just before its release day over its
   hour 1. Launches that were already hot before UTC midnight resemble a
   rolling-unlock launch more than ones that were dark. The case's own ratio
   comes from its `history_hourly`.
4. **Summarise per hour.** For each hour 2–24, the pool's log-ratios give a
   winsorised (±2.5 SD) weighted mean and standard deviation. The mean is
   shrunk 15% toward 0 (toward persistence at hour 1's level) and the spread
   widened 10% for calibration; the quantiles are
   `p1 · exp(mean + z_τ · sd)` with `p1` the case's hour-1 print and `z_τ`
   the standard normal quantile at each level. If the pool is ever empty,
   the program falls back to persistence (every hour at `p1`, spread
   `0.15·√(h−1)`), so a valid forecast file is always written.

The pool threshold, the kernel weighting, the shrinkage and the widening
were chosen by leave-one-out backtests over the comparable set itself (each
launch forecast from the others, scored with the contest's log-scale pinball
loss): on the already-unlocked launches this configuration scores ≈ 0.252,
against ≈ 0.297 for the persistence baseline on the same launches. (The
house baseline's published practice loss is ≈ 0.387; the practice cases are
graded server-side, so no local practice score is claimed.)

## What the model expects for GTA VI

Expressed as multiples of hour 1's print (whatever it turns out to be), the
median curve dips slightly into hour 2 (≈ 0.94×), climbs through the US
unlocks and the transatlantic afternoon, and **peaks around hour 20
(19:00–20:00 UTC) at ≈ 1.9× hour 1**, when EU evening and US afternoon
audiences overlap; hour 24 still sits at ≈ 1.6× hour 1. The 5–95% band
widens from roughly ×/÷ 1.35 in hour 2 to about ×/÷ 2.2 by the evening peak.

## Assumptions

- Hour 1's print is a meaningful anchor: the rolling unlock means hour 1
  (00:00–01:00 UTC, UK unlock) already contains NZ/APAC and central-Europe
  audiences, so the rest of the day is mostly the same audience rotating
  through time zones, not a cold start.
- Launch-day shapes transfer across big launches once size is divided out;
  the main shape risk is an unlock schedule unlike the pool's (a single
  global unlock time, a delay, or a server outage), which the widening
  spread is there to absorb.
- SullyGnome's hourly averages (the comparables and the practice answers)
  measure the same quantity as the final's minute-read means, up to noise.
- No early access, no PC version at launch, single-player: the audience is
  watching streamers, and the evening peak is driven by prime-time
  viewership in EU + US rather than by a content drop later in the day.

## Layout

- `main.py` — the whole entry: input parsing, the pool model, the fallback,
  and the output writer.
- `LICENSE` — MIT (written by the kit's `init`).

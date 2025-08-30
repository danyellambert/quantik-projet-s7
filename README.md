# Quantik AI Suite — Alpha–Beta, MCTS, Genetic & Hybrids

> **One-stop repository** for experimenting with multiple agents for the game **Quantik**: classic **Alpha–Beta** engines (baseline / + / ++), **MCTS** variants (baseline / + / ++), **Genetic Algorithm (GA)** learners (baseline / + / ++), and **hybrid models** (AB+MCTS, GA evaluated by AB). Comes with a **reproducible tournament runner**, rich **diagnostics**, **ratings** (Elo & Glicko‑1), and an optional **GUI** for interactive play and quick smoke‑tests.
>
> This README was refreshed to match the **final state of the codebase**, including the concrete MCTS implementations, the `core/` rules & types, the AI discovery/probe contract, and the runner’s output schemas. Nothing from the original README was removed; new details were **added** and clarified.

---

## Table of Contents

- [Overview](#overview)
- [Feature Highlights](#feature-highlights)
- [Repository Structure](#repository-structure)
- [Game Rules & Types (core)](#game-rules--types-core)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Run the GUI](#run-the-gui)
  - [Run a Tournament (CLI)](#run-a-tournament-cli)
- [Tournament Runner](#tournament-runner)
  - [Configuration Flags](#configuration-flags)
  - [Outputs & Folder Layout](#outputs--folder-layout)
  - [CSV / JSONL Schemas](#csv--jsonl-schemas)
  - [Metrics Explained](#metrics-explained)
  - [Ratings: Elo & Glicko‑1](#ratings-elo--glicko-1)
- [AI Line‑Up (What Each Engine Does)](#ai-line-up-what-each-engine-does)
  - [Alpha–Beta Family: baseline → AB+ → AB++](#alphabeta-family-baseline--ab--ab)
  - [MCTS Family: baseline → MCTS+ → MCTS++](#mcts-family-baseline--mcts--mcts)
  - [Genetic Algorithms Family: baseline → GA+ → GA++](#genetic-algorithms-family-baseline--ga--ga)
  - [Hybrids](#hybrids)
- [Reproducibility & Experimental Design](#reproducibility--experimental-design)
- [How to Add a New AI](#how-to-add-a-new-ai)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Further Reading & Sources](#further-reading--sources)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project is a **learning & evaluation playground** for the game **Quantik**. It implements several agents with ascending sophistication and **instrumented duels** that surface both **strength** (win‑rate, ratings) and **quality/cost metrics** (time per move, tactical misses, blunders that hand immediate wins, search branching).

The suite helps you:

- Compare different search paradigms (Alpha–Beta vs. MCTS vs. GA) under uniform rules.
- Inspect *why* an agent wins/loses (diagnostic counters per move).
- Iterate scientifically (fixed seeds, reproducible logs, confidence intervals).

---

## Feature Highlights

- **Multiple Engines**: Alpha–Beta (baseline/+/++), MCTS (baseline/+/++), Genetic (baseline/+/++), plus **AB+MCTS** and **GA evaluated by AB** hybrids.
- **Reproducible Tournaments**: fixed **seeds**, **round‑robin**, **A/B starters per seed**, optional **SPRT** early stop.
- **Rich Telemetry**: per‑move and per‑game CSV/JSONL, openings, branch factor by turn, forced moves, missed immediate wins, “gave opponent immediate win” flags.
- **Statistical Confidence**: **Wilson 95% CI** for WR; optional **Bootstrap CI**.
- **Ratings**: **Elo (approx.)** and **Glicko‑1 (`r ± RD`)** computed over all games.
- **Time Model**: either **no cap** (measure true compute) or **harmonized per‑move budget** via standard attributes (`time_limit`, `total_time`, `think_time`, `budget`, `max_time_per_move`).
- **Timestamped Artifacts**: every run gets its own **`tournament/out/YYYYMMDD-HHMMSS/`** folder (no overwrites) and a **`run_config.json`** snapshot of the exact run configuration.
- **GUI**: optional graphical interface to play vs. AIs and validate behavior quickly.

---

## Repository Structure

> The exact layout can vary slightly across branches; below is the typical structure used by the tournament runner and GUI. The **MCTS** implementations listed here match the versions in this repository snapshot.

```
quantik-test/
├─ ai_players/
│  ├─ alphabeta_baseline/algorithme.py
│  ├─ alphabeta_adv/algorithme.py
│  ├─ alphabeta_adv+/algorithme.py
│  ├─ mcts_baseline/algorithme.py           
│  ├─ mcts_adv/algorithme.py                
│  ├─ mcts_adv+/algorithme.py               
│  ├─ genetic_baseline/algorithme.py
│  ├─ genetic_adv/algorithme.py
│  ├─ genetic_adv+/algorithme.py
│  ├─ ab_mcts/algorithme.py
│  ├─ ab_mcts_adv/algorithme.py
│  ├─ genetic_ab_eval/algorithme.py
│  ├─ genetic_ab_eval_adv/algorithme.py
│  ├─ random/algorithme.py         
│  └─ template/algorithme.py                # Starter template
│
├─ core/
│  ├─ rules.py        # QuantikBoard, victory check, move legality & application
│  ├─ types.py        # Shape, Piece, Player
│  └─ ai_base.py      # Optional abstract base (constructor + get_move contract)
│
├─ tournament/
│  ├─ runner.py
│  └─ out/
│     └─ YYYYMMDD-HHMMSS/    # per-run artifacts (CSV/JSONL + run_config.json)
│
├─ gui/                         # (optional) interactive UI
│  └─ app.py
│
├─ README_SOURCES.md            # curated bibliography (core)
├─ requirements.txt
└─ README.md
```

---

## Game Rules & Types (core)

The **Quantik** rules implemented in this codebase are explicitly documented in `core/rules.py` and `core/types.py`:

- **Pieces & Players** (`core/types.py`)
  - `Shape` enum: `CIRCLE`, `SQUARE`, `TRIANGLE`, `DIAMOND`.
  - `Player` enum: `PLAYER1`, `PLAYER2`.
  - `Piece(shape, player)` value object with equality semantics (shape+owner).

- **Board & Legality** (`core/rules.py`)
  - `QuantikBoard` manages a **4×4** grid and **four 2×2 zones**.
  - **Legality (project rule variant)**: *You may not place a shape if the **opponent** already has the **same shape** in the same **row**, **column**, or **zone**.*  
    (Repeating **your own** shape in a line/column/zone is allowed here.)
  - **Victory**: a **row/column/zone** is winning if all **4 cells are filled** and the **4 shapes are all different**.
  - Helpers: `zone_index`, `has_valid_moves`, and `raw()` to expose the underlying matrix to AIs.

> The runner uses the same legality checks internally for diagnostics. See **[Further Reading & Sources](#further-reading--sources)** for public rule references of Quantik.

---

## Installation

```bash
# 1) Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2) Install project dependencies
pip install -r requirements.txt

# 3) (Optional) Dev extras
pip install ipython black pytest
```

> **Python**: 3.10+ recommended. No GPU required.

---

## Quick Start

### Run the GUI

The GUI lets you **play Quantik** against any installed AI and is the easiest way to sanity‑check behavior.

```bash
python -m gui.app
# or
python gui/app.py
```

- Select **Player 1** and **Player 2** from the AI dropdowns.
- Start a new game; the interface will call each AI’s `get_move(...)` when it’s their turn.
- Use this for quick **manual testing**, verifying **legality** and **responsiveness**.

### Run a Tournament (CLI)

Run a full round‑robin tournament among all discovered AIs:

```bash
python -m tournament.runner
# or
python tournament/runner.py
```

The runner automatically **discovers agents** in `ai_players/*/algorithme.py` that define **`QuantikAI`** and **`AI_NAME`**, **probes** them for a quick first move (“mute” filter), and then launches the matches.

---

## Tournament Runner

### Configuration Flags

At the top of `tournament/runner.py` (excerpt; defaults shown here match the code):

```python
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 112, 213, 314, 415, 516, 617, 718, 819, 920]
GAMES_PER_SEED = 10

FILTER_MUTE   = True
PROBE_TIMEOUT = 1.75    # seconds to detect "mute" AIs on an empty board

# Time control (optional). None = uncapped. Float = per-move budget.
TIME_PER_MOVE: Optional[float] = None

# Early stopping for A vs B (optional):
USE_SPRT   = False
SPRT_P0    = 0.50
SPRT_P1    = 0.60
SPRT_ALPHA = 0.05
SPRT_BETA  = 0.10

# Bootstrap for empirical CI on win-rate (set 0 to disable):
BOOTSTRAP_N = 2000

# Ratings:
USE_GLICKO = True
GLICKO_START_RATING = 1500.0
GLICKO_START_RD     = 350.0  # clamped to [30, 350]
```

- **Seeds & games**: For each seed we play **two games** per iteration (**A starts** / **B starts**) and repeat `GAMES_PER_SEED` times. Per‑game seeds are derived deterministically: `seed*10000 + 2*g + {1,2}`.
- **Probe filter**: Before the tournament, each AI is given a short **probe** on the empty board. If it cannot return a legal move within **`PROBE_TIMEOUT`**, it’s excluded as “mute” (prevents a bad actor from freezing the round‑robin).
- **Time model**: If `TIME_PER_MOVE` is a float, the runner pushes that budget into any exposed attribute among `time_limit`, `total_time`, `think_time`, `budget`, `max_time_per_move`.
- **SPRT**: Enable to finish a given A vs. B duel early with formal error bounds.

### Outputs & Folder Layout

Each run creates its own timestamped directory:

```
tournament/out/2025MMDD-HHMMSS/
├─ games.csv
├─ moves.csv
├─ moves.jsonl
└─ run_config.json   # exact runner configuration + UTC timestamp + discovered AIs
```

### CSV / JSONL Schemas

**`games.csv`** (one row per game)

| Column | Meaning |
|---|---|
| `game_uid` | Unique id (A vs B, seed, who started) |
| `pair` | Placeholder (reserved) |
| `starter` | `A` or `B` |
| `winner` | `A` or `B` |
| `end_reason` | `victory`, `no_move`, `illegal`, `exception` |
| `plies_total` | Half‑moves in the game |
| `opening` | First *n* plies signature (e.g., `1,1,CIRCLE | 2,2,TRIANGLE | ...`) |
| `time_p1`,`time_p2` | Total think time per player (seconds) |
| `moves_p1`,`moves_p2` | Number of moves each played |
| `branch_sum_p1`,`branch_sum_p2` | Sum of legal moves available each turn (for mean branch/turn) |
| `forced_p1`,`forced_p2` | Number of turns with exactly one legal move |
| `missed_win_p1`,`missed_win_p2` | Immediate wins available but not chosen |
| `gave_opp_win_p1`,`gave_opp_win_p2` | Moves that gave opponent an immediate win next turn |

**`moves.csv`** (one row per ply / half‑move)

| Column | Meaning |
|---|---|
| `game_uid`,`ply`,`player` | Game id, ply index, `A`/`B` |
| `r`,`c`,`shape` | Played move (row, col, shape name) |
| `time_sec` | Think time for this move (seconds) |
| `legal_count` | Legal moves before the move |
| `had_immediate_win` | 1 if a one‑move win was available |
| `chose_immediate_win` | 1 if the move actually took that win |
| `gave_opp_immediate_win` | 1 if the move allowed opponent to win immediately next |
| `is_center` | 1 if square ∈ {(1,1),(1,2),(2,1),(2,2)} |
| `note` | `ok`, `illegal_or_none`, `place_piece_failed`, `exception:...` |

**`moves.jsonl`** mirrors `moves.csv` row‑for‑row for downstream analysis.

### Metrics Explained

- **WR (win‑rate)**: \( \text{wins} / \text{games} \).  
- **Wilson 95% CI**: small‑sample‑friendly interval for binomial proportions.  
- **Bootstrap CI (2.5–97.5%)**: empirical CI by resampling the binary outcomes.  
- **StartsWon / RepliesWon**: wins split by who started first.  
- **TimeUsed & `t/c`**: cumulative time & **average time per move**.  
- **AvgLen (plies)**: average game length in half‑moves.  
- **AvgBranch/turn**: mean legal moves on the side‑to‑move turns.  
- **Forced moves**: turns where only one legal move existed.  
- **Missed immediate wins**: times an agent had a direct win and didn’t take it.  
- **Gave opponent immediate win**: blunders handing a one‑move win to the opponent.  
- **End reasons**: victory vs. no legal move vs. illegal vs. exception/crash.  
- **Openings**: most frequent opening signatures across all games (top‑N printed).

### Ratings: Elo & Glicko‑1

- **Elo (approx.)**: single‑pass update using expected score; good first glance.  
- **Glicko‑1**: **rating `r`** and **rating deviation `RD`**; RD shrinks as more games accumulate. We clamp `RD ∈ [30, 350]` and update after **every game** (no inactivity decay in this runner).

---

## AI Line‑Up (What Each Engine Does)

### Alpha–Beta Family: baseline → AB+ → AB++

- **AB (baseline)**  
  - *Core*: Minimax with **Alpha–Beta pruning** (Negamax style in practice).  
  - *Evaluation*: simple domain features for Quantik (shape availability, center usage, constraints).  
  - *Move ordering*: basic; no transposition table.

- **AB+ (stronger)**  
  Adds classic engine techniques:  
  - **Iterative Deepening** (+ anytime behavior, better PV ordering).  
  - **Principal Variation Search (PVS / NegaScout)** for narrow‑window re‑search.  
  - **Transposition Table (TT)** with **Zobrist hashing**.  
  - **Move Ordering**: **Killer** + **History** heuristics; urgency/tactical first.  
  - **Aspiration Windows** to speed re‑search around previous score.  
  - **Quiescence Search** for “noisy” nodes to fight horizon effects.

- **AB++ (even stronger)**  
  Adds **selective pruning & reductions** and improved TT policy:  
  - **Null‑Move Pruning** (guarded).  
  - **Late Move Reductions (LMR)** on late, quiet moves.  
  - **Futility / Razoring** at shallow depths.  
  - Optional **Multi‑Cut**; depth‑preferred TT replacement.

> *Note:* If your branch only ships MCTS for now, the AB folders will appear when you pull the full repository. The tournament runner is agnostic and will pick up whatever engines exist under `ai_players/`.

### MCTS Family: baseline → MCTS+ → MCTS++

- **MCTS (baseline)** (`ai_players/mcts_baseline/algorithme.py`, `AI_NAME="MCTS (baseline)"`)  
  - **UCT** selection (standard `c≈1.4142`), expansion of one untried move, **pure random rollouts**, **binary backprop** (win if the **player who just moved** eventually wins).  
  - Conservative time budget: **`time_limit ≈ 0.6s`** per move (class default).  
  - Always uses **local clones**; never mutates the global board.

- **MCTS+ (robust)** (`ai_players/mcts_adv/algorithme.py`, `AI_NAME="MCTS+"`)  
  - UCT selection with **immediate‑win check in rollouts**; otherwise **center‑biased random** playouts.  
  - Root shortcut for **immediate wins**.  
  - Time budget default **`≈ 1.2s`** per move.  
  - Still pure UCT (no global priors), but **safer playouts** than the baseline.

- **MCTS++ (optimized)** (`ai_players/mcts_adv+/algorithme.py`, `AI_NAME="MCTS++"`)  
  - UCT selection with **Progressive Bias** using **global CELL priors** learned during rollouts: key `(player, (r,c)) → (wins, plays)`.  
  - Tuned constants: `UCT_C = 1.35`, `PB_LAMBDA = 0.06`. Bias decays with visits.  
  - **Fast & safe rollout policy**: (1) take **immediate win** if available; else (2) **block opponent’s immediate win**; else (3) **center‑biased** random move.  
  - Root shortcut for **immediate wins**; **center preference** on initial move; fallback to center when budget exhausted.  
  - Time budget default **`≈ 1.2s`** per move.

### Genetic Algorithms Family: baseline → GA+ → GA++

> The GA family aims to **learn evaluation weights / rules** for Quantik. Each individual is a **feature vector** or **policy parameterization**.

- **GA (baseline)**  
  - Individuals = **weight vectors** for a handcrafted evaluation.  
  - **Fitness** via mini‑matches (e.g., vs. a fixed baseline) or a static position dataset.  
  - **Selection**, **crossover**, **mutation**, **elitism**.

- **GA+ / GA++**  
  - Improved operators and diversity controls (tournament selection, elitism, adaptive mutation; optional niching).  
  - Caching, dataset augmentation, or **Alpha–Beta‑assisted** fitness.

- **GA evaluated by Alpha–Beta (“GA (eval AB)”)**  
  - Use a **fixed shallow AB evaluator** as the **fitness oracle** across a curated position set.  
  - Transfers AB’s tactical sense to GA‑evolved heuristics with **deterministic, low‑variance** fitness.

### Hybrids

- **AB + MCTS**  
  - Use **AB evaluation** as **progressive bias** in UCT (or implicit minimax backups).  
  - Split by phase: **MCTS** for high‑branching phases, **AB** for tactical resolution.

- **GA × AB (“GA (eval AB)”)**  
  - GA **learns** weights; **AB** provides the **fitness** via shallow search.  
  - The evolved evaluator can be plugged back into **AB+ / AB++** as a drop‑in heuristic.

---

## Reproducibility & Experimental Design

- **Seeds**: Provide a wide set (the default list includes 18 spaced seeds). Per‑game seeds derive deterministically, ensuring **exact reruns**.  
- **GAMES_PER_SEED**: Increase to tighten CIs; `8–10` is a good balance for ~10 AIs.  
- **SPRT**: For large pools, enable to cut A vs. B duels early with controlled error.  
- **Time model**: Choose **uncapped** compute (raw strength) or a fixed **`TIME_PER_MOVE`** (fair budget).  
- **Probe filter**: `PROBE_TIMEOUT` shields the tournament from “mute” engines.

---

## How to Add a New AI

Create `ai_players/<your_engine>/algorithme.py` with a class and constant:

```python
AI_NAME = "My Cool Engine"

class QuantikAI:
    def __init__(self, player):
        self.me = player
        # Any of these, if present, may be set by the runner when TIME_PER_MOVE is not None:
        self.time_limit = None
        self.total_time = None
        self.think_time = None
        self.budget = None
        self.max_time_per_move = None

    def get_move(self, board, pieces_count):
        # board: 4x4 matrix of Piece or None  (use-only; do not mutate)
        # pieces_count: {Player: {Shape: int}} remaining stocks
        # return (row, col, Shape) or None (illegal/None ⇒ loss)
        ...
```

If you prefer a formal interface, see `core/ai_base.py` (abstract `AIBase`).

---

## Troubleshooting & FAQ

- **“Probe excluded my AI as mute”**  
  Your `get_move` didn’t return a legal move on the empty board within `PROBE_TIMEOUT` (1.75 s). Ensure the constructor is light and the first move is produced quickly.

- **“TIME_PER_MOVE is ignored by some engines”**  
  Set `TIME_PER_MOVE = None` to measure raw compute, or pick a budget that your engine honors and that peers can accept via `time_limit/total_time/think_time/budget/max_time_per_move`.

- **Wilson vs. Bootstrap CIs differ**  
  Wilson is analytic; bootstrap is empirical. With small samples, bootstrap can be wider. Prefer larger samples or enable **SPRT**.

- **Glicko shows “± 30 RD” everywhere**  
  We clamp `RD ∈ [30, 350]` and update after each game (no inactivity decay). With many games, RD tends toward ~30.

- **Board rule variant**  
  This project enforces the **opponent‑shape conflict** rule (you may repeat your own shape in a line/column/zone). If you want the stricter “any‑owner shape conflict” variant, adjust `is_valid_move` in `core/rules.py` and the runner’s duplicate in `tournament/runner.py` accordingly.

---

## Further Reading & Sources

See **`README_SOURCES.md`** (included) for a **curated, fully cited** bibliography covering:

- Alpha–Beta (PVS, TT, Zobrist, LMR, null‑move, aspiration, quiescence, etc.).
- MCTS & variants (UCT, RAVE/AMAF, progressive bias/widening, virtual loss).
- Genetic Algorithms for evaluation learning, and the **GA×AB** “eval AB” design.
- AB×MCTS hybrid motivations and patterns from the literature.
- Statistical evaluation & ratings (Wilson CI, Bootstrap, SPRT, Elo, Glicko‑1).
- Sdditional references (e.g., DPW, implicit minimax backups, BayesElo/TrueSkill2, Quantik rule references).

---

## License

Educational / academic use.

---

## Acknowledgments

- The ChessProgramming Wiki and academic papers referenced in **`README_SOURCES.md`**.
- The original Quantik game and its community for inspiration.

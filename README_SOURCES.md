# Sources & Further Reading — What We Borrowed and Why

This section consolidates **all the foundational links, papers, and reference pages** we consulted while designing, implementing, and evaluating the AIs in this project. Each item includes a short note on **how** it informed our work. It is intentionally exhaustive for AB (alpha–beta search), MCTS, Genetic/Evolutionary methods, **hybrids** (AB×GA and AB×MCTS), and **tournament statistics** (Elo, Glicko, Wilson, Bootstrap, SPRT).

---

## 1) Alpha–Beta & Classic Engine Heuristics

- **Alpha–Beta pruning — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Alpha-Beta  
  *Used for:* AB baseline definition, complexity, and canonical pseudocode (negamax form in practice).

- **Principal Variation Search (PVS / NegaScout)**  
  https://www.chessprogramming.org/Principal_Variation_Search  
  *Used for:* Narrow-window re-search on the principal variation; speedups over plain AB in AB+/AB++.

- **Iterative Deepening**  
  https://www.chessprogramming.org/Iterative_Deepening  
  *Used for:* Depth-by-depth search with improved move ordering and anytime behavior.

- **Transposition Tables (TT)**  
  https://www.chessprogramming.org/Transposition_Table  
  *Used for:* Reusing subtrees; cornerstone of AB+ and AB++; required Zobrist keys & TT entry semantics.

- **Zobrist Hashing**  
  https://www.chessprogramming.org/Zobrist_Hashing  
  *Used for:* Fast incremental hashing for TT keys on Quantik boards; stable keys across move/undo.

- **Move Ordering (overview)**  
  https://www.chessprogramming.org/Move_Ordering  
  *Used for:* General strategies to reduce effective branching factor (captures first, killer/history, TT move).

- **Killer Heuristic**  
  https://www.chessprogramming.org/Killer_Heuristic  
  *Used for:* Prioritizing moves that previously caused cutoffs at the same depth (non-captures).

- **History Heuristic**  
  https://www.chessprogramming.org/History_Heuristic  
  *Used for:* Global scoring of moves that frequently produce cutoffs, supporting LMR and ordering.

- **Aspiration Windows**  
  https://www.chessprogramming.org/Aspiration_Windows  
  *Used for:* Speedups by searching in a narrow score window around the last value; re-search on fail.

- **Quiescence Search**  
  https://www.chessprogramming.org/Quiescence_Search  
  *Used for:* Tackling horizon effects by extending “noisy” positions; simplified for Quantik tactics.

- **Null-Move Pruning**  
  https://www.chessprogramming.org/Null_Move_Pruning  
  *Used for:* Aggressive pruning with a “pass” assumption; guarded to avoid zugzwang-like traps.

- **Late Move Reductions (LMR)**  
  https://www.chessprogramming.org/Late_Move_Reductions  
  *Used for:* Reduce search depth for late, non-tactical moves after good ordering; parameterized by depth/move-count.

- **Futility Pruning**  
  https://www.chessprogramming.org/Futility_Pruning  
  *Used for:* Shallow pruning where raising alpha is implausible; tuned for Quantik’s evaluation scale.

- **Razoring**  
  https://www.chessprogramming.org/Razoring  
  *Used for:* Pre-qsearch cut at shallow depths when static eval is far below alpha (guard rails applied).

- **Multi-Cut Pruning**  
  https://www.chessprogramming.org/Multi-Cut  
  *Used for:* Cutting a node when a small sample of children already produce cutoffs; experimental in AB++.

- **MTD(f) (Plaat)**  
  https://people.csail.mit.edu/plaat/mtdf.html  
  *Used for:* Reference on zero-window re-search scheme; benchmarked conceptually vs. PVS in notes.

- **Negamax — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Negamax  
  *Used for:* Unifying alpha–beta logic around a single scoring convention; simplifies code and matches your `player_just_moved` backups and PVS-style search.

- **Transposition Tables (replacement & aging policies)**  
  https://www.chessprogramming.org/Transposition_Table  
  *Used for:* Practical guidance on bucket organizations, replacement schemes (depth-preferred, always-replace), and aging. Helps explain how policy choice impacts TT hit rate, stability, and playing strength in AB+/AB++.

- **Zobrist Hashing — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Zobrist_Hashing  
  *Used for:* Implementation details and pitfalls (incremental key updates on make/undo, randomness requirements, avoiding collisions) beyond the basic definition used for TT keys.

> **Where this shows up:** `ai_players/alphabeta*` families: *AlphaBeta (baseline)* uses clean AB+ID & simple eval; *AlphaBeta+* adds TT, Zobrist, ordering (killer/history), aspiration; *AlphaBeta++* layers qsearch, LMR, futility/razoring, and selective pruning toggles. consistent negamax formulation; TT replacement/aging policy notes for AB+/AB++; Zobrist keys for stable hashing across move/undo.

---

## 2) Monte Carlo Tree Search (MCTS) & Variants

- **Survey: “A Survey of Monte Carlo Tree Search Methods” (Browne et al., 2012)**  
  https://arxiv.org/abs/1204.4510  
  *Used for:* Big-picture of UCT, RAVE/AMAF, progressive bias/widening, backups, parallelization.

- **Coulom (2006): Efficient Selectivity and Backup Operators in MCTS**  
  https://hal.science/inria-00116992/document  
  *Used for:* Early modern MCTS formulation, selection/backup design choices.

- **UCT: Bandit-based Monte-Carlo Planning (Kocsis & Szepesvári, 2006)**  
  https://www.cs.elte.hu/~szepesvari/cikkek/2006-BanditBasedMonteCarloPlanning.pdf  
  *Used for:* UCT selection formula (UCB1 at nodes) and theoretical grounding of exploration/exploitation.

- **UCT — ChessProgramming Wiki**  
  https://www.chessprogramming.org/UCT  
  *Used for:* Implementation tips, normalization, and parameter conventions for UCT variants.

- **RAVE / AMAF — ChessProgramming Wiki**  
  https://www.chessprogramming.org/RAVE  
  *Used for:* Sharing statistics across moves to speed up early estimates (optional in MCTS+).

- **Progressive Bias — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Progressive_Bias  
  *Used for:* Injecting domain heuristics into UCT to guide selection in shallow trees.

- **Progressive Widening — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Progressive_Widening  
  *Used for:* Capping children expansion in high-branching games (Quantik can spike early).

- **Virtual Loss — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Virtual_Loss  
  *Used for:* Thread-friendly parallel MCTS to avoid over-exploring the same node concurrently.

- **MCTS-Solver — ChessProgramming Wiki**  
  https://www.chessprogramming.org/MCTS-Solver  
  *Used for:* Integrating solver flags into nodes when forced wins/losses are proven during rollouts/backups.

  - **Chaslot (2010): *Monte-Carlo Tree Search* (PhD thesis, Maastricht)**  
  https://project.dke.maastrichtuniversity.nl/games/files/phd/Chaslot_thesis.pdf  
  *Used for:* Progressive bias/widening, virtual loss, and parallel MCTS design choices with worked examples (directly relevant to MCTS+ and MCTS++ engineering decisions).

- **Couëtoux et al. (2011): Double Progressive Widening (DPW)**  
  https://hal.science/hal-00696980/document  
  *Used for:* Controlling child expansion in large/wide branching games; provides an effective widening schedule you can enable in MCTS+ when Quantik branching spikes.

- **Lanctot et al. (2014): Implicit Minimax Backups in MCTS**  
  https://arxiv.org/abs/1406.0486  
  *Used for:* Mixing heuristic/minimax-style values into UCT backups—exactly the kind of AB×MCTS blending mentioned in your hybrids and useful for stabilizing playout estimates.

- **Gelly (2010): Computational Experiments with the RAVE Heuristic**  
  https://deepai.org/publication/computational-experiments-with-the-rave-heuristic  
  *Used for:* AMAF/RAVE details and empirical behavior to justify adding RAVE to MCTS+ for quicker early move estimates.

- **Auer, Cesa-Bianchi & Fischer (2002): Finite-time Analysis of the Multi-armed Bandit Problem (UCB1)**  
  https://www.tandfonline.com/doi/abs/10.1080/02331880208997252  
  *Used for:* The UCB1 foundation behind UCT selection; parameter intuition and theory notes for your UCT constant and exploration schedule.

- **Silver et al. (2017): *Mastering Chess and Shogi by Self-Play with a General RL Algorithm* (AlphaZero)**  
  https://arxiv.org/abs/1712.01815  
  *Used for:* PUCT-style selection and policy-prior injection—motivates your progressive bias/priors in MCTS++.

- **Winands et al. (2017): *MCTS-Solver***  
  https://dke.maastrichtuniversity.nl/m.winands/documents/mctswinands2017.pdf  
  *Used for:* Integrating solver flags into nodes when forced wins/losses are proven during search and backups.

> **Where this shows up:** `ai_players/mcts*` families: *MCTS (baseline)* implements UCT with simple playouts; *MCTS+* layers RAVE/progressive bias (domain hints), widening control and (optionally) solver flags. 

---

## 3) Evolutionary / Genetic Algorithms (GA) & Evaluation Tuning

- **Genetic Algorithms — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Genetic_algorithms  
  *Used for:* Patterns for tuning evaluation parameters of board-game engines (encoding, selection, mutation).

- **Genetic Algorithm (overview) — Wikipedia**  
  https://en.wikipedia.org/wiki/Genetic_algorithm  
  *Used for:* Standard operators (selection/crossover/mutation), schema theorem background, and references.

- **Eiben & Smith (2003): *Introduction to Evolutionary Computing***  
  https://link.springer.com/book/10.1007/978-3-662-44874-8  
  *Used for:* Practical parameterization of EA/GA, representation choices, and termination criteria.

- **Holland (1975/1992): *Adaptation in Natural and Artificial Systems***  
  https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/  
  *Used for:* Foundational GA principles; motivated our encoding/selection templates.

- **De Jong (2006): *Evolutionary Computation: A Unified Approach***  
  https://mitpress.mit.edu/9780262041941/evolutionary-computation/  
  *Used for:* Unified view of EC variants and convergence considerations relevant to offline tuning.

- **No Free Lunch (Wolpert & Macready) — overview**  
  https://en.wikipedia.org/wiki/No_free_lunch_in_search_and_optimization  
  *Used for:* Why we benchmark/tune per-domain and don’t expect a universally superior optimizer.
  
- **CMA-ES (Hansen & Ostermeier) — tutorial/overview**  
  https://dspace.mit.edu/handle/1721.1/71941  
  *Used for:* A robust alternative to GA for continuous parameter tuning of evaluation weights; useful reference if you experiment beyond GA for AB-eval tuning.

- **Rubinstein & Kroese: The Cross-Entropy Method (tutorial)**  
  https://web.utk.edu/~cktai/CEtutorial.pdf  
  *Used for:* Another black-box optimizer that often competes with GA/CMA-ES for engine parameter tuning; handy baseline for offline optimization pipelines.

> **Where this shows up:** `ai_players/genetic*` families: *Genetic (baseline)* uses GA for policy/heuristic search; *Genetic (eval AB)* uses GA **specifically to tune the static evaluation used by AB**, i.e., AB search stays exact, but its scoring function parameters are optimized by GA on curated positions/self-play. complements GA by outlining drop-in alternatives (CMA-ES, CEM) for tuning evaluation weights used by AB search.

---

## 4) Hybrids: Why Combine Methods?

### 4.1 GA × Alpha–Beta (Evaluation tuned by GA)
- **Tuning & Parameter Tuning — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Tuning  
  https://www.chessprogramming.org/Parameter_Tuning  
  *Used for:* General methodology for tuning evaluation functions; GA is one viable optimizer among others.
- **Piece-Square Tables / Evaluation — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Piece-Square_Tables  
  https://www.chessprogramming.org/Evaluation  
  *Used for:* Structuring feature weights (tables and linear terms) that a GA can evolve.  
- **(Context) Texel’s Tuning Method**  
  https://www.chessprogramming.org/Texel%27s_Tuning_Method  
  *Used for:* Not GA, but a widely-cited baseline for eval tuning; we use it to frame alternatives and validation protocols.

*How it maps here:* Our **Genetic (eval AB)** pipeline evolves the **parameters of the AB evaluator** (not the search itself). This mirrors long-standing practice in chess/Shogi/Go engines where an offline optimizer (GA, CMA-ES, Texel, etc.) tunes the scoring weights that the exact search (AB/PVS) relies on at the leaf and in move ordering.

### 4.2 MCTS × Alpha–Beta (Search hybrids)
- **MCTS–Minimax Hybrid — ChessProgramming Wiki**  
  https://www.chessprogramming.org/MCTS-Minimax_Hybrid  
  *Used for:* Design patterns for mixing AB-like backups with MCTS selection/expansion.  
- **Implicit Minimax Backups — ChessProgramming Wiki**  
  https://www.chessprogramming.org/Implicit_Minimax_Backups  
  *Used for:* Blending heuristic (minimax-style) values into MCTS backups to stabilize estimates.  
- **Progressive Bias / RAVE (again)**  
  https://www.chessprogramming.org/Progressive_Bias  
  https://www.chessprogramming.org/RAVE  
  *Used for:* Injecting AB-style heuristics or lightweight static eval as bias during UCT selection.
- **Lanctot et al. (2014): Implicit Minimax Backups in MCTS**  
  https://arxiv.org/abs/1406.0486  
  *Used for:* Concrete mechanism for blending AB-style heuristic values into MCTS backups, which stabilizes playout estimates in hybrid engines.
- **Silver et al. (2017): AlphaZero (PUCT)**  
  https://arxiv.org/abs/1712.01815  
  *Used for:* A modern, high-impact example of prior-guided tree search; informs your progressive-bias priors and PUCT-like selection in MCTS++ and AB×MCTS hybrids.
- **Chaslot (2010) thesis**  
  https://project.dke.maastrichtuniversity.nl/games/files/phd/Chaslot_thesis.pdf  
  *Used for:* Practical details on progressive bias and parallelization that carry over to AB×MCTS designs.


*How it maps here:* Our *AlphaBeta+MCTS* variants keep AB for **tactical exactness** while letting MCTS **guide** exploration or finish calculation in wide positions (or vice-versa). Progressive bias/implicit backups give MCTS “a hint” from AB’s static evaluation; conversely, AB benefits from MCTS priors for move ordering in early plies. implicit minimax for MCTS backups; PUCT/prior integration; progressive bias for move ordering interplay.

---

## 5) Quick Map: Which sources influenced which engines?

- **AlphaBeta (baseline):** Alpha–Beta, Iterative Deepening, basic ordering → AB links above.  
- **AlphaBeta+:** +TT & Zobrist; +Killer/History ordering; +Aspiration → AB/TT/move-ordering/aspiration links.  
- **AlphaBeta++:** +Quiescence, +LMR, +Futility/Razoring, +selective pruning → corresponding AB heuristic links.  
- **MCTS (baseline):** UCT selection, simple playouts → UCT/Kocsis–Szepesvári, Coulom.  
- **MCTS+:** +RAVE/AMAF, +Progressive Bias/Widening, +MCTS-Solver → MCTS links above.  
- **Genetic (baseline):** GA mechanics for policy/heuristic search → GA overview/book links.  
- **Genetic (eval AB):** GA tunes **evaluation parameters** used by AB search → GA + Tuning links.  
- **AlphaBeta+MCTS (hybrids):** Progressive bias/implicit minimax backups; optional hybrid leaf evaluation → hybrid links.

---

## 6) Quantik Rules & Engine Interface (core/)

- **Official Rulebook (FR) — “Quantik – Règles du jeu” (Gigamic)**  
  https://pierrickauger.files.wordpress.com/2021/12/quantik1.jpg  
  *Used for:* Canonical statement of the **single placement rule** (“you are **not allowed** to place a shape in a row/column/zone **where the opponent already has that shape**”) and the **win condition** (“first to complete a line/column/zone with the **four different shapes**”). This aligns with our `QuantikBoard.is_valid_move` and `check_victory` logic.

- **Masters of Games — Quantik rules (EN summary)**  
  https://www.mastersofgames.com/rules/quantik-rules.htm  
  *Used for:* English confirmation of the placement restriction (opponent’s identical shape blocks) and the **no-move = loss** clause used in AI playouts & runner diagnostics.

- **Board Game Arena — Quantik (rules overview)**  
  https://en.boardgamearena.com/gamepanel?game=quantik  
  *Used for:* Public **win condition** text and **stalemate rule** (“If a player cannot make a valid move, they lose.”), matching our rollout/runner “blocked ⇒ other wins”.

- **Philibert (FR) — Product page with rule excerpt**  
  https://www.philibertnet.com/fr/gigamic/84424-quantik-3421273314114.html  
  *Used for:* Plain-language restatement of the **opponent-only** shape restriction (mirrors our `is_valid_move` condition) and confirmation of the 4×4 board & 2×2 regions.

- **BoardGameGeek — Quantik**  
  https://boardgamegeek.com/boardgame/143515/quantik  
  *Used for:* Box-rule corroboration, imagery of the four shapes (circle/square/triangle/diamond) and component mapping to `core/types.Shape`.

> **Where this shows up in code:**  
> - `core/types.py` defines **Shape**/**Player** enums and a **Piece** wrapper, mirroring the game’s four shapes and two players.  
> - `core/rules.py::QuantikBoard` implements **4×4 grid**, **2×2 zones**, **opponent-only shape restriction**, **win by 4 different shapes**, and **no-move ⇒ loss** (via callers like playouts/runner).  
> - All AI implementations depend on this interface (pure reads of `board` + `pieces_count`).

### 6.1) Interface & Software Patterns (for the AI plugin API)

- **Strategy Pattern — Gamma et al., *Design Patterns* (1994)**  
  https://en.wikipedia.org/wiki/Strategy_pattern  
  *Used for:* Treating each AI (`QuantikAI`) as a **strategy plug‑in** behind a stable `get_move` interface (`core/ai_base.AIBase`).

- **Python `abc` — Abstract Base Classes (docs)**  
  https://docs.python.org/3/library/abc.html  
  *Used for:* Declaring the **AI contract** via an abstract `get_move(...)` method, enabling discovery/validation in the runner.

- **Python `enum` (docs)**  
  https://docs.python.org/3/library/enum.html  
  *Used for:* `Shape`/`Player` enums — clearer, type‑safe interop between core, AIs, and the runner.

- **Type Hints — PEP 484**  
  https://peps.python.org/pep-0484/  
  *Used for:* Static clarity across modules (`Optional[Tuple[int,int,Shape]]`, etc.), easing refactors and testability.

---

## 7) Tournament Runner — Methodology, Math & Engineering (tournament/runner.py)

### 7.1) Statistical Reporting & Ratings

- **Wilson Score Interval (overview)**  
  https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval  
  *Used for:* 95% CI on match win-rates printed by our runner (`WR`, `95% CI`).

- **Brown, Cai & DasGupta (2001): Interval Estimation for a Binomial Proportion**  
  https://doi.org/10.1002/1097-0258(20010430)20:9%3C1093::AID-SIM514%3E3.0.CO;2-F  
  *Used for:* Modern review & justification of score/Wilson intervals.

- **Bootstrap (Efron, 1979): “Bootstrap Methods: Another Look at the Jackknife”**  
  https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full  
  *Used for:* Empirical 2.5–97.5% confidence band (“boot …”) on WR in our reports.

- **SPRT — ChessProgramming Wiki**  
  https://www.chessprogramming.org/SPRT  
  *Used for:* Sequential testing to stop A vs. B duels early with controlled error (optional in runner).

- **Elo Rating — overview**  
  https://en.wikipedia.org/wiki/Elo_rating_system  
  *Used for:* Baseline rating & expected score; our “Approx. Elo” table is a single-pass estimate.

- **Glicko-1: Introduction & Guide (Mark Glickman)**  
  http://www.glicko.net/glicko.html  
  *Used for:* How rating (r) and rating deviation (RD) work; our runner prints “r ± RD”.

- **Glicko-1 Paper (1999): “Parameter Estimation in Large Dynamic Paired Comparison Experiments”**  
  http://www.glicko.net/research/glicko.pdf  
  *Used for:* Formal derivation of updates; informs our per-game sequential updates.

- **BayesElo — ChessProgramming Wiki**  
  https://www.chessprogramming.org/BayesElo  
  *Used for:* Widely used engine-testing tooling and Bayesian treatment of ratings; complements Elo/Glicko reporting and helps validate rankings in tourneys.

- **Bradley–Terry model — Wikipedia**  
  https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model  
  *Used for:* Pairwise-comparison foundation behind Elo-like systems; useful when fitting win probabilities directly from match results.

- **TrueSkill 2 — Herbrich, Minka & Graepel (2018)**  
  https://www.microsoft.com/en-us/research/publication/trueskill-2-an-improved-bayesian-skill-rating-system/  
  *Used for:* Modern Bayesian rating alternative to Elo/Glicko, with improved variance handling; a good cross-check for leaderboards with uncertainty.

- **SPRT — Wikipedia (foundations)**  
  https://en.wikipedia.org/wiki/Sequential_probability_ratio_test  
  *Used for:* Statistical test behind early stopping in A–B duels; complements the ChessProgramming.org SPRT page already cited.

> **Where this shows up:** `tournament/runner*.py` reports win-rate with Wilson CI, optional **bootstrap CI**, and maintains both **Elo** (approx) and **Glicko-1 (r ± RD)** leaderboards; optional **SPRT** can stop A–B duels early. add optional BayesElo/Bradley–Terry/TrueSkill-2 references alongside Wilson/Bootstrap/Elo/Glicko/SPRT for fuller statistical context.


### 7.2) Engineering & Reproducibility

- **Python `random` (docs)**  
  https://docs.python.org/3/library/random.html  
  *Used for:* Controlled seeding per game (`SEEDS`, `base` scheme) and bootstrap resampling.

- **Python `time` (docs) — `perf_counter`**  
  https://docs.python.org/3/library/time.html#time.perf_counter  
  *Used for:* High‑resolution, monotonic timings of per‑move think time in logs.

- **PEP 418 — Monotonic Clock & `perf_counter`**  
  https://peps.python.org/pep-0418/  
  *Used for:* Rationale behind choosing `perf_counter` for timing vs. wall clocks.

- **Python `threading` (docs)**  
  https://docs.python.org/3/library/threading.html  
  *Used for:* **Mute‑probe** with timeout isolation to exclude non‑responsive AIs safely.

- **Python `csv` & `json` (docs)**  
  https://docs.python.org/3/library/csv.html  
  https://docs.python.org/3/library/json.html  
  *Used for:* Structured output (`games.csv`, `moves.csv`) and streaming **JSON Lines** (`moves.jsonl`).

- **JSON Lines — Format spec**  
  https://jsonlines.org/  
  *Used for:* Append‑friendly move logs for downstream analysis.

- **Python `importlib`, `pkgutil` (docs)**  
  https://docs.python.org/3/library/importlib.html  
  https://docs.python.org/3/library/pkgutil.html  
  *Used for:* **Dynamic discovery** of AI plug‑ins in `ai_players/*/algorithme.py` at runtime.

- **Python `pathlib` (docs)**  
  https://docs.python.org/3/library/pathlib.html  
  *Used for:* Cross‑platform run folders (`tournament/out/YYYYMMDD-HHMMSS`).

- **Python `hashlib` (docs)**  
  https://docs.python.org/3/library/hashlib.html  
  *Used for:* Short SHA‑1 hashes in `run_config.json` to fingerprint modules for reproducibility.

- **Python `math` (docs)**  
  https://docs.python.org/3/library/math.html  
  *Used for:* Numerical routines in Wilson CI, Elo/Glicko updates, and SPRT LLR calculations.

> **Where this shows up in code:**  
> - Pair engine (`run_pair_diagnostic`) logs per‑game lines, **opening keys**, timings, branch factors, forced moves, **missed wins** and **handing the opponent an immediate win**.  
> - Aggregate stage prints **WR with Wilson CI** (and optional **bootstrap CI**), **approx Elo**, **Glicko‑1 r ± RD**, and a **pair matrix**.  
> - `USE_SPRT` lets you stop long matchups early with controlled error rates.

---

### License & Attribution Notes
All links above point to external resources. Please respect their licenses and citation requirements when reusing figures, tables, or code snippets from those sources.

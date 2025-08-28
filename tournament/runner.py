# tournament/runner.py
# ----------------------------------------------------------
# Tournoi "rapide" IA vs IA (multi-seeds, peu de parties)
# - Exclut IAs muettes (probe)
# - Pour chaque pair (A,B) et chaque seed:
#     * 2 parties: A commence, B commence
# - Affiche WR(A) + IC 95%, StartsWon/RepliesWon
# - Résumé agrégé, Elo approximatif et matrice
# ----------------------------------------------------------
from __future__ import annotations
import importlib, pkgutil, pathlib, random, time, math, threading
from typing import List, Dict, Tuple, Optional
from core.types import Shape, Player, Piece
from core.rules import QuantikBoard

# ============= Helpers plateau / compat =============
def raw_board(board: QuantikBoard):
    return board.raw() if hasattr(board, "raw") else board.board

def empty_position():
    b = QuantikBoard()
    pieces = {
        Player.PLAYER1: {s: 2 for s in Shape},
        Player.PLAYER2: {s: 2 for s in Shape},
    }
    return b, pieces

# ============= Découverte des IA =============
def discover_ais():
    base_pkg = "ai_players"
    base_path = pathlib.Path(__file__).resolve().parents[1] / base_pkg
    ais, errors = [], []
    if not base_path.exists():
        return [], ["(ai_players manquant)"]

    for pkg in pkgutil.iter_modules([str(base_path)]):
        if pkg.name == "template":
            continue
        mod_name = f"{base_pkg}.{pkg.name}.algorithme"
        try:
            mod = importlib.import_module(mod_name)
            ai_cls  = getattr(mod, "QuantikAI", None)
            ai_name = getattr(mod, "AI_NAME", pkg.name)
            if ai_cls:
                ais.append({"name": ai_name, "cls": ai_cls})
            else:
                errors.append(f"{mod_name} (QuantikAI introuvable)")
        except Exception as e:
            errors.append(f"{mod_name} (import error: {e})")
    ais.sort(key=lambda x: x["name"].lower())
    return ais, errors

# ============= Probe "IA muette" =============
def probe_ai_speaks(ai_cls, timeout: float = 2.0) -> bool:
    b, pieces = empty_position()
    board_raw = raw_board(b)
    okbox = {"ok": False}

    def worker():
        try:
            ai = ai_cls(Player.PLAYER1)
            mv = ai.get_move(board_raw, pieces)
            if not mv:
                okbox["ok"] = False
                return
            r, c, sh = mv
            okbox["ok"] = bool(b.place_piece(r, c, Piece(sh, Player.PLAYER1)))
        except Exception:
            okbox["ok"] = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    return okbox["ok"]

def filter_mute_ais(ais, do_filter=True):
    if not do_filter:
        return ais, []
    kept, excluded = [], []
    for a in ais:
        if probe_ai_speaks(a["cls"]):
            kept.append(a)
        else:
            excluded.append(a["name"])
    return kept, excluded

# ============= Moteur d'une partie =============
def play_one_game(aiA_cls, aiB_cls, starter: Player, seed: Optional[int] = None) -> Tuple[Player, Dict[str,int]]:
    if seed is not None:
        random.seed(seed)

    board = QuantikBoard()
    pieces = {
        Player.PLAYER1: {s: 2 for s in Shape},
        Player.PLAYER2: {s: 2 for s in Shape},
    }
    aiA = aiA_cls(Player.PLAYER1)  # A est toujours P1
    aiB = aiB_cls(Player.PLAYER2)  # B est toujours P2

    current = starter
    A_started = 1 if starter == Player.PLAYER1 else 0
    A_won_start = 0
    A_won_reply = 0

    while True:
        ai = aiA if current == Player.PLAYER1 else aiB
        move = ai.get_move(raw_board(board), pieces)
        if not move:
            winner = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1
            if winner == Player.PLAYER1:
                if A_started: A_won_start = 1
                else:         A_won_reply = 1
            return winner, {"A_started": A_started, "A_won_start": A_won_start, "A_won_reply": A_won_reply}

        r, c, shape = move
        if not board.place_piece(r, c, Piece(shape, current)):
            winner = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1
            if winner == Player.PLAYER1:
                if A_started: A_won_start = 1
                else:         A_won_reply = 1
            return winner, {"A_started": A_started, "A_won_start": A_won_start, "A_won_reply": A_won_reply}

        pieces[current][shape] -= 1
        if board.check_victory():
            winner = current
            if winner == Player.PLAYER1:
                if A_started: A_won_start = 1
                else:         A_won_reply = 1
            return winner, {"A_started": A_started, "A_won_start": A_won_start, "A_won_reply": A_won_reply}

        current = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1

# ============= Stats =============
def wilson_interval(wins: int, total: int, z: float = 1.96) -> Tuple[float,float]:
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1 + z*z/total
    centre = p + z*z/(2*total)
    margin = z * math.sqrt((p*(1-p)+z*z/(4*total))/total)
    lo = (centre - margin)/denom
    hi = (centre + margin)/denom
    return max(0.0, lo), min(1.0, hi)

def elo_update(ra: float, rb: float, sa: float, k: float = 16.0) -> Tuple[float,float]:
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    return ra + k * (sa - ea), rb + k * ((1 - sa) - eb)

# ============= Duelo multi-seeds (rápido) =============
def run_pair_quick(iaA: Dict, iaB: Dict, seeds: List[int], games_per_seed: int = 1):
    """
    Para cada seed:
      - joga 2*games_per_seed partidas:
         * para cada g: A começa (P1), depois B começa (P2)
    games_per_seed=1 => por seed são 2 partidas (A-start e B-start).
    """
    Aname, Bname = iaA["name"], iaB["name"]
    wA = wB = 0
    A_starts_won = 0
    A_replies_won = 0
    t0 = time.time()

    for seed in seeds:
        # jogamos 'games_per_seed' pares (A-start / B-start)
        for g in range(games_per_seed):
            base = seed * 10_000 + g

            # A começa
            winner, loc = play_one_game(iaA["cls"], iaB["cls"], starter=Player.PLAYER1, seed=base+1)
            if winner == Player.PLAYER1:
                wA += 1
                A_starts_won += loc["A_won_start"]
                A_replies_won += loc["A_won_reply"]
            else:
                wB += 1

            # B começa (A responde)
            winner, loc = play_one_game(iaA["cls"], iaB["cls"], starter=Player.PLAYER2, seed=base+2)
            if winner == Player.PLAYER1:
                wA += 1
                A_starts_won += loc["A_won_start"]
                A_replies_won += loc["A_won_reply"]
            else:
                wB += 1

    elapsed = time.time() - t0
    games = wA + wB
    wr = wA / games if games else 0.0
    lo, hi = wilson_interval(wA, games)
    print(f"{Aname} vs {Bname} -> {wA}-{wB} sur {games} | "
          f"WR(A)={wr:.3f} (95% CI: {lo:.3f}-{hi:.3f}) | "
          f"StartsWon(A)={A_starts_won}, RepliesWon(A)={A_replies_won} | "
          f"{elapsed:.1f}s")
    return {
        "A": Aname, "B": Bname,
        "wA": wA, "wB": wB, "games": games,
        "wrA": wr, "ci_low": lo, "ci_high": hi,
        "time": elapsed,
        "A_starts_won": A_starts_won,
        "A_replies_won": A_replies_won,
    }

# ============= Agregação/relatórios =============
def aggregate(results: List[Dict]):
    # Totais por IA
    totals: Dict[str, Dict] = {}
    for r in results:
        for side in ("A", "B"):
            name = r[side]
            if name not in totals:
                totals[name] = {"wins":0, "losses":0}
        totals[r["A"]]["wins"]  += r["wA"]
        totals[r["A"]]["losses"]+= r["wB"]
        totals[r["B"]]["wins"]  += r["wB"]
        totals[r["B"]]["losses"]+= r["wA"]

    print("\n=== Résumé agrégé par IA (winrate cumulé) ===")
    lines = []
    for name, t in totals.items():
        w, l = t["wins"], t["losses"]
        g = w + l
        wr = w/g if g else 0.0
        lo, hi = wilson_interval(w, g) if g else (0.0, 0.0)
        lines.append((name, w, l, wr, lo, hi))
    lines.sort(key=lambda x: x[3], reverse=True)
    for (name, w, l, wr, lo, hi) in lines:
        print(f"{name:28s} {w:4d}-{l:<4d}  WR={wr:.3f}  (95% CI {lo:.3f}-{hi:.3f})")

    # Elo aproximado
    elo = {name: 1000.0 for name in totals.keys()}
    for r in results:
        A, B = r["A"], r["B"]
        wA, wB = r["wA"], r["wB"]
        games = max(1, wA + wB)
        # atualiza com score agregado dessa dupla
        scoreA = wA / games
        elo[A], elo[B] = elo_update(elo[A], elo[B], scoreA, k=24.0)

    print("\n=== Classement ELO (approx.) ===")
    for name, rating in sorted(elo.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:28s} ELO={rating:.1f}")

    # Matriz A x B
    names = sorted(totals.keys())
    idx = {n:i for i,n in enumerate(names)}
    M = [[None for _ in names] for _ in names]
    for r in results:
        i, j = idx[r["A"]], idx[r["B"]]
        M[i][j] = f"{r['wA']:>3d}-{r['wB']:<3d}"

    print("\n=== Matrice de confrontations (A bat B) ===")
    header = "                         | " + " | ".join(f"{n[:12]:>12s}" for n in names)
    print(header)
    print("-" * len(header))
    for i, ni in enumerate(names):
        row = f"{ni[:25]:<25} | "
        row += " | ".join(f"{(M[i][j] or ''):>12s}" for j in range(len(names)))
        print(row)

# ============= Main (parâmetros “rápidos”) =============
def main():
    # Seeds curtas e poucas partidas por seed (rápido!)
    SEEDS = [101, 202, 303, 404, 505, 606, 707]  # mais seeds
    GAMES_PER_SEED = 2                           # mais partidas por seed
    FILTER_MUTE = True

    ais, errors = discover_ais()
    ais, mute_excluded = filter_mute_ais(ais, do_filter=FILTER_MUTE)
    print(f"IAs découvertes ({len(ais)}): {[a['name'] for a in ais]}")
    if errors:
        print(f"⚠️  Erreurs d’import: {errors}")
    if FILTER_MUTE and mute_excluded:
        print(f"⚠️  Exclues (muettes au probe): {mute_excluded}")

    results = []
    for i in range(len(ais)):
        for j in range(i+1, len(ais)):
            r = run_pair_quick(ais[i], ais[j], seeds=SEEDS, games_per_seed=GAMES_PER_SEED)
            results.append(r)

    aggregate(results)

if __name__ == "__main__":
    main()
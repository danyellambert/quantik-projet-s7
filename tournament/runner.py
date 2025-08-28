# tournament/runner.py
# =====================================================================================
# Tournoi IA vs IA (diagnostic complet + couches « pro »)
# - Découvre les IA dans ai_players/*/algorithme.py (classe QuantikAI, constante AI_NAME)
# - Filtre les IA « muettes » via un probe avec délai (timeout)
# - Round-robin A<B ; pour chaque graine (seed), joue 2 parties (A commence / B commence) × GAMES_PER_SEED
# - Journal par PARTIE (games.csv) et par COUP (moves.csv / moves.jsonl)
# - Mesures : WR (IC de Wilson et optionnellement bootstrap), victoires en début/réponse,
#   temps total et temps moyen par coup, longueur moyenne (plies), ouverture,
#   nombre moyen de branches par tour, coups forcés, victoires immédiates manquées,
#   coups qui donnent une victoire immédiate à l’adversaire.
# - Classements : ELO approx + Glicko-1 (r ± RD)
# - Enregistre dans un dossier unique horodaté en UTC : out/YYYYMMDD-HHMMSS
# - Écrit run_config.json avec toutes les configs + la liste des IA participantes
# =====================================================================================

from __future__ import annotations
import importlib, pkgutil, pathlib, random, time, math, threading, csv, json, hashlib
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime, timezone

from core.types import Shape, Player, Piece
from core.rules import QuantikBoard

# ===================== Configuration ======================
# • Pour ~10 IA, ces valeurs donnent un bon volume/précision.
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 112, 213, 314, 415, 516, 617, 718, 819, 920]
GAMES_PER_SEED = 10

FILTER_MUTE   = True
PROBE_TIMEOUT = 1.75    # permissif pour éviter les faux négatifs

# Contrôle du temps par coup (optionnel).
# None = sans limite (chaque IA utilise le temps qu’elle veut). Pour standardiser, définir p. ex. 1.25.
TIME_PER_MOVE: Optional[float] = None

# SPRT (arrêt anticipé par paire) – utile pour accélérer les grands tournois
USE_SPRT   = False
SPRT_P0    = 0.50     # H0 : p = 0.50
SPRT_P1    = 0.60     # H1 : p = 0.60 (A significativement meilleure)
SPRT_ALPHA = 0.05
SPRT_BETA  = 0.10

# Bootstrap pour IC empirique du taux de victoire (0 = désactivé ; 2000 est une bonne valeur)
BOOTSTRAP_N = 2000

# Glicko-1 (classement) – activé par défaut
USE_GLICKO = True
GLICKO_START_RATING = 1500.0
GLICKO_START_RD     = 350.0   # incertitude initiale

# Verbosité
SHOW_PER_GAME_LINES = True
FIRST_PLIES_TO_LOG  = 4
TOP_OPENINGS_TO_SHOW = 8

# Répertoire racine de sortie (l’exécution crée un sous-dossier horodaté)
OUT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tournament" / "out"
# ===================================================


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

# ====== Règles (pour diagnostic de légalité) ======
N = 4
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]
CENTER = {(1,1), (1,2), (2,1), (2,2)}

def other(p: Player) -> Player:
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board_grid, row: int, col: int, shape: Shape, me: Player) -> bool:
    if board_grid[row][col] is not None:
        return False
    # ligne
    for cc in range(N):
        p = board_grid[row][cc]
        if p is not None and p.shape == shape and p.player != me:
            return False
    # colonne
    for rr in range(N):
        p = board_grid[rr][col]
        if p is not None and p.shape == shape and p.player != me:
            return False
    # zone
    z = zone_index(row, col)
    for (rr, cc) in ZONES[z]:
        p = board_grid[rr][cc]
        if p is not None and p.shape == shape and p.player != me:
            return False
    return True

def generate_valid_moves(board_grid, me: Player, my_counts: Dict[Shape,int]) -> List[Tuple[int,int,Shape]]:
    moves = []
    for sh in Shape:
        if my_counts.get(sh, 0) <= 0:
            continue
        for r in range(N):
            for c in range(N):
                if board_grid[r][c] is None and is_valid_move(board_grid, r, c, sh, me):
                    moves.append((r, c, sh))
    return moves

def opening_key(open_moves: List[Tuple[int,int,Shape]]) -> str:
    return " | ".join(f"{r},{c},{sh.name}" for (r,c,sh) in open_moves)


# ============= Découverte des IA =============
def discover_ais():
    base_pkg = "ai_players"
    base_path = pathlib.Path(__file__).resolve().parents[1] / base_pkg
    ais, errors = [], []
    if not base_path.exists():
        return [], ["(ai_players absent)"]
    for pkg in pkgutil.iter_modules([str(base_path)]):
        if pkg.name == "template":
            continue
        mod_name = f"{base_pkg}.{pkg.name}.algorithme"
        try:
            mod = importlib.import_module(mod_name)
            ai_cls  = getattr(mod, "QuantikAI", None)
            ai_name = getattr(mod, "AI_NAME", pkg.name)
            if ai_cls:
                ais.append({"name": ai_name, "cls": ai_cls, "module": mod_name})
            else:
                errors.append(f"{mod_name} (QuantikAI introuvable)")
        except Exception as e:
            errors.append(f"{mod_name} (erreur d’import : {e})")
    ais.sort(key=lambda x: x["name"].lower())
    return ais, errors


# ============= Probe « IA muette » =============
def probe_ai_speaks(ai_cls, timeout: float = PROBE_TIMEOUT) -> bool:
    b, pieces = empty_position()
    board_raw = raw_board(b)
    okbox = {"ok": False}
    def worker():
        try:
            ai = ai_cls(Player.PLAYER1)
            mv = ai.get_move(board_raw, pieces)
            okbox["ok"] = bool(mv)
        except Exception:
            okbox["ok"] = False
    t = threading.Thread(target=worker, daemon=True)
    t.start(); t.join(timeout)
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


# ============= Budget de temps (optionnel) =============
def set_time_budget(ai_obj, seconds: Optional[float]):
    if seconds is None:
        return
    for attr in ["time_limit", "total_time", "think_time", "budget", "max_time_per_move"]:
        if hasattr(ai_obj, attr):
            try:
                setattr(ai_obj, attr, float(seconds))
            except Exception:
                pass


# ============= Statistiques & classement =============
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

def bootstrap_ci_wins(bin_outcomes: List[int], n_boot: int = 2000) -> Tuple[float,float]:
    if not bin_outcomes or n_boot <= 0:
        return (0.0, 0.0)
    wrs = []
    L = len(bin_outcomes)
    for _ in range(n_boot):
        s = [random.choice(bin_outcomes) for _ in range(L)]
        wrs.append(sum(s)/L)
    wrs.sort()
    return wrs[int(0.025*n_boot)], wrs[int(0.975*n_boot)-1]

def elo_update(ra: float, rb: float, sa: float, k: float = 24.0) -> Tuple[float,float]:
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    return ra + k * (sa - ea), rb + k * ((1 - sa) - eb)

# ---------- Glicko-1 (sans volatilité) ----------
GL_Q = math.log(10) / 400.0
def glicko_g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + (3.0 * (GL_Q**2) * (rd**2)) / (math.pi**2))
def glicko_E(r: float, rj: float, rdj: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-glicko_g(rdj) * (r - rj) / 400.0))
def glicko_update_once(rA: float, rdA: float, rB: float, rdB: float, sA: float) -> Tuple[float,float,float,float]:
    gB = glicko_g(rdB); EA = glicko_E(rA, rB, rdB)
    vA = 1.0 / ((GL_Q**2) * (gB**2) * EA * (1.0 - EA) + 1e-12)
    d2A_inv = (1.0 / (rdA**2)) + (1.0 / vA)
    rdA_new = math.sqrt(1.0 / d2A_inv)
    rA_new  = rA + (GL_Q / d2A_inv) * (gB * (sA - EA))
    # B
    sB = 1.0 - sA
    gA = glicko_g(rdA); EB = glicko_E(rB, rA, rdA)
    vB = 1.0 / ((GL_Q**2) * (gA**2) * EB * (1.0 - EB) + 1e-12)
    d2B_inv = (1.0 / (rdB**2)) + (1.0 / vB)
    rdB_new = math.sqrt(1.0 / d2B_inv)
    rB_new  = rB + (GL_Q / d2B_inv) * (gA * (sB - EB))
    rdA_new = max(30.0, min(350.0, rdA_new))
    rdB_new = max(30.0, min(350.0, rdB_new))
    return rA_new, rdA_new, rB_new, rdB_new


# ============= SPRT =============
def sprt_update_decision(wins: int, losses: int, p0: float, p1: float, alpha: float, beta: float) -> Optional[str]:
    n = wins + losses
    if n == 0:
        return None
    llr = wins * math.log(p1/p0) + losses * math.log((1-p1)/(1-p0))
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    if llr >= A: return "accept_H1"
    if llr <= B: return "accept_H0"
    return None


# ============= Moteur de partie (instrumenté) =============
def play_one_game(aiA_cls, aiB_cls, starter: Player, seed: Optional[int],
                  game_uid: str,
                  csv_games_writer, csv_moves_writer, jsonl_fp) -> Tuple[Player, Dict[str,Any]]:
    if seed is not None:
        random.seed(seed)

    board = QuantikBoard()
    grid = raw_board(board)
    pieces = {
        Player.PLAYER1: {s: 2 for s in Shape},
        Player.PLAYER2: {s: 2 for s in Shape},
    }
    aiA = aiA_cls(Player.PLAYER1)
    aiB = aiB_cls(Player.PLAYER2)
    set_time_budget(aiA, TIME_PER_MOVE)
    set_time_budget(aiB, TIME_PER_MOVE)

    current = starter
    A_started = 1 if starter == Player.PLAYER1 else 0

    time_used = {Player.PLAYER1: 0.0, Player.PLAYER2: 0.0}
    moves_played = {Player.PLAYER1: 0,   Player.PLAYER2: 0}
    opening_moves: List[Tuple[int,int,Shape]] = []
    end_reason = "victory"

    legal_branch_sum = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    forced_count = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    had_win_but_missed = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    gave_opp_immediate_win = {Player.PLAYER1: 0, Player.PLAYER2: 0}

    ply_index = 0
    winner = None

    try:
        while True:
            ai = aiA if current == Player.PLAYER1 else aiB

            legal_moves = generate_valid_moves(grid, current, pieces[current])
            legal_count = len(legal_moves)
            legal_branch_sum[current] += legal_count

            if legal_count == 0:
                winner = other(current); end_reason = "no_move"; break
            if legal_count == 1:
                forced_count[current] += 1

            had_immediate_win = 0
            for (r0, c0, sh0) in legal_moves:
                ok = board.place_piece(r0, c0, Piece(sh0, current))
                if ok and board.check_victory():
                    had_immediate_win = 1
                board.board[r0][c0] = None
                if had_immediate_win: break

            t0 = time.perf_counter()
            move = ai.get_move(grid, pieces)
            dt = time.perf_counter() - t0
            time_used[current] += dt

            illegal_flag = False
            if (not move) or (len(move) != 3):
                illegal_flag = True
            else:
                r, c, sh = move
                if not isinstance(sh, Shape) or r not in range(N) or c not in range(N):
                    illegal_flag = True
                elif not is_valid_move(grid, r, c, sh, current):
                    illegal_flag = True

            if illegal_flag:
                winner = other(current); end_reason = "illegal"
                _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                            current, None, dt, legal_count, had_immediate_win,
                            chose_win=0, gave_opp_win=0, center=0, note="illegal_or_none")
                break

            r, c, sh = move
            ok_place = board.place_piece(r, c, Piece(sh, current))
            if not ok_place:
                winner = other(current); end_reason = "illegal"
                _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                            current, (r,c,sh), dt, legal_count, had_immediate_win,
                            chose_win=0, gave_opp_win=0, center=int((r,c) in CENTER),
                            note="place_piece_failed")
                break

            pieces[current][sh] -= 1
            moves_played[current] += 1
            if len(opening_moves) < FIRST_PLIES_TO_LOG:
                opening_moves.append((r, c, sh))

            chose_win_now = 0
            if had_immediate_win:
                if board.check_victory():
                    chose_win_now = 1
                else:
                    had_win_but_missed[current] += 1

            gave_win = 0
            if not board.check_victory():
                opp = other(current)
                opp_legal = generate_valid_moves(grid, opp, pieces[opp])
                for (rr, cc, ssh) in opp_legal:
                    ok2 = board.place_piece(rr, cc, Piece(ssh, opp))
                    if ok2 and board.check_victory():
                        gave_win = 1
                    board.board[rr][cc] = None
                    if gave_win: break
                if gave_win:
                    gave_opp_immediate_win[current] += 1

            _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                        current, (r,c,sh), dt, legal_count, had_immediate_win,
                        chose_win=chose_win_now, gave_opp_win=gave_win,
                        center=int((r,c) in CENTER), note="ok")

            if board.check_victory():
                winner = current; end_reason = "victory"; break

            ply_index += 1
            current = other(current)

    except Exception as e:
        winner = other(current); end_reason = "exception"
        _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                    current, None, 0.0, 0, 0, 0, 0, 0, note=f"exception:{type(e).__name__}")

    A_won_start = 1 if (winner == Player.PLAYER1 and A_started == 1) else 0
    A_won_reply = 1 if (winner == Player.PLAYER1 and A_started == 0) else 0
    plies_total = moves_played[Player.PLAYER1] + moves_played[Player.PLAYER2]
    opening_str = opening_key(opening_moves)

    _write_game(csv_games_writer, game_uid, starter, winner, end_reason,
                plies_total, opening_str, time_used, moves_played,
                legal_branch_sum, forced_count, had_win_but_missed,
                gave_opp_immediate_win)

    return winner, {
        "A_started": A_started,
        "A_won_start": A_won_start,
        "A_won_reply": A_won_reply,
        "end_reason": end_reason,
        "plies_total": plies_total,
        "opening": opening_str,
        "timeA": time_used[Player.PLAYER1],
        "timeB": time_used[Player.PLAYER2],
        "movesA": moves_played[Player.PLAYER1],
        "movesB": moves_played[Player.PLAYER2],
        "branchA_sum": legal_branch_sum[Player.PLAYER1],
        "branchB_sum": legal_branch_sum[Player.PLAYER2],
        "forcedA": forced_count[Player.PLAYER1],
        "forcedB": forced_count[Player.PLAYER2],
        "missed_winA": had_win_but_missed[Player.PLAYER1],
        "missed_winB": had_win_but_missed[Player.PLAYER2],
        "gave_winA": gave_opp_immediate_win[Player.PLAYER1],
        "gave_winB": gave_opp_immediate_win[Player.PLAYER2],
        "turnsA": moves_played[Player.PLAYER1],
        "turnsB": moves_played[Player.PLAYER2],
        "bin_outcome": 1 if winner == Player.PLAYER1 else 0,
    }


# ===== Écriture des journaux (CSV / JSONL) =====
def _ensure_run_folder():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Run folder: {run_dir}")
    return run_dir

def _open_outputs(run_dir: pathlib.Path):
    games_csv = run_dir / "games.csv"
    moves_csv = run_dir / "moves.csv"
    moves_jsonl = run_dir / "moves.jsonl"
    games_fp = open(games_csv, "w", newline="", encoding="utf-8")
    moves_fp = open(moves_csv, "w", newline="", encoding="utf-8")
    jsonl_fp = open(moves_jsonl, "w", encoding="utf-8")

    csv_games_writer = csv.writer(games_fp)
    csv_games_writer.writerow([
        "game_uid","pair","starter","winner","end_reason","plies_total","opening",
        "time_p1","time_p2","moves_p1","moves_p2",
        "branch_sum_p1","branch_sum_p2","forced_p1","forced_p2",
        "missed_win_p1","missed_win_p2","gave_opp_win_p1","gave_opp_win_p2"
    ])

    csv_moves_writer = csv.writer(moves_fp)
    csv_moves_writer.writerow([
        "game_uid","ply","player","r","c","shape","time_sec","legal_count",
        "had_immediate_win","chose_immediate_win","gave_opp_immediate_win","is_center","note"
    ])

    return games_fp, moves_fp, jsonl_fp, csv_games_writer, csv_moves_writer

def _close_outputs(*fps):
    for fp in fps:
        if fp:
            fp.close()

def _write_game(csv_games_writer, game_uid, starter, winner, end_reason,
                plies_total, opening_str, time_used, moves_played,
                branch_sum, forced_count, missed_win, gave_opp):
    csv_games_writer.writerow([
        game_uid,
        "",  # champ libre
        "A" if starter == Player.PLAYER1 else "B",
        "A" if winner  == Player.PLAYER1 else "B",
        end_reason, plies_total, opening_str,
        f"{time_used[Player.PLAYER1]:.6f}", f"{time_used[Player.PLAYER2]:.6f}",
        moves_played[Player.PLAYER1], moves_played[Player.PLAYER2],
        branch_sum[Player.PLAYER1], branch_sum[Player.PLAYER2],
        forced_count[Player.PLAYER1], forced_count[Player.PLAYER2],
        missed_win[Player.PLAYER1], missed_win[Player.PLAYER2],
        gave_opp[Player.PLAYER1], gave_opp[Player.PLAYER2],
    ])

def _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index, player, mv, tsec,
                legal_count, had_win, chose_win, gave_opp_win, center, note="ok"):
    r = c = None
    sh_name = None
    if mv:
        r, c, sh = mv
        sh_name = sh.name
    csv_moves_writer.writerow([
        game_uid, ply_index, ("A" if player == Player.PLAYER1 else "B") if player else None,
        r, c, sh_name, f"{tsec:.6f}", legal_count,
        had_win, chose_win, gave_opp_win, center, note
    ])
    obj = {
        "game_uid": game_uid,
        "ply": ply_index,
        "player": ("A" if player == Player.PLAYER1 else "B") if player else None,
        "move": {"r": r, "c": c, "shape": sh_name} if mv else None,
        "time_sec": tsec,
        "legal_count": legal_count,
        "had_immediate_win": bool(had_win),
        "chose_immediate_win": bool(chose_win),
        "gave_opp_immediate_win": bool(gave_opp_win),
        "is_center": bool(center),
        "note": note
    }
    jsonl_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============= Duel A×B (avec diagnostics) =============
def run_pair_diagnostic(iaA: Dict, iaB: Dict, seeds: List[int], games_per_seed: int = 1,
                        csv_games_writer=None, csv_moves_writer=None, jsonl_fp=None):
    Aname, Bname = iaA["name"], iaB["name"]

    wA = wB = 0
    A_starts_won = 0
    A_replies_won = 0
    t0_pair = time.time()

    time_used_pair = {Aname: 0.0, Bname: 0.0}
    moves_pair = {Aname: 0, Bname: 0}
    game_lengths: List[int] = []
    end_reasons = Counter()
    opening_counter = Counter()

    branch_sum_A = branch_sum_B = 0
    forced_A = forced_B = 0
    missed_win_A = missed_win_B = 0
    gave_opp_A = gave_opp_B = 0
    turnsA_total = 0
    turnsB_total = 0

    outcomes_bin = []
    sprt_decision = None

    for g in range(games_per_seed):
        for seed in seeds:
            if sprt_decision:
                break
            base = seed * 10_000 + 2*g

            # A commence
            game_uid = f"{Aname}__vs__{Bname}__seed{base+1}__Astart"
            winner, log = play_one_game(iaA["cls"], iaB["cls"], starter=Player.PLAYER1,
                                        seed=base+1, game_uid=game_uid,
                                        csv_games_writer=csv_games_writer,
                                        csv_moves_writer=csv_moves_writer,
                                        jsonl_fp=jsonl_fp)

            time_used_pair[Aname] += log["timeA"]; time_used_pair[Bname] += log["timeB"]
            moves_pair[Aname] += log["movesA"];    moves_pair[Bname] += log["movesB"]
            outcomes_bin.append(log["bin_outcome"])

            game_lengths.append(log["plies_total"])
            end_reasons[log["end_reason"]] += 1
            opening_counter[log["opening"]] += 1
            branch_sum_A += log["branchA_sum"]; branch_sum_B += log["branchB_sum"]
            forced_A += log["forcedA"]; forced_B += log["forcedB"]
            missed_win_A += log["missed_winA"]; missed_win_B += log["missed_winB"]
            gave_opp_A += log["gave_winA"]; gave_opp_B += log["gave_winB"]
            turnsA_total += log["turnsA"]; turnsB_total += log["turnsB"]

            if winner == Player.PLAYER1:
                wA += 1; A_starts_won += log["A_won_start"]; A_replies_won += log["A_won_reply"]
            else:
                wB += 1

            if SHOW_PER_GAME_LINES:
                print(f"  [{Aname} vs {Bname} | seed={base+1} | starter=A] "
                      f"winner={'A' if winner==Player.PLAYER1 else 'B'} "
                      f"len={log['plies_total']}, end={log['end_reason']}, "
                      f"tA={log['timeA']:.2f}s, tB={log['timeB']:.2f}s, opening={log['opening']}")

            if USE_SPRT:
                sprt_decision = sprt_update_decision(wA, wB, SPRT_P0, SPRT_P1, SPRT_ALPHA, SPRT_BETA)
                if sprt_decision: break

            # B commence
            game_uid = f"{Aname}__vs__{Bname}__seed{base+2}__Bstart"
            winner, log = play_one_game(iaA["cls"], iaB["cls"], starter=Player.PLAYER2,
                                        seed=base+2, game_uid=game_uid,
                                        csv_games_writer=csv_games_writer,
                                        csv_moves_writer=csv_moves_writer,
                                        jsonl_fp=jsonl_fp)

            time_used_pair[Aname] += log["timeA"]; time_used_pair[Bname] += log["timeB"]
            moves_pair[Aname] += log["movesA"];    moves_pair[Bname] += log["movesB"]
            outcomes_bin.append(log["bin_outcome"])

            game_lengths.append(log["plies_total"])
            end_reasons[log["end_reason"]] += 1
            opening_counter[log["opening"]] += 1
            branch_sum_A += log["branchA_sum"]; branch_sum_B += log["branchB_sum"]
            forced_A += log["forcedA"]; forced_B += log["forcedB"]
            missed_win_A += log["missed_winA"]; missed_win_B += log["missed_winB"]
            gave_opp_A += log["gave_winA"]; gave_opp_B += log["gave_winB"]
            turnsA_total += log["turnsA"]; turnsB_total += log["turnsB"]

            if winner == Player.PLAYER1:
                wA += 1; A_starts_won += log["A_won_start"]; A_replies_won += log["A_won_reply"]
            else:
                wB += 1

            if SHOW_PER_GAME_LINES:
                print(f"  [{Aname} vs {Bname} | seed={base+2} | starter=B] "
                      f"winner={'A' if winner==Player.PLAYER1 else 'B'} "
                      f"len={log['plies_total']}, end={log['end_reason']}, "
                      f"tA={log['timeA']:.2f}s, tB={log['timeB']:.2f}s, opening={log['opening']}")

            if USE_SPRT:
                sprt_decision = sprt_update_decision(wA, wB, SPRT_P0, SPRT_P1, SPRT_ALPHA, SPRT_BETA)
                if sprt_decision: break

    elapsed_pair = time.time() - t0_pair
    games = wA + wB
    wr = wA / games if games else 0.0
    lo, hi = wilson_interval(wA, games)
    blo, bhi = (0.0, 0.0)
    if BOOTSTRAP_N > 0:
        blo, bhi = bootstrap_ci_wins(outcomes_bin, BOOTSTRAP_N)

    avg_len = sum(game_lengths)/len(game_lengths) if game_lengths else 0.0
    avg_branch_A = (branch_sum_A / max(1, turnsA_total))
    avg_branch_B = (branch_sum_B / max(1, turnsB_total))

    tpmA = time_used_pair[Aname] / max(1, moves_pair[Aname])
    tpmB = time_used_pair[Bname] / max(1, moves_pair[Bname])
    miss_per100_A = 100.0 * missed_win_A / max(1, moves_pair[Aname])
    miss_per100_B = 100.0 * missed_win_B / max(1, moves_pair[Bname])
    gave_per100_A = 100.0 * gave_opp_A   / max(1, moves_pair[Aname])
    gave_per100_B = 100.0 * gave_opp_B   / max(1, moves_pair[Bname])

    sprt_msg = f" | SPRT={sprt_decision}" if sprt_decision else ""
    ci_boot = (f", boot {blo:.3f}-{bhi:.3f}" if BOOTSTRAP_N>0 else "")
    print(f"{Aname} vs {Bname} -> {wA}-{wB} sur {games} | WR(A)={wr:.3f} "
          f"(95% CI: {lo:.3f}-{hi:.3f}{ci_boot}){sprt_msg} | "
          f"StartsWon(A)={A_starts_won}, RepliesWon(A)={A_replies_won} | "
          f"{elapsed_pair:.1f}s total | "
          f"TimeUsed: A={time_used_pair[Aname]:.1f}s ({tpmA:.3f}s/c), "
          f"B={time_used_pair[Bname]:.1f}s ({tpmB:.3f}s/c) | "
          f"AvgLen={avg_len:.2f} plies | AvgBranch/turn: A={avg_branch_A:.2f}, B={avg_branch_B:.2f} | "
          f"miss/100: A={miss_per100_A:.2f}, B={miss_per100_B:.2f} | "
          f"gave/100: A={gave_per100_A:.2f}, B={gave_per100_B:.2f}")

    print(f"  End reasons: {dict(end_reasons)}")
    print(f"  Forced moves: A={forced_A}, B={forced_B}")
    print(f"  Missed immediate wins: A={missed_win_A}, B={missed_win_B}")
    print(f"  Gave opponent immediate win: A={gave_opp_A}, B={gave_opp_B}")

    if opening_counter and TOP_OPENINGS_TO_SHOW > 0:
        print("  Top openings:")
        for (op, cnt) in opening_counter.most_common(TOP_OPENINGS_TO_SHOW):
            print(f"    {cnt:3d}×  {op}")

    return {
        "A": Aname, "B": Bname,
        "wA": wA, "wB": wB, "games": games,
        "wrA": wr, "ci_low": lo, "ci_high": hi,
        "time_total_pair": elapsed_pair,
        "time_A_pair": time_used_pair[Aname],
        "time_B_pair": time_used_pair[Bname],
        "tpm_A": tpmA, "tpm_B": tpmB,
        "missA": miss_per100_A, "missB": miss_per100_B,
        "gaveA": gave_per100_A, "gaveB": gave_per100_B,
        "outcomes": outcomes_bin,  # pour Glicko et bootstrap global
    }


# ============= Agrégation & rapports finaux =============
def aggregate(results: List[Dict]):
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Glicko
    glicko_r: Dict[str, float]  = {}
    glicko_rd: Dict[str, float] = {}
    if USE_GLICKO:
        names = set()
        for r in results:
            names.add(r["A"]); names.add(r["B"])
        for n in names:
            glicko_r[n]  = GLICKO_START_RATING
            glicko_rd[n] = GLICKO_START_RD

    for r in results:
        A, B = r["A"], r["B"]
        totals[A]["wins"]  += r["wA"]; totals[A]["losses"] += r["wB"]
        totals[B]["wins"]  += r["wB"]; totals[B]["losses"] += r["wA"]
        g = max(1, r["games"])
        totals[A]["games"] += g; totals[B]["games"] += g
        totals[A]["time_used"] += r["time_A_pair"]; totals[B]["time_used"] += r["time_B_pair"]
        totals[A]["tpm_sum"]  += r["tpm_A"] * g;    totals[B]["tpm_sum"]  += r["tpm_B"] * g
        totals[A]["miss_sum"] += r["missA"] * g;    totals[B]["miss_sum"] += r["missB"] * g
        totals[A]["gave_sum"] += r["gaveA"] * g;    totals[B]["gave_sum"] += r["gaveB"] * g

        if USE_GLICKO and r.get("outcomes"):
            for sA in r["outcomes"]:
                rA, rdA = glicko_r[A], glicko_rd[A]
                rB, rdB = glicko_r[B], glicko_rd[B]
                rA2, rdA2, rB2, rdB2 = glicko_update_once(rA, rdA, rB, rdB, float(sA))
                glicko_r[A], glicko_rd[A] = rA2, rdA2
                glicko_r[B], glicko_rd[B] = rB2, rdB2

    print("\n=== Résumé agrégé par IA (winrate cumulé) ===")
    for name, t in totals.items():
        w, l = int(t["wins"]), int(t["losses"])
        g = w + l
        wr = w/g if g else 0.0
        lo, hi = wilson_interval(w, g) if g else (0.0, 0.0)
        tpm  = (t["tpm_sum"]/t["games"]) if t["games"] else 0.0
        miss = (t["miss_sum"]/t["games"]) if t["games"] else 0.0
        gave = (t["gave_sum"]/t["games"]) if t["games"] else 0.0
        print(f"{name:28s} {w:4d}-{l:<4d}  WR={wr:.3f}  (95% CI {lo:.3f}-{hi:.3f})  | "
              f"t/c={tpm:.3f}s  |  miss/100={miss:.2f}  |  gave/100={gave:.2f}")

    # Elo approx (un passage)
    elo = {name: 1000.0 for name in totals.keys()}
    for r in results:
        A, B = r["A"], r["B"]
        games = max(1, r["games"])
        scoreA = r["wA"] / games
        elo[A], elo[B] = elo_update(elo[A], elo[B], scoreA, k=24.0)

    print("\n=== Classement ELO (approx.) ===")
    for name, rating in sorted(elo.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:28s} ELO={rating:.1f}")

    if USE_GLICKO:
        print("\n=== Classement Glicko-1 (r ± RD) ===")
        for name in sorted(glicko_r.keys(), key=lambda n: glicko_r[n], reverse=True):
            print(f"{name:28s} r={glicko_r[name]:7.1f} ± {glicko_rd[name]:.1f}")

    # Matrice A×B
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


# ============= Enregistrement de la configuration de l’exécution =============
def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def dump_run_config(run_dir: pathlib.Path, ais_kept: List[Dict], ais_excluded: List[str], import_errors: List[str]):
    cfg = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_dir": str(run_dir),
        "config": {
            "SEEDS": SEEDS,
            "GAMES_PER_SEED": GAMES_PER_SEED,
            "FILTER_MUTE": FILTER_MUTE,
            "PROBE_TIMEOUT": PROBE_TIMEOUT,
            "TIME_PER_MOVE": TIME_PER_MOVE,
            "USE_SPRT": USE_SPRT,
            "SPRT_P0": SPRT_P0, "SPRT_P1": SPRT_P1, "SPRT_ALPHA": SPRT_ALPHA, "SPRT_BETA": SPRT_BETA,
            "BOOTSTRAP_N": BOOTSTRAP_N,
            "USE_GLICKO": USE_GLICKO,
            "GLICKO_START_RATING": GLICKO_START_RATING,
            "GLICKO_START_RD": GLICKO_START_RD,
            "SHOW_PER_GAME_LINES": SHOW_PER_GAME_LINES,
            "FIRST_PLIES_TO_LOG": FIRST_PLIES_TO_LOG,
            "TOP_OPENINGS_TO_SHOW": TOP_OPENINGS_TO_SHOW,
        },
        "ais_discovered": [
            {"name": a["name"], "module": a.get("module", ""), "cls": getattr(a["cls"], "__name__", "QuantikAI")}
            for a in ais_kept
        ],
        "ais_excluded_probe_mute": ais_excluded,
        "import_errors": import_errors,
        # hachages rapides pour la reproductibilité (module)
        "ai_module_hashes": {
            a["name"]: short_hash(a.get("module","")) for a in ais_kept
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


# ===================== Main ======================
def main():
    run_dir = _ensure_run_folder()
    games_fp = moves_fp = jsonl_fp = None
    csv_games_writer = csv_moves_writer = None
    try:
        # fichiers de sortie
        games_fp, moves_fp, jsonl_fp, csv_games_writer, csv_moves_writer = _open_outputs(run_dir)

        # découverte des IA
        ais, errors = discover_ais()
        ais_kept, mute_excluded = filter_mute_ais(ais, do_filter=FILTER_MUTE)
        print(f"IAs découvertes ({len(ais_kept)}): {[a['name'] for a in ais_kept]}")
        if errors:
            print(f"⚠️  Erreurs d’import: {errors}")
        if FILTER_MUTE and mute_excluded:
            print(f"⚠️  Exclues (muettes au probe): {mute_excluded}")

        # enregistre la config de l’exécution (inclut la liste des IA retenues)
        dump_run_config(run_dir, ais_kept, mute_excluded, errors)

        # round-robin
        results = []
        for i in range(len(ais_kept)):
            for j in range(i+1, len(ais_kept)):
                r = run_pair_diagnostic(
                    ais_kept[i], ais_kept[j],
                    seeds=SEEDS, games_per_seed=GAMES_PER_SEED,
                    csv_games_writer=csv_games_writer,
                    csv_moves_writer=csv_moves_writer,
                    jsonl_fp=jsonl_fp
                )
                results.append(r)

    finally:
        _close_outputs(games_fp, moves_fp, jsonl_fp)

    # agrégation finale
    aggregate(results)


if __name__ == "__main__":
    main()

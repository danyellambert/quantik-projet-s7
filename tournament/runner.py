# tournament/runner.py
# =====================================================================================
# Tournoi IA vs IA (diagnostic complet)
# - Découvre les IAs sous ai_players/*/algorithme.py (classe QuantikAI, const AI_NAME)
# - Filtre les IAs “muettes” via un probe (timeout configurable)
# - Pour chaque paire (A,B) et chaque seed :
#     * 2*GAMES_PER_SEED parties : A commence et B commence
# - Collecte & affiche :
#     • WR (IC 95%), StartsWon/RepliesWon
#     • Temps total par IA dans le duel + moyenne par coup
#     • Longueur moyenne (plies), répartition des ouvertures
#     • Raisons de fin : victory / no_move / illegal / exception
#     • Stats par coup : temps, #coups légaux, forcé ?, gain immédiat dispo ? blunders ?
# - Exporte :
#     • CSV par partie (games.csv)
#     • CSV par coup (moves.csv)
#     • JSONL par coup (moves.jsonl)
#
# MOD (2025-08) – corrections clés :
#   • other() corrigée (retournait toujours PLAYER1 dans une version antérieure).
#   • Moyenne de branche (nb coups légaux) désormais par TOUR (turns), pas par partie.
#   • Signature _write_game unifiée et appels corrigés (plus d’argument manquant).
#   • Logs de console uniformisés **en A/B**, jamais en PLAYER1/2.
#   • Graine unique par partie (base = seed*10000 + 2*g ; Astart=+1, Bstart=+2).
# =====================================================================================

from __future__ import annotations
import importlib, pkgutil, pathlib, random, time, math, threading, csv, json
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter

from core.types import Shape, Player, Piece
from core.rules import QuantikBoard

# ===================== Config ======================
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
GAMES_PER_SEED = 2
FILTER_MUTE = True
PROBE_TIMEOUT = 2.0

# Verbosité
SHOW_PER_GAME_LINES = True
SHOW_PER_MOVE_LINES = False  # (les détails de coup vont au CSV/JSONL)
FIRST_PLIES_TO_LOG = 4
TOP_OPENINGS_TO_SHOW = 8

# Export
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "tournament" / "out"
WRITE_CSV = True
WRITE_JSONL = True
# ===================================================


# ============= Helpers plateau / compat =============
def raw_board(board: QuantikBoard):
    """Retourne la grille interne (compat pour versions avec .raw())."""
    return board.raw() if hasattr(board, "raw") else board.board

def empty_position():
    """Position de départ standard (plateau vide + 2 pièces de chaque forme par joueur)."""
    b = QuantikBoard()
    pieces = {
        Player.PLAYER1: {s: 2 for s in Shape},
        Player.PLAYER2: {s: 2 for s in Shape},
    }
    return b, pieces


# ====== Règles (diagnostic : générer l’ensemble des coups légaux) ======
N = 4
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]
CENTER = {(1,1), (1,2), (2,1), (2,2)}

def other(p: Player) -> Player:
    """Joueur adverse (fixé)."""
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board_grid, row: int, col: int, shape: Shape, me: Player) -> bool:
    """
    Légalité Quantik (version utilisée par tes IAs) :
    Interdit si **l’adversaire** a déjà posé la même forme dans la ligne/colonne/zone.
    """
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
    """Liste (r,c,shape) des coups légaux pour `me` selon les stocks."""
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
    """Chaîne courte pour les 1ers coups (utile pour le tri des ouvertures).”
    """
    return " | ".join(f"{r},{c},{sh.name}" for (r,c,sh) in open_moves)


# ============= Découverte des IAs =============
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


# ============= Probe “IA muette” =============
def probe_ai_speaks(ai_cls, timeout: float = PROBE_TIMEOUT) -> bool:
    """
    Lance l’IA sur la position initiale : si aucune réponse légale n’arrive dans
    le délai, on la filtre pour le tournoi (évite les deadlocks).
    """
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
            # undo
            b.board[r][c] = None
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


# ============= Statistiques de base =============
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

def elo_update(ra: float, rb: float, sa: float, k: float = 24.0) -> Tuple[float,float]:
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    return ra + k * (sa - ea), rb + k * ((1 - sa) - eb)


# ============= Moteur d’une partie (instrumentation coup-par-coup) =============
def play_one_game(aiA_cls, aiB_cls, starter: Player, seed: Optional[int],
                  game_uid: str,
                  csv_games_writer, csv_moves_writer, jsonl_fp) -> Tuple[Player, Dict[str,Any]]:
    """
    Conventions :
      • IA A = Player1, IA B = Player2
      • `starter` ∈ {Player1, Player2} indique qui joue en premier
    Retourne : (winner, log_dict)
    Écrit les logs CSV/JSONL si activés.
    """
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

    current = starter
    A_started = 1 if starter == Player.PLAYER1 else 0

    # accumulateurs
    time_used = {Player.PLAYER1: 0.0, Player.PLAYER2: 0.0}
    moves_played = {Player.PLAYER1: 0,   Player.PLAYER2: 0}
    opening_moves: List[Tuple[int,int,Shape]] = []
    end_reason = "victory"

    # stats de diagnostic
    legal_branch_sum = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    forced_count = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    had_win_but_missed = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    gave_opp_immediate_win = {Player.PLAYER1: 0, Player.PLAYER2: 0}

    ply_index = 0
    winner = None

    try:
        while True:
            ai = aiA if current == Player.PLAYER1 else aiB

            # avant le coup : calcule les légaux et gains immédiats dispos
            legal_moves = generate_valid_moves(grid, current, pieces[current])
            legal_count = len(legal_moves)
            legal_branch_sum[current] += legal_count

            if legal_count == 0:
                winner = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1
                end_reason = "no_move"
                break
            if legal_count == 1:
                forced_count[current] += 1

            # victoire immédiate disponible ?
            had_immediate_win = 0
            for (r0, c0, sh0) in legal_moves:
                ok = board.place_piece(r0, c0, Piece(sh0, current))
                if ok and board.check_victory():
                    had_immediate_win = 1
                board.board[r0][c0] = None
                if had_immediate_win:
                    break

            # appel IA
            t0 = time.perf_counter()
            move = ai.get_move(grid, pieces)
            dt = time.perf_counter() - t0
            time_used[current] += dt

            # validation
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
                winner = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1
                end_reason = "illegal"
                _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                            current, None, dt, legal_count, had_immediate_win,
                            chose_win=0, gave_opp_win=0, center=0, note="illegal_or_none")
                break

            # applique le coup
            r, c, sh = move
            ok_place = board.place_piece(r, c, Piece(sh, current))
            if not ok_place:
                winner = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1
                end_reason = "illegal"
                _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                            current, (r,c,sh), dt, legal_count, had_immediate_win,
                            chose_win=0, gave_opp_win=0, center=int((r,c) in CENTER),
                            note="place_piece_failed")
                break

            # compteurs
            pieces[current][sh] -= 1
            moves_played[current] += 1
            if len(opening_moves) < FIRST_PLIES_TO_LOG:
                opening_moves.append((r, c, sh))

            # si victoire immédiate dispo mais non choisie → blunder
            chose_win_now = 0
            if had_immediate_win:
                if board.check_victory():
                    chose_win_now = 1
                else:
                    had_win_but_missed[current] += 1

            # a-t-on donné une victoire immédiate à l’adversaire ?
            gave_win = 0
            if not board.check_victory():
                opp = other(current)
                opp_legal = generate_valid_moves(grid, opp, pieces[opp])
                for (rr, cc, ssh) in opp_legal:
                    ok2 = board.place_piece(rr, cc, Piece(ssh, opp))
                    if ok2 and board.check_victory():
                        gave_win = 1
                    board.board[rr][cc] = None
                    if gave_win:
                        break
                if gave_win:
                    gave_opp_immediate_win[current] += 1

            # log du coup
            _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                        current, (r,c,sh), dt, legal_count, had_immediate_win,
                        chose_win=chose_win_now, gave_opp_win=gave_win,
                        center=int((r,c) in CENTER),
                        note="ok")

            # victoire ?
            if board.check_victory():
                winner = current
                end_reason = "victory"
                break

            # coup suivant
            ply_index += 1
            current = other(current)

    except Exception as e:
        # toute exception côté IA → l’autre gagne
        winner = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1
        end_reason = "exception"
        _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                    current, None, 0.0, 0, 0, 0, 0, 0, note=f"exception:{type(e).__name__}")

    # métriques finales
    A_won_start = 1 if (winner == Player.PLAYER1 and A_started == 1) else 0
    A_won_reply = 1 if (winner == Player.PLAYER1 and A_started == 0) else 0
    plies_total = moves_played[Player.PLAYER1] + moves_played[Player.PLAYER2]
    opening_str = opening_key(opening_moves)

    # ligne games.csv
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
        # nb de tours (pour moyenne de branche correcte)
        "turnsA": moves_played[Player.PLAYER1],
        "turnsB": moves_played[Player.PLAYER2],
    }


# ===== Écriture des logs (CSV / JSONL) =====
def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games_csv = OUT_DIR / "games.csv"
    moves_csv = OUT_DIR / "moves.csv"
    moves_jsonl = OUT_DIR / "moves.jsonl"
    games_fp = open(games_csv, "w", newline="", encoding="utf-8") if WRITE_CSV else None
    moves_fp = open(moves_csv, "w", newline="", encoding="utf-8") if WRITE_CSV else None
    jsonl_fp = open(moves_jsonl, "w", encoding="utf-8") if WRITE_JSONL else None

    csv_games_writer = None
    csv_moves_writer = None
    if games_fp:
        csv_games_writer = csv.writer(games_fp)
        csv_games_writer.writerow([
            "game_uid","pair","starter","winner","end_reason","plies_total","opening",
            "time_p1","time_p2","moves_p1","moves_p2",
            "branch_sum_p1","branch_sum_p2","forced_p1","forced_p2",
            "missed_win_p1","missed_win_p2","gave_opp_win_p1","gave_opp_win_p2"
        ])
    if moves_fp:
        csv_moves_writer = csv.writer(moves_fp)
        csv_moves_writer.writerow([
            "game_uid","ply","player","r","c","shape","time_sec","legal_count",
            "had_immediate_win","chose_immediate_win","gave_opp_immediate_win","is_center","note"
        ])
    return games_fp, moves_fp, jsonl_fp, csv_games_writer, csv_moves_writer

def _close_out(games_fp, moves_fp, jsonl_fp):
    for fp in (games_fp, moves_fp, jsonl_fp):
        if fp:
            fp.close()

def _write_game(csv_games_writer, game_uid, starter, winner, end_reason,
                plies_total, opening_str, time_used, moves_played,
                branch_sum, forced_count, missed_win, gave_opp):
    """Ligne récap par partie (CSV)."""
    if not csv_games_writer:
        return
    csv_games_writer.writerow([
        game_uid,
        "",  # 'pair' (champ non utilisé ici)
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
    """Ligne par coup (CSV + JSONL)."""
    r = c = None
    sh_name = None
    if mv:
        r, c, sh = mv
        sh_name = sh.name
    if csv_moves_writer:
        csv_moves_writer.writerow([
            game_uid, ply_index, ("A" if player == Player.PLAYER1 else "B") if player else None,
            r, c, sh_name, f"{tsec:.6f}", legal_count,
            had_win, chose_win, gave_opp_win, center, note
        ])
    if jsonl_fp:
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


# ============= Duel multi-seeds (avec diagnostics) =============
def run_pair_diagnostic(iaA: Dict, iaB: Dict, seeds: List[int], games_per_seed: int = 1,
                        csv_games_writer=None, csv_moves_writer=None, jsonl_fp=None):
    Aname, Bname = iaA["name"], iaB["name"]

    wA = wB = 0
    A_starts_won = 0
    A_replies_won = 0
    t0_pair = time.time()

    time_used_pair = {Aname: 0.0, Bname: 0.0}
    game_lengths: List[int] = []
    end_reasons = Counter()
    opening_counter = Counter()

    # agrégats par côté
    branch_sum_A = branch_sum_B = 0
    forced_A = forced_B = 0
    missed_win_A = missed_win_B = 0
    gave_opp_A = gave_opp_B = 0
    turnsA_total = 0
    turnsB_total = 0

    for g in range(games_per_seed):
        for seed in seeds:
            # graine unique par PARTIE
            base = seed * 10_000 + 2 * g

            # A commence
            game_uid = f"{Aname}__vs__{Bname}__seed{base+1}__Astart"
            winner, log = play_one_game(iaA["cls"], iaB["cls"], starter=Player.PLAYER1,
                                        seed=base+1, game_uid=game_uid,
                                        csv_games_writer=csv_games_writer,
                                        csv_moves_writer=csv_moves_writer,
                                        jsonl_fp=jsonl_fp)
            time_used_pair[Aname] += log["timeA"]
            time_used_pair[Bname] += log["timeB"]
            game_lengths.append(log["plies_total"])
            end_reasons[log["end_reason"]] += 1
            opening_counter[log["opening"]] += 1
            branch_sum_A += log["branchA_sum"]; branch_sum_B += log["branchB_sum"]
            forced_A += log["forcedA"]; forced_B += log["forcedB"]
            missed_win_A += log["missed_winA"]; missed_win_B += log["missed_winB"]
            gave_opp_A += log["gave_winA"]; gave_opp_B += log["gave_winB"]
            turnsA_total += log["turnsA"]; turnsB_total += log["turnsB"]

            if winner == Player.PLAYER1:  # A gagne
                wA += 1; A_starts_won += log["A_won_start"]; A_replies_won += log["A_won_reply"]
            else:
                wB += 1

            if SHOW_PER_GAME_LINES:
                print(f"  [{Aname} vs {Bname} | seed={base+1} | starter=A] "
                      f"winner={'A' if winner==Player.PLAYER1 else 'B'} "
                      f"len={log['plies_total']}, end={log['end_reason']}, "
                      f"tA={log['timeA']:.2f}s, tB={log['timeB']:.2f}s, opening={log['opening']}")

            # B commence
            game_uid = f"{Aname}__vs__{Bname}__seed{base+2}__Bstart"
            winner, log = play_one_game(iaA["cls"], iaB["cls"], starter=Player.PLAYER2,
                                        seed=base+2, game_uid=game_uid,
                                        csv_games_writer=csv_games_writer,
                                        csv_moves_writer=csv_moves_writer,
                                        jsonl_fp=jsonl_fp)
            time_used_pair[Aname] += log["timeA"]
            time_used_pair[Bname] += log["timeB"]
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

    elapsed_pair = time.time() - t0_pair
    games = wA + wB
    wr = wA / games if games else 0.0
    lo, hi = wilson_interval(wA, games)

    avg_len = sum(game_lengths)/len(game_lengths) if game_lengths else 0.0

    # moyenne de branche par TOUR (et non par partie)
    avg_branch_A = (branch_sum_A / max(1, turnsA_total))
    avg_branch_B = (branch_sum_B / max(1, turnsB_total))

    print(f"{Aname} vs {Bname} -> {wA}-{wB} sur {games} | "
          f"WR(A)={wr:.3f} (95% CI: {lo:.3f}-{hi:.3f}) | "
          f"StartsWon(A)={A_starts_won}, RepliesWon(A)={A_replies_won} | "
          f"{elapsed_pair:.1f}s total | "
          f"TimeUsed(A)={time_used_pair[Aname]:.1f}s, TimeUsed(B)={time_used_pair[Bname]:.1f}s | "
          f"AvgLen={avg_len:.2f} plies | "
          f"AvgBranch/turn: A={avg_branch_A:.2f}, B={avg_branch_B:.2f}")

    print(f"  End reasons: {dict(end_reasons)}")
    print(f"  Forced moves: A={forced_A}, B={forced_B}")
    print(f"  Missed immediate wins: A={missed_win_A}, B={missed_win_B}")
    print(f"  Gave opponent immediate win: A={gave_opp_A}, B={gave_opp_B}")

    if opening_counter:
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
        "avg_len": avg_len,
    }


# ============= Agrégation / rapports finaux =============
def aggregate(results: List[Dict]):
    totals: Dict[str, Dict[str, float]] = {}
    for r in results:
        for side in ("A", "B"):
            name = r[side]
            if name not in totals:
                totals[name] = {"wins":0, "losses":0, "time_used":0.0}
        totals[r["A"]]["wins"]  += r["wA"]
        totals[r["A"]]["losses"]+= r["wB"]
        totals[r["B"]]["wins"]  += r["wB"]
        totals[r["B"]]["losses"]+= r["wA"]
        totals[r["A"]]["time_used"] += r["time_A_pair"]
        totals[r["B"]]["time_used"] += r["time_B_pair"]

    print("\n=== Résumé agrégé par IA (winrate cumulé) ===")
    lines = []
    for name, t in totals.items():
        w, l = int(t["wins"]), int(t["losses"])
        g = w + l
        wr = w/g if g else 0.0
        lo, hi = wilson_interval(w, g) if g else (0.0, 0.0)
        lines.append((name, w, l, wr, lo, hi, t["time_used"]))
    lines.sort(key=lambda x: x[3], reverse=True)
    for (name, w, l, wr, lo, hi, tu) in lines:
        print(f"{name:28s} {w:4d}-{l:<4d}  WR={wr:.3f}  (95% CI {lo:.3f}-{hi:.3f})  | TimeUsed={tu:.1f}s")

    # Elo approximatif (un seul passage)
    elo = {name: 1000.0 for name in totals.keys()}
    for r in results:
        A, B = r["A"], r["B"]
        wA, wB = r["wA"], r["wB"]
        games = max(1, wA + wB)
        scoreA = wA / games
        elo[A], elo[B] = elo_update(elo[A], elo[B], scoreA, k=24.0)

    print("\n=== Classement ELO (approx.) ===")
    for name, rating in sorted(elo.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:28s} ELO={rating:.1f}")

    # Matrice A × B
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


# ===================== Main ======================
def main():
    # préparation sortie
    if WRITE_CSV or WRITE_JSONL:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    games_fp, moves_fp, jsonl_fp, csv_games_writer, csv_moves_writer = _ensure_out()

    # découverte IAs
    ais, errors = discover_ais()
    ais, mute_excluded = filter_mute_ais(ais, do_filter=FILTER_MUTE)
    print(f"IAs découvertes ({len(ais)}): {[a['name'] for a in ais]}")
    if errors:
        print(f"⚠️  Erreurs d’import: {errors}")
    if FILTER_MUTE and mute_excluded:
        print(f"⚠️  Exclues (muettes au probe): {mute_excluded}")

    # duels
    results = []
    for i in range(len(ais)):
        for j in range(i+1, len(ais)):
            r = run_pair_diagnostic(
                ais[i], ais[j],
                seeds=SEEDS, games_per_seed=GAMES_PER_SEED,
                csv_games_writer=csv_games_writer,
                csv_moves_writer=csv_moves_writer,
                jsonl_fp=jsonl_fp
            )
            results.append(r)

    # fermeture fichiers
    _close_out(games_fp, moves_fp, jsonl_fp)

    # agrégat final
    aggregate(results)


if __name__ == "__main__":
    main()
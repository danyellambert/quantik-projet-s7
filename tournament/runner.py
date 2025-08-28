# tournament/runner.py
# =====================================================================================
# Tournoi IA vs IA (diagnostic complet)
# - Découvre les IA dans ai_players/*/algorithme.py (classe QuantikAI, constante AI_NAME)
# - Filtre les IA « muettes » via un probe (timeout configurable)
# - Pour chaque paire (A,B) et chaque seed :
#     * 2*GAMES_PER_SEED parties : A commence et B commence
# - Collecte et affiche :
#     • WR (IC 95%), StartsWon/RepliesWon
#     • Temps total par IA dans la paire + moyenne par partie + moyenne par coup
#     • Longueur moyenne de la partie (plies), distribution des ouvertures
#     • Raisons de fin : victory / no_move / illegal / exception
#     • Statistiques par coup : temps, #coups légaux, forcé ?, victoire immédiate dispo ? blunders ?
# - Exporte :
#     • CSV par partie (games.csv)
#     • CSV par coup (moves.csv)
#     • JSONL par coup (moves.jsonl)
# - À la fin : résumé agrégé par IA, Elo approx., matrice A×B
# =====================================================================================
from __future__ import annotations
import importlib, pkgutil, pathlib, random, time, math, threading, csv, json, os
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict

from core.types import Shape, Player, Piece
from core.rules import QuantikBoard

# ===================== Configuration ======================
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
GAMES_PER_SEED = 2
FILTER_MUTE = True
PROBE_TIMEOUT = 2.0

# Verbosité
SHOW_PER_GAME_LINES = True
SHOW_PER_MOVE_LINES = True
FIRST_PLIES_TO_LOG = 4
TOP_OPENINGS_TO_SHOW = 8

# Export
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "tournament" / "out"
WRITE_CSV = True
WRITE_JSONL = True
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

# ====== Règles (pour diagnostic : générer des coups légaux, etc.) ======
N = 4
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]
CENTER = {(1,1), (1,2), (2,1), (2,2)}

def other(p: Player) -> Player:
    # (retourne l’autre joueur)
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER1

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

def check_win_if_play(board_obj: QuantikBoard, r: int, c: int) -> bool:
    # Utilise la règle officielle de QuantikBoard
    return board_obj.check_victory()

def would_win_immediately(board_obj: QuantikBoard, grid, who: Player, counts, mv: Tuple[int,int,Shape]) -> bool:
    r, c, sh = mv
    ok = board_obj.place_piece(r, c, Piece(sh, who))
    if not ok:
        return False
    win = board_obj.check_victory()
    # annule
    board_obj.board[r][c] = None
    counts[who][sh] += 0  # (on ne touche pas au compteur dans l’appelant ; uniquement au plateau)
    return win

def immediate_wins_available(board_obj: QuantikBoard, grid, who: Player, counts) -> int:
    wins = 0
    for mv in generate_valid_moves(grid, who, counts[who]):
        r, c, sh = mv
        # applique
        ok = board_obj.place_piece(r, c, Piece(sh, who))
        if not ok:
            continue
        if board_obj.check_victory():
            wins += 1
        # annule
        board_obj.board[r][c] = None
    return wins

def opening_key(open_moves: List[Tuple[int,int,Shape]]) -> str:
    return " | ".join(f"{r},{c},{sh.name}" for (r,c,sh) in open_moves)

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

# ============= Probe « IA muette » =============
def probe_ai_speaks(ai_cls, timeout: float = PROBE_TIMEOUT) -> bool:
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

# ============= Moteur d’une partie (instrumenté au niveau du coup) =============
def play_one_game(aiA_cls, aiB_cls, starter: Player, seed: Optional[int],
                  game_uid: str,
                  csv_games_writer, csv_moves_writer, jsonl_fp) -> Tuple[Player, Dict[str,Any]]:
    """
    A = Player1 (IA A), B = Player2 (IA B)
    Renvoie le vainqueur et un dictionnaire avec des métriques agrégées de la partie.
    Écrit également des lignes dans les CSV/JSONL si activé.
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

    # statistiques diagnostiques
    legal_branch_sum = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    forced_count = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    had_win_but_missed = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    gave_opp_immediate_win = {Player.PLAYER1: 0, Player.PLAYER2: 0}

    ply_index = 0
    winner = None

    try:
        while True:
            ai = aiA if current == Player.PLAYER1 else aiB

            # avant le coup : compter les coups légaux et les victoires immédiates possibles
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
            for mv_can in legal_moves:
                r0, c0, sh0 = mv_can
                ok = board.place_piece(r0, c0, Piece(sh0, current))
                if ok and board.check_victory():
                    had_immediate_win = 1
                # annule
                board.board[r0][c0] = None
                if had_immediate_win:
                    break

            # appel à l’IA
            t0 = time.perf_counter()
            move = ai.get_move(grid, pieces)
            dt = time.perf_counter() - t0
            time_used[current] += dt

            # validation basique
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
                # journal du coup illégal
                _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                            current, None, dt, legal_count, had_immediate_win,
                            chose_win=0, gave_opp_win=0, center=0, note="illegal_or_none")
                break

            # applique le coup
            r, c, sh = move
            board_ok = board.place_piece(r, c, Piece(sh, current))
            if not board_ok:
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

            # s’il y avait une victoire immédiate et que le coup choisi n’était pas gagnant → blunder
            chose_win_now = 0
            if had_immediate_win:
                # vérifier si le coup choisi gagne immédiatement
                if board.check_victory():
                    chose_win_now = 1
                else:
                    had_win_but_missed[current] += 1

            # a donné une victoire immédiate à l’adversaire ?
            gave_win = 0
            if not board.check_victory():
                opp = Player.PLAYER1 if current == Player.PLAYER2 else Player.PLAYER2
                opp_legal = generate_valid_moves(grid, opp, pieces[opp])
                for (rr, cc, ssh) in opp_legal:
                    ok2 = board.place_piece(rr, cc, Piece(ssh, opp))
                    if ok2 and board.check_victory():
                        gave_win = 1
                    # annule
                    board.board[rr][cc] = None
                    if gave_win:
                        break
                if gave_win:
                    gave_opp_immediate_win[current] += 1

            # journal du coup
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

            # prochain tour
            ply_index += 1
            current = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1

    except Exception as e:
        # toute exception de l’IA met fin à la partie
        winner = Player.PLAYER2 if current == Player.PLAYER1 else Player.PLAYER1
        end_reason = "exception"
        _write_move(csv_moves_writer, jsonl_fp, game_uid, ply_index,
                    current, None, 0.0, 0, 0, 0, 0, 0, note=f"exception:{type(e).__name__}")

    # métriques finales de la partie
    A_won_start = 1 if (winner == Player.PLAYER1 and A_started == 1) else 0
    A_won_reply = 1 if (winner == Player.PLAYER1 and A_started == 0) else 0
    plies_total = moves_played[Player.PLAYER1] + moves_played[Player.PLAYER2]
    opening_str = opening_key(opening_moves)

    # écrit la ligne du CSV de parties
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
    }

# ===== Écriture des journaux (CSV/JSONL) =====
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
    if not csv_games_writer:
        return
    csv_games_writer.writerow([
        game_uid,
        "",  # champ de la paire rempli au niveau du pair si souhaité (laissé vide ici)
        starter.name, winner.name, end_reason, plies_total, opening_str,
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
    if csv_moves_writer:
        csv_moves_writer.writerow([
            game_uid, ply_index, (player.name if player else None),
            r, c, sh_name, f"{tsec:.6f}", legal_count,
            had_win, chose_win, gave_opp_win, center, note
        ])
    if jsonl_fp:
        obj = {
            "game_uid": game_uid,
            "ply": ply_index,
            "player": (player.name if player else None),
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

    # agrégats par côté (ramification et blunders)
    branch_sum_A = branch_sum_B = 0
    forced_A = forced_B = 0
    missed_win_A = missed_win_B = 0
    gave_opp_A = gave_opp_B = 0

    game_idx = 0

    for seed in seeds:
        for g in range(games_per_seed):
            base = seed * 10_000 + g

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

            if winner == Player.PLAYER1:
                wA += 1; A_starts_won += log["A_won_start"]; A_replies_won += log["A_won_reply"]
            else:
                wB += 1

            if SHOW_PER_GAME_LINES:
                print(f"  [{game_uid}] winner={winner.name} len={log['plies_total']}, "
                      f"end={log['end_reason']}, tA={log['timeA']:.2f}s, tB={log['timeB']:.2f}s, "
                      f"opening={log['opening']}")

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

            if winner == Player.PLAYER1:
                wA += 1; A_starts_won += log["A_won_start"]; A_replies_won += log["A_won_reply"]
            else:
                wB += 1

            if SHOW_PER_GAME_LINES:
                print(f"  [{game_uid}] winner={winner.name} len={log['plies_total']}, "
                      f"end={log['end_reason']}, tA={log['timeA']:.2f}s, tB={log['timeB']:.2f}s, "
                      f"opening={log['opening']}")

            game_idx += 2

    elapsed_pair = time.time() - t0_pair
    games = wA + wB
    wr = wA / games if games else 0.0
    lo, hi = wilson_interval(wA, games)

    avg_len = sum(game_lengths)/len(game_lengths) if game_lengths else 0.0
    avg_branch_A = (branch_sum_A / max(1, sum(1 for x in game_lengths)))  # somme par partie ; interprétation : moyenne de (#légaux par tour d’A)
    avg_branch_B = (branch_sum_B / max(1, sum(1 for x in game_lengths)))

    print(f"{Aname} vs {Bname} -> {wA}-{wB} sur {games} | "
          f"WR(A)={wr:.3f} (95% CI: {lo:.3f}-{hi:.3f}) | "
          f"StartsWon(A)={A_starts_won}, RepliesWon(A)={A_replies_won} | "
          f"{elapsed_pair:.1f}s total | "
          f"TimeUsed(A)={time_used_pair[Aname]:.1f}s, TimeUsed(B)={time_used_pair[Bname]:.1f}s | "
          f"AvgLen={avg_len:.2f} plies")

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

    # Elo approximatif
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
    # prépare la sortie
    if WRITE_CSV or WRITE_JSONL:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    games_fp, moves_fp, jsonl_fp, csv_games_writer, csv_moves_writer = _ensure_out()

    # découvre les IA
    ais, errors = discover_ais()
    ais, mute_excluded = filter_mute_ais(ais, do_filter=FILTER_MUTE)
    print(f"IAs découvertes ({len(ais)}): {[a['name'] for a in ais]}")
    if errors:
        print(f"⚠️  Erreurs d’import: {errors}")
    if FILTER_MUTE and mute_excluded:
        print(f"⚠️  Exclues (muettes au probe): {mute_excluded}")

    # exécute les paires
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

    # ferme les fichiers
    _close_out(games_fp, moves_fp, jsonl_fp)

    # agrégats
    aggregate(results)

if __name__ == "__main__":
    main()

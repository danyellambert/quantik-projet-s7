# ai_players/alphabeta_plus/algorithme.py
# ================================================================
# IA Quantik “AlphaBeta++” (améliorée)
# - Minimax avec élagage alpha–bêta
# - Iterative deepening + fenêtres d’aspiration
# - Ordonnancement fort : VP/TT, victoire, bloc, killers, historique, centre
# - TT stable : (plateau, côté, comptes)
# - Heuristique tactico-positionnelle (menaces immédiates, mobilité, centre)
# - Extension de menace (quiescence légère) lorsqu’il y a mat-en-1
# - Sentinelles finies ; pas de mutation d’état global
# ================================================================

from __future__ import annotations
import time, math, random
from typing import Optional, Tuple, List, Dict, Any

from core.types import Shape, Player, Piece

AI_NAME = "AlphaBeta++"
AI_AUTHOR = ""

# --- Constantes du plateau ---
N = 4
ALL_CELLS = [(r, c) for r in range(N) for c in range(N)]
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]
LINES: List[List[Tuple[int,int]]] = []
for r in range(N): LINES.append([(r, c) for c in range(N)])
for c in range(N): LINES.append([(r, c) for r in range(N)])
LINES.extend(ZONES)

CENTER_CELLS = {(1,1), (1,2), (2,1), (2,2)}

# --- Sentinelles finies ---
WIN_SCORE   = 50_000.0
LOSS_SCORE  = -WIN_SCORE
ALPHA_INIT  = -1_000_000.0
BETA_INIT   = +1_000_000.0

# --- Poids heuristiques ---
W_THREAT = 5000.0   # menaces immédiates (très élevé pour la tactique)
W_MOB    = 2.0
W_CENTER = 1.0

# =========================
# Règles / utilitaires Quantik
# =========================
def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board: List[List[Optional[Piece]]],
                  row: int, col: int,
                  shape: Shape, me: Player) -> bool:
    """
    Coup légal si la case est vide ET que l’ADVERSAIRE n’a pas la même forme
    sur la même ligne/colonne/zone (règle de Quantik utilisée dans le projet).
    """
    if board[row][col] is not None: return False
    # ligne
    for c in range(N):
        p = board[row][c]
        if p is not None and p.shape == shape and p.player != me: return False
    # colonne
    for r in range(N):
        p = board[r][col]
        if p is not None and p.shape == shape and p.player != me: return False
    # zone
    z = zone_index(row, col)
    for (rr, cc) in ZONES[z]:
        p = board[rr][cc]
        if p is not None and p.shape == shape and p.player != me: return False
    return True

def forms_all_different(pieces: List[Optional[Piece]]) -> bool:
    if any(p is None for p in pieces): return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board: List[List[Optional[Piece]]], row: int, col: int) -> bool:
    if forms_all_different([board[row][c] for c in range(N)]): return True
    if forms_all_different([board[r][col] for r in range(N)]): return True
    z = zone_index(row, col)
    if forms_all_different([board[r][c] for (r, c) in ZONES[z]]): return True
    return False

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape, int]) -> List[Tuple[int,int,Shape]]:
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0: continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves

# =========================
# Heuristique : tactique + position
# =========================
def count_immediate_wins(board, counts, who: Player) -> int:
    """Combien de mats-en-1 `who` possède ici."""
    n = 0
    moves = generate_valid_moves(board, who, counts[who])
    for (r, c, sh) in moves:
        board[r][c] = Piece(sh, who)
        if check_victory_after(board, r, c): n += 1
        board[r][c] = None
    return n

def mobility(board, who: Player, counts) -> int:
    return len(generate_valid_moves(board, who, counts[who]))

def center_control(board, me: Player) -> int:
    sc = 0
    for (r, c) in CENTER_CELLS:
        p = board[r][c]
        if p is None: continue
        sc += 1 if p.player == me else -1
    return sc

def heuristic(board, me: Player, counts) -> float:
    opp = Player.PLAYER1 if me == Player.PLAYER2 else Player.PLAYER2
    my_th  = count_immediate_wins(board, counts, me)
    opp_th = count_immediate_wins(board, counts, opp)
    # Remarque : les menaces ont un poids très élevé ; les autres termes affinent
    mob    = mobility(board, me, counts) - mobility(board, opp, counts)
    center = center_control(board, me)
    return W_THREAT*(my_th - opp_th) + W_MOB*mob + W_CENTER*center

# =========================
# Recherche Alpha–Bêta avec approfondissement
# =========================
class Searcher:
    def __init__(self, me: Player, time_limit: float = 1.35):
        self.me = me
        self.opp = Player.PLAYER1 if me == Player.PLAYER2 else Player.PLAYER2
        self.time_limit = time_limit
        self.t0 = 0.0
        # TT : clé -> (profondeur, score, meilleur_coup)
        self.tt: Dict[Any, Tuple[int, float, Optional[Tuple[int,int,Shape]]]] = {}
        # heuristique d’historique et coups tueurs
        self.history: Dict[Tuple[int,int,Shape], int] = {}
        self.killers: Dict[int, List[Tuple[int,int,Shape]]] = {}

    # ---- temps ----
    def time_up(self) -> bool:
        return (time.time() - self.t0) >= self.time_limit

    # ---- clé TT stable (inclut les comptes) ----
    def counts_key(self, counts: Dict[Player, Dict[Shape,int]]) -> Tuple[Tuple[int,...], Tuple[int,...]]:
        p1 = tuple(counts[Player.PLAYER1].get(sh, 0) for sh in Shape)
        p2 = tuple(counts[Player.PLAYER2].get(sh, 0) for sh in Shape)
        return (p1, p2)

    def board_key(self, board, side: Player, counts) -> Tuple:
        cells = []
        for r in range(N):
            for c in range(N):
                p = board[r][c]
                cells.append(None if p is None else (p.shape.value, p.player.value))
        return (tuple(cells), side.value, self.counts_key(counts))

    # ---- outils tactiques ----
    def has_immediate_win(self, board, counts, side: Player) -> Optional[Tuple[int,int,Shape]]:
        for (r, c, sh) in generate_valid_moves(board, side, counts[side]):
            board[r][c] = Piece(sh, side)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (r, c, sh)
            board[r][c] = None
        return None

    def blocks_opponent_win(self, board, counts, side: Player, mv: Tuple[int,int,Shape]) -> int:
        """Renvoie 1 si jouer mv empêche une victoire immédiate de l’adversaire."""
        opp = self.opp if side == self.me else self.me
        r, c, sh = mv
        # si en posant ma pièce ici, cette case n’est plus libre pour que l’adv gagne, on compte.
        # ou si j’utilise la même forme qui donnerait la victoire à l’adversaire et que je la rends illégale.
        # Test ciblé : existait-il une victoire de l’adversaire sur cette même case ?
        # Nous cherchons toutes les victoires de l’adversaire et voyons si ce mv les annule.
        # (optimisation légère : tester uniquement la cellule elle-même)
        for sh2 in Shape:
            if counts[opp].get(sh2, 0) <= 0: continue
            if not is_valid_move(board, r, c, sh2, opp): continue
            board[r][c] = Piece(sh2, opp)
            win = check_victory_after(board, r, c)
            board[r][c] = None
            if win:
                # Si je joue mv sur (r,c), je bloque
                return 1 if (mv[0] == r and mv[1] == c) else 0
        return 0

    # ---- ordonnancement des coups ----
    def order_moves(self, board, counts, side: Player, moves, depth: int) -> List[Tuple[int,int,Shape]]:
        scored = []
        opp = self.opp if side == self.me else self.me

        # Variante principale / TT d’abord, si disponible
        tt_move = None
        key = self.board_key(board, side, counts)
        if key in self.tt and self.tt[key][2] is not None:
            tt_move = self.tt[key][2]

        killer_list = self.killers.get(depth, [])

        for mv in moves:
            r, c, sh = mv
            # victoire immédiate ?
            board[r][c] = Piece(sh, side)
            win = check_victory_after(board, r, c)
            board[r][c] = None

            # blocage valable (comme dans AB+)
            block = 0
            for sh2 in Shape:
                if counts[opp].get(sh2, 0) <= 0: continue
                if not is_valid_move(board, r, c, sh2, opp): continue
                board[r][c] = Piece(sh2, opp)
                if check_victory_after(board, r, c):
                    block = 1
                    board[r][c] = None
                    break
                board[r][c] = None

            center = 1 if (r, c) in CENTER_CELLS else 0
            hist = self.history.get(mv, 0)
            is_killer = 1 if mv in killer_list else 0
            is_tt = 1 if tt_move is not None and mv == tt_move else 0

            key_tuple = (is_tt, 1 if win else 0, block, is_killer, hist, center)
            scored.append((key_tuple, mv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ---- recherche ----
    def search(self, board, counts, side: Player, depth: int,
               alpha: float, beta: float, ply: int) -> Tuple[float, Optional[Tuple[int,int,Shape]]]:
        if self.time_up():
            return heuristic(board, self.me, counts), None

        # Extension de menace (quiescence légère)
        # si l’un ou l’autre camp a un mat-en-1 ici, on étend d’un pli
        ext = 0
        if depth > 0:
            if self.has_immediate_win(board, counts, side) is not None: ext = 1
            elif self.has_immediate_win(board, counts, self.opp if side == self.me else self.me) is not None: ext = 1

        key = self.board_key(board, side, counts)
        if key in self.tt:
            stored_depth, stored_score, stored_move = self.tt[key]
            if stored_depth >= depth:
                return stored_score, stored_move

        moves = generate_valid_moves(board, side, counts[side])
        if not moves:
            return (LOSS_SCORE if side == self.me else WIN_SCORE), None

        # Victoire immédiate : raccourci aussi pour la TT
        for (r, c, sh) in moves:
            board[r][c] = Piece(sh, side)
            if check_victory_after(board, r, c):
                board[r][c] = None
                score = WIN_SCORE if side == self.me else LOSS_SCORE
                self.tt[key] = (depth, score, (r, c, sh))
                return score, (r, c, sh)
            board[r][c] = None

        if depth + ext == 0:
            return heuristic(board, self.me, counts), None

        ordered = self.order_moves(board, counts, side, moves, ply)

        best_move = None

        if side == self.me:
            value = ALPHA_INIT
            for mv in ordered:
                r, c, sh = mv
                board[r][c] = Piece(sh, side)
                counts[side][sh] -= 1

                score, _ = self.search(board, counts, self.opp, depth - 1 + ext, alpha, beta, ply + 1)

                counts[side][sh] += 1
                board[r][c] = None

                if score > value:
                    value = score
                    best_move = mv
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    # coupure bêta : coups tueurs + historique
                    ks = self.killers.setdefault(ply, [])
                    if mv not in ks:
                        if len(ks) < 2: ks.append(mv)
                        else: ks[0] = mv
                    self.history[mv] = self.history.get(mv, 0) + (depth + 1) * 2
                    break

            self.tt[key] = (depth, value, best_move)
            return value, best_move
        else:
            value = BETA_INIT
            for mv in ordered:
                r, c, sh = mv
                board[r][c] = Piece(sh, side)
                counts[side][sh] -= 1

                score, _ = self.search(board, counts, self.me, depth - 1 + ext, alpha, beta, ply + 1)

                counts[side][sh] += 1
                board[r][c] = None

                if score < value:
                    value = score
                    best_move = mv
                if value < beta:
                    beta = value
                if alpha >= beta:
                    ks = self.killers.setdefault(ply, [])
                    if mv not in ks:
                        if len(ks) < 2: ks.append(mv)
                        else: ks[0] = mv
                    self.history[mv] = self.history.get(mv, 0) + (depth + 1) * 2
                    break

            self.tt[key] = (depth, value, best_move)
            return value, best_move

    # ---- approfondissement avec fenêtres d’aspiration ----
    def choose(self, board, counts) -> Optional[Tuple[int,int,Shape]]:
        self.t0 = time.time()
        self.killers.clear()
        self.history.clear()

        best_move = None
        prev_score = 0.0
        # profondeur cible raisonnable pour 4x4
        for depth in range(2, 10):
            if self.time_up(): break

            # fenêtre d’aspiration autour du score précédent
            window = 250.0
            alpha = max(ALPHA_INIT, prev_score - window)
            beta  = min(BETA_INIT,  prev_score + window)

            score, move = self.search(board, counts, self.me, depth, alpha, beta, ply=0)
            if self.time_up(): break

            # échec de la fenêtre ⇒ relancer avec fenêtre complète
            if score <= alpha:
                score, move = self.search(board, counts, self.me, depth, ALPHA_INIT, BETA_INIT, ply=0)
            elif score >= beta:
                score, move = self.search(board, counts, self.me, depth, ALPHA_INIT, BETA_INIT, ply=0)

            if move is not None:
                best_move = move
                prev_score = score

        if best_move is None:
            moves = generate_valid_moves(board, self.me, counts[self.me])
            if not moves: return None
            moves.sort(key=lambda m: ((m[0], m[1]) in CENTER_CELLS), reverse=True)
            return moves[0]
        return best_move

# =========================
# Classe exportée
# =========================
class QuantikAI:
    def __init__(self, player: Player):
        self.me = player
        self.time_limit = 1.25  # s / coup

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }
        engine = Searcher(self.me, time_limit=self.time_limit)
        return engine.choose(board, counts)

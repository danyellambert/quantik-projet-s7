# ai_players/alphabeta_adv/algorithme.py
# ================================================================
# IA Quantik « AlphaBeta+ » – base solide et prévisible
#  • Minimax avec élagage alpha–bêta
#  • Iterative deepening sous contrainte de temps
#  • Ordonnancement des coups (gain immédiat, bloc légal, centre)
#  • Table de transposition (TT) à clé stable : (plateau, stocks, côté)
#  • Heuristique légère : diversité (agnostique à la couleur) + mobilité + centre
#  • MUST-BLOCK à la racine pour éviter de « donner » des victoires en 1 coup
#
# Compatibilité GUI :
#  - AI_NAME = "AlphaBeta+"
#  - Classe exportée : QuantikAI(player)
#  - get_move(board, pieces_count) -> (row, col, Shape) | None
# ================================================================

from __future__ import annotations
import time, math
from typing import Optional, Tuple, List, Dict, Any

from core.types import Shape, Player, Piece

AI_NAME   = "AlphaBeta+"
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
# LINES = lignes + colonnes + zones (pour l’heuristique)
LINES: List[List[Tuple[int,int]]] = []
for r in range(N): LINES.append([(r, c) for c in range(N)])
for c in range(N): LINES.append([(r, c) for r in range(N)])
LINES.extend(ZONES)

CENTER_CELLS = {(1,1), (1,2), (2,1), (2,2)}

# --- Sentinelles finies (éviter ±inf) ---
WIN_SCORE  = 50_000.0
LOSS_SCORE = -WIN_SCORE
ALPHA_INIT = -1_000_000.0
BETA_INIT  = +1_000_000.0


# =========================
# Utilitaires de règles Quantik
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
    Règle (formulation utilisée ici) :
      illégal si l’ADVERSAIRE a déjà placé la MÊME FORME
      sur la même ligne/colonne/zone.
    """
    if board[row][col] is not None:
        return False
    # ligne
    for cc in range(N):
        p = board[row][cc]
        if p is not None and p.shape == shape and p.player != me:
            return False
    # colonne
    for rr in range(N):
        p = board[rr][col]
        if p is not None and p.shape == shape and p.player != me:
            return False
    # zone
    z = zone_index(row, col)
    for (rr, cc) in ZONES[z]:
        p = board[rr][cc]
        if p is not None and p.shape == shape and p.player != me:
            return False
    return True

def forms_all_different(pieces: List[Optional[Piece]]) -> bool:
    """Vrai si les 4 cases sont occupées par 4 formes toutes différentes."""
    if any(p is None for p in pieces): return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board: List[List[Optional[Piece]]], row: int, col: int) -> bool:
    """Après avoir joué (row, col), tester ligne, colonne et zone correspondantes."""
    if forms_all_different([board[row][c] for c in range(N)]): return True
    if forms_all_different([board[r][col] for r in range(N)]): return True
    z = zone_index(row, col)
    if forms_all_different([board[r][c] for (r, c) in ZONES[z]]): return True
    return False

def generate_valid_moves(board,
                         me: Player,
                         my_counts: Dict[Shape,int]) -> List[Tuple[int,int,Shape]]:
    """Énumère tous les coups (r, c, shape) légaux selon le stock restant."""
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves


# =========================
# Heuristique d’évaluation
# =========================
def line_diversity_score(board, cells: List[Tuple[int,int]]) -> int:
    """
    Diversité indépendante de la couleur :
      +3 si (3 formes différentes + 1 vide)
      +1 si (2 formes différentes + 2 vides)
       0 sinon
    """
    pieces = [board[r][c] for (r, c) in cells]
    empties = sum(1 for p in pieces if p is None)
    shapes = {p.shape for p in pieces if p is not None}
    if empties == 1 and len(shapes) == 3: return 3
    if empties == 2 and len(shapes) == 2: return 1
    return 0

def mobility(board, who: Player, counts: Dict[Player, Dict[Shape,int]]) -> int:
    """Mobilité brute : nombre de coups légaux disponibles pour `who`."""
    return len(generate_valid_moves(board, who, counts[who]))

def heuristic(board, me: Player, counts: Dict[Player, Dict[Shape,int]]) -> float:
    """Évaluation statique légère (baseline)."""
    opp = Player.PLAYER1 if me == Player.PLAYER2 else Player.PLAYER2
    my_div  = sum(line_diversity_score(board, cells) for cells in LINES)
    opp_div = sum(line_diversity_score(board, cells) for cells in LINES)  # symétrique
    my_mob  = mobility(board, me,  counts)
    opp_mob = mobility(board, opp, counts)
    center  = 0
    for (r, c) in CENTER_CELLS:
        p = board[r][c]
        if p is None: continue
        center += 1 if p.player == me else -1
    # Pondérations légères pour une base stable
    return 12.0*(my_div - opp_div) + 2.0*(my_mob - opp_mob) + float(center)


# =========================
# MUST-BLOCK (protection à la racine)
# =========================
def find_root_block_move(board, counts, me: Player) -> Optional[Tuple[int,int,Shape]]:
    """
    Si l’adversaire a une victoire immédiate au prochain coup,
    renvoyer un coup légal qui l’empêche :
      (i) occuper la case de la menace, OU
      (ii) rendre son coup illégal via la règle de forme (même forme posée par nous).
    """
    opp = Player.PLAYER1 if me == Player.PLAYER2 else Player.PLAYER2

    # 1) détecter les menaces (gains en 1 coup de l’adversaire)
    threats: List[Tuple[int,int,Shape]] = []
    for (r, c, sh) in generate_valid_moves(board, opp, counts[opp]):
        board[r][c] = Piece(sh, opp)
        if check_victory_after(board, r, c):
            threats.append((r, c, sh))
        board[r][c] = None
    if not threats:
        return None

    threat_cells = {(r, c) for (r, c, _) in threats}
    candidates: List[Tuple[int,int,Shape]] = []

    # (a) occuper la case de la menace
    for (r, c, _sh_opp) in threats:
        for sh_my in Shape:
            if counts[me].get(sh_my, 0) > 0 and is_valid_move(board, r, c, sh_my, me):
                candidates.append((r, c, sh_my))

    # (b) utiliser la MÊME forme que l’adversaire pour rendre son coup illégal (ligne/colonne/zone)
    for (r, c, sh_opp) in threats:
        if counts[me].get(sh_opp, 0) <= 0:
            continue
        for (rr, cc) in ALL_CELLS:
            if board[rr][cc] is not None:
                continue
            if rr == r or cc == c or zone_index(rr, cc) == zone_index(r, c):
                if is_valid_move(board, rr, cc, sh_opp, me):
                    candidates.append((rr, cc, sh_opp))

    if not candidates:
        return None

    # Priorité : même case de menace > centre
    def score(mv):
        r, c, _ = mv
        same = 1 if (r, c) in threat_cells else 0
        center = 1 if (r, c) in CENTER_CELLS else 0
        return (same, center)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


# =========================
# Moteur Alpha–Bêta + deepening + TT
# =========================
class Searcher:
    def __init__(self, me: Player, time_limit: float = 1.25):
        self.me = me
        self.opp = Player.PLAYER1 if me == Player.PLAYER2 else Player.PLAYER2
        self.time_limit = time_limit
        self.t0 = 0.0
        # TT : key -> (depth, score, best_move)
        self.tt: Dict[Any, Tuple[int, float, Optional[Tuple[int,int,Shape]]]] = {}

    # --- gestion du temps ---
    def time_up(self) -> bool:
        return (time.time() - self.t0) >= self.time_limit

    # --- clé TT stable : (plateau_sérialisé, côté, stocks_compactés) ---
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

    # --- ordonnancement des coups ---
    def order_moves(self, board, moves, side: Player, counts) -> List[Tuple[int,int,Shape]]:
        """
        1) gains immédiats
        2) coups qui BLOQUENT une victoire immédiate adverse (coup légal + forme dispo)
        3) préférence centre
        """
        scored = []
        opp = self.opp if side == self.me else self.me
        for (r, c, sh) in moves:
            # 1) gain immédiat ?
            board[r][c] = Piece(sh, side)
            is_win = check_victory_after(board, r, c)
            board[r][c] = None

            # 2) bloc légal (l’adversaire possède la forme, et son coup serait légal et gagnant)
            block = 0
            if board[r][c] is None:
                for sh2 in Shape:
                    if counts[opp].get(sh2, 0) <= 0:
                        continue
                    if not is_valid_move(board, r, c, sh2, opp):
                        continue
                    board[r][c] = Piece(sh2, opp)
                    if check_victory_after(board, r, c):
                        block = 1
                        board[r][c] = None
                        break
                    board[r][c] = None

            center = 1 if (r, c) in CENTER_CELLS else 0
            scored.append(((1 if is_win else 0, block, center), (r, c, sh)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # --- recherche ---
    def search(self, board, counts, side: Player, depth: int,
               alpha: float, beta: float) -> Tuple[float, Optional[Tuple[int,int,Shape]]]:
        if self.time_up():
            return heuristic(board, self.me, counts), None

        key = self.board_key(board, side, counts)
        if key in self.tt:
            stored_depth, stored_score, stored_move = self.tt[key]
            if stored_depth >= depth:
                return stored_score, stored_move

        moves = generate_valid_moves(board, side, counts[side])
        if not moves:
            # aucun coup : le côté au trait perd
            return (LOSS_SCORE if side == self.me else WIN_SCORE), None

        moves = self.order_moves(board, moves, side, counts)

        # court-circuit : victoire immédiate
        for (r, c, sh) in moves:
            board[r][c] = Piece(sh, side)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return ((WIN_SCORE if side == self.me else LOSS_SCORE), (r, c, sh))
            board[r][c] = None

        if depth == 0:
            return heuristic(board, self.me, counts), None

        best_move = None

        if side == self.me:
            value = ALPHA_INIT
            for (r, c, sh) in moves:
                board[r][c] = Piece(sh, side)
                counts[side][sh] -= 1

                score, _ = self.search(board, counts, self.opp, depth - 1, alpha, beta)

                counts[side][sh] += 1
                board[r][c] = None

                if score > value:
                    value = score
                    best_move = (r, c, sh)
                alpha = max(alpha, value)
                if alpha >= beta or self.time_up():
                    break
            self.tt[key] = (depth, value, best_move)
            return value, best_move
        else:
            value = BETA_INIT
            for (r, c, sh) in moves:
                board[r][c] = Piece(sh, side)
                counts[side][sh] -= 1

                score, _ = self.search(board, counts, self.me, depth - 1, alpha, beta)

                counts[side][sh] += 1
                board[r][c] = None

                if score < value:
                    value = score
                    best_move = (r, c, sh)
                beta = min(beta, value)
                if alpha >= beta or self.time_up():
                    break
            self.tt[key] = (depth, value, best_move)
            return value, best_move

    # --- choix final avec deepening ---
    def choose(self, board, counts) -> Optional[Tuple[int,int,Shape]]:
        self.t0 = time.time()

        # Protection à la racine : ne pas « offrir » une victoire en 1 coup
        block = find_root_block_move(board, counts, self.me)
        if block is not None:
            return block

        best_move = None
        for depth in range(2, 8):
            if self.time_up(): break
            score, move = self.search(board, counts, self.me, depth, ALPHA_INIT, BETA_INIT)
            if self.time_up(): break
            if move is not None:
                best_move = move

        if best_move is None:
            moves = generate_valid_moves(board, self.me, counts[self.me])
            if not moves:
                return None
            # filet de sécurité : légère préférence pour le centre
            moves.sort(key=lambda m: ((m[0], m[1]) in CENTER_CELLS), reverse=True)
            return moves[0]
        return best_move


# =========================
# Classe exportée (entrée GUI)
# =========================
class QuantikAI:
    def __init__(self, player: Player):
        self.me = player
        self.time_limit = 1.25  # s par coup (ajustable)

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        # Copier les stocks pour décrémenter/restaurer sans effets de bord
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }

        # Raccourci : si je gagne immédiatement, jouer ce coup
        root_moves = generate_valid_moves(board, self.me, counts[self.me])
        for (r, c, sh) in root_moves:
            board[r][c] = Piece(sh, self.me)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (r, c, sh)
            board[r][c] = None

        engine = Searcher(self.me, time_limit=self.time_limit)
        return engine.choose(board, counts)
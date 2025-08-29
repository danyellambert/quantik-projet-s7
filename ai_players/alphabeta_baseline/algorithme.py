# ai_players/alphabeta_baseline/algorithme.py
# ================================================================
# IA Quantik « AlphaBeta (baseline) »
# ------------------------------------------------
# Objectif : une référence *simple* pour comparer avec vos IA
# plus avancées (AlphaBeta+, AlphaBeta++, MCTS, hybrides).
# Caractéristiques :
#   • Alpha–beta MINIMAX à profondeur fixe (petite)
#   • Aucune table de transposition, aucun PVS, pas d’iterative deepening
#   • Heuristique minimaliste : mobilité + léger bonus centre
#   • Raccourci « victoire immédiate »
#   • Scores finis (pas d’infini) pour éviter les erreurs d’entiers
# API attendue par la GUI :
#   - AI_NAME (str)
#   - class QuantikAI(player)
#       -> get_move(board, pieces_count) -> (row, col, Shape) | None
# ================================================================

from __future__ import annotations
import math, random
from typing import Optional, Tuple, List, Dict

from core.types import Shape, Player, Piece

AI_NAME = "AlphaBeta (baseline)"

# --- Constantes du plateau ---
N = 4
ALL_CELLS = [(r, c) for r in range(N) for c in range(N)]
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]
CENTER = {(1,1), (1,2), (2,1), (2,2)}  # petit bonus « positionnel »

# =========================
# Utilitaires de règles
# =========================
def other(p: Player) -> Player:
    """Retourne l’autre joueur."""
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    """Indice de la zone 2×2 contenant (r,c)."""
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row: int, col: int, shape: Shape, me: Player) -> bool:
    """
    Règle de Quantik (formulation utilisée ici) :
      interdit de poser une forme si l’ADVERSAIRE a déjà posé la même forme
      dans la même ligne / colonne / zone.
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
    """Vrai si 4 cases occupées avec 4 formes toutes différentes."""
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r, c) -> bool:
    """
    Après avoir joué en (r,c), il suffit de tester :
      - la ligne r
      - la colonne c
      - la zone 2×2 de (r,c)
    """
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape, int]) -> List[Tuple[int,int,Shape]]:
    """Liste tous les coups légaux (r,c,shape) pour `me` étant donné son stock."""
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves

# =========================
# Heuristique minimaliste
# =========================
def mobility(board, who: Player, counts: Dict[Player, Dict[Shape,int]]) -> int:
    """Mobilité brute = nombre de coups légaux disponibles pour `who`."""
    return len(generate_valid_moves(board, who, counts[who]))

def heuristic(board, me: Player, counts: Dict[Player, Dict[Shape,int]]) -> int:
    """
    Évaluation simple :
      3 × (mobilité_me – mobilité_adv)
      + (pièces au centre de me – pièces au centre de l’adversaire)
    Valeurs finies (petite échelle) pour rester stables.
    """
    opp = other(me)
    h = 3 * (mobility(board, me, counts) - mobility(board, opp, counts))

    # léger bonus centre
    center_balance = 0
    for (r, c) in CENTER:
        p = board[r][c]
        if p is None:
            continue
        center_balance += 1 if p.player == me else -1
    h += center_balance
    return int(h)

# =========================
# Alpha–Beta (profondeur fixe)
# =========================
def alphabeta(board,
              counts: Dict[Player, Dict[Shape,int]],
              side: Player,
              me: Player,
              depth: int,
              alpha: int,
              beta: int) -> Tuple[int, Optional[Tuple[int,int,Shape]]]:
    """
    Retourne (score, meilleur_coup) depuis le point de vue de `me`.
    Aucune optimisation « lourde » (pas de TT, pas de PVS, pas de deepening).
    """
    # Génération
    moves = generate_valid_moves(board, side, counts[side])

    # Aucun coup : le côté courant est bloqué → l’autre gagne
    if not moves:
        return (-50000, None) if side == me else (50000, None)

    # Feuille
    if depth == 0:
        return heuristic(board, me, counts), None

    best_move = None

    if side == me:
        # Maximise
        value = -10**9
        for (r, c, sh) in moves:
            # Raccourci victoire immédiate
            board[r][c] = Piece(sh, side)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return 40000, (r, c, sh)  # grand mais FINI
            counts[side][sh] -= 1

            score, _ = alphabeta(board, counts, other(side), me, depth-1, alpha, beta)

            counts[side][sh] += 1
            board[r][c] = None

            if score > value:
                value = score
                best_move = (r, c, sh)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        # Minimise (adversaire)
        value = 10**9
        for (r, c, sh) in moves:
            board[r][c] = Piece(sh, side)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return -40000, (r, c, sh)  # très mauvais pour `me`, mais FINI
            counts[side][sh] -= 1

            score, _ = alphabeta(board, counts, other(side), me, depth-1, alpha, beta)

            counts[side][sh] += 1
            board[r][c] = None

            if score < value:
                value = score
                best_move = (r, c, sh)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move

# =========================
# Classe exportée (entrée GUI)
# =========================
class QuantikAI:
    """
    Implémentation « baseline » :
      • profondeur fixe modeste (par défaut 3)
      • assez rapide pour des tests comparatifs
    """
    def __init__(self, player: Player, depth: int = 3):
        self.me = player
        self.depth = depth  # essayez 2/3/4 pour calibrer la force

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        """
        Paramètres attendus :
          - board : matrice 4×4 (liste de listes de Piece | None)
          - pieces_count : {Player: {Shape: int}}
        On travaille sur des copies *des compteurs* pour ne pas modifier l’appelant.
        """
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }

        # Coups racine
        root_moves = generate_valid_moves(board, self.me, counts[self.me])
        if not root_moves:
            return None

        # Si un gain immédiat existe, le jouer tout de suite
        for (r, c, sh) in root_moves:
            board[r][c] = Piece(sh, self.me)
            win = check_victory_after(board, r, c)
            board[r][c] = None
            if win:
                return (r, c, sh)

        # Recherche alpha–beta simple
        score, move = alphabeta(
            board, counts, self.me, self.me, self.depth,
            -10**9, 10**9
        )

        # Filet de sécurité (ne devrait pas arriver)
        if move is None:
            return random.choice(root_moves)
        return move
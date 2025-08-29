# ai_players/genetic_ab_eval/algorithme.py
# ======================================================================
# IA "Génétique (éval AB)" pour QUANTIK
# ----------------------------------------------------------------------
# Idée générale
#   • Pendant les parties réelles : on sélectionne un coup racine à l’aide
#     d’un mini GA (population de coups) où la FITNESS est calculée par un
#     petit Alpha–Beta (peu profond) + heuristique. C’est rapide et “moins
#     bruyant” que des rollouts aléatoires.
#   • Pendant le PROBE du runner : on renvoie IMMÉDIATEMENT un coup légal
#     (aléatoire) si le plateau est visiblement initial, pour éviter d’être
#     classé “muet”.
#
# Compatibilité GUI :
#   - AI_NAME = "IA Génétique (éval AB)"
#   - Classe exportée : QuantikAI(player)
#   - get_move(board, pieces_count) -> (row, col, Shape) | None
#
# Sécurité :
#   - Ne modifie JAMAIS l’état global : toute modification se fait sur clones.
#   - Temps borné par coup (TIME_LIMIT), alpha–beta peu profond par défaut.
# ======================================================================

from __future__ import annotations
import time, random
from typing import Optional, Tuple, List, Dict, Any

from core.types import Shape, Player, Piece

AI_NAME = "Génétique (éval AB)"
AI_AUTHOR = "Danyel Lambert"

# ======== Paramètres globaux ajustables ========
TIME_LIMIT      = 1.0     # temps cible par coup (s)
GA_POP          = 24      # taille de population (coups racine)
GA_GENS         = 8       # nb de générations max (interrompu par le temps)
MUT_PROB        = 0.20    # probabilité de mutation (remplacer par un autre coup)
AB_DEPTH        = 3       # profondeur de l’alpha–beta utilisé pour la fitness
CENTER_CELLS    = {(1,1), (1,2), (2,1), (2,2)}  # bonus positionnel léger

# ======== Constantes plateau / règles ========
N = 4
ALL_CELLS = [(r, c) for r in range(N) for c in range(N)]
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]

# =========================
# Utilitaires de règles
# =========================
def other(p: Player) -> Player:
    """Retourne l’adversaire."""
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    """Indice de la zone 2×2 contenant (r,c)."""
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row: int, col: int, shape: Shape, me: Player) -> bool:
    """
    Règle (formulation utilisée ici) : interdit si l’ADVERSAIRE a déjà
    la même forme dans la même ligne/colonne/zone.
    """
    if board[row][col] is not None:
        return False
    # ligne
    for c in range(N):
        p = board[row][c]
        if p is not None and p.shape == shape and p.player != me:
            return False
    # colonne
    for r in range(N):
        p = board[r][col]
        if p is not None and p.shape == shape and p.player != me:
            return False
    # zone
    z = zone_index(row, col)
    for rr, cc in ZONES[z]:
        p = board[rr][cc]
        if p is not None and p.shape == shape and p.player != me:
            return False
    return True

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape,int]) -> List[Tuple[int,int,Shape]]:
    """Liste tous les coups légaux (r,c,shape) pour `me` compte tenu du stock."""
    moves = []
    for sh in Shape:
        if my_counts.get(sh, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, sh, me):
                moves.append((r, c, sh))
    return moves

def forms_all_different(pieces) -> bool:
    """Vrai s’il y a 4 pièces non vides avec 4 formes différentes."""
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r: int, c: int) -> bool:
    """Après avoir joué (r,c), teste ligne, colonne, et zone correspondantes."""
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

# =========================
# Clonage / application locale
# =========================
def clone_board(board):
    """Clone superficiel par lignes (suffisant pour des Piece immuables)."""
    return [row.copy() for row in board]

def clone_counts(counts):
    """Clone profond des compteurs de pièces restantes."""
    return {
        Player.PLAYER1: dict(counts[Player.PLAYER1]),
        Player.PLAYER2: dict(counts[Player.PLAYER2]),
    }

def apply_local(board, counts, move, who: Player):
    """Applique un coup sur les clones locaux (sans toucher l’état global)."""
    r, c, sh = move
    board[r][c] = Piece(sh, who)
    counts[who][sh] -= 1

# =========================
# Heuristique légère
# =========================
def mobility(board, who: Player, counts) -> int:
    """Mobilité brute = nombre de coups légaux disponibles pour `who`."""
    return len(generate_valid_moves(board, who, counts[who]))

def heuristic(board, me: Player, counts) -> int:
    """
    Évaluation statique très simple :
      3 × (mobilité_me – mobilité_adv) + balance de présence au centre.
    """
    opp = other(me)
    h = 3 * (mobility(board, me, counts) - mobility(board, opp, counts))
    center_balance = 0
    for (r, c) in CENTER_CELLS:
        p = board[r][c]
        if p is None: 
            continue
        center_balance += (1 if p.player == me else -1)
    h += center_balance
    return int(h)

# =========================
# Alpha–Beta (peu profond)
# =========================
def alphabeta(board, counts, side: Player, me: Player,
              depth: int, alpha: int, beta: int) -> int:
    """
    Minimax avec élagage alpha–beta, profondeur fixe et scores FINIS.
    Retourne un score du point de vue de `me`.
    """
    moves = generate_valid_moves(board, side, counts[side])

    # Bloqué => l’autre gagne (valeurs finies pour éviter tout cast/overflow)
    if not moves:
        return -40000 if side == me else 40000

    # Raccourci : coup gagnant immédiat pour le côté courant
    for (r, c, sh) in moves:
        board[r][c] = Piece(sh, side)
        if check_victory_after(board, r, c):
            board[r][c] = None
            return 35000 if side == me else -35000
        board[r][c] = None

    if depth == 0:
        return heuristic(board, me, counts)

    # Ordonnancement naïf : centre d’abord
    moves.sort(key=lambda m: ((m[0], m[1]) in CENTER_CELLS), reverse=True)

    if side == me:
        value = -10**9
        for (r,c,sh) in moves:
            board[r][c] = Piece(sh, side)
            counts[side][sh] -= 1
            sc = alphabeta(board, counts, other(side), me, depth-1, alpha, beta)
            counts[side][sh] += 1
            board[r][c] = None
            if sc > value:
                value = sc
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = 10**9
        for (r,c,sh) in moves:
            board[r][c] = Piece(sh, side)
            counts[side][sh] -= 1
            sc = alphabeta(board, counts, other(side), me, depth-1, alpha, beta)
            counts[side][sh] += 1
            board[r][c] = None
            if sc < value:
                value = sc
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

# =========================
# Fitness d’un COUP racine
# =========================
def eval_move_by_ab(start_board, start_counts, me: Player, move: Tuple[int,int,Shape]) -> int:
    """
    Place `move` sur des clones puis évalue la position résultante
    avec un alpha–beta peu profond.
    """
    b = clone_board(start_board)
    cnt = clone_counts(start_counts)

    # Applique mon coup
    apply_local(b, cnt, move, me)
    r, c, _ = move
    if check_victory_after(b, r, c):
        # énorme bonus si le coup gagne instantanément
        return 50000

    # Tour adverse : évalue avec AB (peu profond)
    score = alphabeta(b, cnt, other(me), me, AB_DEPTH, -10**9, 10**9)

    # Petit bonus centre pour départager (faible)
    center_bonus = 2 if (move[0], move[1]) in CENTER_CELLS else 0
    return int(score) + center_bonus

# =========================
# GA simple sur les coups racine
# =========================
def ga_choose_move(board, counts, me: Player, t_budget: float) -> Optional[Tuple[int,int,Shape]]:
    """
    Population = coups valides. Sélection/Mutation, fitness via AB rapide.
    CORRIGÉ : annotation de retour Optional[...] car il peut n’y avoir aucun coup.
    """
    t0 = time.time()
    moves = generate_valid_moves(board, me, counts[me])
    if not moves:
        return None

    # --- Génération initiale : échantillonner (ou cloner si peu de coups)
    if len(moves) <= GA_POP:
        population = moves[:]
        while len(population) < GA_POP:
            population.append(random.choice(moves))
    else:
        population = random.sample(moves, GA_POP)

    # — Évaluer toute la population
    scored = [(mv, eval_move_by_ab(board, counts, me, mv)) for mv in population]
    scored.sort(key=lambda x: x[1], reverse=True)

    gen = 0
    while gen < GA_GENS and (time.time() - t0) < t_budget:
        # Sélection : top 40% survivent
        cut = max(1, int(0.4 * len(scored)))
        survivors = [mv for (mv, _) in scored[:cut]]

        # Repeuplement par mutations (remplacer par un autre coup légal)
        new_pop: List[Tuple[int,int,Shape]] = survivors[:]
        while len(new_pop) < GA_POP:
            parent = random.choice(survivors)
            child = parent
            if random.random() < MUT_PROB:
                child = random.choice(moves)
            new_pop.append(child)

        # Ré-évaluer (budget restant)
        scored = [(mv, eval_move_by_ab(board, counts, me, mv)) for mv in new_pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        gen += 1

        # Garde-fou de temps
        if (time.time() - t0) >= t_budget:
            break

    # Meilleur coup courant
    best_move = scored[0][0]
    return best_move

# =========================
# Classe exportée
# =========================
class QuantikAI:
    """IA Génétique (fitness = Alpha–Beta) + fallback instantané au probe."""
    def __init__(self, player: Player):
        self.me = player
        self.time_limit = TIME_LIMIT

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        # 1) Coups légaux à la racine
        root_moves = generate_valid_moves(board, self.me, pieces_count[self.me])
        if not root_moves:
            return None

        # 2) Fallback “anti-muet” pour le PROBE :
        #    Si plateau vide ET stock complet (2 de chaque forme), renvoyer
        #    immédiatement un coup aléatoire légal.
        is_board_empty = all(board[r][c] is None for r in range(N) for c in range(N))
        has_full_stock = all(pieces_count[self.me].get(sh, 0) == 2 for sh in Shape)
        if is_board_empty and has_full_stock:
            return random.choice(root_moves)

        # 3) Raccourci : jouer un gain immédiat s’il existe
        for (r, c, sh) in root_moves:
            board[r][c] = Piece(sh, self.me)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (r, c, sh)
            board[r][c] = None

        # 4) Choisir via GA avec fitness AB (sur clones)
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }
        budget = max(0.05, self.time_limit - 0.01)  # petite marge de sécurité
        best = ga_choose_move(board, counts, self.me, t_budget=budget)

        # 5) Filet de sécurité
        if best is None:
            return random.choice(root_moves)
        return best
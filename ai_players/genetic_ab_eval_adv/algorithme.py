# ai_players/genetic_ab_eval+/algorithme.py
# ======================================================================
# IA "Génétique (éval AB)" – version avec limite de temps RIGIDE par coup
# ----------------------------------------------------------------------
# Principales modifications :
#  • Deadline réel par coup (même budget que AlphaBeta++) avec contrôles fréquents
#  • Profondeur adaptative de l’Alpha–Beta utilisé pour la fitness selon
#    le temps restant
#  • Boucles génération/évaluation interrompues dès que le temps est écoulé
#  • Sauvegarde du “meilleur jusqu’ici” pour un retour sûr si le temps expire
# ======================================================================

from __future__ import annotations
import time, random
from typing import Optional, Tuple, List, Dict, Any

from core.types import Shape, Player, Piece

AI_NAME = "Génétique (éval AB)+"
AI_AUTHOR = "Danyel Lambert"

# ======== Paramètres globaux (ajustables) ========
TIME_LIMIT      = 1.25   # pour un match équitable, alignez avec AlphaBeta++
GA_POP          = 24     # taille de la population (coups racine)
GA_GENS         = 8      # nb max de générations (respecte le deadline)
MUT_PROB        = 0.20   # probabilité de mutation (remplacer par un autre coup)
AB_DEPTH        = 3      # profondeur de l’alpha–beta utilisé dans la fitness
CENTER_CELLS    = {(1,1), (1,2), (2,1), (2,2)}  # léger bonus positionnel

# Petite marge de sécurité pour ne pas heurter pile le plafond (en secondes)
SAFETY_SLACK    = 0.005

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
# Utilitaires de temps (deadline)
# =========================
def time_left(deadline: float) -> float:
    """Retourne le temps restant (en secondes) avant l’échéance."""
    return deadline - time.time()

def time_up(deadline: float) -> bool:
    """Vrai si le temps est écoulé (deadline dépassée)."""
    return time_left(deadline) <= 0.0

# =========================
# Règles du jeu (Quantik)
# =========================
def other(p: Player) -> Player:
    """Retourne l’adversaire de p."""
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    """Indice de la zone 2×2 contenant (r,c)."""
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row: int, col: int, shape: Shape, me: Player) -> bool:
    """
    Coup légal si la case est vide ET si l’ADVERSAIRE n’a pas déjà posé la
    même forme dans la même ligne/colonne/zone.
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
    """Enumère tous les coups (r,c,shape) légaux pour `me`, compte tenu de son stock."""
    moves = []
    for sh in Shape:
        if my_counts.get(sh, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, sh, me):
                moves.append((r, c, sh))
    return moves

def forms_all_different(pieces) -> bool:
    """Vrai s’il y a 4 pièces non vides avec 4 formes toutes différentes."""
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r: int, c: int) -> bool:
    """Après (r,c), teste ligne/colonne/zone correspondantes pour une victoire."""
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

# =========================
# Clonage / application locale
# =========================
def clone_board(board):
    """Copie superficielle (par lignes) du plateau."""
    return [row.copy() for row in board]

def clone_counts(counts):
    """Copie profonde des compteurs de pièces restantes."""
    return {
        Player.PLAYER1: dict(counts[Player.PLAYER1]),
        Player.PLAYER2: dict(counts[Player.PLAYER2]),
    }

def apply_local(board, counts, move, who: Player):
    """Applique un coup sur clones locaux (sans effet sur l’état global)."""
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
    Évaluation simple :
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
# Alpha–Beta (profondeur fixe) – utilisé pour la fitness
# =========================
def alphabeta(board, counts, side: Player, me: Player,
              depth: int, alpha: int, beta: int) -> int:
    """
    Minimax avec élagage alpha–beta (profondeur fixe) et scores finis.
    Retourne un score du point de vue de `me`.
    """
    moves = generate_valid_moves(board, side, counts[side])

    # côté bloqué → l’autre gagne (valeurs FINIES pour éviter débordements)
    if not moves:
        return -40000 if side == me else 40000

    # raccourci : gain immédiat pour le côté courant
    for (r, c, sh) in moves:
        board[r][c] = Piece(sh, side)
        if check_victory_after(board, r, c):
            board[r][c] = None
            return 35000 if side == me else -35000
        board[r][c] = None

    if depth == 0:
        return heuristic(board, me, counts)

    # ordre naïf : centre d’abord (aide un peu la poda)
    moves.sort(key=lambda m: ((m[0], m[1]) in CENTER_CELLS), reverse=True)

    if side == me:
        value = -10**9
        for (r,c,sh) in moves:
            board[r][c] = Piece(sh, side)
            counts[side][sh] -= 1
            sc = alphabeta(board, counts, other(side), me, depth-1, alpha, beta)
            counts[side][sh] += 1
            board[r][c] = None
            if sc > value: value = sc
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
            if sc < value: value = sc
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

# ===== Profondeur adaptative selon le temps restant =====
def choose_ab_depth(time_remaining: float) -> int:
    """
    Choisit une profondeur d’AB conservative en fonction du temps restant (4×4) :
      • < 0.02s  → heuristique seule (depth 0)
      • < 0.06s  → depth 1
      • < 0.12s  → depth 2
      • sinon    → AB_DEPTH (souvent 3)
    """
    if time_remaining < 0.02:
        return 0
    if time_remaining < 0.06:
        return 1
    if time_remaining < 0.12:
        return min(2, AB_DEPTH)
    return AB_DEPTH

# =========================
# Fitness d’un coup racine (respecte le deadline)
# =========================
def eval_move_by_ab(start_board, start_counts, me: Player,
                    move: Tuple[int,int,Shape],
                    deadline: float) -> int:
    """Évalue `move` via AB peu profond/heuristique en respectant le temps restant."""
    d = choose_ab_depth(time_left(deadline))

    b = clone_board(start_board)
    cnt = clone_counts(start_counts)

    # appliquer mon coup
    apply_local(b, cnt, move, me)
    r, c, _ = move
    if check_victory_after(b, r, c):
        return 50000  # gain instantané

    if d == 0:
        # temps critique → heuristique pure
        sc = heuristic(b, me, cnt)
    else:
        sc = alphabeta(b, cnt, other(me), me, d, -10**9, 10**9)

    # petit bonus centre
    center_bonus = 2 if (move[0], move[1]) in CENTER_CELLS else 0
    return int(sc) + center_bonus

# =========================
# GA time-boxé sur les coups racine
# =========================
def ga_choose_move(board, counts, me: Player, deadline: float) -> Optional[Tuple[int,int,Shape]]:
    """
    Évolution d’une population de coups racine sous deadline :
      • évalue progressivement la population en vérifiant le temps restant
      • interrompt sélections/mutations/ré-évaluations dès que le temps est écoulé
      • conserve le “meilleur courant” pour un retour sûr
    """
    moves = generate_valid_moves(board, me, counts[me])
    if not moves:
        return None

    # Population initiale (complète jusqu’à GA_POP si peu de coups)
    if len(moves) <= GA_POP:
        population = moves[:]
        while len(population) < GA_POP:
            population.append(random.choice(moves))
    else:
        population = random.sample(moves, GA_POP)

    # Première évaluation (au moins un individu évalué)
    scored: List[Tuple[Tuple[int,int,Shape], int]] = []
    best_mv, best_sc = population[0], -10**9
    for i, mv in enumerate(population):
        if i > 0 and time_up(deadline):
            break
        sc = eval_move_by_ab(board, counts, me, mv, deadline)
        scored.append((mv, sc))
        if sc > best_sc:
            best_mv, best_sc = mv, sc

    scored.sort(key=lambda x: x[1], reverse=True)

    # Boucle des générations (respect strict du deadline)
    gen = 0
    while gen < GA_GENS and not time_up(deadline):
        # Sélection (top 40% au minimum 1)
        cut = max(1, int(0.4 * len(scored)))
        survivors = [mv for (mv, _) in scored[:cut]]

        # Repeuplement avec mutations (contrôle du temps)
        new_pop: List[Tuple[int,int,Shape]] = survivors[:]
        while len(new_pop) < GA_POP and not time_up(deadline):
            parent = random.choice(survivors)
            child = parent
            if random.random() < MUT_PROB:
                child = random.choice(moves)
            new_pop.append(child)

        # Réévaluation time-boxée de la nouvelle population
        new_scored: List[Tuple[Tuple[int,int,Shape], int]] = []
        for mv in new_pop:
            if time_up(deadline):
                break
            sc = eval_move_by_ab(board, counts, me, mv, deadline)
            new_scored.append((mv, sc))
            if sc > best_sc:
                best_mv, best_sc = mv, sc

        if not new_scored:
            break  # plus de temps pour réévaluer → on s’arrête

        new_scored.sort(key=lambda x: x[1], reverse=True)
        scored = new_scored
        gen += 1

    # Meilleur connu à ce stade (sûr même si le temps a expiré)
    return best_mv

# =========================
# Classe exportée (API GUI)
# =========================
class QuantikAI:
    """
    IA Génétique (fitness = Alpha–Beta) avec limite de temps RIGIDE par coup.
    Pour l’équité, alignez `time_limit` sur celui d’AlphaBeta++.
    """
    def __init__(self, player: Player):
        self.me = player
        self.time_limit = TIME_LIMIT

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        # 1) Coups racine légaux
        root_moves = generate_valid_moves(board, self.me, pieces_count[self.me])
        if not root_moves:
            return None

        # 2) Fallback “anti-muet” (probe initial)
        is_board_empty = all(board[r][c] is None for r in range(N) for c in range(N))
        has_full_stock = all(pieces_count[self.me].get(sh, 0) == 2 for sh in Shape)
        if is_board_empty and has_full_stock:
            return random.choice(root_moves)

        # 3) Raccourci : jouer un gain immédiat si disponible
        for (r, c, sh) in root_moves:
            board[r][c] = Piece(sh, self.me)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (r, c, sh)
            board[r][c] = None

        # 4) GA avec deadline strict
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }
        deadline = time.time() + max(0.02, self.time_limit - SAFETY_SLACK)
        best = ga_choose_move(board, counts, self.me, deadline)

        # 5) Filet de sécurité
        if best is None:
            # préférence centre si possible
            root_moves.sort(key=lambda m: ((m[0], m[1]) in CENTER_CELLS), reverse=True)
            return root_moves[0]
        return best
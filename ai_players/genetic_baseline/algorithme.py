# ai_players/genetic_baseline/algorithme.py
# -------------------------------------------------------------------
# IA basée sur un algorithme génétique simplifié :
#  - Chaque individu est un "gène" décrivant un coup (ou priorité de coups)
#  - On évalue la qualité de chaque coup par simulation rapide
#  - Sélection + mutation sur plusieurs générations dans un temps limité
#  - Retourne le meilleur coup trouvé
# -------------------------------------------------------------------

import random
import time
from typing import Optional, Tuple, List
from core.ai_base import AIBase
from core.types import Shape, Player, Piece

AI_NAME = "Génétique (baseline)"
AI_AUTHOR = "Danyel Lambert"
AI_VERSION = "1.0"

# --- Paramètres du GA ---
POP_SIZE = 20         # Taille de la population
GENERATIONS = 15      # Nombre de générations
MUTATION_RATE = 0.2   # Probabilité de mutation d’un individu
TIME_LIMIT = 1.0      # Temps maximum par coup (en secondes)

# Zones 2×2 (utile pour vérifier la validité)
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]

# ============================
# Fonctions utilitaires
# ============================
def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row: int, col: int, shape: Shape, me: Player) -> bool:
    """Vérifie si un coup est légal selon les règles de Quantik."""
    if board[row][col] is not None:
        return False
    # Ligne
    for cc in range(4):
        p = board[row][cc]
        if p is not None and p.shape == shape and p.player != me:
            return False
    # Colonne
    for rr in range(4):
        p = board[rr][col]
        if p is not None and p.shape == shape and p.player != me:
            return False
    # Zone
    z = zone_index(row, col)
    for (rr, cc) in ZONES[z]:
        p = board[rr][cc]
        if p is not None and p.shape == shape and p.player != me:
            return False
    return True

def generate_valid_moves(board, me: Player, my_pieces) -> List[Tuple[int,int,Shape]]:
    """Génère tous les coups valides possibles."""
    moves = []
    for shape in Shape:
        if my_pieces[shape] > 0:
            for r in range(4):
                for c in range(4):
                    if is_valid_move(board, r, c, shape, me):
                        moves.append((r, c, shape))
    return moves

def evaluate_move(board, move, me: Player) -> float:
    """Heuristique simple pour évaluer un coup :
       +1 pour centre, +2 si gagne immédiatement, +bonus si bloque adversaire."""
    r, c, shape = move
    score = 0.0

    # Bonus pour centre
    if (r, c) in [(1,1), (1,2), (2,1), (2,2)]:
        score += 1.0

    # Test victoire immédiate
    temp = [row.copy() for row in board]
    temp[r][c] = Piece(shape, me)
    if check_victory_after(temp, r, c):
        score += 5.0

    # TODO: On peut ajouter un bonus si cela empêche une victoire ennemie
    return score

def check_victory_after(board, r, c) -> bool:
    """Vérifie victoire après avoir joué en (r, c)."""
    def all_diff(pieces):
        return None not in pieces and len({p.shape for p in pieces}) == 4
    # Ligne
    if all_diff([board[r][cc] for cc in range(4)]): return True
    # Colonne
    if all_diff([board[rr][c] for rr in range(4)]): return True
    # Zone
    z = zone_index(r, c)
    if all_diff([board[rr][cc] for rr, cc in ZONES[z]]): return True
    return False

# ============================
# Classe IA
# ============================
class QuantikAI(AIBase):
    def get_move(self, board, pieces_count) -> Optional[Tuple[int, int, Shape]]:
        start_time = time.time()

        # Liste initiale des coups valides
        valid_moves = generate_valid_moves(board, self.me, pieces_count[self.me])
        if not valid_moves:
            return None

        # --- Initialisation population ---
        population = [random.choice(valid_moves) for _ in range(POP_SIZE)]

        # --- Boucle des générations ---
        for _ in range(GENERATIONS):
            # Évaluation de la population
            scored = [(mv, evaluate_move(board, mv, self.me)) for mv in population]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Sélection (top 50%)
            survivors = [mv for mv, _ in scored[:POP_SIZE // 2]]

            # Croisement + mutation pour régénérer la population
            new_pop = survivors.copy()
            while len(new_pop) < POP_SIZE:
                parent = random.choice(survivors)
                child = parent

                # Mutation : changement de coup
                if random.random() < MUTATION_RATE:
                    child = random.choice(valid_moves)
                new_pop.append(child)

            population = new_pop

            # Limite de temps
            if time.time() - start_time > TIME_LIMIT:
                break

        # --- Retour du meilleur coup ---
        best_move = max(population, key=lambda mv: evaluate_move(board, mv, self.me))
        return best_move

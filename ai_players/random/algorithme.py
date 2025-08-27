# ai_players/random/algorithme.py
# -------------------------------------------------------------------
# IA minimaliste : génère tous les coups valides et choisit au hasard.
# Conforme aux règles du sujet et compatible avec la GUI fournie.
# -------------------------------------------------------------------
import random
from typing import Optional, Tuple
from core.ai_base import AIBase
from core.types import Shape, Player, Piece

AI_NAME = "IA aléatoire"
AI_AUTHOR = "Danyel Lambert"
AI_VERSION = "1.0"

# Zones 2×2 (copie locale pour validation rapide)
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]

def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row: int, col: int, shape: Shape, me: Player) -> bool:
    # Case occupée ?
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

class QuantikAI(AIBase):
    def get_move(self, board, pieces_count) -> Optional[Tuple[int, int, Shape]]:
        valid = []
        for shape in Shape:
            if pieces_count[self.me][shape] <= 0:
                continue
            for r in range(4):
                for c in range(4):
                    if is_valid_move(board, r, c, shape, self.me):
                        valid.append((r, c, shape))
        return random.choice(valid) if valid else None
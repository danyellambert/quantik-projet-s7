# ai_players/mcts_baseline/algorithme.py
# ===============================================================
# IA Quantik – MCTS (simple / primitif)
# ---------------------------------------------------------------
# Objectif : fournir une baseline MCTS très simple :
#  - UCT standard (sélection)
#  - Expansion d’un coup non essayé
#  - Rollout aléatoire pur (sans biais ni heuristiques)
#  - Backpropagation binaire (1 si le joueur_ayant_joué_ici gagne)
#
# Compatibilité :
#  - AI_NAME = "MCTS (baseline)"
#  - Classe exportée : QuantikAI(player)
#  - get_move(board, pieces_count) -> (row, col, Shape) | None
#
# Remarques :
#  - Ne modifie JAMAIS le plateau global : clones locaux à chaque itération
#  - Pas d’optimisations : parfait pour servir de référence “primitif”
# ===============================================================

from __future__ import annotations
import math, time, random
from typing import Optional, Tuple, List, Dict

from core.types import Shape, Player, Piece

AI_NAME = "MCTS (baseline)"
AI_AUTHOR = "Danyel Lambert"

# --- Constantes plateau ---
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
    """Renvoie l’autre joueur."""
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    """Indice de la zone 2×2 contenant (r,c)."""
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row, col, shape: Shape, me: Player) -> bool:
    """
    Règle Quantik : il est interdit de poser une forme si l’ADVERSAIRE
    a déjà cette même forme dans la même ligne / colonne / zone.
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
    for (rr, cc) in ZONES[z]:
        p = board[rr][cc]
        if p is not None and p.shape == shape and p.player != me:
            return False
    return True

def forms_all_different(pieces) -> bool:
    """Vrai si 4 cases remplies avec 4 formes toutes différentes."""
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r, c) -> bool:
    """Après avoir joué en (r,c), teste ligne, colonne et zone correspondantes."""
    # ligne
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    # colonne
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    # zone
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape,int]):
    """Liste (r,c,shape) de tous les coups légaux pour `me` étant donné son stock."""
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves


# =========================
# Clonage local (snapshots)
# =========================
def clone_board(board):
    """Copie superficielle du plateau (liste de listes)."""
    return [row.copy() for row in board]

def clone_counts(counts):
    """Copie profonde des compteurs de pièces restantes."""
    return {
        Player.PLAYER1: dict(counts[Player.PLAYER1]),
        Player.PLAYER2: dict(counts[Player.PLAYER2]),
    }

def apply_move_local(board, counts, move, who: Player):
    """Applique un coup sur le clone local (sans toucher l’état global)."""
    r, c, shape = move
    board[r][c] = Piece(shape, who)
    counts[who][shape] -= 1


# =========================
# Nœud MCTS (UCT standard)
# =========================
class Node:
    """
    Nœud MCTS classique basé sur UCT.
    - player_just_moved : joueur ayant joué le coup menant à CE nœud.
    - move : coup appliqué pour arriver ici (None à la racine).
    - untried : liste des coups encore non explorés depuis ce nœud.
    """
    __slots__ = ("parent", "move", "player_just_moved", "untried", "children", "wins", "visits")

    def __init__(self, parent, move, player_just_moved, untried_moves):
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.untried = list(untried_moves)
        self.children: List[Node] = []
        self.wins = 0.0
        self.visits = 0

    def uct_select_child(self, c: float = 1.4142):
        """Sélection UCT : maximise (wins/visits + c * sqrt(log(N)/visits))."""
        logN = math.log(self.visits + 1e-9)
        best, best_val = None, -1e9
        for ch in self.children:
            if ch.visits == 0:
                val = 1e9
            else:
                val = (ch.wins / ch.visits) + c * math.sqrt(logN / ch.visits)
            if val > best_val:
                best, best_val = ch, val
        return best

    def add_child(self, move, child_untried, player_just_moved):
        """Ajoute un enfant pour `move`, retourne le nouveau nœud enfant."""
        ch = Node(self, move, player_just_moved, child_untried)
        self.children.append(ch)
        self.untried.remove(move)
        return ch

    def update(self, winner: Player):
        """Backpropagation binaire : +1 si le `winner` est `player_just_moved`."""
        self.visits += 1
        if winner == self.player_just_moved:
            self.wins += 1.0


# =========================
# Rollout (simulation aléatoire)
# =========================
def rollout(board, counts, player_to_move: Player) -> Player:
    """
    Simule la partie jusqu’à la fin en jouant des coups valides au hasard.
    Aucune heuristique : “primitif”.
    """
    cur = player_to_move
    while True:
        moves = generate_valid_moves(board, cur, counts[cur])
        if not moves:
            # Bloqué => l’autre gagne
            return other(cur)

        mv = random.choice(moves)
        apply_move_local(board, counts, mv, cur)
        r, c, _ = mv
        if check_victory_after(board, r, c):
            return cur
        cur = other(cur)


# =========================
# Boucle MCTS (temps fixe)
# =========================
def mcts_decide(board, counts, me: Player, time_limit: float = 0.6):
    """
    Tour principal MCTS :
      - Racine : coups légaux pour `me`
      - Itère (sélection → expansion → rollout → backprop) jusqu’à time_limit
      - Renvoie le coup le plus visité depuis la racine
    """
    root_moves = generate_valid_moves(board, me, counts[me])
    if not root_moves:
        return None

    # À la racine, personne n’a joué encore : player_just_moved = adversaire
    root = Node(parent=None, move=None, player_just_moved=other(me), untried_moves=root_moves)

    t0 = time.time()
    while (time.time() - t0) < time_limit:
        # Clones locaux à chaque itération
        b = clone_board(board)
        cnt = clone_counts(counts)

        node = root
        cur = me  # c’est à `me` de jouer à la racine

        # 1) Sélection : descend tant qu’il n’y a pas de coup “untried”
        while not node.untried and node.children:
            node = node.uct_select_child()
            apply_move_local(b, cnt, node.move, cur)
            r, c, _ = node.move
            if check_victory_after(b, r, c):
                # Terminal : le gagnant est celui qui vient de jouer (cur)
                winner = cur
                # Backprop immédiate
                n = node
                while n is not None:
                    n.update(winner)
                    n = n.parent
                break
            cur = other(cur)
        else:
            # 2) Expansion si possible
            if node.untried:
                mv = random.choice(node.untried)
                apply_move_local(b, cnt, mv, cur)
                r, c, _ = mv
                if check_victory_after(b, r, c):
                    # Enfant terminal
                    child = node.add_child(mv, [], player_just_moved=cur)
                    # 4) Backprop (gagnant = cur)
                    n = child
                    while n is not None:
                        n.update(cur)
                        n = n.parent
                else:
                    next_player = other(cur)
                    child_untried = generate_valid_moves(b, next_player, cnt[next_player])
                    child = node.add_child(mv, child_untried, player_just_moved=cur)

                    # 3) Rollout aléatoire
                    winner = rollout(b, cnt, next_player)

                    # 4) Backprop
                    n = child
                    while n is not None:
                        n.update(winner)
                        n = n.parent

        # fin d’une itération

    # Choix final : enfant le plus visité
    if not root.children:
        # fallback (peut arriver si time_limit trop court)
        return random.choice(root_moves)
    best = max(root.children, key=lambda ch: ch.visits)
    return best.move


# =========================
# Classe exportée (API GUI)
# =========================
class QuantikAI:
    """
    Point d’entrée attendu par l’interface.
    Version “primitif” : peu de paramètres, rollouts aléatoires.
    """
    def __init__(self, player: Player):
        self.me = player
        self.time_limit = 0.6  # secondes par coup (léger pour rester baseline)

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        # Clones locaux des compteurs (on les décrémente durant les itérations)
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }
        move = mcts_decide(board, counts, self.me, time_limit=self.time_limit)
        return move
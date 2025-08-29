# ai_players/mcts_adv/algorithme.py
# ================================================================
# IA “MCTS+ (robuste)” – Compatible avec l’interface :
#  - AI_NAME = "MCTS+"
#  - Classe exportée : QuantikAI(player)
#  - get_move(board, pieces_count) -> (row, col, Shape) | None
#
# Points clés :
#  • Ne modifie jamais l’état global : chaque itération travaille sur des clones
#  • Schéma MCTS classique avec player_just_moved pour une rétropropagation simple
#  • Sélection / expansion / simulation / backprop propres (sans undo global)
#  • CORRECTIONS : see “Changements” en bas du fichier
# ================================================================

from __future__ import annotations
import math, time, random
from typing import Optional, Tuple, List, Dict

from core.types import Shape, Player, Piece

AI_NAME = "MCTS+"
AI_AUTHOR = "Danyel Lambert"

# --- Constantes ---
N = 4
ALL_CELLS = [(r, c) for r in range(N) for c in range(N)]
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]
CENTER = {(1,1), (1,2), (2,1), (2,2)}


# =========================
# Utilitaires des règles
# =========================
def other(p: Player) -> Player:
    """Retourne l'autre joueur (CORRIGÉ : renvoie bien l’adversaire)."""
    # BUG d’origine : renvoyait toujours PLAYER1.
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    """Retourne l'indice de la zone 2x2 dans laquelle se trouve la case (r,c)."""
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row, col, shape: Shape, me: Player) -> bool:
    """
    Règle Quantik (formulation utilisée ici) :
    interdit si l’ADVERSAIRE a déjà posé la même forme
    dans la même ligne, colonne ou zone.
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

def forms_all_different(pieces: List[Optional[Piece]]) -> bool:
    """Vrai si toutes les cases sont remplies et que toutes les formes sont différentes."""
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r, c) -> bool:
    """Après avoir joué en (r,c), teste la ligne, la colonne et la zone pour une victoire."""
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape, int]) -> List[Tuple[int,int,Shape]]:
    """Génère la liste des coups valides selon la règle de Quantik."""
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves


# =========================
# Clonage / Snapshots locaux
# =========================
def clone_board(board) -> List[List[Optional[Piece]]]:
    """Copie superficielle des lignes du plateau."""
    return [row.copy() for row in board]

def clone_counts(counts) -> Dict[Player, Dict[Shape, int]]:
    """Copie profonde des compteurs de pièces restantes."""
    return {
        Player.PLAYER1: dict(counts[Player.PLAYER1]),
        Player.PLAYER2: dict(counts[Player.PLAYER2]),
    }

def apply_move_local(board, counts, move, who: Player):
    """Applique un coup sur un plateau/clonage local sans affecter l'état global."""
    r, c, shape = move
    board[r][c] = Piece(shape, who)
    counts[who][shape] -= 1


# =========================
# Nœud MCTS (player_just_moved)
# =========================
class Node:
    """Nœud MCTS classique (player_just_moved = joueur qui vient de jouer)."""
    __slots__ = ("parent", "move", "player_just_moved", "untried", "children", "wins", "visits")

    def __init__(self,
                 parent: Optional["Node"],
                 move: Optional[Tuple[int,int,Shape]],
                 player_just_moved: Player,
                 untried: List[Tuple[int,int,Shape]]):
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.untried = list(untried)
        self.children: List[Node] = []
        self.wins = 0.0
        self.visits = 0

    def uct_select_child(self, c: float = 1.4142) -> "Node":
        """Sélection UCT classique."""
        logN = math.log(self.visits + 1e-9)
        best, best_val = None, -1e9
        for ch in self.children:
            if ch.visits == 0:
                val = 1e9
            else:
                exploit = ch.wins / ch.visits
                explore = c * math.sqrt(logN / ch.visits)
                val = exploit + explore
            if val > best_val:
                best, best_val = ch, val
        return best

    def add_child(self, move: Tuple[int,int,Shape], child_untried, player_just_moved: Player) -> "Node":
        """Ajoute un enfant au nœud courant (et retire ce coup de la liste untried)."""
        ch = Node(self, move, player_just_moved, child_untried)
        self.children.append(ch)
        self.untried.remove(move)
        return ch

    def update(self, winner: Player):
        """Backpropagation : +1 si le gagnant est le joueur qui a joué pour arriver ici."""
        self.visits += 1
        if winner == self.player_just_moved:
            self.wins += 1.0


# =========================
# Rollout (simulation)
# =========================
def biased_random_choice(moves: List[Tuple[int,int,Shape]]) -> Tuple[int,int,Shape]:
    """Biais léger vers les cases centrales pendant la simulation."""
    if not moves:
        return None
    weights = [2.0 if (r, c) in CENTER else 1.0 for (r, c, _) in moves]
    tot = sum(weights)
    x = random.random() * tot
    acc = 0.0
    for w, m in zip(weights, moves):
        acc += w
        if x <= acc:
            return m
    return moves[-1]

def rollout(board, counts, player_to_move: Player) -> Player:
    """
    Simule la partie jusqu'à la fin :
      - joue immédiatement un gain si disponible
      - sinon, choix aléatoire (biais centre)
    """
    cur = player_to_move
    while True:
        moves = generate_valid_moves(board, cur, counts[cur])
        if not moves:
            # Aucun coup possible → l'autre joueur gagne
            return other(cur)

        # Vérification gain immédiat (CORRIGÉ : on annule bien l’essai avant de retourner)
        for (r, c, sh) in moves:
            board[r][c] = Piece(sh, cur)
            if check_victory_after(board, r, c):
                board[r][c] = None  # <<< important : nettoyer avant return
                return cur
            board[r][c] = None

        mv = biased_random_choice(moves)
        apply_move_local(board, counts, mv, cur)
        r, c, _ = mv
        if check_victory_after(board, r, c):
            return cur
        cur = other(cur)


# =========================
# MCTS principal (temps limité)
# =========================
def mcts_decide(board, counts, me: Player, time_limit: float = 1.2) -> Optional[Tuple[int,int,Shape]]:
    """Boucle MCTS standard avec clones locaux (pas d'annulation globale)."""
    root_moves = generate_valid_moves(board, me, counts[me])
    if not root_moves:
        return None

    # Raccourci : coup gagnant immédiat
    for (r, c, sh) in root_moves:
        b2 = clone_board(board)
        b2[r][c] = Piece(sh, me)
        if check_victory_after(b2, r, c):
            return (r, c, sh)

    root = Node(parent=None, move=None, player_just_moved=other(me), untried=root_moves)

    t0 = time.time()
    while (time.time() - t0) < time_limit:
        # --- Clones locaux ---
        b = clone_board(board)
        cnt = clone_counts(counts)

        node = root
        cur = me  # à la racine, c’est à ‘me’ de jouer

        # 1) Sélection
        while not node.untried and node.children:
            node = node.uct_select_child()
            apply_move_local(b, cnt, node.move, cur)
            r, c, _ = node.move
            if check_victory_after(b, r, c):
                winner = cur
                n = node
                while n is not None:
                    n.update(winner)
                    n = n.parent
                break  # on termine cette itération
            cur = other(cur)
        else:
            # 2) Expansion
            if node.untried:
                mv = random.choice(node.untried)
                apply_move_local(b, cnt, mv, cur)
                r, c, _ = mv
                if check_victory_after(b, r, c):
                    # le coup mène à une feuille gagnante
                    child = node.add_child(mv, [], player_just_moved=cur)
                    n = child
                    while n is not None:
                        n.update(cur)
                        n = n.parent
                else:
                    next_player = other(cur)
                    child_untried = generate_valid_moves(b, next_player, cnt[next_player])
                    child = node.add_child(mv, child_untried, player_just_moved=cur)
                    winner = rollout(b, cnt, next_player)
                    n = child
                    while n is not None:
                        n.update(winner)
                        n = n.parent

    # Décision finale : enfant le plus visité
    if not root.children:
        return random.choice(root_moves)
    best = max(root.children, key=lambda ch: ch.visits)
    return best.move


# =========================
# Classe exportée
# =========================
class QuantikAI:
    """Entrée pour l’interface : IA MCTS robuste (avec snapshots locaux)."""
    def __init__(self, player: Player):
        self.me = player
        self.time_limit = 1.2  # secondes par coup (modifiable)

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        counts = clone_counts(pieces_count)
        return mcts_decide(board, counts, self.me, time_limit=self.time_limit)
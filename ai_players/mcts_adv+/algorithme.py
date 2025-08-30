# ai_players/mcts_adv+/algorithme.py
# ================================================================
# IA “MCTS++ (optimisé)” – Monte Carlo Tree Search sans alpha–beta
# ---------------------------------------------------------------
# Objectif de cette version :
#  • Garder les atouts “++” qui aident vraiment (win/blocks en rollout,
#    biais centre, priors) en supprimant les surcoûts qui réduisaient
#    trop le nombre de simulations par seconde.
#  • Priorité à la VITESSE des rollouts : plus d’itérations → meilleure
#    estimation statistique → compétitivité face à MCTS+.
#
# Changements vs MCTS++ (v2) :
#  • Rollout “safe léger” : win immédiat > blocage de win adverse > aléatoire
#    biaisé centre. (On retire la détection des coups “suicidaires” et le
#    départage heuristique en fin de rollout, très coûteux.)
#  • Priors uniquement par CELLULE, pas par (cellule+forme), pour réduire
#    la charge mémoire/calcul. Pondération modérée.
#  • UCT légèrement plus agressif pour doper l’exploration.
#
# Compatibilité GUI :
#  - AI_NAME = "MCTS++"
#  - Classe exportée : QuantikAI(player)
#  - get_move(board, pieces_count) -> (row, col, Shape) | None
#
# Règle d’or : ne JAMAIS modifier l’état global (on clone toujours).
# ================================================================

from __future__ import annotations
import math, time, random
from typing import Optional, Tuple, List, Dict

from core.types import Shape, Player, Piece

AI_NAME = "MCTS++"
AI_AUTHOR = "Danyel Lambert"

# --- Constantes de plateau ---
N = 4
ALL_CELLS = [(r, c) for r in range(N) for c in range(N)]
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]
CENTER = {(1,1), (1,2), (2,1), (2,2)}

# --- Hyperparamètres MCTS (réglages “rapides”) ---
UCT_C        = 1.35     # constante d’exploration (un peu agressif)
PB_LAMBDA    = 0.06     # poids modéré de la progressive bias (priors)
TIME_LIMIT_S = 1.2      # secondes par décision (modifiable au niveau de la classe)

# Priors globaux (appris durant les rollouts) : par CELLULE uniquement
# clé: (joueur, (r,c)) -> (wins, plays)
_GLOBAL_PRIORS_CELL: Dict[Tuple[Player, Tuple[int,int]], Tuple[int,int]] = {}

# =========================
# Utilitaires des règles
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
    Règle Quantik utilisée ici :
      Interdit si l’ADVERSAIRE a déjà posé la même forme
      dans la même ligne / colonne / zone.
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

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape,int]) -> List[Tuple[int,int,Shape]]:
    """Enumère les coups légaux (r, c, shape) compte tenu du stock de `me`."""
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves

def forms_all_different(pieces) -> bool:
    """Vrai s’il y a 4 pièces non vides avec 4 formes différentes."""
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r: int, c: int) -> bool:
    """Après avoir joué (r,c), teste ligne, colonne et zone correspondantes."""
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

# =========================
# Clonage / application locale
# =========================
def clone_board(board):
    """Copie superficielle par lignes (suffisant pour des Piece immuables)."""
    return [row.copy() for row in board]

def clone_counts(counts):
    """Copie profonde des compteurs de pièces restantes."""
    return {
        Player.PLAYER1: dict(counts[Player.PLAYER1]),
        Player.PLAYER2: dict(counts[Player.PLAYER2]),
    }

def apply_move_local(board, counts, move, who: Player):
    """Applique un coup localement (sans effet de bord global)."""
    r, c, sh = move
    board[r][c] = Piece(sh, who)
    counts[who][sh] -= 1

# =========================
# Politique “rollout rapide & sûr (léger)”
# =========================
def has_immediate_win(board, counts, who: Player) -> Optional[Tuple[int,int,Shape]]:
    """Retourne un coup gagnant immédiat pour `who` s’il existe (sinon None)."""
    moves = generate_valid_moves(board, who, counts[who])
    for (r, c, sh) in moves:
        board[r][c] = Piece(sh, who)
        if check_victory_after(board, r, c):
            board[r][c] = None
            return (r, c, sh)
        board[r][c] = None
    return None

def cells_where_opponent_wins_next(board, counts, opp: Player) -> set:
    """Retourne l’ensemble des cases (r,c) où l’adversaire peut gagner immédiatement."""
    danger = set()
    moves = generate_valid_moves(board, opp, counts[opp])
    for (r, c, sh) in moves:
        board[r][c] = Piece(sh, opp)
        if check_victory_after(board, r, c):
            danger.add((r, c))
        board[r][c] = None
    return danger

def biased_random_choice(moves: List[Tuple[int,int,Shape]]) -> Tuple[int,int,Shape]:
    """Choix aléatoire biaisé vers le centre (léger)."""
    if not moves:
        return None
    weights = [2.0 if (r, c) in CENTER else 1.0 for (r, c, _) in moves]
    s = sum(weights)
    x, acc = random.random() * s, 0.0
    for w, mv in zip(weights, moves):
        acc += w
        if x <= acc:
            return mv
    return moves[-1]

def rollout_policy_fast_safe(board, counts, who: Player) -> Optional[Tuple[int,int,Shape]]:
    """
    Politique de simulation “fast & safe” :
      1) jouer un GAIN immédiat si dispo
      2) sinon BLOQUER un gain immédiat adverse si possible
      3) sinon, aléatoire biaisé centre
    (Pas de test “suicidaire” ni d’heuristique de départage final.)
    """
    # 1) gain immédiat
    mv = has_immediate_win(board, counts, who)
    if mv is not None:
        return mv

    # 2) blocage de la case gagnante adverse (si existe)
    opp = other(who)
    danger = cells_where_opponent_wins_next(board, counts, opp)
    if danger:
        my_moves = generate_valid_moves(board, who, counts[who])
        for (r, c, sh) in my_moves:
            if (r, c) in danger:
                return (r, c, sh)

    # 3) choix biais centre
    moves = generate_valid_moves(board, who, counts[who])
    if not moves:
        return None
    return biased_random_choice(moves)

# =========================
# Nœud MCTS (player_just_moved)
# =========================
class Node:
    """
    Nœud MCTS classique :
      - player_just_moved : joueur qui a joué pour arriver ici
      - untried : coups encore non explorés à partir de cet état
      - children : sous-nœuds (après avoir joué les coups)
      - wins/visits : statistiques de victoire au profit de player_just_moved
    """
    __slots__ = ("parent", "move", "player_just_moved", "untried",
                 "children", "wins", "visits")

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

    def add_child(self, move: Tuple[int,int,Shape],
                  child_untried: List[Tuple[int,int,Shape]],
                  player_just_moved: Player) -> "Node":
        ch = Node(self, move, player_just_moved, child_untried)
        self.children.append(ch)
        try:
            self.untried.remove(move)
        except ValueError:
            pass
        return ch

    # --- Sélection UCT + progressive bias par priors CELLULE ---
    def uct_select_child(self) -> "Node":
        side_to_play = other(self.player_just_moved)
        logN = math.log(self.visits + 1e-9)

        best, best_val = None, -1e18
        for ch in self.children:
            (r, c, _sh) = ch.move
            w_c, n_c = _GLOBAL_PRIORS_CELL.get((side_to_play, (r, c)), (0, 0))
            prior_cell = (w_c / (n_c + 1e-9)) if n_c > 0 else 0.5

            if ch.visits == 0:
                exploit = 0.5
                explore = 1e6  # forcer l’exploration initiale
                val = exploit + PB_LAMBDA * prior_cell + explore
            else:
                exploit = ch.wins / ch.visits
                explore = UCT_C * math.sqrt(logN / ch.visits)
                prior_term = (PB_LAMBDA * prior_cell) / (1.0 + 0.25 * ch.visits)
                val = exploit + prior_term + explore

            if val > best_val:
                best, best_val = ch, val
        return best

# =========================
# MCTS : simulation et backprop
# =========================
def simulate_to_end(board, counts, player_to_move: Player) -> Tuple[Player, Dict[Player, List[Tuple[int,int,Shape]]]]:
    """
    Simulation rapide “fast & safe” :
      - gagner si possible
      - bloquer si nécessaire
      - sinon choix biais centre
    Retourne (gagnant, historique des coups par joueur) pour mettre à jour les priors.
    """
    cur = player_to_move
    history: Dict[Player, List[Tuple[int,int,Shape]]] = {Player.PLAYER1: [], Player.PLAYER2: []}

    for _ply in range(16):  # borne dure : 16 poses max
        mv = rollout_policy_fast_safe(board, counts, cur)
        if mv is None:
            # cur est bloqué → l’autre gagne
            return other(cur), history

        apply_move_local(board, counts, mv, cur)
        history[cur].append(mv)
        r, c, _ = mv
        if check_victory_after(board, r, c):
            return cur, history

        cur = other(cur)

    # Sécurité : si non terminal à 16 poses, choisir aléatoirement (rare)
    return random.choice([Player.PLAYER1, Player.PLAYER2]), history

def update_priors_from_rollout(winner: Player, moves_by_player: Dict[Player, List[Tuple[int,int,Shape]]]):
    """Met à jour des priors globaux : par CELLULE (seulement)."""
    # gagnant
    for (r, c, _sh) in moves_by_player[winner]:
        wc, nc = _GLOBAL_PRIORS_CELL.get((winner, (r, c)), (0, 0))
        _GLOBAL_PRIORS_CELL[(winner, (r, c))] = (wc + 1, nc + 1)

    # perdant
    loser = other(winner)
    for (r, c, _sh) in moves_by_player[loser]:
        wc, nc = _GLOBAL_PRIORS_CELL.get((loser, (r, c)), (0, 0))
        _GLOBAL_PRIORS_CELL[(loser, (r, c))] = (wc + 0, nc + 1)

def mcts_decide(board, counts, me: Player, time_limit: float = TIME_LIMIT_S) -> Optional[Tuple[int,int,Shape]]:
    """Boucle MCTS : sélection → expansion → simulation → backprop, sous contrainte de temps."""
    root_moves = generate_valid_moves(board, me, counts[me])
    if not root_moves:
        return None

    # Raccourci : coup gagnant immédiat à la racine
    for (r, c, sh) in root_moves:
        b2 = clone_board(board)
        b2[r][c] = Piece(sh, me)
        if check_victory_after(b2, r, c):
            return (r, c, sh)

    root = Node(parent=None, move=None, player_just_moved=other(me), untried=root_moves)

    t0 = time.time()
    while (time.time() - t0) < time_limit:
        # Snapshots locaux
        b = clone_board(board)
        cnt = clone_counts(counts)

        node = root
        cur = me

        # 1) Sélection
        while not node.untried and node.children:
            node = node.uct_select_child()
            apply_move_local(b, cnt, node.move, cur)
            r, c, _ = node.move
            if check_victory_after(b, r, c):
                # Victoire pendant la descente → backprop immédiate
                winner = cur
                n = node
                while n is not None:
                    n.visits += 1
                    if winner == n.player_just_moved:
                        n.wins += 1.0
                    n = n.parent
                break
            cur = other(cur)
        else:
            # 2) Expansion (si coups non essayés)
            if node.untried:
                mv = random.choice(node.untried)
                apply_move_local(b, cnt, mv, cur)
                r, c, _ = mv
                if check_victory_after(b, r, c):
                    child = node.add_child(mv, [], player_just_moved=cur)
                    # rétroprop (feuille gagnante)
                    n = child
                    while n is not None:
                        n.visits += 1
                        if cur == n.player_just_moved:
                            n.wins += 1.0
                        n = n.parent
                else:
                    next_player = other(cur)
                    child_untried = generate_valid_moves(b, next_player, cnt[next_player])
                    child = node.add_child(mv, child_untried, player_just_moved=cur)

                    # 3) Simulation (rollout) jusqu’à fin ou borne
                    winner, moves_by_player = simulate_to_end(b, cnt, next_player)

                    # 4) Backprop + mise à jour des priors
                    update_priors_from_rollout(winner, moves_by_player)
                    n = child
                    while n is not None:
                        n.visits += 1
                        if winner == n.player_just_moved:
                            n.wins += 1.0
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
    """
    Entrée pour l’interface : IA MCTS++ (optimisé).
    Idée directrice : maximiser le nombre de rollouts par seconde tout
    en conservant un minimum de “sécurité” (win/block) et un biais centre.
    """
    def __init__(self, player: Player):
        self.me = player
        self.time_limit = TIME_LIMIT_S

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        # Coups légaux disponibles
        root_moves = generate_valid_moves(board, self.me, pieces_count[self.me])
        if not root_moves:
            return None

        # Anti-muet : plateau initial + stock complet → renvoyer vite (biais centre)
        is_board_empty = all(board[r][c] is None for r in range(N) for c in range(N))
        has_full_stock = all(pieces_count[self.me].get(sh, 0) == 2 for sh in Shape)
        if is_board_empty and has_full_stock:
            centers = [m for m in root_moves if (m[0], m[1]) in CENTER]
            return random.choice(centers) if centers else random.choice(root_moves)

        # Raccourci : victoire immédiate
        for (r, c, sh) in root_moves:
            b2 = clone_board(board)
            b2[r][c] = Piece(sh, self.me)
            if check_victory_after(b2, r, c):
                return (r, c, sh)

        # MCTS sous contrainte de temps
        counts = clone_counts(pieces_count)
        mv = mcts_decide(board, counts, self.me, time_limit=self.time_limit)

        # Filet de sécurité
        if mv is None:
            root_moves.sort(key=lambda m: ((m[0], m[1]) in CENTER), reverse=True)
            return root_moves[0]
        return mv
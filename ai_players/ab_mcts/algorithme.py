# ai_players/ab_mcts/algorithme.py
# ================================================================
# IA Hybride “AlphaBeta+MCTS” – Compatible avec la GUI:
#   - AI_NAME = "AlphaBeta+MCTS"
#   - Classe exportée : QuantikAI(player)
#   - get_move(board, pieces_count) -> (row, col, Shape) | None
#
# Idée:
#  1) Alpha–Beta (iterative deepening, ordonnancement fort) pour trouver
#     1 à 3 meilleurs coups rapidement (et un score approximatif).
#  2) MCTS ciblé sur ces candidats pendant le temps restant pour
#     départager/valider celui qui gagne le plus souvent en simulation.
#
# Changements (robustesse):
#  • Remplacement de ±inf par des sentinelles FINIES (overflow évité).
#  • Garde stricte du budget temps (checks dans les boucles).
#  • Fallback “anti-muet” sur position initiale (renvoie un coup légal direct).
#  • MCTS: au moins un petit budget, choix stable même avec peu de rollouts.
# ================================================================

from __future__ import annotations
import time, math, random
from typing import Optional, Tuple, List, Dict, Any

from core.types import Shape, Player, Piece

AI_NAME = "AlphaBeta+MCTS"
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
CENTER = {(1,1), (1,2), (2,1), (2,2)}

# --- Sentinelles finies (au lieu de ±inf) ---
WIN_SCORE   = 50_000
LOSS_SCORE  = -WIN_SCORE
ALPHA_INIT  = -1_000_000
BETA_INIT   = +1_000_000

# =========================
# Utilitaires de règles
# =========================
def other(p: Player) -> Player:
    """Retourne l’adversaire de p."""
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row: int, col: int, shape: Shape, me: Player) -> bool:
    """Règle du Quantik: interdit si l’ADVERSAIRE a déjà posé la même forme
       dans la ligne/colonne/zone."""
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
    """Vrai si 4 cases remplies et 4 formes toutes différentes (couleur ignorée)."""
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r: int, c: int) -> bool:
    """Après avoir joué (r,c), on teste ligne, colonne et zone correspondantes."""
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape,int]) -> List[Tuple[int,int,Shape]]:
    """Tous les coups valides (r,c,shape) pour `me`."""
    moves = []
    for sh in Shape:
        if my_counts.get(sh, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, sh, me):
                moves.append((r, c, sh))
    return moves

# =========================
# Heuristique Alpha–Beta
# =========================
LINES = []
for r in range(N):
    LINES.append([(r, c) for c in range(N)])
for c in range(N):
    LINES.append([(r, c) for r in range(N)])
LINES.extend(ZONES)

def line_diversity_score(board, cells) -> int:
    """Mesure de ‘proximité victoire’ (diversité de formes, couleur ignorée).
       NB: en Quantik, la condition de victoire ne dépend pas de la couleur.
       Ce terme est donc *symétrique*; les facteurs discriminants seront
       la mobilité et le centre.
    """
    pieces = [board[r][c] for (r, c) in cells]
    empt = sum(1 for p in pieces if p is None)
    if empt == 0:
        return 0
    shapes = {p.shape for p in pieces if p is not None}
    if empt == 1 and len(shapes) == 3:
        return 3
    if empt == 2 and len(shapes) == 2:
        return 1
    return 0

def mobility(board, who: Player, counts: Dict[Player, Dict[Shape,int]]) -> int:
    return len(generate_valid_moves(board, who, counts[who]))

def heuristic(board, me: Player, counts) -> int:
    """Évaluation simple mais efficace:
       • diversité (lignes/colonnes/zones proches de 4 différentes) [symétrique]
       • mobilité (différentielle)
       • léger bonus de centre
    """
    opp = other(me)
    div = sum(line_diversity_score(board, cells) for cells in LINES)
    my_mob  = mobility(board, me, counts)
    opp_mob = mobility(board, opp, counts)

    center_bonus = 0
    for (r, c) in CENTER:
        p = board[r][c]
        if p is None: continue
        center_bonus += (1 if p.player == me else -1)

    # Pondération empirique
    return 12*div + 2*(my_mob - opp_mob) + center_bonus

# =========================
# Alpha–Beta (avec deepening)
# =========================
class ABEngine:
    """Moteur Alpha–Beta avec iterative deepening et bon ordering (budget temps strict)."""
    def __init__(self, me: Player, time_limit: float):
        self.me = me
        self.time_limit = time_limit
        self.t0 = 0.0

    def time_up(self) -> bool:
        return (time.time() - self.t0) >= self.time_limit

    def order_moves(self, board, moves, side: Player) -> List[Tuple[int,int,Shape]]:
        """Ordonnancement: gains immédiats > blocages > centre."""
        scored = []
        opp = other(side)
        for (r, c, sh) in moves:
            # gain immédiat?
            board[r][c] = Piece(sh, side)
            win = check_victory_after(board, r, c)
            board[r][c] = None

            # blocage (grossier): si adv pourrait gagner ici (r,c) (indép. du shape exact)
            block = 0
            # (heuristique volontairement approximative)
            for sh2 in Shape:
                board[r][c] = Piece(sh2, opp)
                if check_victory_after(board, r, c):
                    block = 1
                    board[r][c] = None
                    break
                board[r][c] = None

            center = 1 if (r, c) in CENTER else 0
            scored.append(((1 if win else 0, block, center), (r, c, sh)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def search(self, board, counts, side: Player, depth: int, alpha: int, beta: int) -> Tuple[int, Optional[Tuple[int,int,Shape]]]:
        """Retourne (score, coup) avec garde de temps et sentinelles finies."""
        if self.time_up():
            return heuristic(board, self.me, counts), None

        moves = generate_valid_moves(board, side, counts[side])
        if not moves:
            # côté bloqué perd immédiatement
            return (LOSS_SCORE if side == self.me else WIN_SCORE), None

        # victoire immédiate
        for (r, c, sh) in moves:
            board[r][c] = Piece(sh, side)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (WIN_SCORE if side == self.me else LOSS_SCORE), (r, c, sh)
            board[r][c] = None

        if depth == 0:
            return heuristic(board, self.me, counts), None

        best_move = None
        moves = self.order_moves(board, moves, side)

        if side == self.me:
            value = ALPHA_INIT
            for (r, c, sh) in moves:
                board[r][c] = Piece(sh, side)
                counts[side][sh] -= 1
                score, _ = self.search(board, counts, other(side), depth-1, alpha, beta)
                counts[side][sh] += 1
                board[r][c] = None

                if score > value:
                    value = score
                    best_move = (r, c, sh)
                alpha = max(alpha, value)
                if alpha >= beta or self.time_up():
                    break
            return int(value), best_move
        else:
            value = BETA_INIT
            for (r, c, sh) in moves:
                board[r][c] = Piece(sh, side)
                counts[side][sh] -= 1
                score, _ = self.search(board, counts, other(side), depth-1, alpha, beta)
                counts[side][sh] += 1
                board[r][c] = None

                if score < value:
                    value = score
                    best_move = (r, c, sh)
                beta = min(beta, value)
                if alpha >= beta or self.time_up():
                    break
            return int(value), best_move

    def top_candidates(self, board, counts, k: int = 3) -> List[Tuple[int,int,Shape]]:
        """Deepening progressif pour extraire 1–3 meilleurs coups sous le budget."""
        self.t0 = time.time()
        root_moves = generate_valid_moves(board, self.me, counts[self.me])
        if not root_moves:
            return []

        # ordre initial simple vers centre
        root_moves.sort(key=lambda m: ((m[0], m[1]) in CENTER), reverse=True)
        best_move = None

        # deepening
        for depth in range(2, 7):
            if self.time_up():
                break
            score, move = self.search(board, counts, self.me, depth, ALPHA_INIT, BETA_INIT)
            if not self.time_up() and move is not None:
                best_move = move

        # réordonner une dernière fois si possible
        if not self.time_up():
            moves = generate_valid_moves(board, self.me, counts[self.me])
            ordered = self.order_moves(board, moves, self.me)
        else:
            ordered = root_moves

        # prioriser best_move si trouvé
        if best_move and best_move in ordered:
            ordered = [best_move] + [m for m in ordered if m != best_move]

        return ordered[:max(1, k)]

# =========================
# MCTS ciblé (sur candidats)
# =========================
def clone_board(board):
    return [row.copy() for row in board]

def clone_counts(counts):
    return {Player.PLAYER1: dict(counts[Player.PLAYER1]),
            Player.PLAYER2: dict(counts[Player.PLAYER2])}

def apply_move_local(board, counts, move, who: Player):
    r, c, sh = move
    board[r][c] = Piece(sh, who)
    counts[who][sh] -= 1

def rollout(board, counts, player_to_move: Player) -> Player:
    """Simulation aléatoire (biais centre) + arrêt sur gain immédiat."""
    cur = player_to_move
    while True:
        moves = generate_valid_moves(board, cur, counts[cur])
        if not moves:
            return other(cur)
        # gain immédiat
        for (r, c, sh) in moves:
            board[r][c] = Piece(sh, cur)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return cur
            board[r][c] = None
        # choix biaisé centre
        weights = [2.0 if (r, c) in CENTER else 1.0 for (r, c, _) in moves]
        x = random.random() * sum(weights)
        acc = 0.0
        pick = moves[-1]
        for w, m in zip(weights, moves):
            acc += w
            if x <= acc:
                pick = m; break
        apply_move_local(board, counts, pick, cur)
        r, c, _ = pick
        if check_victory_after(board, r, c):
            return cur
        cur = other(cur)

def mcts_among_candidates(board, counts, me: Player,
                          candidates: List[Tuple[int,int,Shape]],
                          time_limit: float) -> Optional[Tuple[int,int,Shape]]:
    """Exécute des mini-arbres MCTS indépendants par candidat racine
       et choisit celui au meilleur taux de victoire en simulation."""
    if not candidates:
        return None
    t0 = time.time()
    stats = {mv: [0, 0] for mv in candidates}  # mv -> [wins, plays]

    # Gain immédiat ?
    for mv in candidates:
        r, c, sh = mv
        b2 = clone_board(board)
        b2[r][c] = Piece(sh, me)
        if check_victory_after(b2, r, c):
            return mv

    # Boucle de simulations (budget minimal garanti)
    budget = max(0.01, float(time_limit))
    idx = 0
    while (time.time() - t0) < budget:
        mv = candidates[idx % len(candidates)]
        idx += 1

        b = clone_board(board)
        cnt = clone_counts(counts)
        # Joue le candidat puis simule
        apply_move_local(b, cnt, mv, me)
        if check_victory_after(b, mv[0], mv[1]):
            stats[mv][0] += 1
            stats[mv][1] += 1
            continue
        winner = rollout(b, cnt, other(me))
        stats[mv][1] += 1
        if winner == me:
            stats[mv][0] += 1

    # Choix par meilleur winrate (puis centre)
    best_mv = None
    best_wr = -1.0
    best_center = 0
    for mv, (w, n) in stats.items():
        wr = (w / n) if n > 0 else 0.0
        center = 1 if (mv[0], mv[1]) in CENTER else 0
        if (wr > best_wr) or (wr == best_wr and center > best_center):
            best_wr = wr
            best_center = center
            best_mv = mv
    return best_mv if best_mv else candidates[0]

# =========================
# Classe exportée
# =========================
class QuantikAI:
    """Contrôleur hybride: Alpha–Beta pour présélection, MCTS pour trancher."""
    def __init__(self, player: Player):
        self.me = player
        # Budget temps total par coup (ajustez librement)
        self.total_time = 1.6
        # Part de temps pour Alpha–Beta (reste pour MCTS)
        self.ab_fraction = 0.55
        # Nombre de candidats à affiner via MCTS
        self.top_k = 3

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        # Copie locale des compteurs pour la recherche
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }

        # Fallback anti-muet (probe): plateau vide + stocks complets => renvoyer un coup légal instantané
        is_board_empty = all(board[r][c] is None for r in range(N) for c in range(N))
        has_full_stock = all(counts[self.me].get(sh, 0) == 2 for sh in Shape)
        if is_board_empty and has_full_stock:
            root_moves = generate_valid_moves(board, self.me, counts[self.me])
            return random.choice(root_moves) if root_moves else None

        # 0) Aucun coup ?
        root_moves = generate_valid_moves(board, self.me, counts[self.me])
        if not root_moves:
            return None

        # 1) Alpha–Beta: extraire 1–3 meilleurs coups rapidement
        ab_time = max(0.02, self.total_time * self.ab_fraction)
        ab = ABEngine(self.me, ab_time)
        candidates = ab.top_candidates(board, counts, k=self.top_k)
        if not candidates:
            return None

        # Atalho / Raccourci: victoire immédiate ?
        for (r, c, sh) in candidates:
            board[r][c] = Piece(sh, self.me)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (r, c, sh)
            board[r][c] = None

        # 2) MCTS: départager/valider candidats dans le temps restant
        mcts_time = max(0.0, self.total_time - ab_time)
        if mcts_time < 0.01 or len(candidates) <= 1:
            # pas de temps / pas besoin → prendre le 1er candidat d'AB
            return candidates[0]

        choice = mcts_among_candidates(board, counts, self.me, candidates, mcts_time)
        return choice if choice else candidates[0]
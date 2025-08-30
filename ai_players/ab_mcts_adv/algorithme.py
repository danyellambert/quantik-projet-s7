# ai_players/ab_mcts_adv/algorithme.py
# =====================================================================
# IA Hybride “AlphaBeta+MCTS+” – version renforcée
#   - AI_NAME = "AlphaBeta+MCTS+"
#   - Classe exportée : QuantikAI(player)
#   - get_move(board, pieces_count) -> (row, col, Shape) | None
#
# Idée générale :
#  1) Bloc Alpha–Beta FORT (deepening + table de transposition + ordering)
#     pour lister 1–3 meilleurs coups avec leur score.
#  2) Si le meilleur est nettement supérieur au 2ᵉ (MARGIN_LOCK),
#     on **joue directement** ce coup (on ne lance pas MCTS).
#  3) Sinon, MCTS **ciblé** sur les candidats restants :
#     - rollouts avec détection de gain immédiat et **blocage des menaces**
#     - budget de simulations **pondéré** vers le meilleur candidat du AB
#
# Bénéfices :
#  • Quand AB “voit” une tactique gagnante, MCTS ne la dilue pas.
#  • Quand plusieurs coups sont proches, MCTS arbitre mieux les incertitudes.
#
# Notes d’implémentation :
#  • Le plateau global n’est **jamais** modifié durablement : on joue
#    des pièces seulement dans des clones, ou on remet immédiatement à None.
#  • Les compteurs de pièces sont copiés avant toute décrémentation.
#  • Les scores de fin sont **finis** (±50_000) pour éviter des casts/overflows.
# =====================================================================

from __future__ import annotations
import time, math, random
from typing import Optional, Tuple, List, Dict, Any

from core.types import Shape, Player, Piece

AI_NAME = "AlphaBeta+MCTS+"
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
CENTER = {(1,1), (1,2), (2,1), (2,2)}  # léger bonus positionnel

# LIGNES/SECTEURS pour heuristique diversité
LINES = []
for r in range(N): LINES.append([(r, c) for c in range(N)])
for c in range(N): LINES.append([(r, c) for r in range(N)])
LINES.extend(ZONES)

# --- Sentinelles finies (pas d'infini) ---
WIN_SCORE  = 50_000.0
LOSS_SCORE = -WIN_SCORE
ALPHA_INIT = -1_000_000.0
BETA_INIT  = +1_000_000.0


# =========================
# Utilitaires des règles
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
    Règle de Quantik : interdit de poser une forme si l’ADVERSAIRE
    a déjà cette même forme dans la ligne / la colonne / la zone.
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

def forms_all_different(pieces) -> bool:
    """Vrai si 4 cases remplies et 4 formes toutes différentes (couleur ignorée)."""
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r: int, c: int) -> bool:
    """Après avoir joué (r,c), teste uniquement la ligne, la colonne et la zone concernées."""
    # ligne
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    # colonne
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    # zone
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape, int]) -> List[Tuple[int,int,Shape]]:
    """Génère la liste des coups légaux (r,c,shape) pour `me` étant donné son stock."""
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves


# =========================
# Heuristique (diversité + mobilité + centre)
# =========================
def line_diversity_score(board, cells) -> int:
    """+3 si 3 formes diff. + 1 vide ; +1 si 2 formes diff. + 2 vides ; sinon 0."""
    pieces = [board[r][c] for (r, c) in cells]
    empties = sum(1 for p in pieces if p is None)
    shapes = {p.shape for p in pieces if p is not None}
    if empties == 1 and len(shapes) == 3: return 3
    if empties == 2 and len(shapes) == 2: return 1
    return 0

def mobility(board, who: Player, counts: Dict[Player, Dict[Shape,int]]) -> int:
    """Mobilité brute = nombre de coups légaux pour who."""
    return len(generate_valid_moves(board, who, counts[who]))

def heuristic(board, me: Player, counts: Dict[Player, Dict[Shape,int]]) -> float:
    """Évaluation statique rapide, centrée sur la règle du Quantik."""
    opp = other(me)
    my_div  = sum(line_diversity_score(board, cells) for cells in LINES)
    # NB : diversité symétrique sur la grille ; on ne différencie pas la couleur
    opp_div = my_div  # approximation rapide (équilibrée sur 4×4)
    my_mob, opp_mob = mobility(board, me, counts), mobility(board, opp, counts)

    center_bonus = 0
    for (r, c) in CENTER:
        p = board[r][c]
        if p is None: continue
        center_bonus += (1 if p.player == me else -1)

    return 12.0*(my_div - opp_div) + 2.0*(my_mob - opp_mob) + float(center_bonus)


# =========================
# Alpha–Beta fort (TT + deepening + ordering)
# =========================
class ABStrong:
    """Moteur Alpha–Beta “fort” (proche de AlphaBeta++)."""
    def __init__(self, me: Player, time_limit: float = 1.0):
        self.me = me
        self.opp = other(me)
        self.time_limit = time_limit
        self.t0 = 0.0
        # Table de transposition : key -> (depth, score, best_move)
        self.tt: Dict[Any, Tuple[int, float, Optional[Tuple[int,int,Shape]]]] = {}

    def time_up(self) -> bool:
        return (time.time() - self.t0) >= self.time_limit

    def board_key(self, board, side: Player) -> Tuple:
        """Clé sûre : tuple des (shape.value, player.value) + side_to_move."""
        cells = []
        for r in range(N):
            for c in range(N):
                p = board[r][c]
                cells.append(None if p is None else (p.shape.value, p.player.value))
        return (tuple(cells), side.value)

    def order_moves(self, board, moves, side: Player) -> List[Tuple[int,int,Shape]]:
        """Ordonnancement : gagnants > bloqueurs ≈ centre."""
        scored = []
        opp = other(side)
        for (r, c, sh) in moves:
            # gagnant immédiat ?
            board[r][c] = Piece(sh, side)
            is_win = check_victory_after(board, r, c)
            board[r][c] = None
            # bloque (approx) une victoire immédiate de l’adversaire ?
            block = 0
            if board[r][c] is None:
                for sh2 in Shape:
                    board[r][c] = Piece(sh2, opp)
                    if check_victory_after(board, r, c):
                        block = 1
                        board[r][c] = None
                        break
                    board[r][c] = None
            center = 1 if (r, c) in CENTER else 0
            scored.append(((1 if is_win else 0, block, center), (r, c, sh)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def search(self, board, counts, side: Player, depth: int,
               alpha: float, beta: float) -> Tuple[float, Optional[Tuple[int,int,Shape]]]:
        """Retourne (score, meilleur_coup) du point de vue de self.me."""
        if self.time_up():
            return heuristic(board, self.me, counts), None

        key = self.board_key(board, side)
        if key in self.tt:
            d_sto, sc_sto, mv_sto = self.tt[key]
            if d_sto >= depth:
                return sc_sto, mv_sto

        moves = generate_valid_moves(board, side, counts[side])
        if not moves:
            # côté sans coup => perd ; l’autre gagne
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
                    value = score; best_move = (r, c, sh)
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
                score, _ = self.search(board, counts, other(side), depth-1, alpha, beta)
                counts[side][sh] += 1
                board[r][c] = None

                if score < value:
                    value = score; best_move = (r, c, sh)
                beta = min(beta, value)
                if alpha >= beta or self.time_up():
                    break
            self.tt[key] = (depth, value, best_move)
            return value, best_move

    def top_candidates(self, board, counts, k: int = 3) -> Tuple[List[Tuple[int,int,Shape]], List[float]]:
        """Iterative deepening + TT → retourne (k meilleurs coups, scores)."""
        self.t0 = time.time()

        # ordre initial (léger) : centre d’abord
        base = generate_valid_moves(board, self.me, counts[self.me])
        if not base:
            return [], []
        base.sort(key=lambda m: ((m[0], m[1]) in CENTER), reverse=True)

        best_move = None
        scores_map: Dict[Tuple[int,int,Shape], float] = {}

        # deepening progressif
        for depth in range(2, 8):  # 2..7 sur 4×4 = déjà fort
            if self.time_up(): break
            score, mv = self.search(board, counts, self.me, depth, ALPHA_INIT, BETA_INIT)
            if self.time_up(): break
            if mv:
                best_move = mv
                scores_map[mv] = score

            # ré-ordonner d’après la passe courante
            ordered = self.order_moves(board, base, self.me)
            # option : “remplir” quelques scores légers des top si inconnus
            for cand in ordered[:6]:
                if cand not in scores_map:
                    # petite profondeur de validation (rapide)
                    board[cand[0]][cand[1]] = Piece(cand[2], self.me)
                    sc = heuristic(board, self.me, counts)
                    board[cand[0]][cand[1]] = None
                    scores_map[cand] = sc

            base = ordered

        # Prioriser le meilleur vu
        if best_move and best_move in base:
            base = [best_move] + [m for m in base if m != best_move]

        top = base[:max(1, k)]
        top_scores = [scores_map.get(m, 0.0) for m in top]
        return top, top_scores


# =========================
# MCTS ciblé (sur candidats)
# =========================
def clone_board(board):
    """Copie superficielle des lignes du plateau (sûr et rapide)."""
    return [row.copy() for row in board]

def clone_counts(counts):
    """Copie profonde des compteurs de pièces restantes."""
    return {
        Player.PLAYER1: dict(counts[Player.PLAYER1]),
        Player.PLAYER2: dict(counts[Player.PLAYER2]),
    }

def apply_move_local(board, counts, move, who: Player):
    """Applique un coup dans les clones locaux."""
    r, c, sh = move
    board[r][c] = Piece(sh, who)
    counts[who][sh] -= 1

def rollout(board, counts, player_to_move: Player) -> Player:
    """
    Simulation “éclairée” :
      1) jouer un gain immédiat si disponible
      2) sinon, **bloquer** une victoire immédiate adverse, si détectée
      3) sinon, choisir aléatoirement avec léger biais de centre
    """
    cur = player_to_move
    while True:
        moves = generate_valid_moves(board, cur, counts[cur])
        if not moves:
            return other(cur)

        # (1) gain immédiat
        for (r, c, sh) in moves:
            board[r][c] = Piece(sh, cur)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return cur
            board[r][c] = None

        # (2) bloquer menace adverse immédiate
        opp = other(cur)
        opp_moves = generate_valid_moves(board, opp, counts[opp])
        need_block = None
        for (rr, cc, sh2) in opp_moves:
            board[rr][cc] = Piece(sh2, opp)
            if check_victory_after(board, rr, cc):
                need_block = (rr, cc)
                board[rr][cc] = None
                break
            board[rr][cc] = None
        if need_block is not None:
            # joue sur la case critique si possible
            blocked = False
            for (r, c, sh) in moves:
                if (r, c) == need_block:
                    apply_move_local(board, counts, (r, c, sh), cur)
                    if check_victory_after(board, r, c):  # peut finir la sim
                        return cur
                    cur = opp
                    blocked = True
                    break
            if blocked:
                continue  # passe au tour adverse

        # (3) biais de centre
        weights = [2.0 if (r, c) in CENTER else 1.0 for (r, c, _) in moves]
        pick = moves[-1]
        x, acc, tot = random.random() * sum(weights), 0.0, sum(weights)
        for w, mv in zip(weights, moves):
            acc += w
            if x <= acc:
                pick = mv
                break
        apply_move_local(board, counts, pick, cur)
        r, c, _ = pick
        if check_victory_after(board, r, c):
            return cur
        cur = opp

def mcts_among_candidates(board, counts, me: Player,
                          candidates: List[Tuple[int,int,Shape]],
                          time_limit: float) -> Optional[Tuple[int,int,Shape]]:
    """
    MCTS **ciblé** :
      - chaque candidat est testé par playouts
      - budget de temps **pondéré** : 50% sur le meilleur du AB, 50% sur les autres
      - décision par meilleur taux de victoire
    """
    if not candidates:
        return None

    # victoire immédiate depuis la racine ?
    for (r, c, sh) in candidates:
        b2 = clone_board(board)
        b2[r][c] = Piece(sh, me)
        if check_victory_after(b2, r, c):
            return (r, c, sh)

    t0 = time.time()
    stats = {mv: [0, 0] for mv in candidates}  # mv -> [wins, plays]

    # Pondération de l’allocation des simulations
    if len(candidates) == 1:
        weights = [1.0]
    else:
        rest = len(candidates) - 1
        weights = [0.5] + [0.5 / rest] * rest  # 50% sur le 1er, 50% réparti

    while (time.time() - t0) < max(0.02, time_limit):
        # tirage de l’index du candidat selon les poids
        rnd, acc, idx = random.random(), 0.0, 0
        for k, w in enumerate(weights):
            acc += w
            if rnd <= acc:
                idx = k
                break
        mv = candidates[idx]

        b = clone_board(board)
        cnt = clone_counts(counts)
        apply_move_local(b, cnt, mv, me)
        r, c, _ = mv
        if check_victory_after(b, r, c):
            stats[mv][0] += 1
            stats[mv][1] += 1
            continue

        winner = rollout(b, cnt, other(me))
        stats[mv][1] += 1
        if winner == me:
            stats[mv][0] += 1

    # choix final : meilleur winrate, puis préférence centre
    best_mv, best_key = None, (-1.0, 0)
    for mv, (w, n) in stats.items():
        wr = (w / n) if n > 0 else 0.0
        key = (wr, 1 if (mv[0], mv[1]) in CENTER else 0)
        if key > best_key:
            best_key, best_mv = key, mv
    return best_mv if best_mv else candidates[0]


# =========================
# Classe exportée (API GUI)
# =========================
class QuantikAI:
    """
    Contrôleur hybride :
      • Alpha–Beta fort pour présélectionner les meilleurs coups (+ scores)
      • “Margin lock” : si le 1er est bien devant, on le joue sans MCTS
      • Sinon, MCTS ciblé et pondéré pour arbitrer
    """
    def __init__(self, player: Player):
        self.me = player
        # Budget temps total par coup (doit rester raisonnable pour passer le probe)
        self.total_time  = 1.30
        # Part du temps dédiée au AB fort (reste pour MCTS)
        self.ab_fraction = 0.70
        # Nombre de candidats à raffiner via MCTS
        self.top_k       = 2
        # Seuil “conviction AB” : si le meilleur surpasse le 2ᵉ d’au moins ceci,
        # on n’appelle pas MCTS.
        self.MARGIN_LOCK = 2000.0

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        # Copie locale des compteurs
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }

        # Coups racine
        root_moves = generate_valid_moves(board, self.me, counts[self.me])
        if not root_moves:
            return None

        # Fast-path : victoire immédiate
        for (r, c, sh) in root_moves:
            board[r][c] = Piece(sh, self.me)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (r, c, sh)
            board[r][c] = None

        # 1) Alpha–Beta fort : candidats + scores
        ab_time   = max(0.05, min(self.total_time * self.ab_fraction, self.total_time - 0.05))
        ab_engine = ABStrong(self.me, time_limit=ab_time)
        candidates, cand_scores = ab_engine.top_candidates(board, counts, k=self.top_k)

        if not candidates:
            # filet de sécurité
            return root_moves[0]

        # 1.a) si un candidat gagne immédiatement
        for (r, c, sh) in candidates:
            board[r][c] = Piece(sh, self.me)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (r, c, sh)
            board[r][c] = None

        # 1.b) margin lock (évite MCTS si AB est convaincu)
        best = cand_scores[0]
        second = cand_scores[1] if len(cand_scores) >= 2 else (best - 1.0)
        if best - second >= self.MARGIN_LOCK:
            # AB : “le 1ᵉ est nettement devant”
            legal_now = set(generate_valid_moves(board, self.me, counts[self.me]))
            return candidates[0] if candidates[0] in legal_now else (next(iter(legal_now)) if legal_now else None)

        # 2) MCTS ciblé sur les candidats
        mcts_time = max(0.02, self.total_time - ab_time)
        if mcts_time < 0.03 or len(candidates) == 1:
            return candidates[0]

        choice = mcts_among_candidates(board, counts, self.me, candidates, mcts_time)
        return choice if choice else candidates[0]
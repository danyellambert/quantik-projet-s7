# ai_players/genetic_adv/algorithme.py
# ======================================================================
# IA "Génétique++" pour QUANTIK
#  - Architecture compatible avec votre GUI (découverte auto)
#  - N'altère jamais l'état global: travaille sur des clones locaux
#  - Évolue des SÉQUENCES de coups (génome = liste de coups de longueur fixe)
#  - Fitness = taux de victoire estimé via playouts rapides (simulations)
#  - Élites + sélection tournois + croisement 1-point + mutations
#  - Respect strict des règles Quantik (validations à chaque simulation)
#
# Idée: on entraîne rapidement un "plan" de plusieurs demi-coups.
# La première action du meilleur plan est renvoyée à l'UI.
# ======================================================================

from __future__ import annotations
import random, time, math
from typing import Optional, Tuple, List, Dict

from core.ai_base import AIBase
from core.types import Shape, Player, Piece

AI_NAME    = "Génétique+"
AI_AUTHOR  = "Danyel Lambert"
AI_VERSION = "1.0"

# ======== Paramètres GA / Simulation (ajustables) ========
POP_SIZE        = 36    # taille population
ELITE_COUNT     = 6     # nombre d'élites conservées à chaque génération
TOURNEY_K       = 3     # taille de tournoi pour sélection
GENOME_LENGTH   = 6     # nb de demi-coups (plies) planifiés (moi+adversaire+...)
GEN_MAX         = 40    # nombre maximum de générations (soumis au TIME_LIMIT)
MUT_PROB_GENE   = 0.25  # probabilité de muter un gène (coup)
CXP_PROB        = 0.9   # probabilité d'appliquer un croisement
ROLLOUTS_PER_EVAL = 8   # playouts par individu pour estimer la fitness
TIME_LIMIT      = 1.3   # seconde(s) par décision

# Heuristiques légères
CENTER_CELLS = {(1,1), (1,2), (2,1), (2,2)}
CENTER_BIAS  = 1.0     # bonus central léger (utilisé dans choix adversaire)

# ======== Règles/ constantes plateau ========
N = 4
ALL_CELLS = [(r, c) for r in range(N) for c in range(N)]
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],  # TL
    [(0,2), (0,3), (1,2), (1,3)],  # TR
    [(2,0), (2,1), (3,0), (3,1)],  # BL
    [(2,2), (2,3), (3,2), (3,3)],  # BR
]

# =========================
# Utilitaires de règles
# =========================
def other(p: Player) -> Player:
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row: int, col: int, shape: Shape, me: Player) -> bool:
    """Règle Quantik: interdit si l’ADVERSAIRE a déjà cette forme
       sur la même ligne/colonne/zone."""
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
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0:
            continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves

def forms_all_different(pieces) -> bool:
    if any(p is None for p in pieces):
        return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board, r, c) -> bool:
    """Après avoir joué (r,c), tester ligne, colonne, zone; plus rapide."""
    if forms_all_different([board[r][cc] for cc in range(N)]): return True
    if forms_all_different([board[rr][c] for rr in range(N)]): return True
    z = zone_index(r, c)
    if forms_all_different([board[rr][cc] for (rr, cc) in ZONES[z]]): return True
    return False

# =========================
# Clonage / application locale
# =========================
def clone_board(board):  # copie peu coûteuse
    return [row.copy() for row in board]

def clone_counts(counts):
    return {
        Player.PLAYER1: dict(counts[Player.PLAYER1]),
        Player.PLAYER2: dict(counts[Player.PLAYER2]),
    }

def apply_local(board, counts, move, who: Player):
    """Applique un coup localement (sans affecter l’état global)."""
    r, c, sh = move
    board[r][c] = Piece(sh, who)
    counts[who][sh] -= 1

# =========================
# Politique d’adversaire pour playouts
# =========================
def opponent_policy(board, counts, who: Player) -> Optional[Tuple[int,int,Shape]]:
    """Politique adverse rapide:
       1) s’il existe un gain immédiat -> le jouer
       2) sinon, choisir aléatoire (biais centre) parmi les coups valides."""
    moves = generate_valid_moves(board, who, counts[who])
    if not moves:
        return None
    # gain immédiat ?
    for (r, c, sh) in moves:
        board[r][c] = Piece(sh, who)
        if check_victory_after(board, r, c):
            board[r][c] = None
            return (r, c, sh)
        board[r][c] = None
    # biais centre
    weights = [2.0 if (r, c) in CENTER_CELLS else 1.0 for (r, c, _) in moves]
    s = sum(weights)
    x, acc = random.random() * s, 0.0
    for w, mv in zip(weights, moves):
        acc += w
        if x <= acc:
            return mv
    return moves[-1]

# =========================
# GA: représentation & opérateurs
# =========================
def random_gene(board, counts, who: Player) -> Optional[Tuple[int,int,Shape]]:
    """Un gène = un coup valide pour `who`. Retourne None si bloqué."""
    moves = generate_valid_moves(board, who, counts[who])
    if not moves:
        return None
    return random.choice(moves)

def repair_gene(board, counts, gene, who: Player):
    """Si le gène n’est pas légal dans l’état courant, le remplace par un coup valide."""
    (r, c, sh) = gene
    if board[r][c] is None and counts[who].get(sh, 0) > 0 and is_valid_move(board, r, c, sh, who):
        return gene
    mv = random_gene(board, counts, who)
    return mv

def cx_one_point(a: List[Tuple[int,int,Shape]], b: List[Tuple[int,int,Shape]]):
    """Croisement 1-point simple (listes de même longueur)."""
    if len(a) != len(b) or len(a) == 0:
        return a[:], b[:]
    k = random.randint(1, len(a)-1)
    return a[:k] + b[k:], b[:k] + a[k:]

def mutate_gene(gene: Tuple[int,int,Shape]) -> Tuple[int,int,Shape]:
    """Mutation douce: bouger un peu la case OU changer la forme aléatoirement."""
    r, c, sh = gene
    mode = random.randint(0, 2)
    if mode == 0:
        # déplacer dans un voisinage (clip 0..3)
        dr, dc = random.choice([(-1,0),(1,0),(0,-1),(0,1),(0,0)])
        r = max(0, min(3, r + dr))
        c = max(0, min(3, c + dc))
    elif mode == 1:
        # nouvelle case aléatoire
        r, c = random.randint(0,3), random.randint(0,3)
    else:
        # nouvelle forme aléatoire
        sh = random.choice(list(Shape))
    return (r, c, sh)

def mutate_path(path: List[Tuple[int,int,Shape]]) -> List[Tuple[int,int,Shape]]:
    return [mutate_gene(g) if random.random() < MUT_PROB_GENE else g for g in path]

# =========================
# Fitness: playout guidé par un plan (génome)
# =========================
def evaluate_plan(start_board, start_counts, me: Player, genome: List[Tuple[int,int,Shape]]) -> float:
    """Exécute plusieurs playouts en tentant de suivre le plan `genome`:
       aux tours de `me`, on joue les gènes successifs si valides; sinon réparation.
       aux tours de l’adversaire, on utilise opponent_policy. Score = winrate."""
    wins = 0
    for _ in range(ROLLOUTS_PER_EVAL):
        b = clone_board(start_board)
        cnt = clone_counts(start_counts)
        cur = me
        gi  = 0  # index dans le génome

        # petite limite pour éviter boucles excessives (16 poses max)
        for _ply in range(16):
            if cur == me:
                # gène prévu
                mv = None
                if gi < len(genome):
                    mv = repair_gene(b, cnt, genome[gi], me)
                    gi += 1
                if mv is None:
                    mv = random_gene(b, cnt, me)
                if mv is None:
                    # bloqué -> l’autre gagne
                    break
                apply_local(b, cnt, mv, me)
                r, c, _ = mv
                if check_victory_after(b, r, c):
                    wins += 1
                    break
                cur = other(cur)
            else:
                # adversaire
                mv = opponent_policy(b, cnt, cur)
                if mv is None:
                    # adv bloqué -> je gagne
                    wins += 1
                    break
                apply_local(b, cnt, mv, cur)
                r, c, _ = mv
                if check_victory_after(b, r, c):
                    # défaite
                    break
                cur = other(cur)
    return wins / ROLLOUTS_PER_EVAL

# =========================
# IA exportée
# =========================
class QuantikAI(AIBase):
    """IA Génétique++ : renvoie la première action du meilleur plan évolué."""
    def __init__(self, player: Player):
        super().__init__(player)
        self.time_limit = TIME_LIMIT

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        t0 = time.time()

        # 1) Coups valides de base (utile pour quick-wins / fallback)
        root_moves = generate_valid_moves(board, self.me, pieces_count[self.me])
        if not root_moves:
            return None

        # 1.a) Atout: si je peux gagner immédiatement, joue ce coup
        for (r, c, sh) in root_moves:
            b2 = clone_board(board)
            b2[r][c] = Piece(sh, self.me)
            if check_victory_after(b2, r, c):
                return (r, c, sh)

        # 2) Population initiale = plans aléatoires valides (réparés à l’usage)
        def random_plan():
            b = clone_board(board)
            cnt = clone_counts(pieces_count)
            cur = self.me
            plan: List[Tuple[int,int,Shape]] = []
            for _ in range(GENOME_LENGTH):
                if cur == self.me:
                    mv = random_gene(b, cnt, cur)
                    if mv is None:
                        # si bloqué ici, on met un gène "placeholder" (réparé à l’éval)
                        mv = (0,0,random.choice(list(Shape)))
                    apply_local(b, cnt, mv, cur)
                    plan.append(mv)
                    if check_victory_after(b, mv[0], mv[1]):
                        # on peut compléter par des placeholders; peu importe, seule la 1ère action sera renvoyée
                        while len(plan) < GENOME_LENGTH:
                            plan.append(plan[-1])
                        break
                    cur = other(cur)
                else:
                    mv = opponent_policy(b, cnt, cur)
                    if mv is None:
                        # adv bloqué -> remplir placeholders
                        while len(plan) < GENOME_LENGTH:
                            plan.append((0,0,random.choice(list(Shape))))
                        break
                    apply_local(b, cnt, mv, cur)
                    if check_victory_after(b, mv[0], mv[1]):
                        # défaite potentielle pendant l’initialisation: on continue quand même
                        pass
                    cur = other(cur)
            if not plan:
                # garde-fou
                plan = [random.choice(root_moves)]
                while len(plan) < GENOME_LENGTH:
                    plan.append(plan[-1])
            return plan

        population = [random_plan() for _ in range(POP_SIZE)]

        # 3) Boucle d’évolution (sous contrainte temps)
        best_plan = population[0]
        best_fit  = -1.0

        gen = 0
        while gen < GEN_MAX and (time.time() - t0) < self.time_limit:
            # Évaluer
            scored = []
            for ind in population:
                fit = evaluate_plan(board, pieces_count, self.me, ind)
                # léger bonus central sur le premier move du plan
                (r0, c0, _) = ind[0]
                if (r0, c0) in CENTER_CELLS:
                    fit += CENTER_BIAS / 100.0
                scored.append((fit, ind))

            scored.sort(key=lambda x: x[0], reverse=True)

            # Mémoriser le meilleur courant
            if scored[0][0] > best_fit:
                best_fit, best_plan = scored[0]

            # Élites
            new_pop = [ind for (_, ind) in scored[:ELITE_COUNT]]

            # Sélection tournois + croisement + mutation
            def tourney():
                cand = random.sample(scored, k=min(TOURNEY_K, len(scored)))
                cand.sort(key=lambda x: x[0], reverse=True)
                return cand[0][1]

            while len(new_pop) < POP_SIZE:
                p1 = tourney()
                if random.random() < CXP_PROB:
                    p2 = tourney()
                    c1, c2 = cx_one_point(p1, p2)
                else:
                    c1, c2 = p1[:], p1[:]
                c1 = mutate_path(c1)
                c2 = mutate_path(c2)
                new_pop.append(c1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(c2)

            population = new_pop
            gen += 1

        # 4) Choisir le premier coup du meilleur plan, et vérifier la légalité courante
        first = best_plan[0]
        # Si premier gène illégal dans l’état réel -> réparer maintenant
        mv = repair_gene(board, pieces_count, first, self.me)
        if mv is None:
            # fallback: meilleur coup heuristique simple
            # (gagner ─ sinon centre ─ sinon random)
            for (r, c, sh) in root_moves:
                b2 = clone_board(board)
                b2[r][c] = Piece(sh, self.me)
                if check_victory_after(b2, r, c):
                    return (r, c, sh)
            root_moves.sort(key=lambda m: ((m[0], m[1]) in CENTER_CELLS), reverse=True)
            return root_moves[0] if root_moves else None
        return mv

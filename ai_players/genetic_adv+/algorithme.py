# ai_players/genetic_adv+/algorithme.py
# ======================================================================
# IA "Génétique++ (puro)" pour QUANTIK
# ----------------------------------------------------------------------
# Compatible GUI :
#   - AI_NAME = "Génétique++"
#   - Classe exportée : QuantikAI(player)
#   - get_move(board, pieces_count) -> (row, col, Shape) | None
#
# Principes :
#  • GA pur (pas d'Alpha–Beta, pas de MCTS).
#  • Le génome est un PLAN de plusieurs demi-coups (plies).
#  • Fitness = taux de victoire via playouts rapides.
#  • Deepening EVOLUTIF : augmente la longueur du plan si le temps permet.
#  • Politique adverse : gagner / BLOQUER un gain / biais centre.
#  • Cache de playouts par ÉTAT APRÈS LE 1er COUP (réutilisation massive).
#  • Diversité : fitness sharing par cellule du 1er coup.
#  • Micro hill-climb : améliore localement le 1er coup avant de répondre.
#  • Respect strict des règles; pas de mutation d’état global (clones).
# ======================================================================

from __future__ import annotations
import random, time
from typing import Optional, Tuple, List, Dict, Any

# Si votre projet expose AIBase, vous pouvez décommenter la ligne suivante
# et faire QuantikAI(AIBase). Sinon, la classe n'en dépend pas.
# from core.ai_base import AIBase
from core.types import Shape, Player, Piece

AI_NAME    = "Génétique++"
AI_AUTHOR  = "Danyel Lambert"
AI_VERSION = "2.0"

# ======== Paramètres GA / Simulation (ajustables) ========
POP_SIZE          = 36      # taille de population
ELITE_COUNT       = 6       # nombre d'élites conservées à chaque génération
TOURNEY_K         = 3       # taille de tournoi pour sélection
GEN_MAX           = 40      # bornes de générations (soumis au temps)
MUT_PROB_GENE     = 0.25    # probabilité de muter un gène
CXP_PROB          = 0.90    # probabilité de croisement 1-point
ROLLOUTS_PER_EVAL = 8       # playouts par individu pour estimer la fitness
TIME_LIMIT        = 1.30    # secondes par décision (budget total)
# Deepening évolutif : on tente des plans plus longs si temps restant
DEPTH_SCHEDULE    = [4, 6, 8]

# Heuristiques légères
CENTER_CELLS = {(1,1), (1,2), (2,1), (2,2)}

# ======== Cache de playouts (style TT, focalisé "après 1er coup") ========
ROLLOUT_CACHE_TARGET = 6      # nb d’échantillons avant d’exploiter surtout le cache
ROLLOUT_CACHE_MAX    = 50000  # taille max du cache (purgé de temps en temps)

# ======== Règles / constantes plateau ========
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
    return Player.PLAYER1 if p == Player.PLAYER2 else Player.PLAYER2

def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board, row: int, col: int, shape: Shape, me: Player) -> bool:
    """
    Règle Quantik utilisée ici :
      Interdit si l’ADVERSAIRE a déjà posé la même forme
      dans la même ligne/colonne/zone.
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
    return [row.copy() for row in board]

def clone_counts(counts):
    return {
        Player.PLAYER1: dict(counts[Player.PLAYER1]),
        Player.PLAYER2: dict(counts[Player.PLAYER2]),
    }

def apply_local(board, counts, move, who: Player):
    r, c, sh = move
    board[r][c] = Piece(sh, who)
    counts[who][sh] -= 1

# =========================
# Politique d’adversaire (gagne / bloque / centre)
# =========================
def opponent_policy(board, counts, who: Player) -> Optional[Tuple[int,int,Shape]]:
    moves = generate_valid_moves(board, who, counts[who])
    if not moves:
        return None
    # 1) gagner immédiatement ?
    for (r, c, sh) in moves:
        board[r][c] = Piece(sh, who)
        if check_victory_after(board, r, c):
            board[r][c] = None
            return (r, c, sh)
        board[r][c] = None
    # 2) BLOQUER un gain immédiat adverse si possible (forme dispo + coup légal)
    opp = other(who)
    for (r, c, _) in moves:
        for sh2 in Shape:
            if counts[opp].get(sh2, 0) <= 0:
                continue
            if not is_valid_move(board, r, c, sh2, opp):
                continue
            board[r][c] = Piece(sh2, opp)
            if check_victory_after(board, r, c):
                board[r][c] = None
                # jouer le bloc avec une de MES formes légales
                for sh_my in Shape:
                    if counts[who].get(sh_my, 0) > 0 and is_valid_move(board, r, c, sh_my, who):
                        return (r, c, sh_my)
            board[r][c] = None
    # 3) biais centre
    weights = [2.0 if (r, c) in CENTER_CELLS else 1.0 for (r, c, _) in moves]
    s = sum(weights); x = random.random()*s; acc = 0.0
    for w, mv in zip(weights, moves):
        acc += w
        if x <= acc:
            return mv
    return moves[-1]

# =========================
# GA: gènes/opérateurs
# =========================
def random_gene(board, counts, who: Player) -> Optional[Tuple[int,int,Shape]]:
    moves = generate_valid_moves(board, who, counts[who])
    if not moves:
        return None
    return random.choice(moves)

def repair_gene(board, counts, gene, who: Player):
    """Si le gène n’est pas légal dans l’état courant, le remplace par un coup valide."""
    if gene is None:
        return random_gene(board, counts, who)
    (r, c, sh) = gene
    if (
        0 <= r < 4 and 0 <= c < 4 and
        board[r][c] is None and
        counts[who].get(sh, 0) > 0 and
        is_valid_move(board, r, c, sh, who)
    ):
        return gene
    return random_gene(board, counts, who)

def cx_one_point(a: List[Tuple[int,int,Shape]], b: List[Tuple[int,int,Shape]]):
    if len(a) != len(b) or len(a) == 0:
        return a[:], b[:]
    k = random.randint(1, len(a)-1)
    return a[:k] + b[k:], b[:k] + a[k:]

def mutate_gene(gene: Tuple[int,int,Shape]) -> Tuple[int,int,Shape]:
    r, c, sh = gene
    mode = random.randint(0, 2)
    if mode == 0:
        # déplacement voisin
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
# Cache de playouts (clé stable)
# =========================
def counts_key(counts) -> Tuple[Tuple[int,...], Tuple[int,...]]:
    p1 = tuple(counts[Player.PLAYER1].get(sh, 0) for sh in Shape)
    p2 = tuple(counts[Player.PLAYER2].get(sh, 0) for sh in Shape)
    return (p1, p2)

def board_key(board) -> Tuple:
    cells = []
    for r in range(N):
        for c in range(N):
            p = board[r][c]
            cells.append(None if p is None else (p.shape.value, p.player.value))
    return tuple(cells)

def state_key(board, counts, cur: Player) -> Tuple:
    return (board_key(board), counts_key(counts), cur.value)

def cache_get(cache: Dict[Tuple, Tuple[float,int]], key: Tuple):
    v = cache.get(key)
    return v if v is not None else (None, 0)

def cache_update(cache: Dict[Tuple, Tuple[float,int]], key: Tuple, outcome: float):
    mu, n = cache.get(key, (0.0, 0))
    n2 = n + 1
    mu2 = mu + (outcome - mu) / n2
    cache[key] = (mu2, n2)
    # purge simple si trop gros
    if len(cache) > ROLLOUT_CACHE_MAX:
        # retire ~10% des entrées les moins échantillonnées
        victims = sorted(cache.items(), key=lambda kv: kv[1][1])[:len(cache)//10]
        for k, _ in victims:
            cache.pop(k, None)

# =========================
# Playout & évaluation avec cache (centré après 1er coup)
# =========================
def simulate_from_state(b, cnt, me: Player, cur: Player,
                        genome: List[Tuple[int,int,Shape]], gi_start: int) -> Tuple[float, int]:
    """
    Continue la partie depuis l’état (b,cnt) avec cur au trait.
    Suivit du plan 'genome' sur les tours de me à partir de gi_start.
    Retourne (win(0/1), plies_joués_à_partir_d_ici).
    """
    gi = gi_start
    for ply in range(1, 1+16):  # borne douce
        if cur == me:
            mv = repair_gene(b, cnt, genome[gi] if gi < len(genome) else None, me)
            gi += 1
            if mv is None:
                return 0.0, ply
            apply_local(b, cnt, mv, me)
            r, c, _ = mv
            if check_victory_after(b, r, c):
                return 1.0, ply
        else:
            mv = opponent_policy(b, cnt, cur)
            if mv is None:
                return 1.0, ply
            apply_local(b, cnt, mv, cur)
            r, c, _ = mv
            if check_victory_after(b, r, c):
                return 0.0, ply
        cur = other(cur)
    return 0.0, 16

def evaluate_plan_cached(start_board, start_counts, me: Player,
                         genome: List[Tuple[int,int,Shape]],
                         cache: Dict[Tuple, Tuple[float,int]]) -> float:
    """
    Évalue un plan en se focalisant sur l'impact du 1er coup.
    1) Joue le 1er gène (réparé) si possible.
    2) Clef de cache = état après ce 1er coup (adversaire au trait).
    3) Si le cache est suffisant, réutilise sa moyenne; sinon, simule et met à jour.
    4) Ajoute un léger bonus pour victoire rapide.
    """
    wins, bonus = 0.0, 0.0
    for _ in range(ROLLOUTS_PER_EVAL):
        b = clone_board(start_board)
        cnt = clone_counts(start_counts)

        # Déterminer et jouer le 1er coup de ce plan dans l'état réel
        first = repair_gene(b, cnt, genome[0] if genome else None, me)
        if first is None:
            # aucun coup possible pour moi -> perte
            continue
        apply_local(b, cnt, first, me)
        r0, c0, _ = first
        if check_victory_after(b, r0, c0):
            wins += 1.0
            bonus += 0.15  # petit bonus fixe si win immédiate
            continue

        cur = other(me)
        # Clef de cache après 1er coup (partage fort entre individus au même 1er move)
        key = state_key(b, cnt, cur)
        mu, n = cache_get(cache, key)
        if n >= ROLLOUT_CACHE_TARGET:
            wins += mu
            # pas d'info sur la vitesse de victoire -> pas de bonus
            continue

        # Sinon, on SIMULE à partir d'ici en suivant le reste du plan
        w, plies = simulate_from_state(b, cnt, me, cur, genome, gi_start=1)
        wins += w
        if w > 0.5:
            bonus += max(0, 16 - plies) * 0.01  # bonus vitesse doux
        cache_update(cache, key, w)

    # légère préférence centre au 1er coup (départage)
    if genome:
        (r0, c0, _) = genome[0]
        if (r0, c0) in CENTER_CELLS:
            bonus += 0.01

    return (wins / ROLLOUTS_PER_EVAL) + bonus

# =========================
# Plans aléatoires avec longueur donnée (pour deepening)
# =========================
def random_plan_with_len(board, counts, me: Player, gl: int) -> List[Tuple[int,int,Shape]]:
    b = clone_board(board)
    cnt = clone_counts(counts)
    cur = me
    plan: List[Tuple[int,int,Shape]] = []
    for _ in range(gl):
        if cur == me:
            mv = random_gene(b, cnt, cur) or (0,0,random.choice(list(Shape)))
            apply_local(b, cnt, mv, cur)
            plan.append(mv)
            cur = other(cur)
        else:
            mv = opponent_policy(b, cnt, cur)
            if mv is None:
                mv = (0,0,random.choice(list(Shape)))
            apply_local(b, cnt, mv, cur)
            cur = other(cur)
    if not plan:
        # garde-fou
        roots = generate_valid_moves(board, me, counts[me])
        if roots:
            plan = [random.choice(roots)] * gl
        else:
            plan = [(0,0,random.choice(list(Shape)))] * gl
    return plan

# =========================
# Classe exportée
# =========================
# class QuantikAI(AIBase):
class QuantikAI:
    """IA Génétique++ (puro) : retourne le 1er coup du meilleur plan évolué."""
    def __init__(self, player: Player):
        # super().__init__(player)
        self.me = player
        self.time_limit = TIME_LIMIT
        # cache partagé entre coups d'une même partie (peut être vidé si souhaité)
        self.rollout_cache: Dict[Tuple, Tuple[float,int]] = {}

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        t0 = time.time()

        # 1) Coups légaux à la racine
        root_moves = generate_valid_moves(board, self.me, pieces_count[self.me])
        if not root_moves:
            return None

        # 1.a) Win immédiate si dispo
        for (r, c, sh) in root_moves:
            b2 = clone_board(board)
            b2[r][c] = Piece(sh, self.me)
            if check_victory_after(b2, r, c):
                return (r, c, sh)

        best_plan: Optional[List[Tuple[int,int,Shape]]] = None
        best_fit: float = -1.0

        # 2) Deepening évolutif: GL croissant tant qu'il reste du temps
        for GL in DEPTH_SCHEDULE:
            if (time.time() - t0) >= self.time_limit:
                break

            # population initiale
            population = [random_plan_with_len(board, pieces_count, self.me, GL)
                          for _ in range(POP_SIZE)]
            gen = 0

            while gen < GEN_MAX and (time.time() - t0) < self.time_limit:
                # --- ÉVALUATION ---
                scored: List[Tuple[float, List[Tuple[int,int,Shape]]]] = []
                for ind in population:
                    if (time.time() - t0) >= self.time_limit:
                        break
                    fit = evaluate_plan_cached(board, pieces_count, self.me, ind, self.rollout_cache)
                    scored.append((fit, ind))

                if not scored:
                    break
                scored.sort(key=lambda x: x[0], reverse=True)

                # mémoriser meilleur global
                if scored[0][0] > best_fit:
                    best_fit, best_plan = scored[0]

                # --- DIVERSITÉ (fitness sharing par cellule du 1er coup) ---
                first_to_fit: Dict[Tuple[int,int], List[float]] = {}
                for f, ind in scored:
                    r0, c0, _ = ind[0]
                    first_to_fit.setdefault((r0, c0), []).append(f)
                penalty = {k: 0.002*(len(v)-1) for k,v in first_to_fit.items()}
                scored = [(f - penalty[(ind[0][0], ind[0][1])], ind) for (f, ind) in scored]
                scored.sort(key=lambda x: x[0], reverse=True)

                # --- ÉLITES ---
                new_pop = [ind for (f, ind) in scored[:ELITE_COUNT]]

                # --- Tournois + croisement + mutation (sous budget temps) ---
                def tourney():
                    k = min(TOURNEY_K, len(scored))
                    cand = random.sample(scored, k=k)
                    cand.sort(key=lambda x: x[0], reverse=True)
                    return cand[0][1]

                while len(new_pop) < POP_SIZE and (time.time() - t0) < self.time_limit:
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

            # heuristique d'arrêt : si presque sans temps, on s'arrête là
            if (time.time() - t0) >= self.time_limit * 0.95:
                break

        # 3) Choisir le 1er coup du meilleur plan et MICRO-OPTIMISER localement
        #    (hill-climb sur cases voisines, évaluées par la même fitness)
        mv: Optional[Tuple[int,int,Shape]] = None
        if best_plan:
            base = repair_gene(board, pieces_count, best_plan[0], self.me)
            if base is not None:
                candidates = [base]
                r, c, sh = base
                for (dr, dc) in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = max(0, min(3, r+dr)), max(0, min(3, c+dc))
                    if is_valid_move(board, rr, cc, sh, self.me) and pieces_count[self.me].get(sh,0)>0:
                        candidates.append((rr, cc, sh))

                best_loc, best_loc_fit = base, -1.0
                for cand in candidates:
                    if (time.time() - t0) >= self.time_limit:
                        break
                    trial = [cand] + best_plan[1:]
                    f = evaluate_plan_cached(board, pieces_count, self.me, trial, self.rollout_cache)
                    if f > best_loc_fit:
                        best_loc_fit, best_loc = f, cand
                mv = best_loc

        # 4) Filets de sécurité
        if mv is None:
            # (a) win immédiate (déjà testé au début, on re-checke par prudence)
            for (r, c, sh) in root_moves:
                b2 = clone_board(board); b2[r][c] = Piece(sh, self.me)
                if check_victory_after(b2, r, c):
                    return (r, c, sh)
            # (b) préférence centre
            root_moves.sort(key=lambda m: ((m[0], m[1]) in CENTER_CELLS), reverse=True)
            return root_moves[0] if root_moves else None

        return mv

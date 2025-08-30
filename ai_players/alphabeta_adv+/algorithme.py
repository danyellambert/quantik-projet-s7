# ai_players/alphabeta_plus/algorithme.py
# ================================================================
# IA Quantik “AlphaBeta++” (melhorada)
# - Minimax com poda alpha–beta
# - Iterative deepening + aspiration windows
# - Ordenação forte: PV/TT, win, block, killers, history, centro
# - TT estável: (board, side, counts)
# - Heurística tático-posicional (ameaças imediatas, mobilidade, centro)
# - Extensão de ameaça (quiescence light) quando há mate-em-1
# - Sentinelas finitas; sem mutação de estado global
# ================================================================

from __future__ import annotations
import time, math, random
from typing import Optional, Tuple, List, Dict, Any

from core.types import Shape, Player, Piece

AI_NAME = "AlphaBeta++"
AI_AUTHOR = ""

# --- Constantes de tabuleiro ---
N = 4
ALL_CELLS = [(r, c) for r in range(N) for c in range(N)]
ZONES = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(2,2), (2,3), (3,2), (3,3)],
]
LINES: List[List[Tuple[int,int]]] = []
for r in range(N): LINES.append([(r, c) for c in range(N)])
for c in range(N): LINES.append([(r, c) for r in range(N)])
LINES.extend(ZONES)

CENTER_CELLS = {(1,1), (1,2), (2,1), (2,2)}

# --- Sentinelas finitas ---
WIN_SCORE   = 50_000.0
LOSS_SCORE  = -WIN_SCORE
ALPHA_INIT  = -1_000_000.0
BETA_INIT   = +1_000_000.0

# --- Pesos heurísticos ---
W_THREAT = 5000.0   # ameaças imediatas (muito alto para tática)
W_MOB    = 2.0
W_CENTER = 1.0

# =========================
# Regras/utilitários Quantik
# =========================
def zone_index(r: int, c: int) -> int:
    if r < 2 and c < 2:  return 0
    if r < 2 and c >= 2: return 1
    if r >= 2 and c < 2: return 2
    return 3

def is_valid_move(board: List[List[Optional[Piece]]],
                  row: int, col: int,
                  shape: Shape, me: Player) -> bool:
    """
    Legal se a casa está vazia E o ADVERSÁRIO não tem a mesma forma
    na mesma linha/coluna/zona (regra de Quantik usada no projeto).
    """
    if board[row][col] is not None: return False
    # linha
    for c in range(N):
        p = board[row][c]
        if p is not None and p.shape == shape and p.player != me: return False
    # coluna
    for r in range(N):
        p = board[r][col]
        if p is not None and p.shape == shape and p.player != me: return False
    # zona
    z = zone_index(row, col)
    for (rr, cc) in ZONES[z]:
        p = board[rr][cc]
        if p is not None and p.shape == shape and p.player != me: return False
    return True

def forms_all_different(pieces: List[Optional[Piece]]) -> bool:
    if any(p is None for p in pieces): return False
    shapes = {p.shape for p in pieces}
    return len(shapes) == 4

def check_victory_after(board: List[List[Optional[Piece]]], row: int, col: int) -> bool:
    if forms_all_different([board[row][c] for c in range(N)]): return True
    if forms_all_different([board[r][col] for r in range(N)]): return True
    z = zone_index(row, col)
    if forms_all_different([board[r][c] for (r, c) in ZONES[z]]): return True
    return False

def generate_valid_moves(board, me: Player, my_counts: Dict[Shape, int]) -> List[Tuple[int,int,Shape]]:
    moves = []
    for shape in Shape:
        if my_counts.get(shape, 0) <= 0: continue
        for (r, c) in ALL_CELLS:
            if board[r][c] is None and is_valid_move(board, r, c, shape, me):
                moves.append((r, c, shape))
    return moves

# =========================
# Heurística: tática + posição
# =========================
def count_immediate_wins(board, counts, who: Player) -> int:
    """Quantos mates-em-1 `who` possui aqui."""
    n = 0
    moves = generate_valid_moves(board, who, counts[who])
    for (r, c, sh) in moves:
        board[r][c] = Piece(sh, who)
        if check_victory_after(board, r, c): n += 1
        board[r][c] = None
    return n

def mobility(board, who: Player, counts) -> int:
    return len(generate_valid_moves(board, who, counts[who]))

def center_control(board, me: Player) -> int:
    sc = 0
    for (r, c) in CENTER_CELLS:
        p = board[r][c]
        if p is None: continue
        sc += 1 if p.player == me else -1
    return sc

def heuristic(board, me: Player, counts) -> float:
    opp = Player.PLAYER1 if me == Player.PLAYER2 else Player.PLAYER2
    my_th  = count_immediate_wins(board, counts, me)
    opp_th = count_immediate_wins(board, counts, opp)
    # Nota: ameaças têm peso altíssimo; demais termos refinam
    mob    = mobility(board, me, counts) - mobility(board, opp, counts)
    center = center_control(board, me)
    return W_THREAT*(my_th - opp_th) + W_MOB*mob + W_CENTER*center

# =========================
# Busca Alpha–Beta com deepening
# =========================
class Searcher:
    def __init__(self, me: Player, time_limit: float = 1.35):
        self.me = me
        self.opp = Player.PLAYER1 if me == Player.PLAYER2 else Player.PLAYER2
        self.time_limit = time_limit
        self.t0 = 0.0
        # TT: key -> (depth, score, best_move)
        self.tt: Dict[Any, Tuple[int, float, Optional[Tuple[int,int,Shape]]]] = {}
        # history heuristic e killers
        self.history: Dict[Tuple[int,int,Shape], int] = {}
        self.killers: Dict[int, List[Tuple[int,int,Shape]]] = {}

    # ---- tempo ----
    def time_up(self) -> bool:
        return (time.time() - self.t0) >= self.time_limit

    # ---- chave TT estável (inclui counts) ----
    def counts_key(self, counts: Dict[Player, Dict[Shape,int]]) -> Tuple[Tuple[int,...], Tuple[int,...]]:
        p1 = tuple(counts[Player.PLAYER1].get(sh, 0) for sh in Shape)
        p2 = tuple(counts[Player.PLAYER2].get(sh, 0) for sh in Shape)
        return (p1, p2)

    def board_key(self, board, side: Player, counts) -> Tuple:
        cells = []
        for r in range(N):
            for c in range(N):
                p = board[r][c]
                cells.append(None if p is None else (p.shape.value, p.player.value))
        return (tuple(cells), side.value, self.counts_key(counts))

    # ---- utilitários táticos ----
    def has_immediate_win(self, board, counts, side: Player) -> Optional[Tuple[int,int,Shape]]:
        for (r, c, sh) in generate_valid_moves(board, side, counts[side]):
            board[r][c] = Piece(sh, side)
            if check_victory_after(board, r, c):
                board[r][c] = None
                return (r, c, sh)
            board[r][c] = None
        return None

    def blocks_opponent_win(self, board, counts, side: Player, mv: Tuple[int,int,Shape]) -> int:
        """Retorna 1 se jogar mv impede uma vitória imediata do oponente."""
        opp = self.opp if side == self.me else self.me
        r, c, sh = mv
        # se ao colocar minha peça aqui, aquela casa não fica livre pro adv ganhar, conta.
        # ou se eu usar a mesma forma que daria a vitória do adversário e a torne ilegal.
        # Teste focado: existia vitória do adv na mesma casa?
        # Procuramos todas as vitórias do adv e vemos se este mv as anula.
        # (otimização leve: testar apenas a própria célula)
        for sh2 in Shape:
            if counts[opp].get(sh2, 0) <= 0: continue
            if not is_valid_move(board, r, c, sh2, opp): continue
            board[r][c] = Piece(sh2, opp)
            win = check_victory_after(board, r, c)
            board[r][c] = None
            if win:
                # Se eu jogo mv na mesma (r,c), bloqueio
                return 1 if (mv[0] == r and mv[1] == c) else 0
        return 0

    # ---- ordenação de lances ----
    def order_moves(self, board, counts, side: Player, moves, depth: int) -> List[Tuple[int,int,Shape]]:
        scored = []
        opp = self.opp if side == self.me else self.me

        # PV/TT primeiro, se existir
        tt_move = None
        key = self.board_key(board, side, counts)
        if key in self.tt and self.tt[key][2] is not None:
            tt_move = self.tt[key][2]

        killer_list = self.killers.get(depth, [])

        for mv in moves:
            r, c, sh = mv
            # vitória imediata?
            board[r][c] = Piece(sh, side)
            win = check_victory_after(board, r, c)
            board[r][c] = None

            # bloqueio válido (como no AB+)
            block = 0
            for sh2 in Shape:
                if counts[opp].get(sh2, 0) <= 0: continue
                if not is_valid_move(board, r, c, sh2, opp): continue
                board[r][c] = Piece(sh2, opp)
                if check_victory_after(board, r, c):
                    block = 1
                    board[r][c] = None
                    break
                board[r][c] = None

            center = 1 if (r, c) in CENTER_CELLS else 0
            hist = self.history.get(mv, 0)
            is_killer = 1 if mv in killer_list else 0
            is_tt = 1 if tt_move is not None and mv == tt_move else 0

            key_tuple = (is_tt, 1 if win else 0, block, is_killer, hist, center)
            scored.append((key_tuple, mv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ---- busca ----
    def search(self, board, counts, side: Player, depth: int,
               alpha: float, beta: float, ply: int) -> Tuple[float, Optional[Tuple[int,int,Shape]]]:
        if self.time_up():
            return heuristic(board, self.me, counts), None

        # Extensão de ameaça (quiescence light)
        # se qualquer lado tem mate-em-1 aqui, estende uma plie a mais
        ext = 0
        if depth > 0:
            if self.has_immediate_win(board, counts, side) is not None: ext = 1
            elif self.has_immediate_win(board, counts, self.opp if side == self.me else self.me) is not None: ext = 1

        key = self.board_key(board, side, counts)
        if key in self.tt:
            stored_depth, stored_score, stored_move = self.tt[key]
            if stored_depth >= depth:
                return stored_score, stored_move

        moves = generate_valid_moves(board, side, counts[side])
        if not moves:
            return (LOSS_SCORE if side == self.me else WIN_SCORE), None

        # Vitória imediata: atalho também para TT
        for (r, c, sh) in moves:
            board[r][c] = Piece(sh, side)
            if check_victory_after(board, r, c):
                board[r][c] = None
                score = WIN_SCORE if side == self.me else LOSS_SCORE
                self.tt[key] = (depth, score, (r, c, sh))
                return score, (r, c, sh)
            board[r][c] = None

        if depth + ext == 0:
            return heuristic(board, self.me, counts), None

        ordered = self.order_moves(board, counts, side, moves, ply)

        best_move = None

        if side == self.me:
            value = ALPHA_INIT
            for mv in ordered:
                r, c, sh = mv
                board[r][c] = Piece(sh, side)
                counts[side][sh] -= 1

                score, _ = self.search(board, counts, self.opp, depth - 1 + ext, alpha, beta, ply + 1)

                counts[side][sh] += 1
                board[r][c] = None

                if score > value:
                    value = score
                    best_move = mv
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    # beta-cutoff: killers + history
                    ks = self.killers.setdefault(ply, [])
                    if mv not in ks:
                        if len(ks) < 2: ks.append(mv)
                        else: ks[0] = mv
                    self.history[mv] = self.history.get(mv, 0) + (depth + 1) * 2
                    break

            self.tt[key] = (depth, value, best_move)
            return value, best_move
        else:
            value = BETA_INIT
            for mv in ordered:
                r, c, sh = mv
                board[r][c] = Piece(sh, side)
                counts[side][sh] -= 1

                score, _ = self.search(board, counts, self.me, depth - 1 + ext, alpha, beta, ply + 1)

                counts[side][sh] += 1
                board[r][c] = None

                if score < value:
                    value = score
                    best_move = mv
                if value < beta:
                    beta = value
                if alpha >= beta:
                    ks = self.killers.setdefault(ply, [])
                    if mv not in ks:
                        if len(ks) < 2: ks.append(mv)
                        else: ks[0] = mv
                    self.history[mv] = self.history.get(mv, 0) + (depth + 1) * 2
                    break

            self.tt[key] = (depth, value, best_move)
            return value, best_move

    # ---- deepening com aspiration windows ----
    def choose(self, board, counts) -> Optional[Tuple[int,int,Shape]]:
        self.t0 = time.time()
        self.killers.clear()
        self.history.clear()

        best_move = None
        prev_score = 0.0
        # profundidade alvo razoável para 4x4
        for depth in range(2, 10):
            if self.time_up(): break

            # janela de aspiração em torno do score anterior
            window = 250.0
            alpha = max(ALPHA_INIT, prev_score - window)
            beta  = min(BETA_INIT,  prev_score + window)

            score, move = self.search(board, counts, self.me, depth, alpha, beta, ply=0)
            if self.time_up(): break

            # falha da janela => re-buscar com janela cheia
            if score <= alpha:
                score, move = self.search(board, counts, self.me, depth, ALPHA_INIT, BETA_INIT, ply=0)
            elif score >= beta:
                score, move = self.search(board, counts, self.me, depth, ALPHA_INIT, BETA_INIT, ply=0)

            if move is not None:
                best_move = move
                prev_score = score

        if best_move is None:
            moves = generate_valid_moves(board, self.me, counts[self.me])
            if not moves: return None
            moves.sort(key=lambda m: ((m[0], m[1]) in CENTER_CELLS), reverse=True)
            return moves[0]
        return best_move

# =========================
# Classe exportada
# =========================
class QuantikAI:
    def __init__(self, player: Player):
        self.me = player
        self.time_limit = 1.25  # s / lance

    def get_move(self, board, pieces_count) -> Optional[Tuple[int,int,Shape]]:
        counts = {
            Player.PLAYER1: dict(pieces_count[Player.PLAYER1]),
            Player.PLAYER2: dict(pieces_count[Player.PLAYER2]),
        }
        engine = Searcher(self.me, time_limit=self.time_limit)
        return engine.choose(board, counts)
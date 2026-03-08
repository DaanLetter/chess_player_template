import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_move(self, fen: str) -> Optional[str]:
        pass


class TransformerPlayer(Player):
    """
    Tiny LM baseline chess player.

    REQUIRED:
        Subclasses chess_tournament.players.Player
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "PabloSnackbarChessBot (student ID: 2912864)",
        model_id: str = "PabloSnackbar/chess-transformer4.0",
        temperature: float = 1,
        max_new_tokens: int = 8,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"Position: {fen} Best move:"

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    # Main API
    # -------------------------

    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load_model()
        except Exception:
            return self._random_legal(fen)

        prompt = f"Position: {fen} Best move:"
        board = chess.Board(fen)

        #check for immediate checkmate
        checkmate_move = self._find_checkmate(board)
        if checkmate_move:
            return checkmate_move

        #try model up to 5 times
        for attempt in range(5):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True if attempt > 0 else False,  # greedy first, then sample
                        temperature=0.7 if attempt > 0 else self.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = decoded[len(prompt):].strip()

                if len(generated) >= 5 and generated[4] in 'qrbn' and generated[3] in '18':
                    predicted = generated[:5]
                elif len(generated) >= 4:
                    predicted = generated[:4]
                else:
                    continue

                if chess.Move.from_uci(predicted) in board.legal_moves and self._avoids_immediate_checkmate(board, predicted):
                    return predicted

            except Exception:
                continue
        # fallback - at least avoid checkmate if possible
        safe_moves = [m.uci() for m in board.legal_moves
                    if self._avoids_immediate_checkmate(board, m.uci())]
        if safe_moves:
            return random.choice(safe_moves)

        return self._random_legal(fen)

    #extra bits and pieces

    def _find_checkmate(self, board: chess.Board) -> Optional[str]:
        """if there's a checkmate move available, take it immediately"""
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()
        return None

    def _avoids_immediate_checkmate(self, board: chess.Board, move_uci: str) -> bool:
        """check if this move leaves opponent with immediate checkmate"""
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        for opponent_move in board.legal_moves:
            board.push(opponent_move)
            if board.is_checkmate():
                board.pop()
                board.pop()
                return False  # this move is bad, opponent can checkmate
            board.pop()
        board.pop()
        return True  # move is safe


"""
Minimal Chess Utilities for Probe Visualization

This module contains only the chess utility functions needed for probe visualization,
extracted from the chess_llm_interpretability project to make the API self-contained.
"""

import chess
import torch
from torch.nn import functional as F
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
from jaxtyping import Int
from torch import Tensor

# Mapping of chess pieces to integers
PIECE_TO_INT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

INT_TO_PIECE = {value: key for key, value in PIECE_TO_INT.items()}

PIECE_TO_ONE_HOT_MAPPING = {
    -6: 0,
    -5: 1,
    -4: 2,
    -3: 3,
    -2: 4,
    -1: 5,
    0: 6,
    1: 7,
    2: 8,
    3: 9,
    4: 10,
    5: 11,
    6: 12,
}

BLANK_INDEX = PIECE_TO_ONE_HOT_MAPPING[0]
ONE_HOT_TO_PIECE_MAPPING = {value: key for key, value in PIECE_TO_ONE_HOT_MAPPING.items()}

class PlayerColor(Enum):
    WHITE = "White"
    BLACK = "Black"

def board_to_piece_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    The 8x8 array should tell what piece is on each square. A white pawn could be 1, a black pawn could be -1, etc.
    Blank squares should be 0.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""

    # Because state_RR is initialized to all 0s, we only need to change the values of the pieces
    state_RR = torch.zeros((8, 8), dtype=torch.int)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_value = PIECE_TO_INT[piece.piece_type]
            # Multiply by -1 if the piece is black
            if piece.color == chess.BLACK:
                piece_value *= -1
            state_RR[i // 8, i % 8] = piece_value

    return state_RR

def pgn_string_to_board(pgn_string: str) -> chess.Board:
    """Convert a PGN string to a chess.Board object.
    We are making an assumption that the PGN string is in this format:
    ;1.e4 e5 2. or ;1.e4 e5 2.Nf3"""
    board = chess.Board()
    for move in pgn_string.split():
        if "." in move:
            move = move.split(".")[1]
        if move == "":
            continue
        board.push_san(move)
    return board

def find_dots_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]
    return indices

def state_stack_to_one_hot(
    num_modes: int,
    num_rows: int,
    num_cols: int,
    min_val: int,
    max_val: int,
    device: torch.device,
    state_stack_MBLRR: torch.Tensor,
    user_mapping: Optional[dict[int, int]] = None,
) -> torch.Tensor:
    """Input shape: assert(state_stacks_all_chars.shape) == (modes, sample_size, game_length, rows, cols)
    Output shape: assert(state_stacks_one_hot.shape) == (modes, sample_size, game_length, rows, cols, one_hot_range)
    """
    range_size = max_val - min_val + 1

    mapping = {}
    if user_mapping:
        mapping = user_mapping
        min_val = min(mapping.values())
        max_val = max(mapping.values())
        range_size = max_val - min_val + 1
    else:
        for val in range(min_val, max_val + 1):
            mapping[val] = val - min_val

    # Initialize the one-hot tensor
    one_hot_MBLRRC = torch.zeros(
        state_stack_MBLRR.shape[0],  # num modes
        state_stack_MBLRR.shape[1],  # num games
        state_stack_MBLRR.shape[2],  # num moves
        num_rows,
        num_cols,
        range_size,
        device=device,
        dtype=torch.int8,
    )

    for val in mapping:
        one_hot_MBLRRC[..., mapping[val]] = state_stack_MBLRR == val

    return one_hot_MBLRRC

@dataclass
class Config:
    min_val: int
    max_val: int
    custom_board_state_function: callable
    linear_probe_name: str
    custom_indexing_function: callable = find_dots_indices
    num_rows: int = 8
    num_cols: int = 8
    levels_of_interest: Optional[list[int]] = None
    column_name: str = None
    probing_for_skill: bool = False
    # pos_start indexes into custom_indexing_function. Example: if pos_start = 25, for find_dots_indices, selects everything after the first 25 moves
    pos_start: int = 0
    # If pos_end is None, it's set to the length of the shortest game in construct_linear_probe_data()
    pos_end: Optional[int] = None
    player_color: PlayerColor = PlayerColor.WHITE
    othello: bool = False 
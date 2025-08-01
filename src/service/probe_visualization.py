"""
Probe Visualization Service

This module provides functionality to generate heatmaps showing how the model
internally represents the chess board state using trained linear probes.
"""

import os
import sys
import torch
import numpy as np
import chess
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import fancy_einsum with fallback
try:
    from fancy_einsum import einsum
except ImportError:
    einsum = torch.einsum

# Import local chess utilities
try:
    # Import from the local chess_utils_minimal module
    from . import chess_utils_minimal as chess_utils
    from .chess_utils_minimal import Config
except ImportError as e:
    chess_utils = None
    Config = None

# Import transformer_lens
try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError:
    HookedTransformer = None
    HookedTransformerConfig = None

# Piece mappings (matching the notebook exactly)
INT_TO_CHAR = {
    -6: "♔",  # Black king
    -5: "♕",  # Black queen
    -4: "♖",  # Black rook
    -3: "♗",  # Black bishop
    -2: "♘",  # Black knight
    -1: "♙",  # Black pawn
    0: ".",   # Empty
    1: "♟",   # White pawn
    2: "♞",   # White knight
    3: "♝",   # White bishop
    4: "♜",   # White rook
    5: "♛",   # White queen
    6: "♚",   # White king
}

PIECE_TO_ONE_HOT_MAPPING = {
    -6: 0, -5: 1, -4: 2, -3: 3, -2: 4, -1: 5, 0: 6,
    1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12
}

PIECE_TYPE_MAPPING = {
    'white_pawns': 1,
    'white_knights': 2,
    'white_bishops': 3,
    'white_rooks': 4,
    'white_queens': 5,
    'white_kings': 6,
    'black_pawns': -1,
    'black_knights': -2,
    'black_bishops': -3,
    'black_rooks': -4,
    'black_queens': -5,
    'black_kings': -6,
    'empty_squares': 0
}

class ProbeVisualizationService:
    """Service for generating probe visualizations"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.probes = {}
        self.meta = None
        self.encode = None
        self.decode = None
        self.initialized = False
        self.config = None
    
    async def initialize(self):
        """Initialize the probe visualization service"""
        if self.initialized:
            return
            
        try:
            # Check if required dependencies are available
            if chess_utils is None or Config is None:
                raise ImportError("Required chess_utils_minimal modules not available")
            
            if HookedTransformer is None or HookedTransformerConfig is None:
                raise ImportError("Required transformer_lens modules not available")
            
            # Load model metadata - use local models directory
            # Get path to models directory: /Users/muyao.li41/chess/chesstral-api/models/
            current_file = os.path.abspath(__file__)
            service_dir = os.path.dirname(current_file)  # .../chesstral-api/src/service/
            src_dir = os.path.dirname(service_dir)       # .../chesstral-api/src/
            api_root_dir = os.path.dirname(src_dir)      # .../chesstral-api/
            models_dir = os.path.join(api_root_dir, 'models')
            meta_path = os.path.join(models_dir, 'meta.pkl')
            
            with open(meta_path, 'rb') as f:
                self.meta = pickle.load(f)
            
            stoi, itos = self.meta["stoi"], self.meta["itos"]
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: "".join([itos[i] for i in l])
            
            # Load the model using the same approach as the notebook
            model_name = "tf_lens_lichess_8layers_ckpt_no_optimizer"
            n_layers = 8
            
            # Use the same model loading function as the notebook, but with correct path
            # The model file is in the local models directory
            model_path = os.path.join(models_dir, f'{model_name}.pth')
            
            # Create model configuration (same as notebook)
            
            # 8-layer model configuration (same as notebook)
            d_model = 512
            n_heads = 8
            d_mlp = d_model * 4  # 512 * 4 = 2048
            
            cfg = HookedTransformerConfig(
                n_layers=n_layers,
                d_model=d_model,
                d_head=int(d_model / n_heads),
                n_heads=n_heads,
                d_mlp=d_mlp,
                d_vocab=32,
                n_ctx=1023,
                act_fn="gelu",
                normalization_type="LNPre",
            )
            
            self.model = HookedTransformer(cfg)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            
            # Load probe models - use local models/probes directory
            # Probes are in chesstral-api/models/probes/
            probes_dir = os.path.join(models_dir, 'probes')
            
            for layer in range(8):
                probe_path = os.path.join(probes_dir, f'tf_lens_lichess_8layers_ckpt_no_optimizer_chess_piece_probe_layer_{layer}.pth')
                if os.path.exists(probe_path):
                    probe_data = torch.load(probe_path, map_location=self.device)
                    self.probes[layer] = probe_data['linear_probe']
            
            # Set up config (from chess_utils.piece_config)
            self.config = Config(
                min_val=-6,
                max_val=6,
                custom_board_state_function=chess_utils.board_to_piece_state,
                linear_probe_name="chess_piece_probe",
                custom_indexing_function=chess_utils.find_dots_indices,
                num_rows=8,
                num_cols=8,
                levels_of_interest=None,
                column_name=None,
                probing_for_skill=False,
                pos_start=0,
                pos_end=None
            )
            
            self.initialized = True
            
        except Exception as e:
            self.initialized = False

    def _find_dots_indices(self, pgn_string: str) -> List[int]:
        """Find indices of dots (.) in the PGN string - same as chess_utils.find_dots_indices"""
        indices = [index for index, char in enumerate(pgn_string) if char == "."]
        return indices

    def _create_board_state_from_pgn(self, pgn_string: str, char_position: int = None) -> np.ndarray:
        """Create board state from PGN string"""
        # If char_position is provided, extract PGN up to that position
        # Otherwise, use the entire PGN string
        if char_position is not None:
            partial_pgn = pgn_string[:char_position + 1]
        else:
            partial_pgn = pgn_string
        
        # Parse the PGN to get board state
        try:
            board = chess_utils.pgn_string_to_board(partial_pgn)
            board_state = chess_utils.board_to_piece_state(board)
            return board_state
        except Exception as e:
            # Fallback to starting position if parsing fails
            board = chess.Board()
            return chess_utils.board_to_piece_state(board)

    async def analyze_current_position(self, current_pgn: str, layer: int = 5, piece_type: str = 'white_pawns') -> Dict[str, Any]:
        """
        Analyze current position from PGN string - following exact notebook approach
        
        Args:
            current_pgn: PGN string up to current position (e.g., ";1.e4 e5 2.Nf3")
            layer: Layer to probe (0-7)
            piece_type: Type of piece to visualize
        
        Returns:
            Dictionary with heatmap data and analysis results
        """
        if not self.initialized:
            raise Exception("Probe visualization service not initialized")
        
        if layer not in self.probes:
            raise Exception(f"Probe for layer {layer} not found")
        
        # Get piece index
        piece_value = PIECE_TYPE_MAPPING.get(piece_type)
        if piece_value is None:
            raise Exception(f"Unknown piece type: {piece_type}")
        
        piece_index = PIECE_TO_ONE_HOT_MAPPING[piece_value]
        
        # Step 1: Add dummy move to ensure model has processed the complete current move
        # This way we can extract activations at a dot position (where probes work) 
        # but after the model has seen the entire move we want to analyze
        
        # Add a dummy next move so we can extract at the dot position after our real move
        if current_pgn.strip().count(' ') % 2 == 0:
            # If we have even number of moves (last was white), add a dummy black move
            dummy_pgn = current_pgn + " e5 2."
        else:
            # If we have odd number of moves (last was black), add a dummy white move number
            move_num = (current_pgn.strip().count(' ') // 2) + 1
            dummy_pgn = current_pgn + f" {move_num}."
        
        # Step 2: Pad the dummy PGN to 365 characters
        padded_pgn = dummy_pgn.ljust(365)
        
        # Step 3: Find white move indices (dots positions)
        white_move_indices = self._find_dots_indices(padded_pgn)
        if not white_move_indices:
            raise Exception("No white moves found in PGN string")
        
        # Step 4: Use the LAST dot position (which is after our real move)
        current_move_position = len(white_move_indices) - 1
        current_char_index = white_move_indices[current_move_position]
        
        # Step 5: Create board state at current position (ground truth)
        # Use the entire current_pgn to get the board state AFTER all moves have been played
        board_state = self._create_board_state_from_pgn(current_pgn)
        
        # Step 6: Convert board state to one-hot encoding (same as notebook)
        # Add required dimensions: modes, batch, moves, rows, cols
        state_tensor = torch.tensor(board_state, dtype=torch.int8).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 8, 8]
        
        state_one_hot = chess_utils.state_stack_to_one_hot(
            1, self.config.num_rows, self.config.num_cols, 
            self.config.min_val, self.config.max_val, 
            self.device, state_tensor
        )  # [1, 1, 1, 8, 8, 13]
        
        # Extract ground truth for this piece type
        ground_truth = state_one_hot[0, 0, 0, :, :, piece_index]  # [8, 8]
        
        # Step 7: Get model activations (same as notebook)
        encoded_pgn = self.encode(padded_pgn)
        
        model_input = torch.tensor(encoded_pgn[:-1]).unsqueeze(0).to(self.device)  # Remove last char, add batch dim
        
        with torch.inference_mode():
            _, cache = self.model.run_with_cache(model_input, return_type=None)
            resid_post_BlD = cache["resid_post", layer][:, :]  # [batch, seq_len, d_model]
            
            # Extract activation at the current move position
            if current_char_index >= resid_post_BlD.size(1):
                current_char_index = resid_post_BlD.size(1) - 1
            
            resid_post_at_move = resid_post_BlD[:, current_char_index, :]  # [1, d_model]
            resid_post_BLD = resid_post_at_move.unsqueeze(1)  # [1, 1, d_model] to match notebook format
            
            # Step 8: Apply probe (exact einsum from notebook)
            linear_probe = self.probes[layer]  # [1, d_model, 8, 8, 13]
            
            probe_out = einsum(
                "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
                resid_post_BLD,
                linear_probe
            )  # [1, 1, 1, 8, 8, 13]
            
            # Apply log_softmax (same as notebook)
            probe_out = probe_out.log_softmax(-1)
            
            # Extract predictions for this piece type
            piece_predictions = probe_out[0, 0, 0, :, :, piece_index]  # [8, 8]
            
            # Convert to regular probabilities for visualization
            piece_probs = torch.exp(piece_predictions)
            
        # Step 9: Convert to JSON-serializable format
        heatmap_data = piece_probs.cpu().numpy().tolist()
        ground_truth_data = ground_truth.cpu().numpy().tolist()
        board_state_data = board_state.tolist()
        
        return {
            'heatmap': heatmap_data,
            'ground_truth': ground_truth_data,
            'board_state': board_state_data,
            'piece_type': piece_type,
            'layer': layer,
            'current_pgn': current_pgn,
            'padded_pgn': padded_pgn,
            'move_position': current_move_position,
            'char_index': current_char_index,
            'white_move_indices': white_move_indices
        }

    async def generate_heatmap(self, fen: str, pgn_moves: List[str], layer: int, piece_type: str) -> Dict[str, Any]:
        """Generate heatmap for the specified piece type at the given layer (legacy method)"""
        if not self.initialized:
            raise Exception("Probe visualization service not initialized")
        
        if layer not in self.probes:
            raise Exception(f"Probe for layer {layer} not found")
        
        # Get piece index
        piece_value = PIECE_TYPE_MAPPING.get(piece_type)
        if piece_value is None:
            raise Exception(f"Unknown piece type: {piece_type}")
        
        piece_index = PIECE_TO_ONE_HOT_MAPPING[piece_value]
        
        # Format PGN for model input (same as notebook)
        pgn_string = self._format_pgn_for_model(pgn_moves)
        
        # Use the new real-time analysis method
        return await self.analyze_current_position(pgn_string, layer, piece_type)

    def _format_pgn_for_model(self, pgn_moves: List[str]) -> str:
        """Format PGN moves for the model input (same as notebook)"""
        if not pgn_moves:
            return ";"
        
        # Format moves as "1.e4 c5 2.Nf3" etc.
        formatted_moves = []
        for i, move in enumerate(pgn_moves):
            if i % 2 == 0:  # White move
                move_num = (i // 2) + 1
                formatted_moves.append(f"{move_num}.{move}")
            else:  # Black move
                formatted_moves.append(move)
        
        return ";" + " ".join(formatted_moves)

# Global service instance
probe_service = ProbeVisualizationService()

async def generate_probe_heatmap(fen: str, pgn_moves: List[str], layer: int, piece_type: str) -> Dict[str, Any]:
    """Generate probe heatmap for the given position and piece type (legacy)"""
    await probe_service.initialize()
    return await probe_service.generate_heatmap(fen, pgn_moves, layer, piece_type)

async def analyze_current_position(current_pgn: str, layer: int = 5, piece_type: str = 'white_pawns') -> Dict[str, Any]:
    """Generate probe heatmap for current PGN position (real-time analysis)"""
    await probe_service.initialize()
    return await probe_service.analyze_current_position(current_pgn, layer, piece_type) 
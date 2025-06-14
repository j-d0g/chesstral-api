import os
import sys
import torch
import pickle
import asyncio
import chess
from typing import Dict, Any, Optional

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.base import ChessEngine
from core.types import MoveRequest, MoveResponse

# Simple GPT model implementation (copied from nanogpt)
import math
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        original_length = idx.size(1)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # For nanogpt models, 0 is the space index
            # If we break here, we can significantly speed up inference
            # But this is a hardcoded assumption specific to my models
            # Only break on space if we've generated at least one token
            if idx_next == 0 and idx.size(1) > original_length:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class NanoGPTEngine(ChessEngine):
    """NanoGPT chess engine implementation"""
    
    def __init__(self, model_size: str = "small-16", config: Dict[str, Any] = None):
        super().__init__(f"nanogpt-{model_size}", config)
        self.model_size = model_size
        self.model: Optional[GPT] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_vocab_size = None
        self.encode = None
        self.decode = None
        self.model_available = False
        
    async def initialize(self) -> None:
        """Load the NanoGPT model"""
        try:
            # Try to find model files in various locations  
            possible_paths = [
                f"../models/{self.model_size}-ckpt.pt",     # Main path: api/models/ (CORRECT!)
                f"../../models/{self.model_size}-ckpt.pt",  # Alternative path
                f"models/{self.model_size}-ckpt.pt",        # Local path
                f"../chess_gpt_eval/nanogpt/out/{self.model_size}-ckpt.pt",
                f"../../chess_gpt_eval/nanogpt/out/{self.model_size}-ckpt.pt"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                print(f"⚠️  NanoGPT model {self.model_size} not found.")
                print("   To use NanoGPT models, please:")
                print("   1. Train models using the train_ChessGPT directory")
                print("   2. Place model files in one of these locations:")
                for path in possible_paths:
                    print(f"      - {path}")
                print("   3. For now, NanoGPT will provide fallback moves")
                self.model_available = False
                return
            
            print(f"Loading NanoGPT model from: {model_path}")
            
            # Load meta.pkl for vocabulary
            meta_paths = [
                os.path.join(os.path.dirname(model_path), 'meta.pkl'),  # Same dir as model
                "../models/meta.pkl",  # Main path: api/models/ (CORRECT!)
                "../../models/meta.pkl",  # Alternative path
                "../chess_gpt_eval/nanogpt/data/chess/meta.pkl",
                "../../chess_gpt_eval/nanogpt/data/chess/meta.pkl"
            ]
            
            meta = None
            for meta_path in meta_paths:
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    break
            
            if meta:
                self.meta_vocab_size = meta.get('vocab_size', 32)
                stoi = meta.get('stoi', {chr(i): i for i in range(32)})
                itos = meta.get('itos', {i: chr(i) for i in range(32)})
                print(f"Loaded vocabulary with {self.meta_vocab_size} tokens")
            else:
                # Default character-level encoding
                chars = [chr(i) for i in range(32)]
                stoi = {ch: i for i, ch in enumerate(chars)}
                itos = {i: ch for i, ch in enumerate(chars)}
                self.meta_vocab_size = len(chars)
                print("Using default character-level encoding")
                
            self.encode = lambda s: [stoi.get(c, 0) for c in s]
            self.decode = lambda l: ''.join([itos.get(i, f'<UNK{i}>') for i in l])
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model configuration
            model_config = checkpoint.get('model_args', {})
            # Remove vocab_size from model_config if it exists to avoid duplicate
            if 'vocab_size' in model_config:
                del model_config['vocab_size']
            config = GPTConfig(
                vocab_size=self.meta_vocab_size,
                **model_config
            )
            
            # Initialize model
            self.model = GPT(config)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()
            self.model.to(self.device)
            
            self.model_available = True
            print(f"✅ NanoGPT {self.model_size} loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"⚠️  Failed to load NanoGPT model: {e}")
            print("   NanoGPT will provide fallback moves instead")
            self.model_available = False
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def get_move(self, request: MoveRequest) -> MoveResponse:
        """Get a move from NanoGPT with retry logic exactly like the original implementation"""
        try:
            # If model isn't available, fail
            if not self.model_available:
                raise Exception(f"NanoGPT {self.model_size} model not loaded")
            
            # Create board from FEN
            board = chess.Board(request.fen)
            
            print(f"🎯 NANOGPT DEBUG - Starting move generation")
            print(f"   📋 Request details:")
            print(f"      Engine: {self.model_size}")
            print(f"      FEN: {request.fen}")
            print(f"      PGN moves: {request.pgn}")
            print(f"      Temperature: {request.temperature}")
            print(f"   🎲 Legal moves: {[board.san(move) for move in board.legal_moves]}")
            
            # Build prompt for the model using FULL PGN history (critical fix)
            prompt = self._format_pgn_exactly_like_original(request.pgn)
            print(f"🔍 EXACT PROMPT SENT TO MODEL: '{prompt}'")
            print(f"📏 Prompt length: {len(prompt)} characters")
            print(f"🔢 Encoded prompt: {self.encode(prompt)}")
            
            # Try up to 5 times with user-provided temperature as base
            max_attempts = 5
            
            for attempt in range(max_attempts):
                # Use shared temperature calculation from base class
                temperature = self.calculate_retry_temperature(request.temperature, attempt, max_attempts)
                
                print(f"🔄 ATTEMPT {attempt + 1}/{max_attempts}")
                print(f"   🌡️  Temperature: {temperature:.6f} (base: {request.temperature:.6f}, attempt {attempt + 1})")
                
                # Generate move using the model
                move, raw_response = self._generate_move_with_debug(prompt, temperature, attempt + 1)
                
                print(f"   🤖 Model raw output: '{raw_response}'")
                print(f"   ✂️  Extracted move: '{move}'")
                
                # Validate the move
                try:
                    parsed_move = board.parse_san(move.strip())
                    if parsed_move not in board.legal_moves:
                        raise chess.IllegalMoveError(f"Move {move} not in legal moves")
                    
                    print(f"   ✅ LEGAL MOVE FOUND: {move}")
                    print(f"🎯 SUCCESS after {attempt + 1} attempts")
                    
                    return MoveResponse(
                        move=move,
                        raw_response=raw_response,
                        thoughts=f"NanoGPT {self.model_size} suggests: {move} (attempt {attempt + 1}, temp={temperature:.3f})"
                    )
                    
                except Exception as e:
                    print(f"   ❌ ILLEGAL MOVE: '{move}' - {str(e)}")
                    print(f"   🔍 Parse error details: {type(e).__name__}")
                    
                    if attempt < max_attempts - 1:
                        print(f"   🔄 Will retry with higher temperature...")
                        continue
                    else:
                        print(f"💥 FAILED after {max_attempts} attempts")
                        raise Exception(f"NanoGPT failed after {max_attempts} attempts. Last move: '{move}', Raw: '{raw_response}'")
                    
        except Exception as e:
            print(f"💥 NANOGPT ENGINE ERROR: {str(e)}")
            raise Exception(f"NanoGPT {self.model_size} error: {str(e)}")

    def _format_pgn_exactly_like_original(self, pgn_moves: list) -> str:
        """Format PGN moves EXACTLY like the original nanogpt_module.py"""
        print(f"🔧 FORMATTING PGN: Input moves = {pgn_moves}")
        
        if not pgn_moves:
            result = ";"
            print(f"   📝 Empty moves, returning: '{result}'")
            return result
        
        # Build the game state exactly like the original
        game_state = ""
        for i, move in enumerate(pgn_moves):
            if i % 2 == 0:  # White move
                move_number = (i // 2) + 1
                if i > 0:  # Add space before move numbers (except first)
                    game_state += " "
                game_state += f"{move_number}.{move}"
            else:  # Black move  
                game_state += f" {move}"
        
        print(f"   🎯 Game state before regex: '{game_state}'")
        
        # Remove spaces after move numbers exactly like original: "1. e4" -> "1.e4"
        import re
        game_state = re.sub(r"(\d+\.) ", r"\1", game_state)
        print(f"   ✂️  After regex cleanup: '{game_state}'")
        
        # Ensure starts with delimiter exactly like original
        if not game_state.startswith(";"):
            game_state = ";" + game_state
            
        print(f"   ✅ Final formatted prompt: '{game_state}'")
        return game_state
    
    def _generate_move_with_debug(self, prompt: str, temperature: float, attempt_num: int) -> tuple[str, str]:
        """Generate a move with comprehensive debugging"""
        print(f"⚙️  GENERATION DEBUG - Attempt {attempt_num}")
        print(f"   🌡️  Temperature: {temperature}")
        print(f"   📝 Prompt: '{prompt}'")
        
        # Encode the prompt
        encoded_prompt = self.encode(prompt)
        print(f"   🔢 Encoded: {encoded_prompt}")
        
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=self.device)
        print(f"   🎯 Context tensor shape: {context.shape}")
        
        # Generate completion exactly like original (max_new_tokens=10, top_k=200)
        with torch.no_grad():
            print(f"   🤖 Starting model generation...")
            completion_tokens = self.model.generate(
                context,
                max_new_tokens=10,  # Exactly like original
                temperature=temperature,
                top_k=200  # Exactly like original
            )
            print(f"   ✅ Generation complete")
        
        # Decode and process exactly like original
        completion_token_ids = completion_tokens[0].tolist()
        print(f"   🔢 Generated token IDs: {completion_token_ids}")
        
        full_completion = self.decode(completion_token_ids)
        print(f"   📖 Full decoded completion: '{full_completion}'")
        
        # Extract model response (remove prompt) exactly like original
        model_response = full_completion[len(prompt):]
        print(f"   ✂️  Model response (prompt removed): '{model_response}'")
        
        # Stop at semicolon exactly like original
        if ";" in model_response:
            model_response = model_response.split(";")[0]
            print(f"   ✂️  After semicolon split: '{model_response}'")
        
        # Extract move exactly like original
        move = self._extract_move_exactly_like_original(model_response)
        print(f"   🎯 Final extracted move: '{move}'")
        
        return move, model_response
    
    def _extract_move_exactly_like_original(self, completion: str) -> str:
        """Extract move exactly like get_move_from_response in original"""
        print(f"🔍 MOVE EXTRACTION: Input = '{completion}'")
        
        # Parse the response to get only the first move (exactly like original)
        moves = completion.split()
        print(f"   📋 Split into tokens: {moves}")
        
        if not moves:
            print(f"   ❌ No moves found, returning empty string")
            return ""
        
        first_move = moves[0]
        print(f"   🎯 First token: '{first_move}'")
        
        # Handle move numbers exactly like original logic
        if '.' in first_move:
            move_part = first_move.split('.')[-1]  
            result = move_part if move_part else ""
            print(f"   ✂️  Extracted after dot: '{result}'")
        else:
            result = first_move
            print(f"   ✅ Using token as-is: '{result}'")
        
        return result 
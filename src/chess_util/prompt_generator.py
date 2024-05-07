import chess

from chess_util.feature_extractor import extract_board_pos, extract_legal_moves, extract_threats


def generate_game_prompts(game: chess.pgn.Game,
                          pgn=False,
                          fen=False,
                          positions=False,
                          legalmoves=False,
                          threats=False
                          ) -> list[str]:
    """

    :param game: A chess.pgn object representing the current board position.

    :return: prompts: a list of LLM prompts for each move of the game.

    """

    prompts = []
    board = game.board()
    pgn_moves = []

    for move in game.mainline_moves():
        prompts.append(generate_prompt(board, pgn_moves, pgn=pgn, fen=fen, positions=positions, legalmoves=legalmoves,
                                       threats=threats))
        if board.turn == chess.WHITE:
            pgn_moves.append([])
        pgn_moves[-1].append(board.san(move))
        board.push(move)

    return prompts


def generate_move_prompts(game: chess.pgn.Game,
                          move_number: int,
                          context: int,
                          pgn=False,
                          fen=False,
                          positions=False,
                          legalmoves=False,
                          threats=False
                          ) -> list[str]:
    """
    :param game: A chess.pgn object representing the current board position.
    :param move_number: The move number to generate prompts up to and including.
    :param context: The number of moves before the specified move number to include in the prompts.

    :return: prompts: a list of LLM prompts for the specified range of moves.
    """

    prompts = []
    board = game.board()
    pgn_moves = []

    start_move = max(0, move_number - context)
    end_move = move_number + 1

    move_counter = 0
    for move in game.mainline_moves():
        if start_move <= move_counter < end_move:
            prompts.append(
                generate_prompt(board, pgn_moves, pgn=pgn, fen=fen, positions=positions, legalmoves=legalmoves,
                                threats=threats))
        pgn_moves.append(board.san(move))
        board.push(move)
        move_counter += 1

        if move_counter >= end_move:
            break

    return prompts


def generate_prompt(board: chess.Board,
                    pgn=True,
                    fen=False,
                    positions=False,
                    legalmoves=False,
                    threats=False,
                    pgn_moves: list[str] = None
                    ) -> str:
    """
    Extracts and formats the game state into a prompt for the LLM.

    :param board: a chess.Board object representing the current board position
    :param pgn_moves: the output string of chess moves in pgn format

    :return prompts: list of str prompts

    """

    prompt = []

    # Format pgn string
    if pgn:
        pgn_str = pgn_to_str(pgn_moves)
        prompt.append(pgn_str)

    # Format FEN string
    if fen:
        fen_str = f'FEN: {board.fen()}'
        prompt.append(fen_str)

    # Format board positions string
    if positions:
        positions_str = positions_to_str(board)
        prompt.append(positions_str)

    # Format legal moves string
    if legalmoves:
        legal_moves_str = legal_to_str(board)
        prompt.append(legal_moves_str)

    # Format threats string
    if threats:
        threats_str = threats_to_str(board)
        prompt.append(threats_str)

    return "\n".join(prompt)


def pgn_to_str(pgn_moves: list[str]) -> str:
    pgn_str = 'PGN: '
    for index in range(0, len(pgn_moves), 2):
        if index + 1 < len(pgn_moves):
            pgn_str += f"{index // 2 + 1}. {pgn_moves[index]} {pgn_moves[index + 1]} "
        else:
            pgn_str += f"{index // 2 + 1}. {pgn_moves[index]}"
    return pgn_str.strip()


def role_to_str(board: chess.Board) -> str:
    move = board.fullmove_number
    player_turn = 'white' if board.turn is chess.WHITE else 'black'

    role_task_str = f'It is move {move} of a chess game. You are {player_turn}. Your task is to think and reason about the chess-board step-by-step BEFORE returning the best chess move given a board-position. Use any information describing the board-state as well as your knowledge of chess terminology to reason about finding a good move. The queen is the most valuable piece after the king, followed by the rook. The bishop and knight are the same value, but less valuable than the rook. Pawns are the least valuable. When a lower-valued piece is threatening a higher-valued piece, it is generally wise to take it. Please think carefully.'
    return role_task_str


def positions_to_str(board: chess.Board) -> str:
    positions = extract_board_pos(board)
    positions_str = '. '.join([
        f"There is a {'white' if color else 'black'} {chess.piece_name(piece_type)} on {chess.square_name(square)}"
        for color, piece_type, square in positions
    ]) + '.'
    return positions_str


def legal_to_str(board: chess.Board) -> str:
    legal_moves = extract_legal_moves(board)
    legal_moves_str = '. '.join([
        f"The {move['piece_color']} {move['piece_type']} on {move['start_square']} can legally move to {move['end_square']}: ({move['move_san']})"
        for move in legal_moves
    ]) + '.'
    return legal_moves_str


def threats_to_str(board: chess.Board) -> str:
    player_threats, enemy_threats = extract_threats(board)
    player_threats_str = '. '.join([
        f"The {threat['attacker_color']} {threat['attacker_type']} on {threat['attacker_square']} is threatening the "
        f"{threat['defender_color']} {threat['defender_type']} on {threat['defender_square']}: ({threat['san_move']})"
        for threat in
        player_threats
    ])
    enemy_threats_str = '. '.join([
        f"The {threat['attacker_color']} {threat['attacker_type']} on {threat['attacker_square']} is threatening the "
        f"{threat['defender_color']} {threat['defender_type']} on {threat['defender_square']}: ({threat['san_move']})"
        for threat in
        enemy_threats
    ])
    threats = '. '.join([enemy_threats_str, player_threats_str])
    return threats


def completion_to_str(board: chess.Board) -> str:
    completion_str = f'The best move in this position is {board.move}'
    return completion_str


def san_to_str(san_move):
    def is_pawn_move():
        return san_move[0].islower()

    def translate_pawn_move():
        if 'x' in san_move:
            if '=' in san_move:
                file, capture_square, promotion = san_move.split('x')
                promotion_piece = get_promotion_piece(promotion[-1])
                return f"{file}-file pawn captures on {capture_square} and promotes to a {promotion_piece}{get_check_or_checkmate(san_move)}."
            else:
                file, square = san_move.split('x')
                return f"{file}-file pawn captures on {square}{get_check_or_checkmate(san_move)}."
        elif '=' in san_move:
            file, promotion = san_move.split('=')
            promotion_piece = get_promotion_piece(promotion[-1])
            return f"pawn to {file[0]}, promoting to a {promotion_piece}{get_check_or_checkmate(san_move)}."
        else:
            return f"pawn to {san_move}{get_check_or_checkmate(san_move)}."

    def is_piece_move():
        return san_move[0].isupper()

    def translate_piece_move():
        piece_map = {'N': 'knight', 'B': 'bishop', 'R': 'rook', 'Q': 'queen', 'K': 'king'}
        piece = piece_map[san_move[0]]

        if len(san_move) > 4 and san_move[1].isalpha() and san_move[2].isdigit():
            disambiguating_file = san_move[1]
            square = san_move[2:]
            if 'x' in square:
                _, capture_square = square.split('x')
                return f"{piece} on {disambiguating_file}-file to capture on {capture_square}{get_check_or_checkmate(san_move)}."
            else:
                return f"{piece} on {disambiguating_file}-file to {square}{get_check_or_checkmate(san_move)}."
        else:
            if 'x' in san_move:
                _, square = san_move.split('x')
                return f"{piece} captures on {square}{get_check_or_checkmate(san_move)}."
            else:
                square = san_move[1:]
                return f"{piece} moves to {square}{get_check_or_checkmate(san_move)}."

    def is_castling_move():
        return san_move in ['O-O', 'O-O-O']

    def translate_castling_move():
        if san_move == 'O-O':
            return "short castling on the king-side."
        elif san_move == 'O-O-O':
            return "long castling on the queen-side."

    def get_promotion_piece(piece_code):
        promotion_map = {'Q': 'queen', 'R': 'rook', 'B': 'bishop', 'N': 'knight'}
        return promotion_map[piece_code]

    def get_check_or_checkmate():
        if san_move.endswith('+'):
            return ", giving check"
        elif san_move.endswith('#'):
            return ", resulting in checkmate"
        else:
            return ""

    if is_castling_move():
        return translate_castling_move()
    if is_pawn_move():
        return translate_pawn_move()
    elif is_piece_move():
        return translate_piece_move()
    else:
        return "Unknown move format"


# Example usage
san_moves = [
    'e4', 'd5', 'exd5', 'Qxd5', 'Nc3', 'Qa5', 'Nf3', 'Nf6', 'Bc4', 'c6', 'd3', 'Bg4', 'h3', 'Bh5', 'O-O', 'e6',
    'Be3', 'Bb4', 'Qe2', 'Nbd7', 'Bd2', 'Rd8', 'a3', 'Ba5', 'b4', 'Bb6', 'Na4', 'Qc7', 'Nxb6', 'axb6', 'c4', 'O-O',
    'Rac1', 'h6', 'Be3', 'Rfe8', 'Rfe1', 'Nd5', 'Bd2', 'N7f6', 'a4', 'Qd6', 'Ne5', 'Nh7', 'Bxh6', 'gxh6', 'Qxh5',
    'Nxe3',
    'Rxe3', 'Qg6', 'Qxg6+', 'fxg6', 'Rce1', 'Kf7', 'f4', 'Rg8', 'Kf2', 'Rad8', 'Ke2', 'Nf8', 'g3', 'Nd7', 'Kd2', 'Nb8',
    'Nd1', 'Nc6', 'Nc3', 'Rd7', 'Rf1', 'Rh8', 'Nd1', 'Rhd8', 'Ne3', 'Kg7', 'Rff3', 'Rh8', 'Nc2', 'Rhh7', 'Ne1', 'Na5',
    'bxa5', 'bxa5', 'Rb3', 'Rc7', 'Rfb1', 'Kf6', 'Rb6', 'Ra7', 'R1b5', 'Rd2+', 'Ke1', 'Rdd7', 'Kf2', 'Rf7', 'Kg2',
    'Rfd7',
    'Kh2', 'Rf7', 'Kg2', 'Ke7', 'Kf2', 'Kd6', 'Ke2', 'c5', 'Kd2', 'Kc6', 'Kc3', 'Kb6', 'Kb3', 'e5', 'fxe5', 'Rxe5',
    'Rxb7+', 'Kxb7', 'Rb6+', 'Kc7', 'Rxa6', 'Kb7', 'Ra5', 'Kb6', 'Rxc5', 'Kb7', 'a5', 'Rfe7', 'Kb4', 'Re4+', 'Kb5',
    'R7e5',
    'a6+', 'Ka7', 'Ra5', 'Kb8', 'c5', 'Rc4', 'Rb5', 'Rxc5', 'Rxc5', 'Rd4', 'a7+', 'Kb7', 'Rb5+', 'Kc6', 'Kc4', 'Rd6',
    'Rb6', 'Rxb6', 'cxb6', 'Kxb6', 'Kd5', 'Kxa7', 'Ke6', 'Kb6', 'Kf6', 'a5', 'Kxg6', 'a4', 'Kxh6', 'a3', 'g4', 'a2',
    'g5', 'a1=Q', 'g6', 'Qb2', 'g7', 'Qg2', 'g8=Q', 'Qxg8+', 'Kh7', 'Qg5#'
]


def translate_game(game: chess.pgn.Game):
    def game_to_san_moves() -> list[str]:
        """convert a chess game object into a list of its moves in SAN format"""
        moves = []
        board = game.board()
        for move in game.mainline_moves():
            san_moves.append(board.san(move))
            board.push(move)
        return moves

    san_moves = game_to_san_moves()
    player_colors = ['White', 'Black'] * (len(san_moves) // 2)  # Alternate player colors

    move_number = 1
    for move, color in zip(san_moves, player_colors):
        turn = f'{move_number}. {move}: {color} plays '
        sentence = san_to_str(move)
        print(turn + sentence)

        if color == 'Black':
            move_number += 1


def system_chess_prompt(turn: str = 'b') -> str:
    """ Returns a system template for Mistral chat."""
    colour = 'white' if turn == 'w' else 'black'
    # Original with PF-BLT
    original = (
        'You are an auto-regressive language model that is brilliant at reasoning and playing chess at a grandmaster-level. '
        'Your goal is to use your reasoning and chess skills to produce the best chess move given a board position. '
        'Since you are autoregressive, each token you produce is another opportunity to use computation, therefore '
        'you always spend a few sentences first analysing the most recent move, before discussing your step-by-step thought '
        'process to deduce the best move. Use your knowledge of chess rules, strategy, tactics, and the current board-state. '
        'Your thoughts should be concise and clearly structured. It is vital that you always return a move that is both '
        'legal given the board position, and formatted correctly given the notation specified (SAN).'
    )

    # Original shortened
    original_short = ('You are an auto-regressive language model that is brilliant at complex reasoning.'
                      'Since you are autoregressive, each token you produce is another opportunity to use computation, therefore '
                      'you always spend a few sentences analysing the problem step-by-step before giving an answer.'
                      )

    # Best Performance with Best thoughts, though they only stay relevant up to move 8.
    best_thoughts = (
        f"You are a chess coach playing as {colour} and your goal is to win in as few moves as possible. Before making a move, you will first "
        "make a qualitative comment on my last move, followed by a series of thought-steps towards choosing your next move. "
        "You should respond as if you were teaching a student how to play chess, explaining your reasoning and strategy. "
        "I will give you the move sequence, and you will return your next move. Return your move as a JSON object with the following format: 'thoughts': 'Your commentary', ''move': 'Your move in SAN notation'."
    )

    # OK.
    best_commentary = (
        f"You are a chess coach playing as {colour} and your goal is to win in as few moves as possible. I will give you the move sequence, and you will return your next move "
        "with qualitative comments about the last move and the board-state. "
        "Return your move as a JSON object with the following format: 'thoughts': 'Your commentary', ''move': 'Your move in SAN notation'."
    )

    # Gets the furthest number of moves before hallucination
    commentary_ahead = (
        f"You are a chess coach playing as {colour} and your goal is to win in as few moves as possible. I will give you the move sequence, and you will return your next move."
        "Before returning your move, you will first make a qualitative comment on my last move, and the current board-state. You are encouraged to think ahead a few steps. "
        "Return your move as a JSON object with the following format: 'thoughts': 'Your commentary', ''move': 'Your move in SAN notation'."
    )

    commentary_general = (
        f"You are a chess coach playing as {colour} and your goal is to win in as few moves as possible. I will give you the move sequence, and you will return your next move."
        "Return your move as a JSON object with the following format: 'thoughts': 'Your commentary', ''move': 'Your move in SAN notation'."
    )

    commentary_brief = (
        f"You are a chess coach playing as {colour} and your goal is to win in as few moves as possible. I will give you the move sequence, and you will return your next move, alongside a brief, single-sentence commentary on the last move."
        "Return your move as a JSON object with the following format: 'thoughts': 'Your commentary', 'move': 'Your move in SAN notation'."
    )

    # Best Performance on Mistral-7B
    best = (
        f"You are a chess grandmaster playing as {colour} and your goal is to win in as few moves as possible."
        "I will give you the move sequence, and you will return your next move. Return your move as a JSON object with the following format: 'move': 'Your move in SAN notation'."
    )

    different_thoughts = (
        f"You are a chess grandmaster playing as {colour} and your goal is to win in as few moves as possible. Before making a move, you will first "
        "consider the options you have, and the consequences of each move. You should respond objectively and with clarity, and remain concise with your explanations. "
        "I will give you the move sequence, and you will return your next move. Return your move as a JSON object with the following format: 'thoughts': 'Your commentary', ''move': 'Your move in SAN notation'."
    )

    return best_commentary


def user_chess_prompt(board: chess.Board, pgn_moves: list[str], flags) -> str:
    """ Returns a content template for Mistral chat."""
    role = role_to_str(board) if "r" in flags else ""
    schema = {
        "thoughts": "Qualitative comment on my last move.",
        "move": "Your move in PGN notation."
    }
    json_str = f'Provide your thoughts and move in the correct JSON format: {schema}.' if "j" in flags else ""

    board_str = generate_prompt(board, pgn=('p' in flags), fen=('f' in flags),
                                positions=('b' in flags), legalmoves=('l' in flags),
                                threats=('t' in flags),
                                pgn_moves=pgn_moves)

    prompt = " ".join([role, board_str, json_str])

    if not pgn_moves:
        prompt += 'Your move.'

    return prompt

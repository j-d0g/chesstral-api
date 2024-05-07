import chess
import chess.pgn
import pandas as pd


def read_pgn(games: list, file_path: str, num_games: int) -> list[chess.pgn.Game]:
    """

    Extracts chess game objects from PGN file

    :param num_games: the number of pgn games to extract from each file, for testing purposes
    :param file_path: str object representing path to PGN file
    :param games: list of chess.pgn objects, each corresponding to a complete chess game

    """
    with open(file_path) as pgn_file:
        while len(games) < num_games:
            game = chess.pgn.read_game(pgn_file)
            num_of_moves = len(list(game.mainline_moves()))

            is_valid = 70 <= num_of_moves <= 100

            if is_valid:
                games.append((
                    game,
                    num_of_moves
                ))

    return games


def extract_features(game: chess.pgn):
    """

    Extracts features from a python-chess game object from a row in a DataFrame.

    :param game: A chess.pgn object representing the current board position.

    :return features:
        A pd.Series object corresponding to a dictionary of features:
        - Game object
        - List of moves
        - Nested list of legal moves
        - Nested list of threats given/received
        - Headers and Meta-Data
        - Potentially Evaluation?

    """
    board = game.board()  # Initialize board once
    opening = None if "ECO" not in game.headers else game.headers["ECO"]
    result = None if "Result" not in game.headers else game.headers["Result"]

    features = {
        "game": game,
        "moves": list(game.mainline_moves()),  # Materialize moves upfront
        "result": result,
        "opening": opening,
        "legal_moves": [],  # Start as an empty list
        "player_threats": [],
        "enemy_threats": [],
        "num_of_moves": len(list(game.mainline_moves()))
    }

    for move in game.mainline_moves():  # Iterate directly over moves
        features["legal_moves"].append(list(board.legal_moves))  # Append legal moves directly
        player_threats, enemy_threats = extract_threats(board)  # Get attacked pieces
        features["player_threats"].append(player_threats)  # Append to lists directly
        features["enemy_threats"].append(enemy_threats)
        board.push(move)

    return pd.Series(features)


def get_board_position(game: chess.pgn.Game, move_number: int) -> str:
    """ Generates a board position from a given game at a specific move number. """
    board = game.board()
    moves = list(game.mainline_moves())
    for _ in range(min(move_number, len(moves))):
        board.push(moves.pop(0))
    return board.fen()


def extract_board_pos(board: chess.Board) -> list[tuple]:
    """ Iterates through board, and extracts piece color, type and position into tuple.

    :param board: A chess.Board object representing the current board position.

    :return board_positions: a list of board-position tuples, where tuple(piece_color, piece_type, piece_position)

    """
    board_positions = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        # If square contains piece
        if piece is not None:
            # Add piece details
            pos = (piece.color, piece.piece_type, square)
            board_positions.append(pos)

    return board_positions


def extract_legal_moves(board: chess.Board) -> list[dict]:
    """

    Extract all the legal moves for a given board position.

    :param board: A chess.Board object representing the current board position.

    :return legal_moves: List of dicts (legal moves). Each dict encodes the piece-to-move's color, type, and move.

    """
    moves_details = []
    for move in board.legal_moves:
        # Extracting basic move information
        start_square = move.from_square
        end_square = move.to_square
        moving_piece = board.piece_at(start_square)

        # Check for None to avoid AttributeError for empty squares
        if moving_piece:
            piece_type = chess.piece_name(moving_piece.piece_type)
            piece_color = "white" if moving_piece.color == chess.WHITE else "black"
        else:
            piece_type = None
            piece_color = None

        # Storing move details in a dictionary
        move_detail = {
            "piece_color": piece_color,
            "piece_type": piece_type,
            "start_square": chess.square_name(start_square),
            "end_square": chess.square_name(end_square),
            "move_san": board.san(move)
        }
        moves_details.append(move_detail)

    return moves_details


def extract_threats(board: chess.Board):
    """

    Retrieves a list of threatened pieces for the current (player), and also the opponent (enemy).

    :param board: A chess.Board object representing the current board position.

    :return threats: A tuple containing two lists:
        - player_threats: A list of the current player's pieces that are threatening the opponent's pieces
        - enemy_threats: A list of the current player's pieces that are under threat from the opponent's pieces

    """
    player_threats = []
    enemy_threats = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attackers = board.attackers(not piece.color, square)
            for attacker_square in attackers:
                attacker_piece = board.piece_at(attacker_square)
                if attacker_piece:  # Making sure there's actually a piece
                    threat_detail = {
                        "attacker_color": "white" if attacker_piece.color else "black",
                        "attacker_type": chess.piece_name(attacker_piece.piece_type),
                        "attacker_square": chess.square_name(attacker_square),  # Corrected to attacker's square
                        "defender_color": "white" if piece.color else "black",
                        "defender_type": chess.piece_name(piece.piece_type),
                        "defender_square": chess.square_name(square),
                        "san_move": board.san(chess.Move(attacker_square, square))
                    }
                    if piece.color == board.turn:
                        enemy_threats.append(threat_detail)
                    else:
                        player_threats.append(threat_detail)

    return player_threats, enemy_threats


def to_san(fen, last_move):
    """
    Pushes the last move to the board and extracts the SAN move.

    :param fen:
    :param last_move:
    :return:
    """
    uci_move = last_move['from'] + last_move['to']
    chess_move = chess.Move.from_uci(uci_move)
    board = chess.Board(fen)

    return board.san(chess_move)

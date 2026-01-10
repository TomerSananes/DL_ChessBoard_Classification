import numpy as np
import chess.svg
import io
from PIL import Image
from cairosvg import svg2png


# map to number
piece_to_idx = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    'empty': 0
}


def fen_to_matrix(fen):
    # create matrix from fen
    board_str = fen.split(' ')[0]
    matrix = np.zeros((8, 8), dtype=int)
    row, col = 0, 0

    for char in board_str:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            matrix[row, col] = piece_to_idx[char]
            col += 1
    return matrix


def matrix_to_fen(matrix):
    idx_to_piece = {v: k for k, v in piece_to_idx.items() if k != 'empty'}

    fen_rows = []
    for row in range(8):
        empty_count = 0
        row_str = ""
        for col in range(8):
            piece_val = matrix[row, col]
            if piece_val == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += idx_to_piece[piece_val]

        if empty_count > 0:
            row_str += str(empty_count)
        fen_rows.append(row_str)
    return "/".join(fen_rows)


def create_fen_image(fen_str):
    board = chess.Board(fen_str)
    svg_data = chess.svg.board(board=board, size=300)
    png_data = svg2png(bytestring=svg_data)
    return Image.open(io.BytesIO(png_data))
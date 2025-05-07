import numpy as np
import math

def reassemble_from_blocks(blocks_list, padded_height, padded_width):
    """
    Собирает 2D матрицу (канал изображения) из списка блоков NxN.

    Аргументы:
        blocks_list (list[np.ndarray]): Список блоков NxN в порядке чтения
                                       (слева направо, сверху вниз).
        padded_height (int): Высота собранного изображения (должна быть кратна N).
        padded_width (int): Ширина собранного изображения (должна быть кратна N).

    Возвращает:
        np.ndarray: Собранная 2D матрица.
    """
    if not blocks_list:
        return np.array([], dtype=np.uint8).reshape(0,0)

    block_size = blocks_list[0].shape[0]
    if block_size == 0 or blocks_list[0].shape[1] != block_size:
        raise ValueError("Блоки в списке должны быть непустыми и квадратными.")
    if padded_height % block_size != 0 or padded_width % block_size != 0:
        raise ValueError("Высота и ширина для сборки должны быть кратны размеру блока.")

    num_blocks_h = padded_height // block_size
    num_blocks_w = padded_width // block_size

    if len(blocks_list) != num_blocks_h * num_blocks_w:
        raise ValueError(f"Количество блоков ({len(blocks_list)}) не соответствует "
                         f"ожидаемому ({num_blocks_h * num_blocks_w}) для данных размеров.")

    dtype = blocks_list[0].dtype
    reassembled_image = np.zeros((padded_height, padded_width), dtype=dtype)

    block_index = 0
    for r_idx in range(num_blocks_h):
        for c_idx in range(num_blocks_w):
            if block_index >= len(blocks_list):
                 raise IndexError("Индекс блока вышел за пределы списка.")

            start_row = r_idx * block_size
            end_row = start_row + block_size
            start_col = c_idx * block_size
            end_col = start_col + block_size

            current_block = blocks_list[block_index]
            if current_block.shape != (block_size, block_size):
                 raise ValueError(f"Блок {block_index} имеет неверный размер {current_block.shape}, ожидался {(block_size, block_size)}")

            reassembled_image[start_row:end_row, start_col:end_col] = current_block
            block_index += 1

    return reassembled_image
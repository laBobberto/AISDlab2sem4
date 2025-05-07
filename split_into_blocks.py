import numpy as np
import math


def split_into_blocks(image_channel, block_size, fill_value=0):
    """
    Разбивает канал изображения на блоки NxN.
    Если размеры канала не делятся на N, канал дополняется значением fill_value.

    Аргументы:
        image_channel (np.ndarray): Входная 2D матрица (один цветовой канал).
        block_size (int): Размер блока N (блоки будут NxN).
        fill_value (int or float): Значение для дополнения неполных блоков. По умолчанию 0.

    Возвращает:
        list[np.ndarray]: Список блоков NxN. Блоки идут в порядке чтения:
                          слева направо, затем сверху вниз.
    """
    if not isinstance(image_channel, np.ndarray):
        raise TypeError("Входная матрица должна быть массивом NumPy.")
    if image_channel.ndim != 2:
        raise ValueError("Входная матрица должна быть двумерной (один канал).")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Размер блока должен быть положительным целым числом.")

    original_height, original_width = image_channel.shape

    pad_height = (block_size - (original_height % block_size)) % block_size
    pad_width = (block_size - (original_width % block_size)) % block_size

    if pad_height > 0 or pad_width > 0:
        padded_image = np.pad(
            image_channel,
            ((0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=fill_value
        )
    else:
        padded_image = image_channel

    padded_height, padded_width = padded_image.shape

    num_blocks_h = padded_height // block_size
    num_blocks_w = padded_width // block_size

    blocks = []
    for r_idx in range(num_blocks_h):
        for c_idx in range(num_blocks_w):
            start_row = r_idx * block_size
            end_row = start_row + block_size
            start_col = c_idx * block_size
            end_col = start_col + block_size

            block = padded_image[start_row:end_row, start_col:end_col]
            blocks.append(block)

    return blocks
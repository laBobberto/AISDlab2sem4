import numpy as np
import math

def downsample_channel_420(channel_matrix):
    """
    Уменьшает разрешение матрицы одного цветового канала в 2 раза по каждой оси (4:2:0).
    Для каждого блока 2x2 пикселя в исходном канале берется среднее значение.
    Если размеры нечетные, последние строки/столбцы усредняются по доступным пикселям.

    Аргументы:
        channel_matrix (np.ndarray): Входная 2D матрица (один цветовой канал, например, Cb или Cr).
                                     Ожидаются значения в диапазоне [0, 255], тип uint8.

    Возвращает:
        np.ndarray: Уменьшенная 2D матрица.
                    Размеры: (ceil(height/2), ceil(width/2)).
                    Тип: uint8.
    """
    if not isinstance(channel_matrix, np.ndarray):
        raise TypeError("Входная матрица должна быть массивом NumPy.")
    if channel_matrix.ndim != 2:
        raise ValueError("Входная матрица должна быть двумерной (один канал).")

    original_height, original_width = channel_matrix.shape

    new_height = math.ceil(original_height / 2)
    new_width = math.ceil(original_width / 2)

    downsampled_matrix = np.zeros((new_height, new_width), dtype=np.float32)

    for r_new in range(new_height):
        for c_new in range(new_width):
            r_orig_start = r_new * 2
            c_orig_start = c_new * 2

            block = channel_matrix[r_orig_start:min(r_orig_start + 2, original_height),
                    c_orig_start:min(c_orig_start + 2, original_width)]

            if block.size > 0:
                downsampled_matrix[r_new, c_new] = np.mean(block)
            else:
                downsampled_matrix[r_new, c_new] = 0

    return np.round(downsampled_matrix).astype(np.uint8)


def upsample_channel_nearest_neighbor(channel_matrix, target_height, target_width):
    """
    Увеличивает разрешение матрицы канала до целевых размеров, используя метод ближайшего соседа.
    Каждый пиксель исходной матрицы дублируется для формирования блока 2x2 в целевой.

    Аргументы:
        channel_matrix (np.ndarray): Входная 2D матрица (один цветовой канал, например, Cb или Cr).
        target_height (int): Целевая высота.
        target_width (int): Целевая ширина.

    Возвращает:
        np.ndarray: Увеличенная 2D матрица с размерами (target_height, target_width).
    """
    if not isinstance(channel_matrix, np.ndarray):
        raise TypeError("Входная матрица должна быть массивом NumPy.")
    if channel_matrix.ndim != 2:
        raise ValueError("Входная матрица должна быть двумерной (один канал).")

    if channel_matrix.size == 0:
        print(f"Предупреждение: Апсэмплинг пустого канала. Возвращаем массив нулей ({target_height}x{target_width}).")
        return np.zeros((target_height, target_width), dtype=np.uint8)

    upsampled = channel_matrix.repeat(2, axis=0).repeat(2, axis=1)

    final_height = min(target_height, upsampled.shape[0])
    final_width = min(target_width, upsampled.shape[1])

    result = np.zeros((target_height, target_width), dtype=upsampled.dtype)
    result[:final_height, :final_width] = upsampled[:final_height, :final_width]

    return result
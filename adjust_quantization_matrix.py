import numpy as np


def adjust_quantization_matrix(base_matrix, quality_factor):
    """
    Изменяет базовую матрицу квантования в зависимости от уровня качества.

    Аргументы:
        base_matrix (np.ndarray): Базовая матрица квантования (например, NxN).
        quality_factor (int): Уровень качества от 1 до 100.
                              1 = худшее качество, макс. сжатие.
                              100 = лучшее качество, мин. сжатие.

    Возвращает:
        np.ndarray: Измененная матрица квантования с целочисленными значениями.
    """
    if not isinstance(base_matrix, np.ndarray):
        raise TypeError("Базовая матрица должна быть массивом NumPy.")
    if not (1 <= quality_factor <= 100):
        raise ValueError("Уровень качества должен быть в диапазоне от 1 до 100.")

    base_matrix_float = base_matrix.astype(np.float64)

    if quality_factor < 1:
        quality_factor = 1
    if quality_factor > 100:
        quality_factor = 100

    if quality_factor < 50:
        scale_factor = 5000.0 / quality_factor
    else:
        scale_factor = 200.0 - 2.0 * quality_factor

    adjusted_matrix_float = (base_matrix_float * scale_factor + 50.0) / 100.0
    adjusted_matrix_int = np.floor(adjusted_matrix_float)

    adjusted_matrix_int[adjusted_matrix_int < 1] = 1
    adjusted_matrix_int[adjusted_matrix_int > 255] = 255

    return adjusted_matrix_int.astype(np.uint8)
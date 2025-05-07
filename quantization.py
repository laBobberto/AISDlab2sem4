import numpy as np


def quantize(dct_coeffs_block, quantization_matrix):
    """
    Квантует блок коэффициентов DCT с использованием заданной матрицы квантования.

    Аргументы:
        dct_coeffs_block (np.ndarray): NxN блок коэффициентов DCT (тип float).
        quantization_matrix (np.ndarray): NxN матрица квантования (тип int, значения >= 1).

    Возвращает:
        np.ndarray: NxN блок квантованных коэффициентов DCT (тип int, обычно np.int16 или np.int32).
    """
    if dct_coeffs_block.shape != quantization_matrix.shape:
        raise ValueError("Размеры блока коэффициентов и матрицы квантования должны совпадать.")
    if not np.all(quantization_matrix >= 1):
        raise ValueError("Все значения в матрице квантования должны быть >= 1.")

    quantized_coeffs = np.round(dct_coeffs_block / quantization_matrix.astype(np.float64))

    return quantized_coeffs.astype(np.int32)


def dequantize(quantized_coeffs_block, quantization_matrix):
    """
    Выполняет обратное квантование (де-квантование) блока коэффициентов DCT.

    Аргументы:
        quantized_coeffs_block (np.ndarray): NxN блок квантованных коэффициентов DCT (тип int).
        quantization_matrix (np.ndarray): NxN матрица квантования, использованная для квантования (тип int).

    Возвращает:
        np.ndarray: NxN блок де-квантованных коэффициентов DCT (тип float).
    """
    if quantized_coeffs_block.shape != quantization_matrix.shape:
        raise ValueError("Размеры блока квантованных коэффициентов и матрицы квантования должны совпадать.")

    dequantized_coeffs = quantized_coeffs_block.astype(np.float64) * quantization_matrix.astype(np.float64)

    return dequantized_coeffs
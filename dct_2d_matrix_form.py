import numpy as np


def _get_C_factor(k_val):
    """Вспомогательная функция для коэффициента C(k) в формулах DCT/IDCT."""
    if k_val == 0:
        return 1.0 / np.sqrt(2.0)
    else:
        return 1.0


def _create_dct_1d_transform_matrix(N_val):
    """
    Создает матрицу 1D DCT-II T размером N x N.
    T_kn = cos((2*n + 1) * k * pi / (2*N))
    где k - индекс частоты (строка), n - пространственный индекс (столбец).
    """
    T = np.zeros((N_val, N_val), dtype=np.float64)
    for k_idx in range(N_val):
        for n_idx in range(N_val):
            T[k_idx, n_idx] = np.cos((2 * n_idx + 1) * k_idx * np.pi / (2 * N_val))
    return T


def dct_2d_matrix_transform(input_block):
    """
    Выполняет прямое 2D DCT-II для блока NxN, используя матричные операции.
    Формула из ITU-T T.81: S_vu = (1/4)C(v)C(u) * P_vu
    где P_vu - результат двумерного косинусного суммирования.
    В матричной форме: DCT_COEFFS = SCALING_MATRIX * (T @ INPUT_BLOCK @ T.T)

    Аргументы:
        input_block (np.ndarray): Входной блок NxN (например, 8x8).
                                  Значения должны быть уже сдвинуты по уровню (например, вещественные числа в диапазоне ~ -128..127).

    Возвращает:
        np.ndarray: Блок NxN коэффициентов DCT.
    """
    N = input_block.shape[0]
    if input_block.shape[1] != N:
        raise ValueError("Входной блок должен быть квадратным.")
    if input_block.dtype == np.uint8:
        input_block_float = input_block.astype(np.float64) - 128.0
    else:
        input_block_float = input_block.astype(np.float64)

    T_matrix = _create_dct_1d_transform_matrix(N)

    dct_unscaled = T_matrix @ input_block_float @ T_matrix.T

    C_array = np.array([_get_C_factor(k_idx) for k_idx in range(N)], dtype=np.float64)
    Cv_factor_col_vec = C_array.reshape(N, 1)
    Cu_factor_row_vec = C_array.reshape(1, N)

    scaling_coeffs_matrix = Cv_factor_col_vec * Cu_factor_row_vec
    full_scaling_matrix = (1.0 / 4.0) * scaling_coeffs_matrix

    dct_coeffs = full_scaling_matrix * dct_unscaled

    return dct_coeffs


def idct_2d_matrix_transform(dct_coeffs):
    """
    Выполняет обратное 2D DCT-II для блока NxN, используя матричные операции.
    Формула из ITU-T T.81: s_xy = (1/4) * sum_u sum_v C(u)C(v)S_vu cos_x cos_y
    В матричной форме: RECON_BLOCK = (1/4) * (T.T @ S_PRIME @ T)
    где (S_PRIME)_vu = C(v)C(u)*S_vu.

    Аргументы:
        dct_coeffs (np.ndarray): Входной блок NxN коэффициентов DCT.

    Возвращает:
        np.ndarray: Блок NxN восстановленных значений (с уровнем сдвига).
    """
    N = dct_coeffs.shape[0]
    if dct_coeffs.shape[1] != N:
        raise ValueError("Входной блок коэффициентов DCT должен быть квадратным.")

    T_matrix = _create_dct_1d_transform_matrix(N)

    C_array = np.array([_get_C_factor(k_idx) for k_idx in range(N)], dtype=np.float64)
    Cv_factor_col_vec = C_array.reshape(N, 1)
    Cu_factor_row_vec = C_array.reshape(1, N)

    S_prime_scaling_matrix = Cv_factor_col_vec * Cu_factor_row_vec
    S_prime = S_prime_scaling_matrix * dct_coeffs

    rec_block_unscaled = T_matrix.T @ S_prime @ T_matrix

    rec_block = (1.0 / 4.0) * rec_block_unscaled

    return rec_block
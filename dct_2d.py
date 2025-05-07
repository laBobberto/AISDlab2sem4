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


def dct_2d_transform(input_block):
    """
    Выполняет прямое 2D DCT-II для блока NxN, используя матричные операции.
    Формула из ITU-T T.81 (A.3.3): S_vu = (1/4)C(v)C(u) * sum_x sum_y s_xy cos_xv cos_yu
    В матричной форме: DCT_COEFFS = (1/4) * (C_v_col @ C_u_row) * (T @ INPUT_BLOCK @ T.T)
                       где C_v_col @ C_u_row создает матрицу C(v)C(u).
                       Используется нормализация DCT-II, где множитель 2/N для 1D DCT
                       приводит к (4/N^2) для 2D DCT, но в JPEG используется scaling.
                       Формула JPEG (стр. 27, A.3.3) S_vu = 1/4 * C(u)C(v) sum sum s_xy cos cos.
                       T матрица здесь не включает sqrt(2/N) или sqrt(1/N) множители,
                       они учтены в C(u)C(v) и общем множителе 1/4.

    Аргументы:
        input_block (np.ndarray): Входной блок NxN (например, 8x8).
                                  Значения должны быть уже сдвинуты по уровню
                                  (например, вещественные числа в диапазоне ~ -128..127 для 8-битного входа).

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
    dct_intermediate = T_matrix @ input_block_float @ T_matrix.T

    C_array = np.array([_get_C_factor(k_idx) for k_idx in range(N)], dtype=np.float64)
    Cv_factor_col_vec = C_array.reshape(N, 1)
    Cu_factor_row_vec = C_array.reshape(1, N)

    C_vu_matrix = Cv_factor_col_vec * Cu_factor_row_vec
    dct_coeffs = (1.0/4.0) * C_vu_matrix * dct_intermediate
    return dct_coeffs


def idct_2d_transform(dct_coeffs):
    """
    Выполняет обратное 2D DCT-II для блока NxN, используя матричные операции.
    Формула из ITU-T T.81 (A.3.3): s_xy = (1/4) * sum_u sum_v C(u)C(v)S_vu cos_xu cos_yv
    В матричной форме: RECON_BLOCK = (1/4) * (T.T @ (C_v_col @ C_u_row * DCT_COEFFS) @ T)

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

    C_vu_matrix = Cv_factor_col_vec * Cu_factor_row_vec
    S_prime = C_vu_matrix * dct_coeffs

    rec_intermediate = T_matrix.T @ S_prime @ T_matrix
    rec_block = (1.0/4.0) * rec_intermediate

    return rec_block
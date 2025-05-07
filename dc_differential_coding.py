import numpy as np


def dpcm_encode_dc(dc_coefficients):
    """
    Выполняет разностное кодирование (DPCM) для списка DC коэффициентов.
    Первый DC коэффициент остается без изменений (или считается разностью с 0).
    Последующие кодируются как разность с предыдущим восстановленным DC.
    В JPEG PRED инициализируется нулем для первого блока каждого компонента
    в начале скана и в начале каждого интервала перезапуска (ITU-T.81, F.1.1.5.1 ).
    Здесь мы реализуем простой DPCM для последовательности.
    Предполагаем, что dc_coefficients - это уже извлеченные DC значения для одного компонента.

    Аргументы:
        dc_coefficients (list[int] or np.ndarray): Список или массив DC коэффициентов.

    Возвращает:
        list[int]: Список разностно-кодированных DC коэффициентов.
                   Тип элементов будет int.
    """
    if not isinstance(dc_coefficients, (list, np.ndarray)):
        raise TypeError("Входные DC коэффициенты должны быть списком или массивом NumPy.")
    if len(dc_coefficients) == 0:
        return []

    dc_coeffs_np = np.array(dc_coefficients, dtype=np.int32)

    diff_dc = np.empty_like(dc_coeffs_np)

    pred_dc = 0
    diff_dc[0] = dc_coeffs_np[0] - pred_dc

    for i in range(1, len(dc_coeffs_np)):
        diff_dc[i] = dc_coeffs_np[i] - dc_coeffs_np[i - 1]

    return diff_dc.tolist()


def dpcm_decode_dc(diff_dc_coefficients):
    """
    Выполняет обратное разностное кодирование для списка DC коэффициентов.

    Аргументы:
        diff_dc_coefficients (list[int] or np.ndarray): Список или массив разностно-кодированных
                                                        DC коэффициентов.
    Возвращает:
        list[int]: Список восстановленных DC коэффициентов.
    """
    if not isinstance(diff_dc_coefficients, (list, np.ndarray)):
        raise TypeError("Входные разностные DC коэффициенты должны быть списком или массивом NumPy.")
    if len(diff_dc_coefficients) == 0:
        return []

    diff_dc_coeffs_np = np.array(diff_dc_coefficients, dtype=np.int32)
    dc_coeffs_reconstructed = np.empty_like(diff_dc_coeffs_np)

    pred_dc = 0
    dc_coeffs_reconstructed[0] = diff_dc_coeffs_np[0] + pred_dc

    for i in range(1, len(diff_dc_coeffs_np)):
        dc_coeffs_reconstructed[i] = diff_dc_coeffs_np[i] + dc_coeffs_reconstructed[i - 1]

    return dc_coeffs_reconstructed.tolist()
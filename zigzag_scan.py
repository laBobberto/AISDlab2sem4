import numpy as np

def zigzag_scan(matrix_block):
    """
    Выполняет зигзаг-сканирование квадратной матрицы NxN.

    Аргументы:
        matrix_block (np.ndarray): Входная квадратная матрица NxN.

    Возвращает:
        np.ndarray: Одномерный массив, содержащий элементы матрицы
                    в порядке зигзаг-сканирования.
    """
    if not isinstance(matrix_block, np.ndarray):
        raise TypeError("Входная матрица должна быть массивом NumPy.")
    if matrix_block.ndim != 2 or matrix_block.shape[0] != matrix_block.shape[1]:
        raise ValueError("Входная матрица должна быть квадратной и двумерной.")

    n = matrix_block.shape[0]
    result = np.empty(n * n, dtype=matrix_block.dtype)
    index = 0
    row, col = 0, 0
    going_up = True

    for _ in range(n * n):
        result[index] = matrix_block[row, col]
        index += 1

        if going_up:
            if col == n - 1:
                row += 1
                going_up = False
            elif row == 0:
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == n - 1:
                col += 1
                going_up = True
            elif col == 0:
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1
    return result

def inverse_zigzag_scan(flat_array, n):
    """
    Восстанавливает квадратную матрицу NxN из одномерного массива,
    полученного зигзаг-сканированием.

    Аргументы:
        flat_array (np.ndarray): Одномерный массив после зигзаг-сканирования.
        n (int): Размерность исходной квадратной матрицы (N).

    Возвращает:
        np.ndarray: Восстановленная квадратная матрица NxN.
    """
    if not isinstance(flat_array, np.ndarray):
        raise TypeError("Входной массив должен быть массивом NumPy.")
    if flat_array.ndim != 1 or flat_array.size != n * n:
        raise ValueError(f"Размер входного массива ({flat_array.size}) "
                         f"не соответствует квадрату n*n ({n*n}).")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Размерность n должна быть положительным целым числом.")


    matrix_block = np.empty((n, n), dtype=flat_array.dtype)
    index = 0
    row, col = 0, 0
    going_up = True

    for _ in range(n * n):
        matrix_block[row, col] = flat_array[index]
        index += 1

        if going_up:
            if col == n - 1:
                row += 1
                going_up = False
            elif row == 0:
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == n - 1:
                col += 1
                going_up = True
            elif col == 0:
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1
    return matrix_block
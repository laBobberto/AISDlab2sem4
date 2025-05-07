import numpy as np

def rle_encode_ac_coefficients(ac_coeffs_zigzag):
    """
    Выполняет Run-Length Encoding (RLE) для AC коэффициентов после зигзаг-сканирования.
    Кодирует последовательности нулей.

    Аргументы:
        ac_coeffs_zigzag (list[int] or np.ndarray): Одномерный массив AC коэффициентов
                                                   в зигзаг-порядке (DC коэффициент исключен).

    Возвращает:
        list[tuple]: Список кортежей (run_length, value) или специальный маркер (0,0) для EOB.
                     run_length (RRRR в JPEG): количество нулей перед value.
                     value: само ненулевое значение AC коэффициента.
                            Для маркера конца блока (EOB) используется (0,0).
                     Специальный кортеж (15, 0) (ZRL - Zero Run Length) используется для
                     кодирования 16 последовательных нулей.
    """
    if not isinstance(ac_coeffs_zigzag, (list, np.ndarray)):
        raise TypeError("Входные AC коэффициенты должны быть списком или массивом NumPy.")

    rle_encoded = []
    zero_run_length = 0

    for i, coeff in enumerate(ac_coeffs_zigzag):
        if coeff == 0:
            zero_run_length += 1
            if zero_run_length == 16:
                rle_encoded.append((15, 0))
                zero_run_length = 0
        else:
            rle_encoded.append((zero_run_length, coeff))
            zero_run_length = 0

    rle_encoded.append((0, 0))

    return rle_encoded

def rle_decode_ac_coefficients(rle_encoded_ac, num_ac_coeffs=63):
    """
    Декодирует AC коэффициенты из RLE представления.
    Стала более устойчивой к некорректным RLE-последовательностям.

    Аргументы:
        rle_encoded_ac (list[tuple]): Список RLE-кортежей (run_length, value) или (0,0) для EOB,
                                      или (15,0) для ZRL.
        num_ac_coeffs (int): Общее количество AC коэффициентов в блоке (обычно 63).

    Возвращает:
        list[int]: Восстановленный одномерный список AC коэффициентов.
    """
    if not isinstance(rle_encoded_ac, list):
        raise TypeError("Входные RLE AC коэффициенты должны быть списком.")

    ac_coeffs_zigzag = []
    for idx, (run_length, value) in enumerate(rle_encoded_ac):
        current_len = len(ac_coeffs_zigzag)

        if current_len >= num_ac_coeffs:
             if run_length == 0 and value == 0:
                  break
             else:
                  print(f"Предупреждение: Достигнут лимит ({current_len}/{num_ac_coeffs}), но RLE данные продолжаются: ({run_length},{value}) в rle_encoded_ac[{idx}]. Игнорируется остаток.")
                  break

        if run_length == 0 and value == 0:
            remaining_zeros = num_ac_coeffs - current_len
            if remaining_zeros > 0:
                ac_coeffs_zigzag.extend([0] * remaining_zeros)
            break
        elif run_length == 15 and value == 0:
            num_zeros_to_add = 16
            if current_len + num_zeros_to_add > num_ac_coeffs:
                print(f"Предупреждение: ZRL привел бы к превышению лимита ({current_len} + {num_zeros_to_add} > {num_ac_coeffs}). Добавляем только {num_ac_coeffs - current_len} нулей.")
                num_zeros_to_add = num_ac_coeffs - current_len
            ac_coeffs_zigzag.extend([0] * max(0, num_zeros_to_add))
            if len(ac_coeffs_zigzag) >= num_ac_coeffs:
                 break

        else:
            if run_length < 0 or run_length > 15:
                 print(f"Предупреждение: Некорректный run_length={run_length} в RLE паре. Игнорируется пара.")
                 continue

            num_zeros_to_add = run_length
            if current_len + num_zeros_to_add > num_ac_coeffs:
                 print(f"Предупреждение: Run-length {run_length} для значения {value} привел бы к превышению лимита ({current_len} + {num_zeros_to_add} > {num_ac_coeffs}). Добавляем только {num_ac_coeffs - current_len} нулей.")
                 num_zeros_to_add = num_ac_coeffs - current_len
                 ac_coeffs_zigzag.extend([0] * max(0, num_zeros_to_add))
                 break
            else:
                 ac_coeffs_zigzag.extend([0] * num_zeros_to_add)

            if len(ac_coeffs_zigzag) < num_ac_coeffs:
                ac_coeffs_zigzag.append(value)
            else:
                 if value != 0:
                    print(f"Предупреждение: После добавления {run_length} нулей не осталось места для значения {value} (достигнут лимит {num_ac_coeffs}). Значение пропущено.")
                 break

    if len(ac_coeffs_zigzag) < num_ac_coeffs:
         remaining_zeros = num_ac_coeffs - len(ac_coeffs_zigzag)
         ac_coeffs_zigzag.extend([0] * remaining_zeros)

    return ac_coeffs_zigzag[:num_ac_coeffs]
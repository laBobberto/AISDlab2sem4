import numpy as np
from vli_coding import get_vli_category_and_value, decode_vli

def encode_dc_coefficient(dc_difference):
    """
    Подготавливает разность DC коэффициента для кодирования Хаффманом.

    Согласно ITU-T.81, F.1.2.1, символ Хаффмана для DC состоит только из SSSS (категории).
    Дополнительные биты представляют собой VLI представление разности.

    Аргументы:
        dc_difference (int): Разность между текущим DC и предсказанным DC.

    Возвращает:
        tuple[int, str]: Кортеж (category, extra_bits_string).
                         category: Категория разности (SSSS).
                         extra_bits_string: Строка дополнительных битов (VLI).
    """
    category, extra_bits = get_vli_category_and_value(dc_difference)
    return category, extra_bits

def encode_ac_coefficients(rle_encoded_ac):
    """
    Подготавливает RLE-кодированные AC коэффициенты для кодирования Хаффманом.

    Согласно ITU-T.81, F.2.2.4, символ Хаффмана для AC состоит из (RUN, SIZE).
    RUN - длина серии нулей перед ненулевым коэффициентом.
    SIZE - категория (SSSS) ненулевого коэффициента.
    Дополнительные биты представляют собой VLI представление ненулевого коэффициента.

    Особые случаи:
    (0,0) (EOB - End of Block): Маркер конца блока. Нет дополнительных битов.
    (15,0) (ZRL - Zero Run Length): Маркер серии из 16 нулей. Нет дополнительных битов.

    Аргументы:
        rle_encoded_ac (list[tuple[int, int]]): Список пар (run_length, value)
                                                из RLE-кодирования AC коэффициентов.
                                                value = 0 для EOB и ZRL.

    Возвращает:
        list[tuple[tuple[int, int], str]]: Список кортежей ((run, size), extra_bits_string).
                                           run: Длина серии нулей.
                                           size: Категория ненулевого коэффициента.
                                           extra_bits_string: Строка дополнительных битов (VLI).
                                           Для (0,0) и (15,0) extra_bits_string будет пустой.
    """
    encoded_ac = []
    for run_length, value in rle_encoded_ac:
        if (run_length, value) == (0, 0):
            encoded_ac.append(((0, 0), ""))
        elif (run_length, value) == (15, 0):
             encoded_ac.append(((15, 0), ""))
        elif value != 0:
            category, extra_bits = get_vli_category_and_value(value)
            encoded_ac.append(((run_length, category), extra_bits))

    return encoded_ac
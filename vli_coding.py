import numpy as np
import math


def get_vli_category_and_value(number):
    """
    Определяет категорию (размер в битах) и дополнительные биты для числа
    в формате Variable Length Integer (VLI), используемом в JPEG.
    См. ITU-T T.81, Таблица F.1 для DC, Таблица F.2 для AC.

    Аргументы:
        number (int): Входное число (разность DC или значение AC).

    Возвращает:
        tuple[int, str]: Кортеж (категория, строка_дополнительных_бит).
                         Категория - это SSSS (количество доп. бит).
                         Если число 0, категория 0, строка пустая.
    """
    if number == 0:
        return 0, ""

    magnitude = abs(number)

    category = 0
    if magnitude > 0:
        category = magnitude.bit_length()

    if number > 0:
        value_bits = bin(magnitude)[2:].zfill(category)
    else:
        temp_val_for_neg = (1 << category) - 1 - magnitude
        value_bits = bin(temp_val_for_neg)[2:].zfill(category)

    return category, value_bits


def decode_vli(category, value_bits_str):
    """
    Декодирует число из его VLI категории и строки дополнительных бит.

    Аргументы:
        category (int): Категория (SSSS, количество дополнительных бит).
        value_bits_str (str): Строка дополнительных бит.

    Возвращает:
        int: Декодированное число.
    """
    if category == 0:
        if value_bits_str == "":
            return 0
        else:
            raise ValueError("Для категории 0 строка бит должна быть пустой.")

    if len(value_bits_str) != category:
        raise ValueError(f"Длина строки бит ({len(value_bits_str)}) не совпадает с категорией ({category}).")

    value_from_bits = int(value_bits_str, 2)

    sign_threshold = 1 << (category - 1)

    if value_from_bits >= sign_threshold:
        return value_from_bits
    else:
        return value_from_bits - ((1 << category) - 1)
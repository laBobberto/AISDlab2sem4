import numpy as np

def pack_bitstream(bit_string):
    """
    Упаковывает битовую строку в байтовую последовательность, применяя правило
    "stuffing" байта 0x00 после байта 0xFF согласно стандарту JPEG (B.2.2).

    Аргументы:
        bit_string (str): Строка из символов '0' и '1', представляющая битовый поток.

    Возвращает:
        bytes: Последовательность байтов.
    """
    if not isinstance(bit_string, str):
        raise TypeError("Вход должен быть строкой битов.")
    if not all(c in '01' for c in bit_string):
        raise ValueError("Входная строка должна содержать только '0' или '1'.")

    byte_list = []
    current_byte = 0
    bits_in_byte = 0

    for bit in bit_string:
        current_byte = (current_byte << 1) | int(bit)
        bits_in_byte += 1

        if bits_in_byte == 8:
            byte_list.append(current_byte)
            if current_byte == 0xFF:
                byte_list.append(0x00)
            current_byte = 0
            bits_in_byte = 0

    if bits_in_byte > 0:
        current_byte = (current_byte << (8 - bits_in_byte)) | ((1 << (8 - bits_in_byte)) - 1)
        byte_list.append(current_byte)

    return bytes(byte_list)

def unpack_bitstream(byte_data):
    """
    Распаковывает байтовую последовательность обратно в битовую строку, удаляя
    stuffing байты 0x00 после 0xFF согласно стандарту JPEG (B.2.2).
    Предполагается, что byte_data является сегментом энтропийно-кодированных данных,
    не содержащим маркеров JPEG (кроме, возможно, маркера конца скана EOI).

    Аргументы:
        byte_data (bytes): Входная последовательность байтов.

    Возвращает:
        str: Строка из символов '0' и '1', представляющая восстановленный битовый поток.
    """
    if not isinstance(byte_data, bytes):
        raise TypeError("Вход должен быть последовательностью байтов.")

    bit_string = ""
    i = 0
    while i < len(byte_data):
        byte = byte_data[i]

        if byte == 0xFF:
            bit_string += bin(byte)[2:].zfill(8)
            i += 1
            if i < len(byte_data):
                next_byte = byte_data[i]
                if next_byte == 0x00:
                    i += 1
                else:
                    break
        else:
            bit_string += bin(byte)[2:].zfill(8)
            i += 1
    return bit_string
import heapq
from collections import Counter, defaultdict
import io
import numpy as np


DEFAULT_DC_LUMINANCE_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
DEFAULT_DC_LUMINANCE_HUFFVAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

DEFAULT_DC_CHROMINANCE_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
DEFAULT_DC_CHROMINANCE_HUFFVAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

DEFAULT_AC_LUMINANCE_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
DEFAULT_AC_LUMINANCE_HUFFVAL = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]

DEFAULT_AC_CHROMINANCE_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
DEFAULT_AC_CHROMINANCE_HUFFVAL = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]

class HuffmanTable:
    """
    Класс для представления и работы с таблицей Хаффмана в формате JPEG.
    Генерирует канонические коды Хаффмана на основе спецификации BITS и HUFFVAL.
    """
    def __init__(self, bits, huffval):
        """
        Инициализирует таблицу Хаффмана.

        Аргументы:
            bits (list[int]): Список из 16 элементов. bits[i] - количество кодов длины (i+1).
            huffval (list[int]): Список значений символов (0-255), упорядоченный по длинам кодов.
        """
        if len(bits) != 16:
            raise ValueError("Список BITS должен содержать 16 элементов.")
        total_codes_in_bits = sum(bits)
        if total_codes_in_bits != len(huffval):
             raise ValueError(f"Сумма BITS ({total_codes_in_bits}) не равна длине HUFFVAL ({len(huffval)}).")

        self.bits = list(bits)
        self.huffval = list(huffval)
        self.encode_table = {}
        self.decode_table = {}
        self.max_code_len = 0

        self._generate_huffman_codes()
        self._build_decode_structure()

    def get_spec(self):
        """Возвращает спецификацию таблицы в формате BITS, HUFFVAL."""
        return self.bits, self.huffval

    def _generate_huffman_codes(self):
        """Генерирует канонические коды Хаффмана по спецификации BITS и HUFFVAL."""
        huffcode = []
        code = 0
        si = 1
        num_codes_generated = 0
        huffval_idx = 0
        for i in range(16):
            num_codes_of_length_si = self.bits[i]
            for j in range(num_codes_of_length_si):
                if huffval_idx >= len(self.huffval):
                     raise IndexError(f"Ошибка генерации кодов: индекс HUFFVAL ({huffval_idx}) выходит за пределы "
                                      f"для длины {si} (i={i}, j={j}). Сумма BITS={sum(self.bits)}, длина HUFFVAL={len(self.huffval)}")
                symbol = self.huffval[huffval_idx]
                self.encode_table[symbol] = (code, si)
                huffcode.append(code)
                huffval_idx += 1
                code += 1
            num_codes_generated += num_codes_of_length_si
            code <<= 1
            si += 1
            if num_codes_of_length_si > 0:
                self.max_code_len = i + 1

        if num_codes_generated != len(self.huffval):
             print(f"Предупреждение: Сгенерировано {num_codes_generated} кодов, но HUFFVAL содержит {len(self.huffval)} символов.")


    def _build_decode_structure(self):
        """Строит структуру для быстрого декодирования (словарь префиксов)."""
        self.decode_table = {}
        for symbol, (code, length) in self.encode_table.items():
            code_str = format(code, f'0{length}b')
            self.decode_table[code_str] = symbol

    def get_code(self, symbol):
        """
        Возвращает код и его длину для заданного символа.

        Аргументы:
            symbol (int): Символ для кодирования (0-255).

        Возвращает:
            tuple[int, int]: (код, длина_кода) или None, если символ не найден.
        """
        return self.encode_table.get(symbol)

    def decode_symbol(self, bit_reader):
        """
        Декодирует следующий символ Хаффмана из битового потока.

        Аргументы:
            bit_reader (BitReader): Объект для чтения бит.

        Возвращает:
            int: Декодированный символ или None, если достигнут конец потока или ошибка.
        """
        current_code_str = ""
        for _ in range(self.max_code_len):
            bit = bit_reader.read_bit()
            if bit is None:
                if current_code_str:
                     print(f"Ошибка декодирования: конец потока после неполного кода '{current_code_str}'")
                return None
            current_code_str += str(bit)
            if current_code_str in self.decode_table:
                return self.decode_table[current_code_str]
        print(f"Ошибка декодирования: не найден символ для кода '{current_code_str}' (макс. длина {self.max_code_len})")
        return None


class BitWriter:
    """Класс для записи бит в байтовый поток с JPEG байт-стаффингом."""
    def __init__(self):
        self._buffer = 0
        self._bit_count = 0
        self._byte_stream = bytearray()

    def write_bit(self, bit):
        """Записывает один бит (0 или 1)."""
        if bit not in (0, 1):
            raise ValueError("Бит должен быть 0 или 1")
        self._buffer = (self._buffer << 1) | bit
        self._bit_count += 1
        if self._bit_count == 8:
            self._flush_byte()

    def write_bits(self, value, num_bits):
        """Записывает num_bits младших бит из value."""
        if num_bits < 0:
             raise ValueError("Количество бит не может быть отрицательным")
        if num_bits == 0:
             return

        mask = (1 << num_bits) - 1
        bits_to_write = value & mask

        for i in range(num_bits - 1, -1, -1):
            bit = (bits_to_write >> i) & 1
            self.write_bit(bit)

    def _flush_byte(self):
        """Записывает заполненный буфер в байтовый поток, выполняя стаффинг."""
        if self._bit_count != 8:
             raise Exception("Внутренняя ошибка: попытка сбросить неполный байт")

        byte_to_write = self._buffer
        self._byte_stream.append(byte_to_write)
        if byte_to_write == 0xFF:
            self._byte_stream.append(0x00)
        self._buffer = 0
        self._bit_count = 0

    def get_byte_string(self):
        """Возвращает итоговую байтовую строку, добавляя паддинг."""
        if self._bit_count > 0:
            padding_bits = 8 - self._bit_count
            pad_mask = (1 << padding_bits) - 1
            self._buffer = (self._buffer << padding_bits) | pad_mask
            self._bit_count = 8
            self._flush_byte()

        return bytes(self._byte_stream)


class BitReader:
    """Класс для чтения бит из байтового потока с JPEG байт-стаффингом."""
    def __init__(self, byte_data):
        self._byte_stream = io.BytesIO(byte_data)
        self._current_byte = 0
        self._bit_pos = 8
        self._marker_found = False

    def _load_byte(self):
        """Загружает следующий байт, пропуская стаффинг."""
        if self._marker_found:
            return None

        byte = self._byte_stream.read(1)
        if not byte:
            return None

        val = byte[0]
        if val == 0xFF:
            next_byte = self._byte_stream.read(1)
            if not next_byte:
                 self._marker_found = True
                 return None
            next_val = next_byte[0]
            if next_val == 0x00:
                self._current_byte = 0xFF
                self._bit_pos = 0
                return True
            else:
                self._byte_stream.seek(-2, io.SEEK_CUR)
                self._marker_found = True
                return None
        else:
            self._current_byte = val
            self._bit_pos = 0
            return True

    def read_bit(self):
        """Читает один бит."""
        if self._bit_pos > 7:
            if not self._load_byte():
                return None

        bit = (self._current_byte >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        return bit

    def read_bits(self, num_bits):
        """Читает num_bits бит и возвращает их как целое число."""
        if num_bits < 0:
            raise ValueError("Количество бит не может быть отрицательным")
        if num_bits == 0:
             return 0

        value = 0
        for i in range(num_bits):
            bit = self.read_bit()
            if bit is None:
                raise EOFError(f"Неожиданный конец потока/маркер при попытке чтения {num_bits} бит (прочитано {i} бит).")
            value = (value << 1) | bit
        return value


from vli_coding import get_vli_category_and_value

def huffman_encode_data(data_units, dc_table, ac_table):
    """
    Кодирует список обработанных блоков данных (DC + AC RLE) Хаффманом.

    Аргументы:
        data_units (list): Список кортежей, каждый кортеж содержит данные одного блока:
                           (dc_category, dc_vli_bits, [(rle_ac_run, ac_value), ...])
                           ac_rle_pairs получается из rle_encode_ac_coefficients.
        dc_table (HuffmanTable): Таблица Хаффмана для DC категорий.
        ac_table (HuffmanTable): Таблица Хаффмана для AC RLE пар (run/size).

    Возвращает:
        bytes: Закодированная байтовая строка.
    """
    bit_writer = BitWriter()

    for dc_category, dc_vli_bits, ac_rle_pairs in data_units:
        dc_huff_code_info = dc_table.get_code(dc_category)
        if dc_huff_code_info is None:
             raise ValueError(f"Символ DC категории {dc_category} не найден в таблице Хаффмана.")
        dc_code, dc_len = dc_huff_code_info
        bit_writer.write_bits(dc_code, dc_len)
        if dc_category > 0:
            if len(dc_vli_bits) != dc_category:
                 raise ValueError(f"Неверная длина VLI бит для DC={dc_category}: '{dc_vli_bits}' (длина {len(dc_vli_bits)})")
            dc_vli_val = int(dc_vli_bits, 2)
            bit_writer.write_bits(dc_vli_val, dc_category)

        for run_length, ac_value in ac_rle_pairs:
            if run_length == 0 and ac_value == 0:
                ac_symbol = 0x00
                ac_huff_code_info = ac_table.get_code(ac_symbol)
                if ac_huff_code_info is None:
                    raise ValueError("Символ EOB (0x00) не найден в AC таблице Хаффмана.")
                ac_code, ac_len = ac_huff_code_info
                bit_writer.write_bits(ac_code, ac_len)
                break
            elif run_length == 15 and ac_value == 0:
                ac_symbol = 0xF0
                ac_huff_code_info = ac_table.get_code(ac_symbol)
                if ac_huff_code_info is None:
                    raise ValueError("Символ ZRL (0xF0) не найден в AC таблице Хаффмана.")
                ac_code, ac_len = ac_huff_code_info
                bit_writer.write_bits(ac_code, ac_len)
            else:
                ac_category, ac_vli_bits = get_vli_category_and_value(ac_value)
                if ac_category == 0:
                    raise ValueError(f"Получена нулевая категория для ненулевого AC: {ac_value}")
                if ac_category > 15:
                    raise ValueError(f"AC VLI категория {ac_category} не может быть > 15")
                if not (0 <= run_length <= 15):
                     raise ValueError(f"Недопустимый run_length {run_length} в AC RLE.")

                ac_symbol = (run_length << 4) | ac_category
                ac_huff_code_info = ac_table.get_code(ac_symbol)
                if ac_huff_code_info is None:
                    raise ValueError(f"Символ AC (run={run_length}, size={ac_category}, sym=0x{ac_symbol:02X}) не найден в таблице Хаффмана.")
                ac_code, ac_len = ac_huff_code_info
                bit_writer.write_bits(ac_code, ac_len)

                if len(ac_vli_bits) != ac_category:
                     raise ValueError(f"Неверная длина VLI бит для AC={ac_value} (кат={ac_category}): '{ac_vli_bits}' (длина {len(ac_vli_bits)})")
                ac_vli_val = int(ac_vli_bits, 2)
                bit_writer.write_bits(ac_vli_val, ac_category)

    return bit_writer.get_byte_string()


from vli_coding import decode_vli

def huffman_decode_data(byte_data, dc_table, ac_table, num_blocks):
    """
    Декодирует Хаффман-закодированные данные для нескольких блоков.

    Аргументы:
        byte_data (bytes): Входная байтовая строка.
        dc_table (HuffmanTable): Таблица Хаффмана для DC категорий.
        ac_table (HuffmanTable): Таблица Хаффмана для AC RLE пар (run/size).
        num_blocks (int): Ожидаемое количество блоков для декодирования.

    Возвращает:
        list: Список кортежей, формат совпадает с входом huffman_encode_data:
              [(dc_category, dc_vli_bits, [(rle_ac_run, ac_value), ...]), ...]
              где ac_value - восстановленное значение AC.
    """
    bit_reader = BitReader(byte_data)
    decoded_units = []

    try:
        for block_index in range(num_blocks):
            dc_category = dc_table.decode_symbol(bit_reader)
            if dc_category is None:
                 raise EOFError(f"Не удалось декодировать DC категорию блока {block_index + 1}.")

            dc_vli_bits_str = ""
            if dc_category > 15:
                raise ValueError(f"Декодирована некорректная DC категория {dc_category} > 15.")
            if dc_category > 0:
                dc_vli_val = bit_reader.read_bits(dc_category)
                dc_vli_bits_str = format(dc_vli_val, f'0{dc_category}b')

            ac_rle_pairs = []
            ac_count = 0
            while ac_count < 64:
                ac_symbol = ac_table.decode_symbol(bit_reader)
                if ac_symbol is None:
                    raise EOFError(f"Не удалось декодировать AC символ в блоке {block_index + 1} после {len(ac_rle_pairs)} пар.")

                if ac_symbol == 0x00:
                    ac_rle_pairs.append((0, 0))
                    break
                elif ac_symbol == 0xF0:
                    ac_rle_pairs.append((15, 0))
                    ac_count += 16
                else:
                    run_length = (ac_symbol >> 4) & 0x0F
                    ac_category = ac_symbol & 0x0F
                    if ac_category == 0 or ac_category > 15:
                        raise ValueError(f"Некорректный AC символ 0x{ac_symbol:02X} (run={run_length}, size={ac_category})")

                    ac_vli_val = bit_reader.read_bits(ac_category)
                    ac_vli_bits_str = format(ac_vli_val, f'0{ac_category}b')
                    ac_value = decode_vli(ac_category, ac_vli_bits_str)
                    ac_rle_pairs.append((run_length, ac_value))
                    ac_count += run_length + 1

                if ac_count > 63 :
                     print(f"Предупреждение: Счетчик AC ({ac_count}) превысил 63 в блоке {block_index + 1}. Возможно, лишние данные.")

            if ac_count > 63 and ac_symbol != 0x00:
                 print(f"Предупреждение: Цикл декодирования AC завершился с ac_count={ac_count} > 63 и без EOB.")


            decoded_units.append((dc_category, dc_vli_bits_str, ac_rle_pairs))

    except EOFError as e:
         print(f"Предупреждение: Ошибка конца потока при декодировании блока {len(decoded_units) + 1}: {e}. Декодировано {len(decoded_units)} блоков.")
         pass
    except ValueError as e:
         print(f"Ошибка значения при декодировании блока {len(decoded_units) + 1}: {e}")
         pass

    return decoded_units
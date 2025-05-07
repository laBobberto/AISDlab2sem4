import numpy
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt

from constants import Bites_for_param, ByteOrder, Channels


def image_to_raw_data(image_path: str) -> [int, int, int, str]:
    try:
        with Image.open(image_path) as img:
            raw_data = img.tobytes()
            width, height = img.size
            mode = img.mode
            return raw_data, width, height, mode
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути: {image_path}", file=sys.stderr)
        return None, None, None, None
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {e}", file=sys.stderr)
        return None, None, None, None

def convert_image_to_raw_data(image_path: str, raw_image_path: str) -> None:
    raw_data, width, height, mode = image_to_raw_data(image_path)
    if mode == "RGB":
        mode = 0

    with open(raw_image_path, "wb") as f:
        f.write(mode.to_bytes(Bites_for_param, byteorder=ByteOrder))
        f.write(width.to_bytes(Bites_for_param, byteorder=ByteOrder))
        f.write(height.to_bytes(Bites_for_param, byteorder=ByteOrder))
        f.write(raw_data)

def get_raw_image_param(raw_image_path: str) -> [int, int]:
    with open(raw_image_path, "rb") as f:
        mode = int.from_bytes(f.read(Bites_for_param), byteorder=ByteOrder)
        width = int.from_bytes(f.read(Bites_for_param), byteorder=ByteOrder)
        height = int.from_bytes(f.read(Bites_for_param), byteorder=ByteOrder)
    return mode, width, height

def raw_image_to_matrix(raw_image_path: str) -> np.ndarray:
    with open(raw_image_path, "rb") as f:
        mode = int.from_bytes(f.read(Bites_for_param), byteorder=ByteOrder)
        width = int.from_bytes(f.read(Bites_for_param), byteorder=ByteOrder)
        height = int.from_bytes(f.read(Bites_for_param), byteorder=ByteOrder)

        image_matrix_1d = np.fromfile(f, dtype=np.uint8)

        if width * height * Channels != image_matrix_1d.size:
            print("Неверный размер изображения", file=sys.stderr)
            return None
        else:
            image_matrix_2d = image_matrix_1d.reshape((height, width, Channels))
            return image_matrix_2d

def display_image_matrix(image_matrix: np.ndarray, title: str = "Image"):
    if image_matrix is None:
        print("Неверные данные", file=sys.stderr)
        return

    plt.figure()
    plt.imshow(image_matrix)
    plt.title(title)
    plt.axis('off')
    plt.show()

def split_into_blocks(image_matrix: np.ndarray, block_size: int, padding_value: int = 0) -> np.ndarray:
    """
    Разбивает матрицу изображения на блоки NxN, добавляя заполнение при необходимости.

    Args:
        image_matrix: Входная матрица изображения (NumPy array).
                      Может быть 2D (градации серого) или 3D (цветное: Height x Width x Channels).
        block_size: Размер блока N (N x N). Должен быть положительным целым числом.
        padding_value: Значение, которым будут заполнены неполные блоки (по умолчанию 0).

    Returns:
        NumPy array, содержащий блоки изображения.
        Форма выходного массива:
        (num_blocks_v, num_blocks_h, block_size, block_size, ...)
        где ... - размеры каналов, если изображение цветное.
        Возвращает None в случае ошибки.
    """
    if not isinstance(image_matrix, np.ndarray):
        print("Ошибка: Входные данные не являются массивом NumPy.", file=sys.stderr)
        return None

    if image_matrix.ndim not in [2, 3]:
        print("Ошибка: Входные данные должны быть 2D или 3D массивом NumPy.", file=sys.stderr)
        return None

    if not isinstance(block_size, int) or block_size <= 0:
        print("Ошибка: block_size должен быть положительным целым числом.", file=sys.stderr)
        return None

    height, width = image_matrix.shape[:2]
    num_channels = image_matrix.shape[2] if image_matrix.ndim == 3 else 1

    height_padded = ((height + block_size - 1) // block_size) * block_size
    width_padded = ((width + block_size - 1) // block_size) * block_size

    pad_height = height_padded - height
    pad_width = width_padded - width

    if image_matrix.ndim == 2:
        padding_config = ((0, pad_height), (0, pad_width))
    else:
        padding_config = ((0, pad_height), (0, pad_width), (0, 0))

    padding_value_typed = np.array(padding_value, dtype=image_matrix.dtype)
    padded_image = np.pad(image_matrix, padding_config, mode='constant', constant_values=padding_value_typed)

    num_blocks_v = height_padded // block_size
    num_blocks_h = width_padded // block_size

    if image_matrix.ndim == 2:
        blocks = padded_image.reshape(num_blocks_v, block_size, num_blocks_h, block_size)
        blocks = blocks.transpose(0, 2, 1, 3)
    else:
        blocks = padded_image.reshape(num_blocks_v, block_size, num_blocks_h, block_size, num_channels)
        blocks = blocks.transpose(0, 2, 1, 3, 4)

    return blocks
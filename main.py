import csv
import os
from pathlib import Path
from jpeg_compressor import compress_image
from jpeg_decompressor import decompress_image

def write_table(filename, new_data):
    file_exists = Path(filename).exists()

    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Коэффициент качества сжатия", "Размер сжатого файла"])
        writer.writerow(new_data)




files = os.listdir("data")

for file in files:
    os.makedirs(file[:-4], exist_ok=True)
    current_folder = file[:-4]

    for i in range(0, 101, 5):
        if i == 0:
            quality = 1
        else:
            quality = i

        current_raw_file = f"{current_folder}/{file[:-4]} {quality}.raw"
        compress_image(f"data/{file}", current_raw_file, quality=quality)

        if i in [0, 20, 40, 60, 80, 100]:
            decompress_image(current_raw_file, f"{current_folder}/{file[:-4]} {i}.png")

        write_table(f"{current_folder}/{file[:-4]}.csv", [i, os.path.getsize(current_raw_file)])
        os.remove(current_raw_file)
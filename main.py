from jpeg_compressor import compress_image
from jpeg_decompressor import decompress_image

for i in range(1, 101, 5):
    compress_image("data/Lenna.png", "data/Lenna.raw", quality=i)
    # display_compressed_image("data/Tree.raw")
    decompress_image("data/Lenna.raw", f"decompressed/Lenna {i}.png")

# compress_image("data/Lenna.png", "data/Lenna.raw", quality=1)
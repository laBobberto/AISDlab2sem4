from PIL import Image
import sys

def image_to_raw_data(image_path: str) -> [int, int, int, int]:
    try:
        with Image.open(image_path) as img:
            raw_data = img.tobytes()
            width, height = img.size
            mode = img.mode
            return raw_data, width, height, mode
    except FileNotFoundError:
        print(f"Error: File not found at path: {image_path}", file=sys.stderr)
        return None, None, None, None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}", file=sys.stderr)
        return None, None, None, None

def convert_image_to_raw_data(image_path: str, raw_data_path: str):
    raw_data, width, height, mode = image_to_raw_data(image_path)

    with open(raw_data_path, "wb") as f:
        f.write(raw_data)



a = 100
print(int.from_bytes(a.to_bytes(40, byteorder="little"), byteorder="little"))

#
# if __name__ == '__main__':
#     # Example usage (requires Pillow and test image files)
#     # Create dummy files or use your own for testing
#     # Example: Save a simple image
#     # from PIL import Image
#     # img_rgb = Image.new('RGB', (60, 30), color = 'red')
#     # img_rgb.save("test_image.png")
#     # img_rgb.save("test_image.jpg")
#
#     png_file = "data/Lenna.png"
#     jpeg_file = "data/Tree.jpg"
#     non_existent_file = "non_existent.png"
#
#     print(f"Attempting to read: {png_file}")
#     raw_png_data, png_width, png_height, png_mode = image_to_raw_data(png_file)
#     if raw_png_data is not None:
#         print(f"Read successful. Size: {png_width}x{png_height}, Mode: {png_mode}, Data length: {len(raw_png_data)}")
#
#     print("-" * 20)
#
#     print(f"Attempting to read: {jpeg_file}")
#     raw_jpeg_data, jpeg_width, jpeg_height, jpeg_mode = image_to_raw_data(jpeg_file)
#     if raw_jpeg_data is not None:
#         print(f"Read successful. Size: {jpeg_width}x{jpeg_height}, Mode: {jpeg_mode}, Data length: {len(raw_jpeg_data)}")
#
#     print("-" * 20)
#
#     print(f"Attempting to read: {non_existent_file}")
#     raw_none_data, none_width, none_height, none_mode = image_to_raw_data(non_existent_file)
#     if raw_none_data is None:
#         print("As expected, file was not read.")
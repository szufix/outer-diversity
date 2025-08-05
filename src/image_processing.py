from PIL import Image

def create_image_grid(image_paths, a, b, output_path="output.png"):
    if len(image_paths) != a * b:
        raise ValueError(
            f"Number of images ({len(image_paths)}) does not match grid size {a}x{b} ({a * b})")

    # Open all images and ensure they're the same size
    images = [Image.open(path).convert("RGBA") for path in image_paths]
    widths, heights = zip(*(img.size for img in images))

    # Ensure all images are the same size (optional: you can resize if needed)
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        raise ValueError(
            "All images must be the same size. Resize them before using this function.")

    img_width, img_height = images[0].size

    # Create a new blank image to paste into
    grid_image = Image.new('RGBA', (a * img_width, b * img_height))

    # Paste images into grid
    for idx, img in enumerate(images):
        x = (idx % a) * img_width
        y = (idx // a) * img_height
        grid_image.paste(img, (x, y))

    # Save output
    grid_image.save(output_path)
    print(f"Saved grid image to {output_path}")

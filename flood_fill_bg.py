from PIL import Image, ImageDraw
import sys

def remove_bg_floodfill(image_path, output_path, tolerance=50):
    img = Image.open(image_path).convert("RGBA")
    
    # Create a mask by flood filling from the top-left corner (0,0)
    # We'll use a temporary image to find the background
    # PIL ImageDraw.floodfill requires RGB, so we work on RGB version
    rgb_img = img.convert("RGB")
    
    # The background color is assumed to be the color at (0,0)
    bg_color = rgb_img.getpixel((0, 0))
    
    # We create a mask image initialized to black (0)
    mask = Image.new("L", rgb_img.size, 0)
    
    # Unfortunately PIL's floodfill modifies the image directly and doesn't return a mask easily.
    # Instead, let's just do a simple BFS flood fill manually or use scipy/skimage.
    # Since we might not have them, let's write a simple BFS.
    
    width, height = img.size
    pixels = img.load()
    mask_pixels = mask.load()
    
    def color_diff(c1, c2):
        return sum(abs(a - b) for a, b in zip(c1[:3], c2[:3]))
    
    # Start points: 4 corners
    stack = [(0,0), (width-1, 0), (0, height-1), (width-1, height-1)]
    visited = set(stack)
    
    while stack:
        x, y = stack.pop()
        
        # If it's similar to bg color, make it transparent
        if color_diff(pixels[x, y], bg_color) < tolerance:
            pixels[x, y] = (0, 0, 0, 0) # Make transparent
            
            # Add neighbors
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))

    img.save(output_path)
    print(f"Flood-filled {image_path} -> {output_path}")

if __name__ == "__main__":
    base_dir = "assets/images/"
    files = ["hero_avatar.png", "detective_avatar.png", "data_avatar.png"]
    
    # Since we previously overwrote the files with the simple script,
    # let's hope the artifacts are still in the original locations to start fresh.
    # Actually, we can just process the current files. The background is already transparent 
    # except for the "cone sides" (corners). 
    # Wait! The corners are NOT transparent. If we flood-fill from corners, it will remove them!
    
    for file in files:
        path = base_dir + file
        try:
            remove_bg_floodfill(path, path, tolerance=100)
        except Exception as e:
            print(f"Error processing {file}: {e}")

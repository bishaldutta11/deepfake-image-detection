import sys
from PIL import Image
import numpy as np

def make_white_transparent(image_path, output_path, tolerance=220):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    data = np.array(img)
    
    # Get RGB components
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # Find pixels where R, G, and B are all greater than tolerance (near white)
    white_areas = (r > tolerance) & (g > tolerance) & (b > tolerance)
    
    # Set alpha channel to 0 for those pixels
    data[:,:,3][white_areas] = 0
    
    # To improve edges, we can apply a small anti-aliasing or just keep it simple
    img_out = Image.fromarray(data)
    img_out.save(output_path)
    print(f"Processed {image_path} -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tolerance = int(sys.argv[1])
    else:
        tolerance = 230
        
    base_dir = "assets/images/"
    files = ["hero_avatar.png", "detective_avatar.png", "data_avatar.png"]
    
    for file in files:
        path = base_dir + file
        try:
            make_white_transparent(path, path, tolerance)
        except Exception as e:
            print(f"Error processing {file}: {e}")

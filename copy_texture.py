import os
import shutil 
import skimage 
NEED_HORIZONTAL_FLIP = False

def main():
    input_dir = '/pure/t1/project/vggt_bary/data/shiny_extended_1008/cake'
    output_dir = '/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake_obj_v3'
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(os.listdir(input_dir))
    for idx, file in enumerate(files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"{idx:02d}.png")
        if NEED_HORIZONTAL_FLIP:
            image = skimage.io.imread(input_path)
            flipped_image = image[:, ::-1]  # Flip horizontally
            skimage.io.imsave(output_path, flipped_image)
            print(f"Flipped and saved {input_path} to {output_path}")
        else:
            # If no flipping is needed, just copy the file
            shutil.copy(input_path, output_path)
        print(f"Copied {input_path} to {output_path}")

if __name__ == "__main__":
    main()
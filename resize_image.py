import skimage 
import os 
from tqdm.auto import tqdm

def main():
    input_dir = '/pure/t1/project/vggt_bary/data/shiny_extended/cake'
    output_dir = '/pure/t1/project/vggt_bary/data/shiny_extended_1008/cake'
    files = os.listdir(input_dir)
    print(files)
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        image = skimage.io.imread(input_path)
        resized_image = skimage.transform.resize(image, (756, 1008), anti_aliasing=True)
        resized_image = skimage.img_as_ubyte(resized_image)
        skimage.io.imsave(output_path, resized_image)


if __name__ == "__main__":
    main()
import os
import shutil 

def main():
    input_dir = '/pure/t1/project/vggt_bary/data/shiny_extended_1008/cake'
    output_dir = '/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake_obj'
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    for idx, file in enumerate(files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"{idx:02d}.png")
        shutil.copy(input_path, output_path)
        print(f"Copied {input_path} to {output_path}")

if __name__ == "__main__":
    main()
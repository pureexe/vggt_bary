import trimesh
import os

def convert_obj_dir_to_glb_dir(input_dir, output_dir):
    """
    Converts all OBJ files in an input directory (and their associated MTLs)
    to GLB files, saving them to an output directory.

    Args:
        input_dir (str): The path to the directory containing .obj and .mtl files.
        output_dir (str): The path to the directory where converted .glb files will be saved.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Scanning input directory: {input_dir}")
    converted_count = 0
    skipped_count = 0

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.obj'):
            obj_name = os.path.splitext(filename)[0] # Get base name without extension
            obj_path = os.path.join(input_dir, filename)
            output_glb_path = os.path.join(output_dir, f"{obj_name}.glb")

            print(f"\n--- Processing '{filename}' ---")
            
            try:
                # trimesh automatically looks for the .mtl file (same base name)
                # and associated textures in the same directory as the .obj.
                print(f"Loading OBJ file: {obj_path}")
                scene = trimesh.load(obj_path)
                print("OBJ file loaded successfully.")

                print(f"Exporting to GLB file: {output_glb_path}")
                scene.export(output_glb_path, file_type='glb')
                print(f"Successfully converted '{filename}' to '{os.path.basename(output_glb_path)}'")
                converted_count += 1

            except Exception as e:
                print(f"An error occurred during conversion of '{filename}': {e}")
                skipped_count += 1
                
    print("\n--- Conversion Summary ---")
    print(f"Total OBJ files converted: {converted_count}")
    print(f"Total OBJ files skipped due to errors: {skipped_count}")
    print(f"Output GLB files saved to: {output_dir}")


if __name__ == "__main__":
    # --- Configuration ---
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define your input and output directory names
    # Example: If your obj/mtl files are in a 'input_models' folder
    # and you want to save GLB files to 'output_glbs' folder,
    # both relative to where your script is.
    input_directory_name = '/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake_obj' # <--- IMPORTANT: Change this to your input directory name
    output_directory_name = '/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake_glb' # <--- IMPORTANT: Change this to your desired output directory name

    # Construct the full paths for the directories
    input_dir_path = os.path.join(script_dir, input_directory_name)
    output_dir_path = os.path.join(script_dir, output_directory_name)

    # Call the conversion function
    convert_obj_dir_to_glb_dir(input_dir_path, output_dir_path)

    # Example of how to use it with absolute paths if directories are elsewhere:
    # convert_obj_dir_to_glb_dir('/path/to/your/input/folder', '/path/to/save/output/folder')
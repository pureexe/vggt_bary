import os 
from safetensors.torch import save_file, load_file
import torch 
import numpy as np
from tqdm.auto import tqdm

def save_mesh_as_obj(coords: np.ndarray, obj_filename: str, texture_filename: str = "texture.png"):
    """
    Save a structured mesh as a .obj file with UVs and texture reference.

    Args:
        coords (np.ndarray): Array of shape [H, W, 3] containing 3D coordinates.
        obj_filename (str): Path to save the .obj file.
        texture_filename (str): Texture file name to reference in .mtl file (assumed to be in same folder).
    """
    H, W, _ = coords.shape
    mtl_filename = obj_filename.replace('.obj', '.mtl')
    obj_basename = obj_filename.split('/')[-1]
    mtl_basename = mtl_filename.split('/')[-1]

    with open(obj_filename, 'w') as f:
        # Reference to material
        f.write(f"mtllib {mtl_basename}\n")
        f.write(f"usemtl material_0\n")

        # Write vertices
        for y in range(H):
            for x in range(W):
                vx, vy, vz = coords[y, x]
                f.write(f"v {vx} {vy} {vz}\n")

        # Write UV coordinates (normalized [0,1])
        for y in range(H):
            for x in range(W):
                u = x / (W - 1)
                v = 1.0 - y / (H - 1)  # flip v to match image coords
                f.write(f"vt {u:.6f} {v:.6f}\n")

        # Write face definitions (vertices + texture coords)
        def vertex_index(y, x):
            return y * W + x + 1  # OBJ indices are 1-based

        for y in range(H - 1):
            for x in range(W - 1):
                v0 = vertex_index(y, x)
                v1 = vertex_index(y, x + 1)
                v2 = vertex_index(y + 1, x + 1)
                v3 = vertex_index(y + 1, x)

                # Each face has both vertex and texture coordinate index (v/vt)
                f.write(f"f {v0}/{v0} {v1}/{v1} {v2}/{v2}\n")
                f.write(f"f {v0}/{v0} {v2}/{v2} {v3}/{v3}\n")

    # Write corresponding MTL file
    with open(mtl_filename, 'w') as f:
        f.write("newmtl material_0\n")
        f.write("Ka 0.000 0.000 0.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.0\n")
        f.write("illum 0\n")
        f.write(f"map_Kd {texture_filename}\n")


@torch.inference_mode()
def main():
    INPUT_FILE = "/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake.safetensors"
    # read point map 
    data = load_file(INPUT_FILE)
    point_map = data["point_map_by_unprojection"]
    # print(f"Point map shape: {point_map.shape}")
    for image_id in tqdm(range(point_map.shape[0])):
        coords = point_map[image_id].cpu().numpy()
        # print(f"Coords shape: {coords.shape}")
        # Save as OBJ file
        obj_filename = f"/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake_obj/{image_id:02d}.obj"
        save_mesh_as_obj(coords, obj_filename, texture_filename=f"{image_id:02d}.png")
        print(f"Saved mesh to {obj_filename}")

if __name__ == "__main__":
    main()
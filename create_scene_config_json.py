import os 
from safetensors.torch import save_file, load_file
import torch 
import numpy as np
from tqdm.auto import tqdm
import json

def convert_opencv_to_opengl_extrinsic(extrinsic_cv):
    """
    Convert a [3x4] OpenCV extrinsic matrix to OpenGL [4x4] view matrix.

    Args:
        extrinsic_cv (np.ndarray): shape [3,4], OpenCV extrinsic matrix [R|t]

    Returns:
        np.ndarray: shape [4,4], OpenGL-style view matrix
    """
    assert extrinsic_cv.shape == (3, 4), "Input must be 3x4 matrix"

    # Convert to 4x4 matrix
    extrinsic_cv_4x4 = np.eye(4)
    extrinsic_cv_4x4[:3, :4] = extrinsic_cv

    # Coordinate system conversion matrix (flip Y and Z axes)
    flip_yz = np.diag([1, -1, -1, 1])

    # Apply the coordinate system flip
    extrinsic_gl = flip_yz @ extrinsic_cv_4x4

    return extrinsic_gl

@torch.inference_mode()
def main():
    INPUT_FILE = "/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake.safetensors"
    OUTPUT_FILE = "/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake.json"
    # read point map 
    data = load_file(INPUT_FILE)
    intrinsic = data["intrinsic"]
    extrinsic = data["extrinsic"]
    output = []
    for image_id in tqdm(range(intrinsic.shape[0])):
        intrinsic_matrix = intrinsic[image_id].cpu().numpy()
        extrinsic_matrix = extrinsic[image_id].cpu().numpy()
        extrinsic_matrix = convert_opencv_to_opengl_extrinsic(extrinsic_matrix)
        # print(f"Intrinsic shape: {intrinsic_matrix.shape}, Extrinsic shape: {extrinsic_matrix.shape}")
        output.append({
            "glb": f"http://localhost/vggt_bary/data/cake/{image_id:02d}.glb",
            "intrinsic": intrinsic_matrix.tolist(),
            "extrinsic": extrinsic_matrix.tolist()
        })
    # save to json file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()
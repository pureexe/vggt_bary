import os 
from safetensors.torch import save_file, load_file
import json 
import numpy as np

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


def main():
    INPUT_FILE = "/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake.safetensors"
    data = load_file(INPUT_FILE)
    intrinsic = data["intrinsic"][0]
    extrinsic = data["extrinsic"][0]
    output = []
    for i in range(intrinsic.shape[0]):
        output.append({
            "intrinsic": intrinsic[i].cpu().numpy().tolist(),
            "extrinsic": extrinsic[i].cpu().numpy().tolist(),  #convert_opencv_to_opengl_extrinsic(extrinsic[i].cpu().numpy()).tolist(),
            "glb": f'http://localhost/vggt_bary/data/cake/{i:02d}.glb'
        })
    output_file = "/pure/t1/project/vggt_bary/output/shiny_extended_1008/cake_direct.json"
    with open(output_file, 'w') as f:
        print(len(output), "GLB JSON config entries")
        json.dump(output, f, indent=4)
    print(f"Saved GLB JSON config to {output_file}")

if __name__ == "__main__":
    main()
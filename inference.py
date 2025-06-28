import os
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from safetensors.torch import save_file

INPUT_DIR = "/pure/t1/project/vggt_bary/data/shiny_extended_1008/cake"
OUPUT_DIR = "/pure/t1/project/vggt_bary/output/shiny_extended_1008/"
OUTPUT_FILE = "cake.safetensors"
os.makedirs(OUPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
image_names = [os.path.join(INPUT_DIR, file) for file in sorted(os.listdir(INPUT_DIR)) if file.endswith(('.png', '.jpg', '.jpeg'))]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        # predictions = model(images)
        print(f"Input images shape: {images.shape}")
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
    
        print(f"Compute camera head")
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]

        print(f"Pose encoding shape: {pose_enc.shape}")
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # save extrinsic and intrinsic matrices
        output = {}
        output["extrinsic"] = extrinsic
        output["intrinsic"] = intrinsic

        print(f"Saved extrinsic and intrinsic matrices to {OUPUT_DIR}") 
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        output["depth_map"] = depth_map
        output["depth_conf"] = depth_conf
        
        # Predict Point Maps
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

        print(f"Depth map shape: {depth_map.shape}, Depth confidence shape: {depth_conf.shape}")
         # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))
        
        output["point_map_by_unprojection"] = torch.from_numpy(point_map_by_unprojection)
        output["point_map"] = point_map
        # save point map
        point_map_path = os.path.join(OUPUT_DIR, OUTPUT_FILE)
        save_file(output, point_map_path)
        print(f"Saved extrinsic, intrinsic, depth map, and point map to {OUPUT_DIR}")




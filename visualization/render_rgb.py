import open3d as o3d
import numpy as np

# Load mesh
mesh = o3d.io.read_triangle_mesh("data/scannetv2/input/org_ply/org_val_ply/scene0011_00_vh_clean_2.ply")
mesh.compute_vertex_normals()

# Extract intrinsics
K = np.loadtxt("data/scannetv2/input/scannetv2_images/val/scene0011_00/intrinsics/intrinsic_color.txt")
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
width, height = round(cx * 2), round(cy * 2)  # Use actual ScanNet image resolution

# Load and invert pose
pose = np.loadtxt("data/scannetv2/input/scannetv2_images/val/scene0011_00/pose/0.txt")
extrinsic = np.linalg.inv(pose)

# Create intrinsic object
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Create scene and renderer
render = o3d.visualization.rendering.OffscreenRenderer(width, height)
render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background
render.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())

# Set camera
render.setup_camera(intrinsic, extrinsic)

# Render to image
image = render.render_to_image()
o3d.io.write_image("render_000000.png", image)

import trimesh
import numpy as np
import os
import cv2

def generate_mtl(obj_path, material_type='soft'):
    """
    Auto-generate a PBR-like .mtl file mapping for TripoSR.
    """
    base_name = os.path.splitext(obj_path)[0]
    mtl_path = f"{base_name}.mtl"
    obj_filename = os.path.basename(obj_path)
    mtl_filename = os.path.basename(mtl_path)
    material_name = f"mat_{os.path.splitext(obj_filename)[0]}"
    
    with open(mtl_path, 'w') as f:
        f.write(f"# Auto-generated MTL by CAST Enhanced\n")
        f.write(f"newmtl {material_name}\n")
        
        # PBR fake attributes
        if 'metal' in material_type:
            f.write("Ka 0.200 0.200 0.200\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.800 0.800 0.800\n")
            f.write("Ns 200.000\n")  # High gloss
            f.write("illum 2\n")
        elif 'glass' in material_type:
            f.write("Ka 0.000 0.000 0.000\n")
            f.write("Kd 1.000 1.000 1.000\n")
            # If reflection focus, boost specular, if transp focus, boost transparent
            if 'refl' in material_type:
                f.write("Ks 1.000 1.000 1.000\nNs 300.000\nd 0.8\nillum 3\n")
            else:
                f.write("Ks 0.800 0.800 0.800\nNs 150.000\nd 0.3\nillum 4\n")
        elif 'plastic' in material_type:
            f.write("Ka 0.100 0.100 0.100\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.300 0.300 0.300\n")
            f.write("Ns 80.000\n")  # Semi-gloss
            f.write("illum 2\n")
        elif 'wood' in material_type:
            f.write("Ka 0.100 0.100 0.100\n")
            f.write("Kd 0.800 0.800 0.800\n")
            f.write("Ks 0.100 0.100 0.100\n")
            f.write("Ns 20.000\n")  # Rough
            f.write("illum 2\n")
        elif 'stone' in material_type:
            f.write("Ka 0.050 0.050 0.050\n")
            f.write("Kd 0.700 0.700 0.700\n")
            f.write("Ks 0.050 0.050 0.050\n")
            f.write("Ns 10.000\n")  # Very rough
            f.write("illum 2\n")
        else:  # soft / fabric / default
            f.write("Ka 0.100 0.100 0.100\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write("Ns 5.000\n")  # Very matte
            f.write("illum 1\n")
            
        # Try to bind the generated texture if it exists
        texture_path = f"{base_name}.png"
        if os.path.exists(texture_path):
            f.write(f"map_Kd {os.path.basename(texture_path)}\n")

    return mtl_filename, material_name

def link_mtl_to_obj(obj_path, mtl_filename, material_name):
    """ Inject mtllib and usemtl into the obj, removing old ones """
    with open(obj_path, 'r') as f:
        raw_lines = f.readlines()
        
    # Clean out old material mappings from TripoSR
    lines = [line for line in raw_lines if not line.startswith('mtllib') and not line.startswith('usemtl')]
        
    for idx, line in enumerate(lines):
        if line.startswith('v '):
            lines.insert(idx, f"mtllib {mtl_filename}\nusemtl {material_name}\n")
            break
            
    with open(obj_path, 'w') as f:
        f.writelines(lines)

def enhance_texture_super_res(image_path):
    """
    Simulate Texture Super Resolution (Sharpness + Denoising)
    Since Real-ESRGAN dependencies are heavy, we use OpenCV USM directly as a lightweight fast replacement.
    """
    if not os.path.exists(image_path):
        return
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None: return
    
    # Handle RGBA (Alpha channel crash prevention)
    has_alpha = False
    if img.shape[-1] == 4:
        has_alpha = True
        alpha_channel = img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Unsharp Masking (USM) for sharpening
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    
    # Restore Alpha if it existed
    if has_alpha:
        sharpened = cv2.cvtColor(sharpened, cv2.COLOR_BGR2BGRA)
        sharpened[:, :, 3] = alpha_channel
        
    cv2.imwrite(image_path, sharpened)

def refine_mesh(obj_path, material_type='soft', texture_img_path=None):
    """
    Refine 3D mesh: Remove outliers, smooth surface, and cleanup clusters.
    material_type: 'soft', 'hard', 'metal', 'glass', 'wood'
    """
    if not os.path.exists(obj_path):
        print(f"File not found: {obj_path}")
        return
        
    # Enhance texture if provided
    if texture_img_path and os.path.exists(texture_img_path):
        print("  Super-Resolving & Sharpening Base Texture...")
        enhance_texture_super_res(texture_img_path)
        
    try:
        # PBR Material Generation
        mtl_filename, mat_name = generate_mtl(obj_path, material_type)
        print(f"  Generated PBR Material: {mtl_filename}")

        mesh = trimesh.load(obj_path)
        
        # Handle scenes (if GLB/GLTF)
        if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
            mesh = list(mesh.geometry.values())[0]
            
        print(f"Refining {os.path.basename(obj_path)} ({len(mesh.vertices)} vertices)...")

        # 1. Cluster Cleanup: Threshold-based approach (Keep parts > 1% volume)
        mesh = mesh.process(validate=True)
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            total_faces = sum(len(c.faces) for c in components)
            threshold = 0.01 * total_faces
            valid_components = [c for c in components if len(c.faces) >= threshold]
            print(f"  Found {len(components)} parts, keeping {len(valid_components)} essential parts (threshold: 1%).")
            mesh = trimesh.util.concatenate(valid_components)

        # 2. Smoothing: Material-Aware Volume-Preserving Smoothing (Taubin)
        if 'fabric' in material_type or 'silk' in material_type or 'soft' in material_type:
            # Using Taubin instead of Laplacian to prevent Shrinkage (Volume Loss)
            trimesh.smoothing.filter_taubin(mesh, iterations=12)
        elif 'wood' in material_type or 'stone' in material_type:
            # Less smoothing to preserve rough micro-details
            trimesh.smoothing.filter_taubin(mesh, iterations=4)
        else: # Metal, glass_transp, glass_refl, hard plastics
            # Taubin smoothing to preserve sharp edges and physics
            trimesh.smoothing.filter_taubin(mesh, iterations=10)

        # Explicitly compute perfect vertex normals to fix lighting/specularity issues
        mesh.fix_normals()
        
        # Export back with strict normals included for PBR
        mesh.export(obj_path, include_normals=True)
        
        # Link MTL
        link_mtl_to_obj(obj_path, mtl_filename, mat_name)
        
        print(f"  Refinement complete.")
        
    except Exception as e:
        print(f"  Error refining mesh: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        refine_mesh(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'soft')

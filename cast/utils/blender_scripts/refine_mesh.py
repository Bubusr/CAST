import bpy
import sys
import os

def refine_mesh(input_path, output_path, metallic, roughness):
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import the mesh
    if input_path.endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=input_path)
    elif input_path.endswith('.glb') or input_path.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=input_path)
    else:
        print(f"Unsupported file format: {input_path}")
        return

    # Get the imported objects
    selected = bpy.context.selected_objects
    if not selected:
        print("No objects selected after import")
        return

    # In case there are multiple parts, join them
    bpy.context.view_layer.objects.active = selected[0]
    if len(selected) > 1:
        bpy.ops.object.join()
        
    obj = bpy.context.active_object

    print(f"Refining object: {obj.name} with M={metallic}, R={roughness}")

    # 1. Subdivision Surface (Smooths mesh while PRESERVING UVs)
    subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = 1
    subsurf.render_levels = 2
    # No need to apply. GLB exporter will apply it automatically.

    # 2. Corrective Smooth (Smooth surface without losing volume)
    smooth = obj.modifiers.new(name="Corrective Smooth", type='CORRECTIVE_SMOOTH')
    smooth.iterations = 5
    smooth.use_only_smooth = True

    # 3. PBR Material Setup
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="PBR_Material")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    
    if principled:
        principled.inputs['Metallic'].default_value = metallic
        principled.inputs['Roughness'].default_value = roughness
        print(f"Applied PBR: Metallic={metallic}, Roughness={roughness}")

    # 4. Weighted Normals (For sharp looks on flat surfaces)
    wn = obj.modifiers.new(name="Weighted Normal", type='WEIGHTED_NORMAL')
    wn.keep_sharp = True
    # Exporter handles application

    # Export
    if output_path.endswith('.glb'):
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
    else:
        bpy.ops.wm.obj_export(filepath=output_path)
    
    print(f"Refined mesh saved to: {output_path}")

if __name__ == "__main__":
    # Args: -- [input] [output] [metallic] [roughness]
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx+1:]
        input_p = args[0]
        output_p = args[1]
        metal = float(args[2])
        rough = float(args[3])
        refine_mesh(input_p, output_p, metal, rough)
    except Exception as e:
        print(f"Error in blender script: {e}")

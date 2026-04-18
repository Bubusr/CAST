import bpy
import sys
import os

def clear_scene():
    """Xóa objects rác rưởi mặc định (Cube, Light, Camera)"""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in bpy.context.scene.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

def apply_quad_remesh(obj):
    """Sử dụng Voxel Remesh để nung chảy lưới tam giác lởm chởm thành Grid chuẩn"""
    remesh_mod = obj.modifiers.new(name="Remesh", type='REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = 0.02 # Càng thấp lưới càng mịn nhưng nặng
    remesh_mod.use_smooth_shade = True
    
    # Apply modifier
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Remesh")

def apply_corrective_smooth(obj, iterations=20):
    """Mượt lưới vải không bị móp/teo (Volume Preserving)"""
    smooth_mod = obj.modifiers.new(name="CorrectiveSmooth", type='CORRECTIVE_SMOOTH')
    smooth_mod.iterations = iterations
    smooth_mod.use_only_smooth = True
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="CorrectiveSmooth")

def apply_weighted_normal(obj):
    """Trị lởm chởm mặt kính/gỗ, ép các góc bóng loáng tăm tắp"""
    # Mở auto smooth trước khi apply
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = 3.14159 / 3.0 # 60 độ

    wn_mod = obj.modifiers.new(name="WeightedNormal", type='WEIGHTED_NORMAL')
    wn_mod.keep_sharp = True
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="WeightedNormal")

def assign_pbr_material(obj, material_type, texture_path=None):
    """Tạo Shader PBR chuẩn (Dựa vào tag vật liệu AI)"""
    mat = bpy.data.materials.new(name=f"Mat_{material_type}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    
    # Màu hoặc Texture
    if texture_path and os.path.exists(texture_path):
        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(texture_path)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    
    # Tinh chỉnh thông số Shader Nodes
    if 'glass_transp' in material_type:
        bsdf.inputs['Transmission'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = 0.05
        mat.blend_method = 'BLEND'
    elif 'glass_refl' in material_type:
        bsdf.inputs['Transmission'].default_value = 0.2
        bsdf.inputs['Roughness'].default_value = 0.0
        bsdf.inputs['Specular'].default_value = 1.0
    elif 'metal' in material_type:
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = 0.15
    elif 'fabric' in material_type or 'silk' in material_type:
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.9 # Rất lì
        # Tùy bản Blender (2.8 -> 3.x), có thông số Sheen cho lụa
        if 'Sheen' in bsdf.inputs:
            bsdf.inputs['Sheen'].default_value = 0.8
    elif 'wood' in material_type:
        bsdf.inputs['Roughness'].default_value = 0.7
        bsdf.inputs['Metallic'].default_value = 0.0

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def process_file(input_obj, output_glb, material_type='generic', texture_path=None):
    print(f"--- Đang chạy Blender Headless Pipeline cho: {input_obj} ---")
    clear_scene()
    
    # Import
    bpy.ops.import_scene.obj(filepath=input_obj, use_edges=True, use_smooth_groups=True, use_split_objects=False)
    
    # Lấy object vừa import
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # 1. Trị bệnh đa giác lởm chởm bằng Auto-Quad Remesh (Voxel)
    # apply_quad_remesh(obj) # Bật lên nếu VRAM trên Colab cho phép
    
    # 2. Xử lý bề mặt tùy chất liệu
    if material_type in ['fabric', 'silk']:
        apply_corrective_smooth(obj, iterations=25)
    elif material_type in ['wood', 'metal', 'glass_transp', 'glass_refl']:
        apply_weighted_normal(obj)
        
    # 3. Gắn PBR Shader chuyên nghiệp
    assign_pbr_material(obj, material_type, texture_path)
    
    # 4. Xuất GLB gộp chung ánh sáng/vật liệu
    print(f"--- Đang xuất cực phẩm 3D: {output_glb} ---")
    bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')


if __name__ == "__main__":
    # Cách Colab gọi file này: 
    # blender -b -P blender_process.py -- input.obj output.glb fabric /path/to/texture_4k.png
    argv = sys.argv
    try:
        index = argv.index("--") + 1
        input_obj = argv[index]
        output_glb = argv[index + 1]
        mat_type = argv[index + 2] if len(argv) > index + 2 else "generic"
        tex_path = argv[index + 3] if len(argv) > index + 3 else None
        
        process_file(input_obj, output_glb, mat_type, tex_path)
    except ValueError:
        print("Lỗi: Thiết lập sai đối số Blender truyền vào.")

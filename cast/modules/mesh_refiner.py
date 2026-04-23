import os
import subprocess
import gc
from pathlib import Path
from typing import List, Optional
import torch

from ..core.common import DetectedObject, Mesh3D

class MeshRefiner:
    """Module for refining 3D meshes using Headless Blender"""
    
    def __init__(self):
        import sys
        self.python_path = sys.executable
        self.script_path = Path(__file__).parent.parent / "utils" / "blender_scripts" / "refine_mesh.py"
        
    def refine_mesh(self, mesh_path: Path, obj_data: DetectedObject, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Refine a mesh using Blender headless script
        """
        if not mesh_path.exists():
            print(f"Mesh file not found: {mesh_path}")
            return None
            
        if output_dir := output_path or mesh_path.parent:
            # If output is same as input, we might need a temporary name or overwrite
            final_output = output_dir / f"{mesh_path.stem}_refined.glb"
        
        print(f"Refining mesh {mesh_path.name} using Blender...")
        
        # Build the command
        # Build the command using python interpreter (since bpy is installed via pip)
        cmd = [
            self.python_path,
            str(self.script_path),
            "--",
            str(mesh_path),
            str(final_output),
            str(obj_data.metallic),
            str(obj_data.roughness)
        ]
        
        try:
            # Run blender as a subprocess to keep memory isolated
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Blender output: {result.stdout}")
            
            if final_output.exists():
                print(f"Successfully refined mesh: {final_output}")
                return final_output
            else:
                print("Blender finished but output file not found")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error running Blender: {e}")
            print(f"Blender error output: {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error in MeshRefiner: {e}")
            return None
        finally:
            # Cleanup just in case
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def batch_refine(self, meshes: List[Mesh3D], objects: List[DetectedObject]) -> List[Optional[Path]]:
        """Process a batch of meshes"""
        results = []
        for mesh, obj in zip(meshes, objects):
            if mesh and mesh.file_path:
                refined_path = self.refine_mesh(mesh.file_path, obj)
                results.append(refined_path)
            else:
                results.append(None)
        return results

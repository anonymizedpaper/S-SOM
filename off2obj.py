import pymeshlab
import argparse
import os
import glob

def convert_off_to_obj(off_file, output_file):
    """Convert a single OFF file to OBJ format."""
    try:
        # Load the OFF file using pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(off_file)
        
        # Save as OBJ file
        ms.save_current_mesh(output_file)
        print(f"Converted: {off_file} -> {output_file}")
        return True
    except Exception as e:
        print(f"Error converting {off_file}: {str(e)}")
        return False

def main(folder):
    """Convert all .off files in the specified folder to .obj files."""
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory")
        return
    
    # Find all .off files in the folder
    off_pattern = os.path.join(folder, "*.off")
    off_files = glob.glob(off_pattern)
    
    if not off_files:
        print(f"No .off files found in directory: {folder}")
        return
    
    print(f"Found {len(off_files)} .off file(s) in directory: {folder}")
    
    # Convert each .off file to .obj
    success_count = 0
    for off_file in off_files:
        # Generate output filename (same name, different extension)
        base_name = os.path.splitext(off_file)[0]
        output_file = f"{base_name}.obj"
        
        if convert_off_to_obj(off_file, output_file):
            success_count += 1
    
    print(f"\nConversion complete: {success_count}/{len(off_files)} files converted successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert all .off files to .obj files in a folder')
    parser.add_argument('--folder', type=str, required=True,
                      help='Path to the folder containing .off files')
    
    args = parser.parse_args()
    main(args.folder)

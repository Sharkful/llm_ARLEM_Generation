import json

def struct_to_array(val):
    """Converts {"x": 1, "y": 2, "z": 3} to [1, 2, 3]"""
    if isinstance(val, dict) and 'x' in val and 'y' in val and 'z' in val:
        if 'w' in val:  # Handle Quaternions
            return [val['x'], val['y'], val['z'], val['w']]
        return [val['x'], val['y'], val['z']]
    return val

def convert_lab_data():
    with open('Artifacts/Data/Raw/Full_Lab_Transmission.json', 'r') as f:
        data = json.load(f)
    
    # Get Original size for stats comparison
    input_size = len(json.dumps(data))

    for i, module_str in enumerate(data.get("ActivityModules", [])):
        # Parse the nested stringified module
        module = json.loads(module_str)
        
        # 1. Add moduleType discriminator
        module["moduleType"] = "DemoModule" if module.get("prefabName") == "demoPrefab" else "StandardModule"
        module.pop("prefabName", None) 
        
        new_objects = []
        for obj in module.get("objects", []):
            # 2. Extract out the Nav Buttons entirely
            obj_name = obj.get("name", "").lower()
            if "brb" in obj_name or "brb" in obj.get("parentName", "").lower():
                continue
                
            # 3. Unity structs to arrays
            for key in ["position", "scale", "eulerAngles", "rotation", "color"]:
                if key in obj:
                    obj[key] = struct_to_array(obj[key])
                    
            # 4. Handle nested components & rename discriminator
            components = []
            for comp_str in obj.get("componentsToAdd", []):
                comp = json.loads(comp_str) if isinstance(comp_str, str) else comp_str
                if "name" in comp:
                    comp["componentType"] = comp.pop("name")
                components.append(comp)
                
            # 5. Move TMP into components and rename
            if "tmp" in obj:
                tmp_data = json.loads(obj["tmp"]) if isinstance(obj["tmp"], str) else obj["tmp"]
                tmp_data["componentType"] = "textMeshPro"
                components.append(tmp_data)
                del obj["tmp"]
                
            obj["componentsToAdd"] = components
            new_objects.append(obj)
            
        module["objects"] = new_objects
        
        # 6. Make Object Changes Sparse
        for clip in module.get("clips", []):
            new_changes = []
            for change in clip.get("objectChanges", []):
                # Remove Nav Button clip changes
                if "brb" in change.get("name", "").lower():
                    continue
                    
                # Strip unchanged struct fields and boolean flags
                for state_key, new_flag in [("position", "newPosition"), ("eulerAngles", "newEulerAngles"), ("scale", "newScale")]:
                    if not change.get(new_flag, False):
                        change.pop(state_key, None)
                    change.pop(new_flag, None) 
                
                # Strip default false/0 values to make it sparse
                if change.get("activationConditions") == 0:
                    del change["activationConditions"]
                if change.get("reactiveObject") is False:
                    del change["reactiveObject"]
                    
                # Convert remaining structs to arrays
                for key in ["position", "scale", "eulerAngles", "rotation", "color"]:
                    if key in change:
                        change[key] = struct_to_array(change[key])
                        
                # Rename tmp -> textMeshPro
                if "tmp" in change:
                    tmp_data = json.loads(change["tmp"]) if isinstance(change["tmp"], str) else change["tmp"]
                    change["textMeshPro"] = tmp_data
                    del change["tmp"]
                    
                new_changes.append(change)
            clip["objectChanges"] = new_changes
            
        # Put the object back in the array as a native dictionary, NOT a string!
        data["ActivityModules"][i] = module
        
    output_path = 'Artifacts/Data/Processed/gemini_optomized_moon_lab.json'
    # Write the clean data back to a new file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    # Stats
    output_size = len(json.dumps(data))
    print(f"Input size:  {input_size:>8,} chars")
    print(f"Output size: {output_size:>8,} chars")
    print(f"Reduction:   {(1 - output_size/input_size)*100:.1f}%")
    print(f"Written to:  {output_path}")
    print("Conversion Complete! Saved to gemini_optomized_moon_lab.json")

if __name__ == "__main__":
    convert_lab_data()
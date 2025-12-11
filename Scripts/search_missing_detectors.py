import os
import xml.etree.ElementTree as ET
from collections import defaultdict

def search_missing_detectors():
    """Search for missing detectors in XML files"""
    
    missing_detectors = ['AID02112', 'AID02204', 'AID04122', 'AID05224']
    
    print("SEARCHING FOR MISSING DETECTORS IN XML FILES")
    print("="*50)
    
    # Search in a few sample XML files
    sample_dirs = [
        r'c:\Users\maxch\MaxProjects\NLP\202212_modified',
        r'c:\Users\maxch\MaxProjects\NLP\202301_modified',
        r'c:\Users\maxch\MaxProjects\NLP\202302_modified'
    ]
    
    detector_found = defaultdict(list)
    
    for dir_path in sample_dirs:
        if not os.path.exists(dir_path):
            continue
            
        print(f"\nSearching in {dir_path}...")
        
        # Check first few files in each directory
        xml_files = [f for f in os.listdir(dir_path) if f.endswith('.xml')][:3]
        
        for xml_file in xml_files:
            file_path = os.path.join(dir_path, xml_file)
            
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Find all detector IDs in this file
                detectors_in_file = set()
                for detector in root.findall('.//detector'):
                    detector_id = detector.get('id')
                    if detector_id:
                        detectors_in_file.add(detector_id)
                
                # Check if any missing detectors are found
                for missing_det in missing_detectors:
                    if missing_det in detectors_in_file:
                        detector_found[missing_det].append(f"{dir_path}/{xml_file}")
                        
                print(f"  {xml_file}: {len(detectors_in_file)} detectors found")
                
            except Exception as e:
                print(f"  Error reading {xml_file}: {e}")
    
    print(f"\nRESULTS:")
    for detector in missing_detectors:
        if detector in detector_found:
            print(f"  {detector}: FOUND in {len(detector_found[detector])} files")
            for file_path in detector_found[detector][:2]:  # Show first 2 files
                print(f"    - {file_path}")
        else:
            print(f"  {detector}: NOT FOUND in sample files")
    
    # Also check what detectors ARE commonly found
    print(f"\nSAMPLE OF DETECTORS FOUND IN XML:")
    try:
        sample_file = os.path.join(sample_dirs[0], os.listdir(sample_dirs[0])[0])
        tree = ET.parse(sample_file)
        root = tree.getroot()
        
        all_detectors = []
        for detector in root.findall('.//detector'):
            detector_id = detector.get('id')
            if detector_id:
                all_detectors.append(detector_id)
        
        print(f"  Total detectors in {sample_file}: {len(set(all_detectors))}")
        print(f"  Sample: {sorted(set(all_detectors))[:10]}")
        
    except Exception as e:
        print(f"  Error reading sample file: {e}")

if __name__ == "__main__":
    search_missing_detectors()
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
    all_detectors_found = set()
    
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
                
                # Find all detector IDs in this file using correct XML structure
                detectors_in_file = set()
                for period in root.findall('.//period'):
                    for detector in period.findall('.//detector'):
                        detector_id_elem = detector.find('detector_id')
                        if detector_id_elem is not None:
                            detector_id = detector_id_elem.text
                            if detector_id:
                                detectors_in_file.add(detector_id)
                                all_detectors_found.add(detector_id)
                
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
    
    # Show what detectors ARE commonly found
    print(f"\nSAMPLE OF DETECTORS FOUND IN XML:")
    print(f"  Total unique detectors found: {len(all_detectors_found)}")
    print(f"  Sample detectors: {sorted(list(all_detectors_found))[:15]}")
    
    # Check if any of the missing detectors appear with slight variations
    print(f"\nCHECKING FOR SIMILAR DETECTOR IDs:")
    for missing_det in missing_detectors:
        similar = [d for d in all_detectors_found if missing_det[3:] in d or missing_det[:6] in d]
        if similar:
            print(f"  {missing_det} - Similar found: {similar}")
        else:
            print(f"  {missing_det} - No similar detectors found")

if __name__ == "__main__":
    search_missing_detectors()
import os
import xml.etree.ElementTree as ET
import glob
from collections import defaultdict

def check_detector_availability():
    """Check which detectors are actually available in XML files"""
    
    base_dir = r'c:\Users\maxch\MaxProjects\NLP'
    
    # Expected detectors from data_preprocessing_2.py
    expected_detectors = {
        'WHT': ['AID03204', 'AID03205', 'AID03206', 'AID03207', 'AID03208', 'AID03209', 'AID03210', 'AID03211',
                'AID04218', 'AID04219', 'AID03103', 'AID04104'],
        'CHT': ['AID01108', 'AID01109', 'AID01110', 'TDSIEC10002', 'TDSIEC10003', 'TDSIEC10004',
                'AID01208', 'AID01209', 'AID01211', 'AID01212', 'AID01213', 'AID05224', 'AID05225', 'AID05226', 'AID05109'],
        'EHT': ['AID04210', 'AID04212', 'AID04106', 'AID04107', 'AID04122', 'AID04110',
                'AID02204', 'AID02205', 'AID02206', 'AID02207', 'AID02208', 'AID02209', 'AID02210', 'AID02211', 'AID02212', 'AID02213', 'AID02214', 'AID07226']
    }
    
    # Find modified folders
    modified_folders = [d for d in os.listdir(base_dir) if d.endswith('_modified')]
    
    found_detectors = set()
    
    # Sample a few XML files to check detector availability
    for folder in modified_folders[:2]:  # Check first 2 folders
        folder_path = os.path.join(base_dir, folder)
        xml_files = glob.glob(os.path.join(folder_path, "*_processed.xml"))[:5]  # Sample 5 files
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for detector in root.findall('.//detector'):
                    detector_id = detector.find('detector_id').text
                    found_detectors.add(detector_id)
                    
            except:
                continue
    
    print("DETECTOR AVAILABILITY CHECK")
    print("="*50)
    
    all_expected = set()
    for tunnel_detectors in expected_detectors.values():
        all_expected.update(tunnel_detectors)
    
    for tunnel, detectors in expected_detectors.items():
        available = [d for d in detectors if d in found_detectors]
        missing = [d for d in detectors if d not in found_detectors]
        
        print(f"\n{tunnel} Tunnel:")
        print(f"  Expected: {len(detectors)}")
        print(f"  Available: {len(available)}")
        print(f"  Missing: {missing}")
    
    print(f"\nOverall:")
    print(f"  Expected total: {len(all_expected)}")
    print(f"  Found total: {len(found_detectors & all_expected)}")
    print(f"  Missing: {all_expected - found_detectors}")

if __name__ == "__main__":
    check_detector_availability()
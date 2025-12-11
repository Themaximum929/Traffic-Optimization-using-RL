import os
import xml.etree.ElementTree as ET
import pandas as pd

def debug_preprocessing():
    """Debug why WHT detectors are not in CSV output"""
    
    # Test with one XML file
    xml_file = r'c:\Users\maxch\MaxProjects\NLP\202508_modified\2025-08-30_processed.xml'
    
    detector_tunnel_map = {
        'AID03204': 'WHT', 'AID03205': 'WHT', 'AID03206': 'WHT', 'AID03207': 'WHT', 'AID03208': 'WHT',
        'AID03209': 'WHT', 'AID03210': 'WHT', 'AID03211': 'WHT',
        'AID04218': 'WHT', 'AID04219': 'WHT', 'AID03103': 'WHT', 'AID04104': 'WHT'
    }
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        date = root.find('date').text
        
        print(f"Processing file: {xml_file}")
        print(f"Date: {date}")
        
        wht_found = []
        all_detectors = []
        
        for period in root.findall('.//period'):
            time_slot = period.find('period_from').text
            
            for detector in period.findall('.//detector'):
                detector_id = detector.find('detector_id').text
                all_detectors.append(detector_id)
                
                if detector_id in detector_tunnel_map:
                    tunnel = detector_tunnel_map[detector_id]
                    direction = detector.find('direction').text
                    
                    lanes = detector.findall('.//lane')
                    print(f"Found {detector_id} ({tunnel}) at {time_slot}, direction: {direction}, lanes: {len(lanes)}")
                    
                    if tunnel == 'WHT':
                        wht_found.append(detector_id)
                        
                        for lane in lanes:
                            valid = lane.find('valid').text
                            speed = lane.find('speed').text
                            print(f"  Lane {lane.find('lane_id').text}: valid={valid}, speed={speed}")
        
        print(f"\nWHT detectors found: {set(wht_found)}")
        print(f"All unique detectors: {len(set(all_detectors))}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_preprocessing()
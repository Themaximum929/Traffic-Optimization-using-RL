import os
import xml.etree.ElementTree as ET
import pandas as pd

# Simplified test without sklearn
detector_tunnel_map = {
    'AID03204': 'WHT', 'AID03205': 'WHT', 'AID03206': 'WHT', 'AID03207': 'WHT', 'AID03208': 'WHT',
    'AID03209': 'WHT', 'AID03210': 'WHT', 'AID03211': 'WHT', 'AID04218': 'WHT', 'AID04219': 'WHT', 
    'AID03103': 'WHT', 'AID04104': 'WHT'
}

xml_file = r'c:\Users\maxch\MaxProjects\NLP\202508_modified\2025-08-30_processed.xml'

try:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    date = root.find('date').text
    
    data = []
    for period in root.findall('.//period'):
        time_slot = period.find('period_from').text
        timestamp = f"{date} {time_slot}"
        
        for detector in period.findall('.//detector'):
            detector_id = detector.find('detector_id').text
            
            # Check if detector is in tunnel map
            tunnel = detector_tunnel_map.get(detector_id)
            if tunnel is None:
                continue
                
            print(f"Processing {detector_id} -> {tunnel}")
            
            direction = detector.find('direction').text
            
            for lane in detector.findall('.//lane'):
                data.append({
                    'detector_id': detector_id,
                    'timestamp': timestamp,
                    'tunnel': tunnel,
                    'direction': direction,
                    'speed': float(lane.find('speed').text or 0),
                    'valid': lane.find('valid').text == 'Y'
                })
    
    df = pd.DataFrame(data)
    print(f"\nTotal records: {len(df)}")
    print(f"WHT records: {len(df[df['tunnel'] == 'WHT'])}")
    print(f"Valid records: {len(df[df['valid'] == True])}")
    
except Exception as e:
    print(f"Error: {e}")
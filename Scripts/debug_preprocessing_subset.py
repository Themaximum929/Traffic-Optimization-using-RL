import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob

class DebugPreprocessor:
    def __init__(self):
        self.detector_tunnel_map = {
            # WHT detectors
            'AID03204': 'WHT', 'AID03205': 'WHT', 'AID03206': 'WHT', 'AID03207': 'WHT', 'AID03208': 'WHT',
            'AID03209': 'WHT', 'AID03210': 'WHT', 'AID03211': 'WHT', 'AID04218': 'WHT', 'AID04219': 'WHT', 
            'AID03103': 'WHT', 'AID04104': 'WHT',
            # CHT detectors
            'AID01108': 'CHT', 'AID01109': 'CHT', 'AID01110': 'CHT', 'TDSIEC10002': 'CHT', 'TDSIEC10003': 'CHT', 
            'TDSIEC10004': 'CHT', 'AID01208': 'CHT', 'AID01209': 'CHT', 'AID01211': 'CHT', 'AID01212': 'CHT', 
            'AID01213': 'CHT', 'AID05224': 'CHT', 'AID05225': 'CHT', 'AID05226': 'CHT', 'AID05109': 'CHT',
            # EHT detectors
            'AID04210': 'EHT', 'AID04212': 'EHT', 'AID04106': 'EHT', 'AID04107': 'EHT', 'AID04122': 'EHT', 
            'AID04110': 'EHT', 'AID02204': 'EHT', 'AID02205': 'EHT', 'AID02206': 'EHT', 'AID02207': 'EHT', 
            'AID02208': 'EHT', 'AID02209': 'EHT', 'AID02210': 'EHT', 'AID02211': 'EHT', 'AID02212': 'EHT', 
            'AID02213': 'EHT', 'AID02214': 'EHT', 'AID07226': 'EHT', 'AID02112': 'EHT'
        }
    
    def parse_xml_debug(self, xml_file):
        """Parse XML with detailed debugging"""
        print(f"\n=== PARSING {os.path.basename(xml_file)} ===")
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            date = root.find('date').text
            print(f"Date: {date}")
            
            data = []
            detector_counts = {'WHT': 0, 'CHT': 0, 'EHT': 0, 'Unknown': 0}
            
            for period in root.findall('.//period'):
                time_slot = period.find('period_from').text
                
                for detector in period.findall('.//detector'):
                    detector_id = detector.find('detector_id').text
                    direction = detector.find('direction').text
                    
                    # Check tunnel mapping
                    tunnel = self.detector_tunnel_map.get(detector_id)
                    if tunnel is None:
                        detector_counts['Unknown'] += 1
                        continue
                    
                    detector_counts[tunnel] += 1
                    
                    # Process lanes
                    lanes = detector.findall('.//lane')
                    for lane in lanes:
                        lane_id = lane.find('lane_id').text
                        speed = float(lane.find('speed').text or 0)
                        occupancy = float(lane.find('occupancy').text or 0)
                        volume = int(lane.find('volume').text or 0)
                        valid = lane.find('valid').text == 'Y'
                        
                        data.append({
                            'detector_id': detector_id,
                            'tunnel': tunnel,
                            'direction': direction,
                            'lane_id': lane_id,
                            'speed': speed,
                            'occupancy': occupancy,
                            'volume': volume,
                            'valid': valid,
                            'time_slot': time_slot
                        })
            
            print(f"Detector counts: {detector_counts}")
            df = pd.DataFrame(data)
            
            if not df.empty:
                print(f"Total records: {len(df)}")
                print(f"Tunnel distribution:")
                for tunnel in ['WHT', 'CHT', 'EHT']:
                    count = len(df[df['tunnel'] == tunnel])
                    print(f"  {tunnel}: {count} records")
                
                # Sample WHT data
                wht_data = df[df['tunnel'] == 'WHT']
                if not wht_data.empty:
                    print(f"\nSample WHT data (first 3 records):")
                    sample = wht_data.head(3)[['detector_id', 'tunnel', 'speed', 'occupancy', 'valid']]
                    print(sample.to_string(index=False))
                else:
                    print("\nNO WHT DATA FOUND!")
            
            return df
            
        except Exception as e:
            print(f"ERROR: {e}")
            return pd.DataFrame()
    
    def validate_xml_vs_scraped(self, xml_file):
        """Validate scraped data matches XML attributes"""
        print(f"\n=== VALIDATION {os.path.basename(xml_file)} ===")
        
        # Parse with debug
        df = self.parse_xml_debug(xml_file)
        
        if df.empty:
            print("No data to validate")
            return
        
        # Check specific detector from XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find first WHT detector in XML
        for detector in root.findall('.//detector'):
            detector_id = detector.find('detector_id').text
            if detector_id in ['AID03204', 'AID03205', 'AID03206']:
                direction_xml = detector.find('direction').text
                
                # Find same detector in scraped data
                scraped = df[df['detector_id'] == detector_id]
                if not scraped.empty:
                    direction_scraped = scraped.iloc[0]['direction']
                    print(f"\nValidation for {detector_id}:")
                    print(f"  XML direction: {direction_xml}")
                    print(f"  Scraped direction: {direction_scraped}")
                    print(f"  Match: {direction_xml == direction_scraped}")
                    
                    # Check lane data
                    lanes_xml = detector.findall('.//lane')
                    lanes_scraped = len(scraped)
                    print(f"  XML lanes: {len(lanes_xml)}")
                    print(f"  Scraped lanes: {lanes_scraped}")
                    
                    if len(lanes_xml) > 0:
                        first_lane = lanes_xml[0]
                        speed_xml = float(first_lane.find('speed').text)
                        speed_scraped = scraped.iloc[0]['speed']
                        print(f"  XML speed: {speed_xml}")
                        print(f"  Scraped speed: {speed_scraped}")
                        print(f"  Speed match: {speed_xml == speed_scraped}")
                else:
                    print(f"\n{detector_id} NOT FOUND in scraped data!")
                break
    
    def process_folder(self):
        """Process 202508_modified folder"""
        folder_path = r'C:\Users\maxch\MaxProjects\NLP\202508_modified'
        xml_files = glob.glob(os.path.join(folder_path, "*_processed.xml"))[:3]  # Test first 3 files
        
        print(f"Found {len(xml_files)} XML files to test")
        
        all_data = []
        for xml_file in xml_files:
            df = self.parse_xml_debug(xml_file)
            if not df.empty:
                all_data.append(df)
            
            # Validate first file
            if xml_file == xml_files[0]:
                self.validate_xml_vs_scraped(xml_file)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"\n=== COMBINED RESULTS ===")
            print(f"Total records: {len(combined)}")
            print(f"Tunnel distribution:")
            for tunnel in ['WHT', 'CHT', 'EHT']:
                count = len(combined[combined['tunnel'] == tunnel])
                unique_detectors = combined[combined['tunnel'] == tunnel]['detector_id'].nunique()
                print(f"  {tunnel}: {count} records, {unique_detectors} unique detectors")
            
            # Save sample for inspection
            output_file = r'C:\Users\maxch\MaxProjects\NLP\Scripts\debug_sample.csv'
            combined.to_csv(output_file, index=False)
            print(f"\nSample data saved to: {output_file}")
        else:
            print("\nNO DATA PROCESSED!")

if __name__ == "__main__":
    debugger = DebugPreprocessor()
    debugger.process_folder()
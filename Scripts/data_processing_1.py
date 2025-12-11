import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import glob
from collections import defaultdict
import statistics

# Configuration
DETECTOR_LIST = ["AID03102","AID03104","AID04220","AID03103","AID03101","AID04219","AID04218","AID03106","AID03211","AID03210","AID03209","AID03208","AID03207","AID03206",
"AID03205","AID03204","AID04104","AID04221","AID01110","AID01109","AID01108","TDSIEC10004","TDSIEC10003","TDSIEC10002","TDSIEC10001","AID01112","AID01111","AID05110",
"AID01114","AID01115","AID01115","AID05227","AID05111","AID01213","AID05109","AID01212","AID05226","AID01211","AID01210","AID01209","AID01208","AID05225","AID01214",
"TDSIEC20002","AID04110","AID04109","AID04109","AID04121","AID04107","AID04106","AID04212","AID04210","AID04211","AID02104","AID02104","AID02214","AID02213","AID02212",
"AID02211","AID02210","AID02209","AID07226","AID02208","AID02207","AID02206","AID02205","AID04214","AID04111"] 
MORNING_START, MORNING_END = 7, 11  # 7-11 AM
EVENING_START, EVENING_END = 17, 21  # 5-9 PM
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def is_time_in_range(time_str):
    """Check if time is within 7-11AM or 5-9PM"""
    hour = int(time_str.split(':')[0])
    return (MORNING_START <= hour < MORNING_END) or (EVENING_START <= hour < EVENING_END)

def time_to_5min_slot(time_str):
    """Convert time to 5-minute slot (e.g., 07:03:30 -> 07:00:00)"""
    hour, minute, _ = map(int, time_str.split(':'))
    slot_minute = (minute // 5) * 5
    return f"{hour:02d}:{slot_minute:02d}:00"

def aggregate_lane_data(lane_data_list):
    """Aggregate lane data: sum volume, mean for others"""
    if not lane_data_list:
        return None
    
    total_volume = sum(d['volume'] for d in lane_data_list)
    speeds = [d['speed'] for d in lane_data_list if d['speed'] > 0]
    avg_speed = statistics.mean(speeds) if speeds else 0
    avg_occupancy = statistics.mean(d['occupancy'] for d in lane_data_list)
    avg_sd = statistics.mean(d['s.d.'] for d in lane_data_list)
    
    return {
        'lane_id': lane_data_list[0]['lane_id'],
        'speed': round(avg_speed, 1),
        'occupancy': round(avg_occupancy, 1),
        'volume': total_volume,
        's.d.': round(avg_sd, 1),
        'valid': 'Y'
    }

def process_xml_file(file_path):
    """Process single XML file and extract relevant data"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        date = root.find('date').text
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        directions = {}  # Store detector directions
        
        for period in root.findall('.//period'):
            period_from = period.find('period_from').text
            
            if not is_time_in_range(period_from):
                continue
                
            time_slot = time_to_5min_slot(period_from)
            
            for detector in period.findall('.//detector'):
                detector_id = detector.find('detector_id').text
                
                if detector_id not in DETECTOR_LIST:
                    continue
                    
                direction = detector.find('direction').text
                if detector_id not in directions:
                    directions[detector_id] = direction
                
                for lane in detector.findall('.//lane'):
                    lane_id = lane.find('lane_id').text
                    
                    lane_data = {
                        'lane_id': lane_id,
                        'speed': float(lane.find('speed').text or 0),
                        'occupancy': float(lane.find('occupancy').text or 0),
                        'volume': int(lane.find('volume').text or 0),
                        's.d.': float(lane.find('s.d.').text or 0)
                    }
                    
                    data[time_slot][detector_id][lane_id].append(lane_data)
        
        return date, data, directions
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def create_output_xml(date, aggregated_data, directions, output_path):
    """Create output XML with aggregated 5-minute data"""
    root = ET.Element('raw_speed_volume_list')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 
             'https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/SpeedVolOcc-BR.xsd')
    
    date_elem = ET.SubElement(root, 'date')
    date_elem.text = date
    
    periods_elem = ET.SubElement(root, 'periods')
    
    for time_slot in sorted(aggregated_data.keys()):
        period_elem = ET.SubElement(periods_elem, 'period')
        
        period_from_elem = ET.SubElement(period_elem, 'period_from')
        period_from_elem.text = time_slot
        
        # Calculate period_to (5 minutes later)
        hour, minute, _ = map(int, time_slot.split(':'))
        end_minute = minute + 5
        if end_minute >= 60:
            hour += 1
            end_minute -= 60
        period_to = f"{hour:02d}:{end_minute:02d}:00"
        
        period_to_elem = ET.SubElement(period_elem, 'period_to')
        period_to_elem.text = period_to
        
        detectors_elem = ET.SubElement(period_elem, 'detectors')
        
        for detector_id, lanes_data in aggregated_data[time_slot].items():
            detector_elem = ET.SubElement(detectors_elem, 'detector')
            
            detector_id_elem = ET.SubElement(detector_elem, 'detector_id')
            detector_id_elem.text = detector_id
            
            # Get direction from stored data
            direction_elem = ET.SubElement(detector_elem, 'direction')
            direction_elem.text = directions.get(detector_id, "Unknown")
            
            lanes_elem = ET.SubElement(detector_elem, 'lanes')
            
            for lane_id, lane_data in lanes_data.items():
                lane_elem = ET.SubElement(lanes_elem, 'lane')
                
                for key, value in lane_data.items():
                    elem = ET.SubElement(lane_elem, key.replace('.', '.'))
                    elem.text = str(value)
    
    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def get_processed_dates(output_dir):
    """Get list of already processed dates"""
    if not os.path.exists(output_dir):
        return set()
    processed_files = glob.glob(os.path.join(output_dir, "*_processed.xml"))
    return {os.path.basename(f).replace('_processed.xml', '') for f in processed_files}

def process_daily_data(date, date_data, directions, output_dir):
    """Process and save data for a single date"""
    print(f"Processing date: {date}")
    aggregated_daily = {}
    
    for time_slot, detectors in date_data.items():
        aggregated_daily[time_slot] = {}
        
        for detector_id, lanes in detectors.items():
            aggregated_daily[time_slot][detector_id] = {}
            
            for lane_id, lane_data_list in lanes.items():
                aggregated_lane = aggregate_lane_data(lane_data_list)
                if aggregated_lane:
                    aggregated_daily[time_slot][detector_id][lane_id] = aggregated_lane
    
    if aggregated_daily:
        output_file = os.path.join(output_dir, f"{date}_processed.xml")
        create_output_xml(date, aggregated_daily, directions, output_file)
        print(f"Created: {output_file}")
        return True
    else:
        print(f"No valid data for date: {date}")
        return False

def process_monthly_data(month_folder):
    """Process all XML files in a month folder"""
    print(f"Processing {month_folder}...")
    
    # Find subfolder
    subfolders = [d for d in os.listdir(month_folder) if os.path.isdir(os.path.join(month_folder, d))]
    if not subfolders:
        print(f"No subfolders found in {month_folder}")
        return
    
    subfolder = os.path.join(month_folder, subfolders[0])
    xml_files = glob.glob(os.path.join(subfolder, "*.xml"))
    
    if not xml_files:
        print(f"No XML files found in {subfolder}")
        return
    
    # Create output directory
    month_name = os.path.basename(month_folder)
    output_dir = os.path.join(BASE_DIR, f"{month_name}_modified")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check already processed dates
    processed_dates = get_processed_dates(output_dir)
    print(f"Already processed {len(processed_dates)} dates")
    
    print(f"Found {len(xml_files)} XML files to process")
    
    # Process files day by day
    daily_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    daily_directions = defaultdict(dict)
    current_date = None
    
    for i, xml_file in enumerate(xml_files, 1):
        #print(f"Processing file {i}/{len(xml_files)}: {os.path.basename(xml_file)}")
        date, data, directions = process_xml_file(xml_file)
        
        if date and data:
            # If new date and we have accumulated data, process previous date
            if current_date and current_date != date and current_date not in processed_dates:
                if daily_data[current_date]:
                    process_daily_data(current_date, daily_data[current_date], daily_directions[current_date], output_dir)
                    daily_data[current_date].clear()
                    daily_directions[current_date].clear()
            
            current_date = date
            
            # Skip if already processed
            if date in processed_dates:
                continue
                
            # Store directions
            daily_directions[date].update(directions)
                
            # Accumulate data for current date
            for time_slot, detectors in data.items():
                for detector_id, lanes in detectors.items():
                    for lane_id, lane_data_list in lanes.items():
                        daily_data[date][time_slot][detector_id][lane_id].extend(lane_data_list)
    
    # Process final date
    if current_date and current_date not in processed_dates and daily_data[current_date]:
        process_daily_data(current_date, daily_data[current_date], daily_directions[current_date], output_dir)

def main():
    """Main processing function"""
    # Find all month folders under base directory
    month_folders = [d for d in os.listdir(BASE_DIR) 
                    if os.path.isdir(os.path.join(BASE_DIR, d)) and d.isdigit() and len(d) == 6]
    
    if not month_folders:
        print(f"No month folders found in {BASE_DIR}")
        return
    
    # Filter out folders that already have _modified versions
    pending_folders = []
    for folder in month_folders:
        modified_folder = f"{folder}_modified"
        if not os.path.exists(os.path.join(BASE_DIR, modified_folder)):
            pending_folders.append(folder)
        else:
            print(f"Skipping {folder} - already processed")
    
    if not pending_folders:
        print("All folders already processed")
        return
    
    pending_folders.sort()
    print(f"Processing {len(pending_folders)} folders: {pending_folders}")
    
    for month_folder in pending_folders:
        folder_path = os.path.join(BASE_DIR, month_folder)
        process_monthly_data(folder_path)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
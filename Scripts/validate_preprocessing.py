import os
import xml.etree.ElementTree as ET
import glob

def validate_preprocessing():
    """Validate data_processing_1.py preprocessing"""
    base_dir = r'D:\MAIE5532_Resources'
    
    # Check if modified folders exist
    modified_folders = [d for d in os.listdir(base_dir) if d.endswith('_modified')]
    print(f"Found {len(modified_folders)} modified folders")
    
    if not modified_folders:
        print("❌ No preprocessed data found")
        return False
    
    # Validate sample files
    sample_folder = os.path.join(base_dir, modified_folders[0])
    xml_files = glob.glob(os.path.join(sample_folder, "*_processed.xml"))[:3]
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Check structure
            date = root.find('date')
            periods = root.find('periods')
            
            if date is None or periods is None:
                print(f"❌ Invalid structure in {xml_file}")
                return False
            
            # Check time filtering (should only have 7-11AM, 17-21PM)
            for period in periods.findall('period'):
                time_str = period.find('period_from').text
                hour = int(time_str.split(':')[0])
                if not ((7 <= hour < 11) or (17 <= hour < 21)):
                    print(f"❌ Invalid time range {time_str} in {xml_file}")
                    return False
            
            print(f"✅ {os.path.basename(xml_file)} - valid structure and time filtering")
            
        except Exception as e:
            print(f"❌ Error validating {xml_file}: {e}")
            return False
    
    print(f"✅ Preprocessing validation passed - {len(modified_folders)} folders processed")
    return True

if __name__ == "__main__":
    validate_preprocessing()
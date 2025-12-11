import sys
sys.path.append(r'c:\Users\maxch\MaxProjects\NLP\Scripts')
from data_preprocessing_2 import MLDataPreprocessor

# Test with single file
preprocessor = MLDataPreprocessor()
xml_file = r'c:\Users\maxch\MaxProjects\NLP\202508_modified\2025-08-30_processed.xml'

print("Testing single XML file processing...")
df = preprocessor.parse_xml_to_dataframe(xml_file)

if not df.empty:
    print(f"Total records: {len(df)}")
    print(f"Unique detectors: {df['detector_id'].nunique()}")
    print(f"Tunnels: {df['tunnel'].value_counts()}")
    print(f"WHT records: {len(df[df['tunnel'] == 'WHT'])}")
    print(f"Sample WHT data:")
    wht_data = df[df['tunnel'] == 'WHT'].head()
    print(wht_data[['detector_id', 'tunnel', 'speed', 'occupancy', 'valid']])
else:
    print("No data returned!")
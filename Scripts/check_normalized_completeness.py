import pandas as pd
import numpy as np

def check_normalized_completeness():
    """Check if normalized CSV has all expected detectors"""
    
    print("Checking normalized CSV completeness...")
    
    # Load normalized data
    try:
        df = pd.read_csv(r'c:\Users\maxch\MaxProjects\NLP\ML_Data\traffic_data_normalized.csv')
        print(f"Loaded normalized CSV: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading normalized CSV: {e}")
        return
    
    # Check basic structure
    print(f"\nColumns: {list(df.columns)}")
    if 'date' in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    elif 'timestamp' in df.columns:
        print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check unique detectors
    if 'detector_id' in df.columns:
        unique_detectors = df['detector_id'].unique()
        print(f"\nFound {len(unique_detectors)} unique detectors in normalized CSV")
        
        # Show sample detectors
        print(f"Sample detectors: {list(unique_detectors[:10])}")
        
        # Check for missing data
        # Check for missing data using available columns
        agg_dict = {}
        if 'speed_mean' in df.columns:
            agg_dict['speed_mean'] = lambda x: x.isna().sum()
        if 'volume_total' in df.columns:
            agg_dict['volume_total'] = lambda x: x.isna().sum()
            
        missing_by_detector = df.groupby('detector_id').agg(agg_dict)
        
        print(f"\nMissing data summary:")
        if 'speed_mean' in missing_by_detector.columns:
            print(f"Detectors with missing speed data: {(missing_by_detector['speed_mean'] > 0).sum()}")
        if 'volume_total' in missing_by_detector.columns:
            print(f"Detectors with missing volume data: {(missing_by_detector['volume_total'] > 0).sum()}")
        
        # Check data coverage by detector
        data_coverage = df.groupby('detector_id').size()
        print(f"\nData points per detector:")
        print(f"Min: {data_coverage.min()}")
        print(f"Max: {data_coverage.max()}")
        print(f"Mean: {data_coverage.mean():.1f}")
        
        # Show detectors with least data
        low_coverage = data_coverage.sort_values().head(10)
        print(f"\nDetectors with lowest coverage:")
        for det, count in low_coverage.items():
            print(f"  {det}: {count} records")
            
    else:
        print("No 'detector_id' column found")
        
    # Check for expected detector patterns
    detector_patterns = {
        'AID': 0,
        'TDSIEC': 0,
        'Other': 0
    }
    
    if 'detector_id' in df.columns:
        for detector in unique_detectors:
            if 'AID' in str(detector):
                detector_patterns['AID'] += 1
            elif 'TDSIEC' in str(detector):
                detector_patterns['TDSIEC'] += 1
            else:
                detector_patterns['Other'] += 1
                
        print(f"\nDetector type distribution:")
        for pattern, count in detector_patterns.items():
            print(f"  {pattern}: {count} detectors")

if __name__ == "__main__":
    check_normalized_completeness()
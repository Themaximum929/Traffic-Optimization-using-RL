import pandas as pd

def compare_detector_completeness():
    """Compare expected detectors vs actual detectors in normalized CSV"""
    
    # Import detector mapping from data_preprocessing_2.py
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    try:
        from data_preprocessing_2 import MLDataPreprocessor
        preprocessor = MLDataPreprocessor()
        expected_detectors = set(preprocessor.detector_tunnel_map.keys())
        print(f"Loaded detector mapping from data_preprocessing_2.py: {len(expected_detectors)} detectors")
    except ImportError as e:
        print(f"Warning: Could not import from data_preprocessing_2.py: {e}")
        print("Using fallback detector list...")
        # Fallback to known available detectors
        expected_detectors = {
            # WHT detectors
            'AID03204', 'AID03205', 'AID03206', 'AID03207', 'AID03208',
            'AID03209', 'AID03210', 'AID03211', 'AID04218', 'AID04219', 'AID03103', 'AID04104',
            # CHT detectors  
            'AID01108', 'AID01109', 'AID01110', 'TDSIEC10002', 'TDSIEC10003', 'TDSIEC10004',
            'AID01208', 'AID01209', 'AID01211', 'AID01212', 'AID01213', 'AID05225', 'AID05226', 'AID05109',
            # EHT detectors
            'AID04210', 'AID04212', 'AID04106', 'AID04107', 'AID04110',
            'AID02205', 'AID02206', 'AID02207', 'AID02208', 'AID02209', 'AID02210', 
            'AID02211', 'AID02212', 'AID02213', 'AID02214', 'AID07226'
        }
    
    print("DETECTOR COMPLETENESS COMPARISON")
    print("="*50)
    print(f"Expected detectors: {len(expected_detectors)}")
    
    # Load normalized data
    try:
        df = pd.read_csv(r'c:\Users\maxch\MaxProjects\NLP\ML_Data\traffic_data_normalized.csv')
        actual_detectors = set(df['detector_id'].unique())
        print(f"Actual detectors in CSV: {len(actual_detectors)}")
    except Exception as e:
        print(f"Error loading normalized CSV: {e}")
        return
    
    # Find missing and extra detectors
    missing_detectors = expected_detectors - actual_detectors
    extra_detectors = actual_detectors - expected_detectors
    
    print(f"\nMISSING DETECTORS ({len(missing_detectors)}):")
    if missing_detectors:
        for detector in sorted(missing_detectors):
            print(f"  {detector}")
    else:
        print("  None - All expected detectors present!")
    
    print(f"\nEXTRA DETECTORS ({len(extra_detectors)}):")
    if extra_detectors:
        for detector in sorted(extra_detectors):
            print(f"  {detector}")
    else:
        print("  None")
    
    # Get tunnel mapping from preprocessor if available
    try:
        detector_tunnel_map = preprocessor.detector_tunnel_map
    except:
        # Fallback tunnel mapping for available detectors only
        detector_tunnel_map = {
            # WHT
            'AID03204': 'WHT', 'AID03205': 'WHT', 'AID03206': 'WHT', 'AID03207': 'WHT', 'AID03208': 'WHT',
            'AID03209': 'WHT', 'AID03210': 'WHT', 'AID03211': 'WHT', 'AID04218': 'WHT', 'AID04219': 'WHT', 
            'AID03103': 'WHT', 'AID04104': 'WHT',
            # CHT
            'AID01108': 'CHT', 'AID01109': 'CHT', 'AID01110': 'CHT', 'TDSIEC10002': 'CHT', 'TDSIEC10003': 'CHT', 
            'TDSIEC10004': 'CHT', 'AID01208': 'CHT', 'AID01209': 'CHT', 'AID01211': 'CHT', 'AID01212': 'CHT', 
            'AID01213': 'CHT', 'AID05225': 'CHT', 'AID05226': 'CHT', 'AID05109': 'CHT',
            # EHT
            'AID04210': 'EHT', 'AID04212': 'EHT', 'AID04106': 'EHT', 'AID04107': 'EHT', 
            'AID04110': 'EHT', 'AID02205': 'EHT', 'AID02206': 'EHT', 'AID02207': 'EHT', 
            'AID02208': 'EHT', 'AID02209': 'EHT', 'AID02210': 'EHT', 'AID02211': 'EHT', 'AID02212': 'EHT', 
            'AID02213': 'EHT', 'AID02214': 'EHT', 'AID07226': 'EHT'
        }
    
    print(f"\nMISSING BY TUNNEL:")
    for tunnel in ['WHT', 'CHT', 'EHT']:
        tunnel_expected = {d for d, t in detector_tunnel_map.items() if t == tunnel}
        tunnel_missing = tunnel_expected - actual_detectors
        print(f"  {tunnel}: {len(tunnel_missing)} missing - {sorted(tunnel_missing)}")
    
    # Check data coverage for present detectors
    print(f"\nDATA COVERAGE FOR PRESENT DETECTORS:")
    coverage_stats = df.groupby('detector_id').size().describe()
    print(f"  Min records: {coverage_stats['min']:.0f}")
    print(f"  Max records: {coverage_stats['max']:.0f}")
    print(f"  Mean records: {coverage_stats['mean']:.0f}")
    
    return missing_detectors, extra_detectors

if __name__ == "__main__":
    compare_detector_completeness()
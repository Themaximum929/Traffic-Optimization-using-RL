def get_updated_detector_mappings():
    """Updated detector mappings based on actual available data"""
    
    # Updated tunnel mapping - removed non-existent detectors
    detector_tunnel_map = {
        # WHT Southbound Kowloon sequence (Complete)
        'AID03204': 'WHT', 'AID03205': 'WHT', 'AID03206': 'WHT', 'AID03207': 'WHT', 'AID03208': 'WHT',
        'AID03209': 'WHT', 'AID03210': 'WHT', 'AID03211': 'WHT',
        # WHT Northbound sequence (Complete)
        'AID04218': 'WHT', 'AID04219': 'WHT', 'AID03103': 'WHT', 'AID04104': 'WHT',
        
        # CHT Northbound Route 1 (Complete)
        'AID01108': 'CHT', 'AID01109': 'CHT', 'AID01110': 'CHT',
        # CHT Northbound Route 2 (Complete)
        'TDSIEC10002': 'CHT', 'TDSIEC10003': 'CHT', 'TDSIEC10004': 'CHT',
        # CHT Southbound Route 1 (Complete)
        'AID01208': 'CHT', 'AID01209': 'CHT', 'AID01211': 'CHT', 'AID01212': 'CHT', 'AID01213': 'CHT',
        # CHT Southbound Route 2 & 3 (Missing AID05224)
        'AID05225': 'CHT', 'AID05226': 'CHT', 'AID05109': 'CHT',
        
        # EHT Northbound Route 1 & 2 (Missing AID04122)
        'AID04210': 'EHT', 'AID04212': 'EHT', 'AID04106': 'EHT', 'AID04107': 'EHT', 'AID04110': 'EHT',
        # EHT Southbound Main & Alt (Missing AID02204, using AID02207 for inter-tunnel)
        'AID02205': 'EHT', 'AID02206': 'EHT', 'AID02207': 'EHT', 'AID02208': 'EHT',
        'AID02209': 'EHT', 'AID02210': 'EHT', 'AID02211': 'EHT', 'AID02212': 'EHT', 'AID02213': 'EHT',
        'AID02214': 'EHT', 'AID07226': 'EHT'
    }
    
    # Updated direction mapping - removed non-existent detectors
    detector_direction_map = {
        # WHT Southbound Kowloon sequence
        'AID03204': 'Southbound', 'AID03205': 'Southbound', 'AID03206': 'Southbound', 'AID03207': 'Southbound',
        'AID03208': 'Southbound', 'AID03209': 'Southbound', 'AID03210': 'Southbound', 'AID03211': 'Southbound',
        # WHT Northbound sequence
        'AID04218': 'Northbound', 'AID04219': 'Northbound', 'AID03103': 'Northbound', 'AID04104': 'Southbound',
        
        # CHT Northbound Route 1 & 2
        'AID01108': 'Northbound', 'AID01109': 'Northbound', 'AID01110': 'Northbound',
        'TDSIEC10002': 'Northbound', 'TDSIEC10003': 'Northbound', 'TDSIEC10004': 'Northbound',
        # CHT Southbound Route 1, 2 & 3
        'AID01208': 'Southbound', 'AID01209': 'Southbound', 'AID01211': 'Southbound', 'AID01212': 'Southbound', 'AID01213': 'Southbound',
        'AID05225': 'Southbound', 'AID05226': 'Southbound', 'AID05109': 'Southbound',
        
        # EHT Northbound Route 1 & 2
        'AID04210': 'Northbound', 'AID04212': 'Northbound',
        'AID04106': 'Northbound', 'AID04107': 'Northbound', 'AID04110': 'Northbound',
        # EHT Southbound Main & Alt (AID02207 used for inter-tunnel connection)
        'AID02205': 'Southbound', 'AID02206': 'Southbound', 'AID02207': 'Southbound',
        'AID02208': 'Southbound', 'AID02209': 'Southbound', 'AID02210': 'Southbound', 'AID02211': 'Southbound',
        'AID02212': 'Southbound', 'AID02213': 'Northbound', 'AID02214': 'Northbound', 'AID07226': 'Southbound'
    }
    
    return detector_tunnel_map, detector_direction_map

def print_updated_mappings():
    """Print the updated detector mappings"""
    
    tunnel_map, direction_map = get_updated_detector_mappings()
    
    print("UPDATED DETECTOR MAPPINGS")
    print("="*50)
    
    print(f"\nTotal detectors: {len(tunnel_map)}")
    
    # Count by tunnel
    tunnel_counts = {}
    for detector, tunnel in tunnel_map.items():
        tunnel_counts[tunnel] = tunnel_counts.get(tunnel, 0) + 1
    
    print("\nDetectors by tunnel:")
    for tunnel, count in tunnel_counts.items():
        print(f"  {tunnel}: {count} detectors")
    
    print("\nDetectors by tunnel and direction:")
    for tunnel in ['WHT', 'CHT', 'EHT']:
        tunnel_detectors = [d for d, t in tunnel_map.items() if t == tunnel]
        northbound = [d for d in tunnel_detectors if direction_map.get(d) == 'Northbound']
        southbound = [d for d in tunnel_detectors if direction_map.get(d) == 'Southbound']
        
        print(f"\n{tunnel}:")
        print(f"  Northbound: {len(northbound)} detectors")
        print(f"  Southbound: {len(southbound)} detectors")
    
    print("\nREMOVED DETECTORS:")
    removed = ['AID02112', 'AID02204', 'AID04122', 'AID05224']
    for detector in removed:
        if detector == 'AID02112':
            print(f"  {detector}: Excluded - using AID02207 for inter-tunnel link")
        else:
            print(f"  {detector}: Not found in source data")

if __name__ == "__main__":
    print_updated_mappings()
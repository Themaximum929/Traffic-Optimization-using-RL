def check_all_mappings():
    """Check if all detectors in tunnel_map have direction mappings"""
    
    detector_tunnel_map = {
        # WHT Southbound Kowloon sequence
        'AID03204': 'WHT', 'AID03205': 'WHT', 'AID03206': 'WHT', 'AID03207': 'WHT', 'AID03208': 'WHT',
        'AID03209': 'WHT', 'AID03210': 'WHT', 'AID03211': 'WHT',
        # WHT Northbound sequence
        'AID04218': 'WHT', 'AID04219': 'WHT', 'AID03103': 'WHT',
        # CHT Northbound Route 1
        'AID01108': 'CHT', 'AID01109': 'CHT', 'AID01110': 'CHT',
        # CHT Northbound Route 2
        'TDSIEC10002': 'CHT', 'TDSIEC10003': 'CHT', 'TDSIEC10004': 'CHT',
        # CHT Southbound Route 1
        'AID01208': 'CHT', 'AID01209': 'CHT', 'AID01211': 'CHT', 'AID01212': 'CHT', 'AID01213': 'CHT',
        # CHT Southbound Route 2 & 3
        'AID05224': 'CHT', 'AID05225': 'CHT', 'AID05226': 'CHT', 'AID05109': 'CHT',
        # EHT Northbound Route 1 & 2
        'AID04210': 'EHT', 'AID04212': 'EHT', 'AID04106': 'EHT', 'AID04107': 'EHT', 'AID04122': 'EHT', 'AID04110': 'EHT',
        # EHT Southbound Main & Alt
        'AID02204': 'EHT', 'AID02205': 'EHT', 'AID02206': 'EHT', 'AID02207': 'EHT', 'AID02208': 'EHT',
        'AID02209': 'EHT', 'AID02210': 'EHT', 'AID02211': 'EHT', 'AID02212': 'EHT', 'AID02213': 'EHT',
        'AID02214': 'EHT', 'AID07226': 'EHT',
        # Inter-tunnel linkages
        'AID04104': 'WHT'
    }
    
    detector_direction_map = {
        # WHT Southbound Kowloon sequence
        'AID03204': 'Southbound', 'AID03205': 'Southbound', 'AID03206': 'Southbound', 'AID03207': 'Southbound',
        'AID03208': 'Southbound', 'AID03209': 'Southbound', 'AID03210': 'Southbound', 'AID03211': 'Southbound',
        # WHT Northbound sequence
        'AID04218': 'Northbound', 'AID04219': 'Northbound', 'AID03103': 'Northbound',
        # CHT Northbound Route 1 & 2
        'AID01108': 'Northbound', 'AID01109': 'Northbound', 'AID01110': 'Northbound',
        'TDSIEC10002': 'Northbound', 'TDSIEC10003': 'Northbound', 'TDSIEC10004': 'Northbound',
        # CHT Southbound Route 1, 2 & 3
        'AID01208': 'Southbound', 'AID01209': 'Southbound', 'AID01211': 'Southbound', 'AID01212': 'Southbound', 'AID01213': 'Southbound',
        'AID05224': 'Southbound', 'AID05225': 'Southbound', 'AID05226': 'Southbound', 'AID05109': 'Southbound',
        # EHT Northbound Route 1 & 2
        'AID04210': 'Northbound', 'AID04212': 'Northbound',
        'AID04106': 'Northbound', 'AID04107': 'Northbound', 'AID04122': 'Northbound', 'AID04110': 'Northbound',
        # EHT Southbound Main & Alt
        'AID02204': 'Southbound', 'AID02205': 'Southbound', 'AID02206': 'Southbound', 'AID02207': 'Southbound',
        'AID02208': 'Southbound', 'AID02209': 'Southbound', 'AID02210': 'Southbound', 'AID02211': 'Southbound',
        'AID02212': 'Southbound', 'AID02213': 'Southbound', 'AID02214': 'Southbound', 'AID07226': 'Southbound',
        # Inter-tunnel connection points
        'AID04104': 'Southbound',
        'AID02112': 'Southbound'
    }
    
    print("MAPPING COMPLETENESS CHECK")
    print("="*50)
    
    # Check if all tunnel detectors have direction mappings
    missing_directions = []
    for detector in detector_tunnel_map.keys():
        if detector not in detector_direction_map:
            missing_directions.append(detector)
    
    # Check by tunnel
    wht_detectors = [d for d, t in detector_tunnel_map.items() if t == 'WHT']
    cht_detectors = [d for d, t in detector_tunnel_map.items() if t == 'CHT']
    eht_detectors = [d for d, t in detector_tunnel_map.items() if t == 'EHT']
    
    print(f"WHT detectors ({len(wht_detectors)}): {wht_detectors}")
    print(f"CHT detectors ({len(cht_detectors)}): {cht_detectors}")
    print(f"EHT detectors ({len(eht_detectors)}): {eht_detectors}")
    
    if missing_directions:
        print(f"\nMISSING DIRECTION MAPPINGS: {missing_directions}")
    else:
        print(f"\nAll {len(detector_tunnel_map)} detectors have direction mappings - OK")
    
    # Check for extra directions
    extra_directions = []
    for detector in detector_direction_map.keys():
        if detector not in detector_tunnel_map:
            extra_directions.append(detector)
    
    if extra_directions:
        print(f"EXTRA DIRECTION MAPPINGS: {extra_directions}")
    
    return len(missing_directions) == 0

if __name__ == "__main__":
    check_all_mappings()
def check_network_completeness():
    """Check if tunnel network has complete inter-tunnel connectivity"""
    
    # Current inter-tunnel links from data_preprocessing_2.py
    inter_tunnel_links = {
        'Link1_HK_Side': ['AID04104', 'TDSIEC10004', 'AID04106'],     # CHT->CHT->EHT
        'Link2_Kowloon_Side': ['AID03210', 'AID05224', 'AID02112']    # WHT->CHT->EHT
    }
    
    # Tunnel assignments
    tunnel_map = {
        'AID04104': 'WHT', 'TDSIEC10004': 'CHT', 'AID04106': 'EHT',  # HK Side
        'AID03210': 'WHT', 'AID05224': 'CHT', 'AID02112': 'EHT'      # Kowloon Side
    }
    
    print("Network Completeness Check:")
    print("\nHK Island Side (Link1):")
    hk_tunnels = []
    for detector in inter_tunnel_links['Link1_HK_Side']:
        tunnel = tunnel_map[detector]
        hk_tunnels.append(tunnel)
        print(f"   {detector} -> {tunnel}")
    
    print("\nKowloon Side (Link2):")
    kowloon_tunnels = []
    for detector in inter_tunnel_links['Link2_Kowloon_Side']:
        tunnel = tunnel_map[detector]
        kowloon_tunnels.append(tunnel)
        print(f"   {detector} -> {tunnel}")
    
    # Check completeness
    required_tunnels = {'WHT', 'CHT', 'EHT'}
    hk_coverage = set(hk_tunnels)
    kowloon_coverage = set(kowloon_tunnels)
    
    print(f"\nHK Side Coverage: {hk_coverage}")
    print(f"Kowloon Side Coverage: {kowloon_coverage}")
    
    hk_complete = required_tunnels.issubset(hk_coverage)
    kowloon_complete = required_tunnels.issubset(kowloon_coverage)
    
    print(f"\nHK Side Complete: {hk_complete}")
    print(f"Kowloon Side Complete: {kowloon_complete}")
    
    if not hk_complete:
        missing_hk = required_tunnels - hk_coverage
        print(f"HK Side Missing: {missing_hk}")
    
    if not kowloon_complete:
        missing_kowloon = required_tunnels - kowloon_coverage
        print(f"Kowloon Side Missing: {missing_kowloon}")
    
    overall_complete = hk_complete and kowloon_complete
    print(f"\nNetwork Complete: {overall_complete}")
    
    return overall_complete

if __name__ == "__main__":
    check_network_completeness()
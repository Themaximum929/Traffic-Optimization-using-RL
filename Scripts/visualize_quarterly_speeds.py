import json

with open('d:\\MAIE5532_Resources\\Scripts\\proposal_metrics_complete.json', 'r') as f:
    data = json.load(f)

tunnels = data['tunnel_groups']

print("\n" + "="*80)
print("QUARTERLY TUNNEL SPEED ANALYSIS")
print("="*80)

for name, info in tunnels.items():
    if 'analysis' in info and info['analysis'] and 'quarterly_breakdown' in info['analysis']:
        tunnel = name.replace('_tunnel', '')
        print(f"\n{tunnel} TUNNEL:")
        print("-" * 80)
        
        quarters = info['analysis']['quarterly_breakdown']
        for q_name, q_data in quarters.items():
            if 'combined' in q_data:
                speed = q_data['combined']['avg_speed']
                volume = q_data['combined']['avg_volume']
                events = q_data['combined']['congestion_events']
                print(f"{q_name:15} | Speed: {speed:5.1f} km/h | Volume: {volume:5.1f} | Events: {events:3}")

print("\n" + "="*80)

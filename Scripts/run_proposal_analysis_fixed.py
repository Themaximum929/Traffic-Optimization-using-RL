import pandas as pd
import os
from datetime import datetime

def run_fixed_analysis():
    """Run proposal analysis using normalized CSV data"""
    
    print("RUNNING FIXED PROPOSAL ANALYSIS")
    print("="*50)
    
    # Load normalized data
    df = pd.read_csv(r'c:\Users\maxch\MaxProjects\NLP\ML_Data\traffic_data_normalized.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded data: {len(df):,} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Define toll implementation date
    toll_date = datetime(2023, 12, 17)
    
    # Split pre/post toll
    pre_toll = df[df['timestamp'] < toll_date]
    post_toll = df[df['timestamp'] >= toll_date]
    
    print(f"Pre-toll: {len(pre_toll):,} records")
    print(f"Post-toll: {len(post_toll):,} records")
    
    # Analyze congestion (using speed_mean < 30 and occupancy_mean > 50)
    def analyze_period(data, period_name):
        # Peak hours only
        peak_data = data[data['is_peak'] == True]
        
        # Congested conditions
        congested = peak_data[(peak_data['speed_mean'] < 30) & (peak_data['occupancy_mean'] > 50)]
        
        # Severe congestion
        severe = congested[congested['speed_mean'] < 20]
        
        return {
            'period': period_name,
            'total_records': len(data),
            'peak_records': len(peak_data),
            'congested_events': len(congested),
            'severe_events': len(severe),
            'avg_peak_speed': peak_data['speed_mean'].mean(),
            'avg_congested_speed': congested['speed_mean'].mean() if len(congested) > 0 else 0
        }
    
    # Analyze both periods
    pre_analysis = analyze_period(pre_toll, 'Pre-toll')
    post_analysis = analyze_period(post_toll, 'Post-toll')
    
    # Calculate improvements
    speed_improvement = ((post_analysis['avg_peak_speed'] - pre_analysis['avg_peak_speed']) / 
                        pre_analysis['avg_peak_speed'] * 100) if pre_analysis['avg_peak_speed'] > 0 else 0
    
    congestion_reduction = ((pre_analysis['severe_events'] - post_analysis['severe_events']) / 
                           pre_analysis['severe_events'] * 100) if pre_analysis['severe_events'] > 0 else 0
    
    # Results
    results = {
        'analysis_summary': {
            'data_source': 'Normalized CSV with 42 available detectors',
            'total_detectors': len(df['detector_id'].unique()),
            'analysis_period': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
            'toll_implementation': '2023-12-17'
        },
        'pre_toll_metrics': pre_analysis,
        'post_toll_metrics': post_analysis,
        'improvements': {
            'speed_improvement_percent': speed_improvement,
            'congestion_reduction_percent': congestion_reduction
        },
        'detector_completeness': {
            'expected_detectors': 46,
            'available_detectors': 42,
            'missing_detectors': ['AID02112', 'AID02204', 'AID04122', 'AID05224'],
            'missing_reasons': {
                'AID02112': 'Excluded - using AID02207 for inter-tunnel link',
                'AID02204': 'Not found in source data',
                'AID04122': 'Not found in source data', 
                'AID05224': 'Not found in source data'
            }
        }
    }
    
    print("\nANALYSIS RESULTS:")
    print(f"Pre-toll avg peak speed: {pre_analysis['avg_peak_speed']:.1f} km/h")
    print(f"Post-toll avg peak speed: {post_analysis['avg_peak_speed']:.1f} km/h")
    print(f"Speed improvement: {speed_improvement:.1f}%")
    print(f"Pre-toll severe events: {pre_analysis['severe_events']:,}")
    print(f"Post-toll severe events: {post_analysis['severe_events']:,}")
    print(f"Congestion reduction: {congestion_reduction:.1f}%")
    
    # Save results
    import json
    output_file = os.path.join(r'c:\Users\maxch\MaxProjects\NLP\Scripts', 'proposal_metrics_fixed.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    return results

if __name__ == "__main__":
    run_fixed_analysis()
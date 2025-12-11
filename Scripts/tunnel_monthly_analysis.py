import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class TunnelMonthlyAnalyzer:
    def __init__(self, base_dir=r'C:\Users\maxch\MaxProjects\NLP'):
        self.base_dir = base_dir
        self.toll_date = datetime(2023, 12, 17)
        self.max_workers = min(16, mp.cpu_count())
        
        self.tunnel_groups = {
            'WHT': {
                'entrance_detectors': ['AID03211', 'AID03210', 'AID04104'],
                'weights': [0.71, 0.73, 0.66]
            },
            'CHT': {
                'entrance_detectors': ['AID01213', 'AID01110', 'TDSIEC10004'],
                'weights': [0.59, 0.60, 0.5]
            },
            'EHT': {
                'entrance_detectors': ['AID02214', 'AID04110', 'AID04212'],
                'weights': [0.59, 0.47, 0.80]
            }
        }
    
    def load_xml_data(self, xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            date = root.find('date').text
            data = []
            
            for period in root.findall('.//period'):
                time_slot = period.find('period_from').text
                for detector in period.findall('.//detector'):
                    detector_id = detector.find('detector_id').text
                    for lane in detector.findall('.//lane'):
                        data.append({
                            'date': date, 'time': time_slot, 'detector_id': detector_id,
                            'lane_id': lane.find('lane_id').text,
                            'speed': float(lane.find('speed').text or 0),
                            'occupancy': float(lane.find('occupancy').text or 0),
                            'volume': int(lane.find('volume').text or 0)
                        })
            return pd.DataFrame(data)
        except:
            return pd.DataFrame()
    
    def get_xml_files_for_month(self, year, month):
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        xml_files = []
        current = start_date
        while current <= end_date:
            month_folder = current.strftime('%Y%m') + '_modified'
            xml_file = os.path.join(self.base_dir, month_folder, current.strftime('%Y-%m-%d') + '_processed.xml')
            if os.path.exists(xml_file):
                xml_files.append(xml_file)
            current += timedelta(days=1)
        return xml_files
    
    def analyze_month(self, year, month, tunnel_name):
        xml_files = self.get_xml_files_for_month(year, month)
        if not xml_files:
            return None
        
        detector_list = self.tunnel_groups[tunnel_name]['entrance_detectors']
        weights = self.tunnel_groups[tunnel_name]['weights']
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(self.load_xml_data, xml_files))
        
        dfs = [df for df in dfs if not df.empty]
        if not dfs:
            return None
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['hour'] = pd.to_datetime(combined_df['time']).dt.hour
        
        tunnel_data = combined_df[combined_df['detector_id'].isin(detector_list)]
        peak_data = tunnel_data[tunnel_data['hour'].isin([7, 8, 9, 17, 18, 19])]
        
        if peak_data.empty:
            return None
        
        # Get slowest lane per time period
        slowest = peak_data.loc[peak_data.groupby(['date', 'time', 'detector_id'])['speed'].idxmin()]
        
        # Weighted average
        all_speeds, all_volumes = [], []
        for i, det in enumerate(detector_list):
            det_data = slowest[slowest['detector_id'] == det]
            if not det_data.empty:
                weight = weights[i]
                for _ in range(int(weight * 100)):
                    all_speeds.extend(det_data['speed'].tolist())
                    all_volumes.extend(det_data['volume'].tolist())
        
        if not all_speeds:
            return None
        
        return {
            'year': year,
            'month': month,
            'avg_speed': np.mean(all_speeds),
            'min_speed': np.min(all_speeds),
            'avg_volume': np.mean(all_volumes),
            'congestion_events': len([s for s in all_speeds if s < 20])
        }
    
    def analyze_all_months(self):
        results = {tunnel: [] for tunnel in self.tunnel_groups.keys()}
        
        # Jan 2023 to Aug 2025
        for year in range(2023, 2026):
            start_month = 1
            end_month = 12 if year < 2025 else 8
            
            for month in range(start_month, end_month + 1):
                print(f"Analyzing {year}-{month:02d}...")
                
                for tunnel in self.tunnel_groups.keys():
                    result = self.analyze_month(year, month, tunnel)
                    if result:
                        results[tunnel].append(result)
        
        return results
    
    def plot_monthly_trends(self, results):
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        for tunnel_idx, (tunnel, data) in enumerate(results.items()):
            if not data:
                continue
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            df['is_post_toll'] = df['date'] >= self.toll_date
            
            # Speed
            ax = axes[tunnel_idx, 0]
            ax.plot(df['date'], df['avg_speed'], marker='o', linewidth=2)
            ax.axvline(self.toll_date, color='red', linestyle='--', label='Toll Start')
            ax.set_title(f'{tunnel} - Average Speed', fontweight='bold')
            ax.set_ylabel('Speed (km/h)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Volume
            ax = axes[tunnel_idx, 1]
            ax.plot(df['date'], df['avg_volume'], marker='o', linewidth=2, color='orange')
            ax.axvline(self.toll_date, color='red', linestyle='--', label='Toll Start')
            ax.set_title(f'{tunnel} - Average Volume', fontweight='bold')
            ax.set_ylabel('Volume (veh/5min)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Congestion Events
            ax = axes[tunnel_idx, 2]
            ax.plot(df['date'], df['congestion_events'], marker='o', linewidth=2, color='red')
            ax.axvline(self.toll_date, color='red', linestyle='--', label='Toll Start')
            ax.set_title(f'{tunnel} - Congestion Events', fontweight='bold')
            ax.set_ylabel('Events (<20 km/h)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'Scripts', 'individual_results', 'tunnel_monthly_analysis.png'), dpi=300, bbox_inches='tight')
        print("✓ Plot saved: individual_results/tunnel_monthly_analysis.png")
        plt.close()
    
    def save_results(self, results):
        output = {}
        for tunnel, data in results.items():
            output[tunnel] = {
                'monthly_data': data,
                'pre_toll_avg': np.mean([d['avg_speed'] for d in data if datetime(d['year'], d['month'], 1) < self.toll_date]),
                'post_toll_avg': np.mean([d['avg_speed'] for d in data if datetime(d['year'], d['month'], 1) >= self.toll_date])
            }
        
        output_file = os.path.join(self.base_dir, 'Scripts', 'tunnel_monthly_results.json')
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✓ Results saved: {output_file}")

if __name__ == "__main__":
    analyzer = TunnelMonthlyAnalyzer()
    results = analyzer.analyze_all_months()
    analyzer.save_results(results)
    analyzer.plot_monthly_trends(results)

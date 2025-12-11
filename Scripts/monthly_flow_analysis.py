import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class MonthlyFlowAnalyzer:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.toll_date = datetime(2023, 12, 17)
        self.max_workers = min(16, mp.cpu_count())
        
        self.tunnel_groups = {
            'WHT': ['AID03211', 'AID03210', 'AID04104'],
            'CHT': ['AID01213', 'AID01110', 'TDSIEC10004'],
            'EHT': ['AID02214', 'AID04110', 'AID04212']
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
                            'volume': int(lane.find('volume').text or 0)
                        })
            return pd.DataFrame(data)
        except:
            return pd.DataFrame()
    
    def get_xml_files_for_month(self, year, month):
        start_date = datetime(year, month, 1)
        end_date = datetime(year, month + 1, 1) - timedelta(days=1) if month < 12 else datetime(year, 12, 31)
        
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
        
        detector_list = self.tunnel_groups[tunnel_name]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(self.load_xml_data, xml_files))
        
        dfs = [df for df in dfs if not df.empty]
        if not dfs:
            return None
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['hour'] = pd.to_datetime(combined_df['time']).dt.hour
        
        tunnel_data = combined_df[combined_df['detector_id'].isin(detector_list)]
        
        peak_flow = tunnel_data[tunnel_data['hour'].isin([7, 8, 9, 17, 18, 19])]['volume'].sum()
        non_peak_flow = tunnel_data[~tunnel_data['hour'].isin([7, 8, 9, 17, 18, 19])]['volume'].sum()
        
        peak_records = len(tunnel_data[tunnel_data['hour'].isin([7, 8, 9, 17, 18, 19])])
        non_peak_records = len(tunnel_data[~tunnel_data['hour'].isin([7, 8, 9, 17, 18, 19])])
        
        return {
            'year': year,
            'month': month,
            'peak_avg_flow': peak_flow / peak_records if peak_records > 0 else 0,
            'non_peak_avg_flow': non_peak_flow / non_peak_records if non_peak_records > 0 else 0
        }
    
    def analyze_all_months(self):
        results = {tunnel: [] for tunnel in self.tunnel_groups.keys()}
        
        for year in range(2023, 2026):
            end_month = 12 if year < 2025 else 8
            
            for month in range(1, end_month + 1):
                print(f"Analyzing {year}-{month:02d}...")
                
                for tunnel in self.tunnel_groups.keys():
                    result = self.analyze_month(year, month, tunnel)
                    if result:
                        results[tunnel].append(result)
        
        return results
    
    def plot_flow_comparison(self, results):
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        for tunnel_idx, (tunnel, data) in enumerate(results.items()):
            if not data:
                continue
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            
            ax = axes[tunnel_idx]
            x = np.arange(len(df))
            width = 0.35
            
            ax.bar(x - width/2, df['peak_avg_flow'], width, label='Peak Hours (7-9, 17-19)', color='#e74c3c')
            ax.bar(x + width/2, df['non_peak_avg_flow'], width, label='Non-Peak Hours', color='#3498db')
            
            toll_idx = np.where(df['date'] >= self.toll_date)[0]
            if len(toll_idx) > 0:
                ax.axvline(toll_idx[0] - 0.5, color='black', linestyle='--', linewidth=2, label='Toll Start')
            
            ax.set_title(f'{tunnel} - Average Car Flow per 5-min Interval', fontweight='bold', fontsize=14)
            ax.set_ylabel('Vehicles per 5-min', fontsize=12)
            ax.set_xticks(x[::3])
            ax.set_xticklabels(df['date'].dt.strftime('%Y-%m')[::3], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'Scripts', 'individual_results', 'monthly_flow_comparison.png'), dpi=300, bbox_inches='tight')
        print("✓ Plot saved: individual_results/monthly_flow_comparison.png")
        plt.close()
    
    def save_results(self, results):
        output_file = os.path.join(self.base_dir, 'Scripts', 'monthly_flow_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved: {output_file}")

if __name__ == "__main__":
    analyzer = MonthlyFlowAnalyzer()
    results = analyzer.analyze_all_months()
    analyzer.save_results(results)
    analyzer.plot_flow_comparison(results)

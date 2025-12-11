import os
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
import json
from collections import defaultdict

class CongestionAnalyzer:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.toll_date = datetime(2023, 12, 17)
        
        # Tunnel routes - last detector is nearest to tunnel entrance
        self.routes = {
            'WHT_Southbound_Kowloon': ['AID03204', 'AID03205', 'AID03206', 'AID03207', 'AID03208', 'AID03209', 'AID03210', 'AID03211'],
            'WHT_Northbound_Main': ['AID04218', 'AID04219', 'AID03103'],
            'CHT_Northbound_Route1': ['AID01108', 'AID01109', 'AID01110'],
            'CHT_Northbound_Route2': ['TDSIEC10002', 'TDSIEC10003', 'TDSIEC10004'],
            'CHT_Southbound_Route1': ['AID01208', 'AID01209', 'AID01211', 'AID01212', 'AID01213'],
            'CHT_Southbound_Route2': ['AID05224', 'AID05225', 'AID05226', 'AID01213'],
            'CHT_Southbound_Route3': ['AID05109', 'AID01213'],
            'EHT_Northbound_Route1': ['AID04210', 'AID04212'],
            'EHT_Northbound_Route2': ['AID04106', 'AID04107', 'AID04122', 'AID04110'],
            'EHT_Southbound_Main': ['AID02204', 'AID02205', 'AID02206', 'AID02207', 'AID02208', 'AID02209', 'AID02210', 'AID02211', 'AID02212', 'AID02213', 'AID02214']
        }
    
    def load_xml(self, xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            date = root.find('date').text
            data = []
            
            for period in root.findall('.//period'):
                time_slot = period.find('period_from').text
                for detector in period.findall('.//detector'):
                    detector_id = detector.find('detector_id').text
                    
                    # Only process detectors from routes
                    route_detectors = set()
                    for route_list in self.routes.values():
                        route_detectors.update(route_list)
                    
                    if detector_id not in route_detectors:
                        continue
                        
                    for lane in detector.findall('.//lane'):
                        data.append({
                            'date': date,
                            'time': time_slot,
                            'detector_id': detector_id,
                            'lane_id': lane.find('lane_id').text,
                            'speed': float(lane.find('speed').text or 0),
                            'occupancy': float(lane.find('occupancy').text or 0),
                            'volume': int(lane.find('volume').text or 0)
                        })
            return pd.DataFrame(data)
        except:
            return pd.DataFrame()
    
    def get_xml_files(self, start_date, end_date):
        current = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
        end = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
        files = []
        
        while current <= end:
            month_folder = current.strftime('%Y%m') + '_modified'
            xml_file = os.path.join(self.base_dir, month_folder, current.strftime('%Y-%m-%d') + '_processed.xml')
            if os.path.exists(xml_file):
                files.append(xml_file)
            current += timedelta(days=1)
        return files
    
    def detect_congestion_propagation(self, df, route_detectors, date):
        """Detect how far congestion propagates from tunnel entrance"""
        # Get slowest lane per detector per time period
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        peak_data = df[(df['hour'].isin([7, 8, 9, 17, 18, 19])) & (df['date'] == date)]
        
        if peak_data.empty:
            return None
        
        # Get slowest lane for each detector-time combination
        slowest = peak_data.loc[peak_data.groupby(['detector_id', 'time'])['speed'].idxmin()]
        
        # Check from tunnel entrance (last detector) backward
        congested_detectors = []
        for detector in reversed(route_detectors):
            det_data = slowest[slowest['detector_id'] == detector]
            if det_data.empty:
                continue
            
            avg_speed = det_data['speed'].mean()
            if avg_speed < 30:  # Congested
                congested_detectors.append(detector)
            else:
                break  # Stop when reaching non-congested detector
        
        return list(reversed(congested_detectors)) if congested_detectors else None
    
    def analyze_congestion_events(self, start_date, end_date):
        """Count congestion events per detector"""
        xml_files = self.get_xml_files(start_date, end_date)
        print(f"Loading {len(xml_files)} files from {start_date} to {end_date}...")
        
        all_data = []
        for xml_file in xml_files:
            df = self.load_xml(xml_file)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return {}
        
        combined = pd.concat(all_data, ignore_index=True)
        combined['hour'] = pd.to_datetime(combined['time']).dt.hour
        combined['weekday'] = pd.to_datetime(combined['date']).dt.weekday
        
        # Get slowest lane data for peak hours
        peak = combined[(combined['hour'].isin([7, 8, 9, 17, 18, 19])) & (combined['weekday'] < 5)]
        slowest = peak.loc[peak.groupby(['detector_id', 'date', 'time'])['speed'].idxmin()]
        
        # Count congestion events (speed < 30 km/h)
        congested = slowest[slowest['speed'] < 30]
        events_per_detector = congested.groupby('detector_id').size().to_dict()
        
        return events_per_detector
    
    def analyze_propagation_depth(self, start_date, end_date):
        """Analyze how far congestion propagates on weekdays"""
        xml_files = self.get_xml_files(start_date, end_date)
        print(f"Analyzing propagation for {len(xml_files)} days...")
        
        all_data = []
        for xml_file in xml_files:
            df = self.load_xml(xml_file)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return {}
        
        combined = pd.concat(all_data, ignore_index=True)
        combined['weekday'] = pd.to_datetime(combined['date']).dt.weekday
        weekday_data = combined[combined['weekday'] < 5]
        
        route_propagation = {}
        for route_name, detectors in self.routes.items():
            dates = weekday_data['date'].unique()
            propagation_counts = defaultdict(int)
            
            for date in dates:
                congested = self.detect_congestion_propagation(weekday_data, detectors, date)
                if congested:
                    # Record furthest detector affected
                    furthest = congested[0] if congested else None
                    if furthest:
                        propagation_counts[furthest] += 1
            
            if propagation_counts:
                route_propagation[route_name] = dict(propagation_counts)
        
        return route_propagation
    
    def run_analysis(self):
        print("🚦 Starting Congestion Analysis...")
        
        # 1. Overall congestion events per detector
        print("\n1️⃣ Analyzing overall congestion events...")
        overall_events = self.analyze_congestion_events('2022-12-01', '2025-08-31')
        
        # 2. Pre-toll vs Post-toll
        print("\n2️⃣ Analyzing pre-toll period...")
        pre_toll_events = self.analyze_congestion_events('2022-12-01', '2023-12-16')
        
        print("\n3️⃣ Analyzing post-toll period...")
        post_toll_events = self.analyze_congestion_events('2023-12-17', '2025-08-31')
        
        # 3. Propagation depth analysis
        print("\n4️⃣ Analyzing congestion propagation (pre-toll)...")
        pre_propagation = self.analyze_propagation_depth('2022-12-01', '2023-12-16')
        
        print("\n5️⃣ Analyzing congestion propagation (post-toll)...")
        post_propagation = self.analyze_propagation_depth('2023-12-17', '2025-08-31')
        
        # Compile results
        results = {
            'analysis_period': {
                'start': '2022-12-01',
                'end': '2025-08-31',
                'toll_implementation': '2023-12-17'
            },
            'congestion_events_overall': overall_events,
            'congestion_events_pre_toll': pre_toll_events,
            'congestion_events_post_toll': post_toll_events,
            'congestion_reduction_by_detector': {
                detector: {
                    'pre_toll': pre_toll_events.get(detector, 0),
                    'post_toll': post_toll_events.get(detector, 0),
                    'reduction': pre_toll_events.get(detector, 0) - post_toll_events.get(detector, 0),
                    'reduction_percent': ((pre_toll_events.get(detector, 0) - post_toll_events.get(detector, 0)) / pre_toll_events.get(detector, 1) * 100) if pre_toll_events.get(detector, 0) > 0 else 0
                }
                for detector in set(list(pre_toll_events.keys()) + list(post_toll_events.keys()))
            },
            'weekday_propagation_depth': {
                'pre_toll': pre_propagation,
                'post_toll': post_propagation
            },
            'route_summary': {
                route: {
                    'detectors': detectors,
                    'tunnel_entrance': detectors[-1],
                    'pre_toll_congestion_days': sum(pre_propagation.get(route, {}).values()),
                    'post_toll_congestion_days': sum(post_propagation.get(route, {}).values())
                }
                for route, detectors in self.routes.items()
            }
        }
        
        # Save to JSON
        output_file = os.path.join(self.base_dir, 'Scripts', 'congestion_analysis_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("CONGESTION ANALYSIS SUMMARY")
        print("="*60)
        
        total_pre = sum(pre_toll_events.values())
        total_post = sum(post_toll_events.values())
        reduction = ((total_pre - total_post) / total_pre * 100) if total_pre > 0 else 0
        
        print(f"\n📊 Total Congestion Events:")
        print(f"   Pre-toll:  {total_pre:,}")
        print(f"   Post-toll: {total_post:,}")
        print(f"   Reduction: {reduction:.1f}%")
        
        print(f"\n🚇 Top 5 Most Congested Detectors (Overall):")
        sorted_detectors = sorted(overall_events.items(), key=lambda x: x[1], reverse=True)[:5]
        for detector, count in sorted_detectors:
            print(f"   {detector}: {count:,} events")
        
        print("\n" + "="*60)
        
        return results

if __name__ == "__main__":
    analyzer = CongestionAnalyzer()
    results = analyzer.run_analysis()

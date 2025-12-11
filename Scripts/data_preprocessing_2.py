import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob
import json

class MLDataPreprocessor:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.scaler = StandardScaler()
        
        # Detectors with inter-tunnel linkage preserved
        self.detector_tunnel_map = {
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
            'AID04104': 'WHT'  # AID04104=WHT exit
        }
        
        # Inter-tunnel connection mapping - AID04104 as WHT exit
        self.inter_tunnel_links = {
            'Link1_HK_Side': ['AID04104', 'TDSIEC10002', 'AID04106'],  # WHT(exit)->CHT->EHT
            'Link2_Kowloon_Side': ['AID03210', 'AID05224', 'AID02207']  # WHT->CHT->EHT
        }
        
        # Direction mappings preserving network connectivity
        self.detector_direction_map = {
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
            'AID04104': 'Southbound',  # WHT exit (after AID03211)
        }
    
    def parse_xml_to_dataframe(self, xml_file):
        """Parse XML file to DataFrame"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            date = root.find('date').text
            
            data = []
            for period in root.findall('.//period'):
                time_slot = period.find('period_from').text
                timestamp = f"{date} {time_slot}"
                
                for detector in period.findall('.//detector'):
                    detector_id = detector.find('detector_id').text
                    
                    # Process all detectors that have tunnel mapping (handles time-varying availability)
                    tunnel = self.detector_tunnel_map.get(detector_id)
                    if tunnel is None:
                        continue
                    
                    direction = detector.find('direction').text
                    mapped_direction = self.detector_direction_map.get(detector_id, direction)
                    
                    for lane in detector.findall('.//lane'):
                        data.append({
                            'detector_id': detector_id,
                            'timestamp': pd.to_datetime(timestamp),
                            'lane_id': lane.find('lane_id').text,
                            'speed': float(lane.find('speed').text or 0),
                            'occupancy': float(lane.find('occupancy').text or 0),
                            'volume': int(lane.find('volume').text or 0),
                            'valid': lane.find('valid').text == 'Y',
                            'tunnel': tunnel,
                            'direction': mapped_direction
                        })
            
            return pd.DataFrame(data)
        except:
            return pd.DataFrame()
    
    def aggregate_lane_features(self, df):
        """Aggregate lane-level data to detector level"""
        # Filter valid data only
        valid_df = df[df['valid'] == True].copy()
        
        # Aggregate by detector and timestamp
        agg_df = valid_df.groupby(['detector_id', 'timestamp', 'tunnel', 'direction']).agg({
            'speed': ['mean', 'min', 'max', 'std'],
            'occupancy': ['mean', 'max'],
            'volume': 'sum'
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['detector_id', 'timestamp', 'tunnel', 'direction',
                         'speed_mean', 'speed_min', 'speed_max', 'speed_std',
                         'occupancy_mean', 'occupancy_max', 'volume_total']
        
        # Fill NaN values
        agg_df['speed_std'] = agg_df['speed_std'].fillna(0)
        
        return agg_df
    
    def add_temporal_features(self, df):
        """Add temporal features"""
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6])
        df['is_peak'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
        return df
    
    def normalize_features(self, df, feature_cols):
        """Normalize numerical features"""
        df_norm = df.copy()
        df_norm[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df_norm
    
    def process_all_data(self):
        """Process all XML files to ML-ready format"""
        print("🔄 Processing XML files to ML format...")
        
        # Find modified folders
        modified_folders = [d for d in os.listdir(self.base_dir) if d.endswith('_modified')]
        
        all_data = []
        for folder in modified_folders:
            folder_path = os.path.join(self.base_dir, folder)
            xml_files = glob.glob(os.path.join(folder_path, "*_processed.xml"))
            
            print(f"Processing {folder}: {len(xml_files)} files")
            
            for xml_file in xml_files:
                df = self.parse_xml_to_dataframe(xml_file)
                if not df.empty:
                    all_data.append(df)
        
        if not all_data:
            print("❌ No data found")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 Total records: {len(combined_df):,}")
        
        # Aggregate lane features
        agg_df = self.aggregate_lane_features(combined_df)
        print(f"📊 After aggregation: {len(agg_df):,}")
        
        # Add temporal features
        final_df = self.add_temporal_features(agg_df)
        
        # Define feature columns for normalization
        feature_cols = ['speed_mean', 'speed_min', 'speed_max', 'speed_std',
                       'occupancy_mean', 'occupancy_max', 'volume_total']
        
        # Normalize features
        normalized_df = self.normalize_features(final_df, feature_cols)
        
        # Save datasets
        output_dir = os.path.join(self.base_dir, 'ML_Data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Raw aggregated data
        final_df.to_csv(os.path.join(output_dir, 'traffic_data_raw.csv'), index=False)
        
        # Normalized data
        normalized_df.to_csv(os.path.join(output_dir, 'traffic_data_normalized.csv'), index=False)
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
        
        # Data summary
        summary = {
            'total_records': len(final_df),
            'date_range': {
                'start': final_df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': final_df['timestamp'].max().strftime('%Y-%m-%d')
            },
            'detectors': final_df['detector_id'].nunique(),
            'tunnels': final_df['tunnel'].value_counts().to_dict(),
            'feature_columns': feature_cols,
            'categorical_columns': ['detector_id', 'tunnel', 'direction'],
            'temporal_columns': ['hour', 'weekday', 'is_weekend', 'is_peak']
        }
        
        with open(os.path.join(output_dir, 'data_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ ML data saved to: {output_dir}")
        print(f"📈 Features: {len(feature_cols)} numerical, 4 temporal, 3 categorical")
        
        return final_df, normalized_df

if __name__ == "__main__":
    preprocessor = MLDataPreprocessor()
    raw_df, norm_df = preprocessor.process_all_data()
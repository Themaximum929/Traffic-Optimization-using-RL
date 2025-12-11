import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

print("Loading CSV...")
df = pd.read_csv(r'c:\Users\maxch\MaxProjects\NLP\ML_Data\traffic_data_normalized.csv')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Convert to optimal dtypes
print("Optimizing data types...")
df['detector_id'] = df['detector_id'].astype('category')
df['tunnel'] = df['tunnel'].astype('category')
df['direction'] = df['direction'].astype('category')
for col in ['speed_mean', 'speed_min', 'speed_max', 'speed_std', 'occupancy_mean', 'occupancy_max']:
    df[col] = df[col].astype('float32')
df['volume_total'] = df['volume_total'].astype('int32')
for col in ['hour', 'weekday', 'is_weekend', 'is_peak']:
    df[col] = df[col].astype('int8')

print("Saving to Parquet...")
df.to_parquet(
    r'c:\Users\maxch\MaxProjects\NLP\ML_Data\traffic_data_normalized.parquet',
    engine='pyarrow',
    compression='snappy',
    index=False
)

print("Done! Parquet file created.")

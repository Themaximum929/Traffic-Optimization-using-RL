import pandas as pd

def fix_proposal_analysis():
    """Quick fix for the proposal analysis column issue"""
    
    print("DEBUGGING PROPOSAL ANALYSIS ISSUE")
    print("="*50)
    
    # Check normalized CSV structure
    try:
        df = pd.read_csv(r'c:\Users\maxch\MaxProjects\NLP\ML_Data\traffic_data_normalized.csv')
        print(f"Normalized CSV columns: {list(df.columns)}")
        print(f"Sample data shape: {df.shape}")
        
        # Check if we have speed vs speed_mean
        if 'speed_mean' in df.columns and 'speed' not in df.columns:
            print("✓ Found 'speed_mean' column (correct for normalized data)")
        elif 'speed' in df.columns:
            print("✓ Found 'speed' column (raw data format)")
        else:
            print("❌ No speed column found")
            
    except Exception as e:
        print(f"Error loading normalized CSV: {e}")
    
    print("\nRECOMMENDATION:")
    print("The proposal_analysis_complete.py script loads XML files directly,")
    print("so it should have 'speed' column. The error suggests the data")
    print("processing is returning empty DataFrames.")
    
    print("\nQUICK FIX:")
    print("1. The script should use XML data (has 'speed' column)")
    print("2. If using normalized CSV, change 'speed' to 'speed_mean'")
    print("3. Check if XML files exist for the analysis period")

if __name__ == "__main__":
    fix_proposal_analysis()
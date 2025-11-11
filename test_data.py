# test_your_data.py - Test with your exact data structure
from pathlib import Path
import pandas as pd

print("Testing NSE Data Access")
print("="*60)

# Your exact path
data_path = Path("./data/07112025/Equity")

if data_path.exists():
    print(f"✅ Found data folder: {data_path}")
    
    # List all files
    all_files = list(data_path.iterdir())
    print(f"\n📁 Total files: {len(all_files)}")
    
    # Categorize files
    excel_files = [f for f in all_files if f.suffix in ['.xlsx', '.xls']]
    csv_files = [f for f in all_files if f.suffix == '.csv']
    dat_files = [f for f in all_files if f.suffix.upper() == '.DAT']
    zip_files = [f for f in all_files if f.suffix == '.zip']
    
    print(f"\n📊 File types:")
    print(f"  Excel files: {len(excel_files)}")
    print(f"  CSV files: {len(csv_files)}")
    print(f"  DAT files: {len(dat_files)}")
    print(f"  ZIP files: {len(zip_files)}")
    
    # Try to read bhavdata
    print("\n🔍 Looking for bhavdata file...")
    bhav_file = data_path / "sec_bhavdata_full_07112025"
    
    if bhav_file.exists():
        print(f"  Found: {bhav_file.name}")
        try:
            # Try reading as Excel
            df = pd.read_excel(bhav_file)
            print(f"  ✅ Successfully read {len(df)} rows")
            print(f"  Columns: {', '.join(df.columns[:5])}...")
        except:
            print(f"  ❌ Could not read as Excel, may need conversion")
    else:
        print(f"  ❌ Not found as expected")
    
    # List actual files for reference
    print("\n📝 Files in folder:")
    for f in all_files[:10]:  # Show first 10
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
else:
    print(f"❌ Data folder not found: {data_path}")
    print(f"\nCurrent working directory: {Path.cwd()}")
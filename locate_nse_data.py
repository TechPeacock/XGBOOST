# locate_nse_data.py - Find where your NSE data actually is
import os
from pathlib import Path
import shutil

def find_nse_data_files():
    """Search for NSE data files in common locations"""
    
    # Update these paths based on where you mentioned your data is
    search_paths = [
        Path("C:/Users/Madhan/Desktop/Fundamental/Scanner/data"),
        Path("C:/Users/Madhan/Desktop/Fundamental/Scanner/Data"),
        Path("C:/Users/Madhan/Desktop/Data"),
        Path("C:/Users/Madhan/Downloads"),
        Path("C:/Users/Madhan/Documents"),
    ]
    
    print("="*60)
    print("SEARCHING FOR NSE DATA FILES")
    print("="*60)
    
    found_locations = []
    
    for search_path in search_paths:
        if search_path.exists():
            print(f"\nSearching in: {search_path}")
            
            # Look for date folders (10112025 format as mentioned in your description)
            for item in search_path.iterdir():
                if item.is_dir():
                    # Check if it's a date folder (8 digits)
                    if len(item.name) == 8 and item.name.isdigit():
                        print(f"  📁 Found date folder: {item.name}")
                        
                        # Check for Equity folder (capital E as in your description)
                        equity_folder = item / "Equity"
                        if not equity_folder.exists():
                            equity_folder = item / "equity"
                        
                        if equity_folder.exists():
                            csv_files = list(equity_folder.glob("*.csv")) + list(equity_folder.glob("*.CSV"))
                            if csv_files:
                                print(f"    ✓ Found {len(csv_files)} CSV files in Equity folder")
                                for f in csv_files[:3]:
                                    print(f"      - {f.name}")
                                found_locations.append({
                                    'date': item.name,
                                    'path': equity_folder,
                                    'files': csv_files
                                })
                        
                        # Check for fo folder
                        fo_folder = item / "fo"
                        if fo_folder.exists():
                            print(f"    ✓ Found F&O folder")
                        
                        # Check for dept/debt folder
                        debt_folder = item / "dept"
                        if not debt_folder.exists():
                            debt_folder = item / "debt"
                        if debt_folder.exists():
                            print(f"    ✓ Found Debt folder")
            
            # Also check for direct CSV files
            csv_files = list(search_path.glob("*.csv")) + list(search_path.glob("*.CSV"))
            if csv_files:
                print(f"  Found {len(csv_files)} CSV files directly in folder")
                for f in csv_files[:5]:
                    if 'bhav' in f.name.lower():
                        print(f"    📊 {f.name}")
    
    return found_locations

def copy_data_to_xgboost_folder(source_locations):
    """Copy found data to XGBOOST project folder"""
    if not source_locations:
        print("\n❌ No data locations found!")
        return False
    
    print("\n" + "="*60)
    print("COPYING DATA TO XGBOOST PROJECT")
    print("="*60)
    
    # Create the data structure
    xgboost_data = Path("./data")
    xgboost_equity = xgboost_data / "equity"
    xgboost_fo = xgboost_data / "fo"
    xgboost_debt = xgboost_data / "debt"
    
    # Create directories
    for folder in [xgboost_data, xgboost_equity, xgboost_fo, xgboost_debt]:
        folder.mkdir(exist_ok=True)
    
    # Copy the most recent date's data
    latest_data = sorted(source_locations, key=lambda x: x['date'])[-1]
    
    print(f"\nCopying from date: {latest_data['date']}")
    print(f"Source: {latest_data['path']}")
    print(f"Destination: {xgboost_equity}")
    
    copied_count = 0
    for file in latest_data['files'][:50]:  # Copy first 50 files to start
        try:
            dest_file = xgboost_equity / file.name
            shutil.copy2(file, dest_file)
            copied_count += 1
            if copied_count <= 5:
                print(f"  ✓ Copied: {file.name}")
        except Exception as e:
            print(f"  ✗ Error copying {file.name}: {e}")
    
    print(f"\n✅ Copied {copied_count} files to data/equity/")
    
    # Also check for other folders in the same date directory
    parent_dir = latest_data['path'].parent
    
    # Copy F&O data if exists
    fo_source = parent_dir / "fo"
    if fo_source.exists():
        fo_files = list(fo_source.glob("*.csv")) + list(fo_source.glob("*.CSV"))
        if fo_files:
            print(f"\nCopying F&O data...")
            for file in fo_files[:20]:
                try:
                    dest_file = xgboost_fo / file.name
                    shutil.copy2(file, dest_file)
                except:
                    pass
            print(f"  ✓ Copied {len(fo_files)} F&O files")
    
    return True

if __name__ == "__main__":
    # Find data
    locations = find_nse_data_files()
    
    if locations:
        print(f"\n✅ Found data in {len(locations)} location(s)")
        
        # Ask user if they want to copy
        print("\n" + "="*60)
        print("Would you like to copy the data to your XGBOOST project?")
        print("This will copy files to: ./data/equity/")
        print("="*60)
        
        response = input("\nType 'yes' to copy, or press Enter to skip: ")
        
        if response.lower() == 'yes':
            copy_data_to_xgboost_folder(locations)
            
            # Test if files are now accessible
            print("\n" + "="*60)
            print("VERIFYING DATA SETUP")
            print("="*60)
            
            equity_path = Path("./data/equity")
            if equity_path.exists():
                csv_files = list(equity_path.glob("*.csv")) + list(equity_path.glob("*.CSV"))
                print(f"✅ Found {len(csv_files)} files in data/equity/")
                
                # Find bhavdata file
                bhav_files = [f for f in csv_files if 'bhav' in f.name.lower()]
                if bhav_files:
                    print(f"✅ Found bhavdata file: {bhav_files[0].name}")
                else:
                    print("⚠️  No bhavdata file found - checking file names...")
                    for f in csv_files[:5]:
                        print(f"  - {f.name}")
        else:
            print("\nPlease manually copy your data files to:")
            print("  ./data/equity/ - for equity files")
            print("  ./data/fo/ - for F&O files")
            print("  ./data/debt/ - for debt files")
    else:
        print("\n❌ No NSE data found in expected locations!")
        print("\nPlease provide the correct path to your data folder.")
        custom_path = input("Enter the full path to your NSE data folder: ")
        
        if custom_path:
            custom_path = Path(custom_path)
            if custom_path.exists():
                print(f"\nChecking {custom_path}...")
                # List contents
                for item in custom_path.iterdir():
                    print(f"  - {item.name}")
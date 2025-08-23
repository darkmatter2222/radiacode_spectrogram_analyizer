#!/usr/bin/env python3
"""
Extract individual isotope files from the master isotope_baselines_22.json database.
Organizes files by category into subdirectories within master_spectra/
"""

import json
import os
from pathlib import Path

def create_directory_structure(base_path):
    """Create the directory structure for organizing isotope files by category."""
    categories = ['background', 'calibration', 'industrial', 'medical']
    
    for category in categories:
        category_path = base_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {category_path}")

def extract_isotope_files(input_file, output_base_dir):
    """
    Extract individual isotope files from the master JSON database.
    
    Args:
        input_file (str): Path to the master isotope_baselines_22.json file
        output_base_dir (str): Base directory to create the master_spectra structure
    """
    
    # Load the master database
    print(f"Loading master database from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract detector configuration and energy axis
    detector_config = data['detector']
    energy_axis = data['energy_axis_keV']
    isotopes = data['isotopes']
    
    print(f"Found {len(isotopes)} isotopes in the database")
    
    # Create output directory structure
    output_path = Path(output_base_dir)
    create_directory_structure(output_path)
    
    # Track statistics
    category_counts = {}
    
    # Process each isotope
    for isotope_name, isotope_data in isotopes.items():
        category = isotope_data['category']
        
        # Count by category
        category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create individual isotope file
        isotope_file_data = {
            'detector': detector_config,
            'isotope_name': isotope_name,
            'isotope_data': isotope_data,
            'energy_axis_keV': energy_axis
        }
        
        # Create filename (sanitize isotope name for filesystem)
        safe_name = isotope_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename = f"{safe_name}.json"
        
        # Determine output path based on category
        category_dir = output_path / category
        output_file = category_dir / filename
        
        # Write individual isotope file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(isotope_file_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total isotopes processed: {len(isotopes)}")
    print("\nFiles created by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category:12s}: {count:2d} files")
    
    print(f"\nAll files created in: {output_path}")
    
    # Create a summary file
    summary_data = {
        'extraction_info': {
            'source_file': str(input_file),
            'extraction_date': '2025-08-21',
            'total_isotopes': len(isotopes),
            'categories': category_counts
        },
        'detector_config': detector_config,
        'isotope_list': {
            category: [name for name, data in isotopes.items() if data['category'] == category]
            for category in category_counts.keys()
        }
    }
    
    summary_file = output_path / 'extraction_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created summary file: {summary_file}")

def main():
    """Main execution function."""
    
    # Define file paths
    script_dir = Path(__file__).parent
    input_file = script_dir / 'master_spectra' / 'isotope_baselines_22.json'
    output_dir = script_dir / 'master_spectra'
    
    # Check if input file exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please ensure isotope_baselines_22.json is in the same directory as this script.")
        return
    
    print("Isotope Database File Extractor")
    print("="*40)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Extract the files
    try:
        extract_isotope_files(input_file, output_dir)
        print("\n✅ Extraction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

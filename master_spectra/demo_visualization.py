#!/usr/bin/env python3
"""
Demo script to showcase the spectrum visualization capabilities.
This script demonstrates various visualization features in a responsible manner.
"""

import subprocess
import time
import os
from pathlib import Path

def run_visualization_demo():
    """Run a comprehensive demo of the visualization capabilities."""
    
    print("🎯 GAMMA SPECTRUM VISUALIZATION DEMO")
    print("="*50)
    print("This demo showcases responsible visualization of simulated gamma spectroscopy data.")
    print("All visualizations include safety warnings and educational context.")
    print("="*50)
    print()
    
    # Create plots directory
    plots_dir = Path("master_spectra/plots")
    plots_dir.mkdir(exist_ok=True)
    
    demos = [
        {
            "title": "📊 Listing Available Isotopes",
            "command": ["python", "visualize_spectra.py", "--list"],
            "description": "Shows all available isotopes organized by category"
        },
        {
            "title": "🔬 Single Isotope Analysis (Cs-137 Calibration Source)",
            "command": ["python", "visualize_spectra.py", "--single", "calibration", "Cs-137", "--save"],
            "description": "Detailed analysis of Cs-137 with gamma ray peak annotations"
        },
        {
            "title": "🏭 Industrial Isotopes Overview",
            "command": ["python", "visualize_spectra.py", "--category", "industrial", "--save"],
            "description": "Overview of all industrial isotopes in the database"
        },
        {
            "title": "🏥 Medical Isotope Comparison",
            "command": ["python", "visualize_spectra.py", "--compare", "medical", "Tc-99m", "medical", "I-131", "--save"],
            "description": "Side-by-side comparison of common medical isotopes"
        },
        {
            "title": "🎯 Detector Resolution Analysis",
            "command": ["python", "visualize_spectra.py", "--resolution", "--save"],
            "description": "Analysis of NaI(Tl) detector energy resolution characteristics"
        },
        {
            "title": "🌍 Background Radiation Sources",
            "command": ["python", "visualize_spectra.py", "--category", "background", "--save"],
            "description": "Natural background radiation sources (K-40, U-238, Th-232 series)"
        }
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"{i}. {demo['title']}")
        print(f"   Description: {demo['description']}")
        print(f"   Command: {' '.join(demo['command'])}")
        
        # Ask user if they want to run this demo
        response = input(f"   Run this demo? (y/n/q): ").strip().lower()
        
        if response == 'q':
            print("Demo stopped by user.")
            break
        elif response == 'y':
            print("   Running...")
            try:
                result = subprocess.run(demo['command'], capture_output=True, text=True, cwd='.')
                if result.returncode == 0:
                    print("   ✅ Success!")
                    # Show relevant output lines (skip the safety notice for brevity)
                    output_lines = result.stdout.split('\n')
                    relevant_lines = [line for line in output_lines if 
                                    'Plot saved:' in line or 'saved:' in line or 
                                    'Total:' in line or '📊' in line]
                    for line in relevant_lines[-3:]:  # Show last few relevant lines
                        if line.strip():
                            print(f"   {line}")
                else:
                    print(f"   ❌ Error: {result.stderr}")
            except Exception as e:
                print(f"   ❌ Error running demo: {str(e)}")
        else:
            print("   Skipped.")
        
        print()
        time.sleep(0.5)  # Brief pause between demos
    
    # Show final summary
    print("🎉 DEMO SUMMARY")
    print("="*30)
    
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        print(f"📁 Plots created: {len(plot_files)}")
        for plot_file in sorted(plot_files):
            print(f"   • {plot_file.name}")
        print(f"\n📂 All plots saved in: {plots_dir}")
    
    print("\n🔬 VISUALIZATION FEATURES DEMONSTRATED:")
    print("   • Safety warnings and educational context")
    print("   • Individual isotope spectrum analysis")
    print("   • Category-based organization")
    print("   • Multi-isotope comparisons")
    print("   • Detector resolution analysis")
    print("   • Gamma ray peak annotations")
    print("   • Both linear and logarithmic scales")
    print("   • Nuclear data integration (half-life, energies)")
    print("   • High-quality plot exports")
    
    print("\n📖 EDUCATIONAL VALUE:")
    print("   • Proper scientific presentation")
    print("   • Radiation safety awareness")
    print("   • Detector physics principles")
    print("   • Nuclear decay properties")
    print("   • Spectroscopy analysis techniques")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    run_visualization_demo()

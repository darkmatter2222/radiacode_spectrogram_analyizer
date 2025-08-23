#!/usr/bin/env python3
"""
Responsible Gamma Spectrum Visualization Tool

This script provides comprehensive visualization capabilities for gamma spectroscopy data
from the extracted isotope database. It includes safety warnings, educational context,
and proper scientific presentation of radioactive material data.

SAFETY NOTICE:
This tool is for educational and research purposes only. All data represents simulated
detector responses. Real radioactive materials require proper licensing, training,
and safety protocols. Always follow local radiation safety regulations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import warnings

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpectrumVisualizer:
    """
    A comprehensive gamma spectrum visualization tool with safety considerations
    and educational context.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize the visualizer with the base path to the master_spectra directory.
        
        Args:
            base_path (str): Path to the master_spectra directory
        """
        self.base_path = Path(base_path)
        self.categories = ['background', 'calibration', 'industrial', 'medical']
        self.detector_config = None
        self.energy_axis = None
        
        # Safety and educational messaging
        self._display_safety_notice()
        
    def _display_safety_notice(self):
        """Display important safety and educational information."""
        print("="*80)
        print("🔬 GAMMA SPECTRUM VISUALIZATION TOOL")
        print("="*80)
        print("⚠️  IMPORTANT SAFETY NOTICE:")
        print("   • This tool displays SIMULATED detector responses only")
        print("   • Real radioactive materials require proper licensing & training")
        print("   • Always follow radiation safety protocols and regulations")
        print("   • For educational and research purposes only")
        print("="*80)
        print()
    
    def load_isotope_data(self, category: str, isotope_name: str) -> Dict:
        """
        Load data for a specific isotope.
        
        Args:
            category (str): Isotope category (background, calibration, industrial, medical)
            isotope_name (str): Name of the isotope (without .json extension)
            
        Returns:
            Dict: Loaded isotope data
        """
        file_path = self.base_path / category / f"{isotope_name}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Isotope file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Cache detector config and energy axis
        if self.detector_config is None:
            self.detector_config = data['detector']
            self.energy_axis = np.array(data['energy_axis_keV'])
        
        return data
    
    def get_available_isotopes(self) -> Dict[str, List[str]]:
        """
        Get all available isotopes organized by category.
        
        Returns:
            Dict[str, List[str]]: Dictionary of categories and their isotopes
        """
        isotopes = {}
        
        for category in self.categories:
            category_path = self.base_path / category
            if category_path.exists():
                isotopes[category] = []
                for file_path in category_path.glob("*.json"):
                    isotope_name = file_path.stem
                    isotopes[category].append(isotope_name)
                isotopes[category].sort()
        
        return isotopes
    
    def plot_single_spectrum(self, category: str, isotope_name: str, 
                           save_plot: bool = False, show_peaks: bool = True) -> plt.Figure:
        """
        Plot a single isotope spectrum with detailed annotations.
        
        Args:
            category (str): Isotope category
            isotope_name (str): Isotope name
            save_plot (bool): Whether to save the plot
            show_peaks (bool): Whether to annotate gamma ray peaks
            
        Returns:
            plt.Figure: The created figure
        """
        # Load data
        data = self.load_isotope_data(category, isotope_name)
        isotope_data = data['isotope_data']
        spectrum = np.array(isotope_data['spectrum_counts'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Main spectrum plot (linear scale)
        ax1.plot(self.energy_axis, spectrum, linewidth=1.5, alpha=0.8, 
                label=f"{isotope_name} ({category})")
        ax1.fill_between(self.energy_axis, spectrum, alpha=0.3)
        
        # Annotate gamma ray peaks
        if show_peaks and 'gamma_energies_keV' in isotope_data:
            for i, (energy, weight) in enumerate(zip(isotope_data['gamma_energies_keV'], 
                                                   isotope_data['gamma_weights_rel'])):
                if energy <= max(self.energy_axis):
                    ax1.axvline(x=energy, color='red', linestyle='--', alpha=0.7)
                    ax1.annotate(f'{energy:.1f} keV\n(rel: {weight:.2f})', 
                               xy=(energy, max(spectrum)*0.8), 
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               fontsize=9, ha='left')
        
        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel('Counts')
        ax1.set_title(f'Gamma Spectrum: {isotope_name} (Category: {category.title()})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Log scale plot for better peak visibility
        ax2.semilogy(self.energy_axis, spectrum + 1, linewidth=1.5, alpha=0.8)  # +1 to avoid log(0)
        ax2.fill_between(self.energy_axis, spectrum + 1, alpha=0.3)
        ax2.set_xlabel('Energy (keV)')
        ax2.set_ylabel('Counts (log scale)')
        ax2.set_title(f'Log Scale View - {isotope_name}')
        ax2.grid(True, alpha=0.3)
        
        # Add isotope information
        info_text = f"""Isotope Information:
Half-life: {isotope_data.get('half_life', 'N/A')}
Collection time: {isotope_data.get('collection_time_s', 'N/A')} seconds
Category: {category.title()}
Total counts: {sum(spectrum):,.0f}

Detector: {self.detector_config['type']}
Energy range: {self.detector_config['energy_range_keV'][0]}-{self.detector_config['energy_range_keV'][1]} keV
Channels: {self.detector_config['channels']}"""
        
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        
        # Save if requested
        if save_plot:
            output_dir = self.base_path / 'plots'
            output_dir.mkdir(exist_ok=True)
            filename = f"{category}_{isotope_name}_spectrum.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_dir / filename}")
        
        return fig
    
    def compare_spectra(self, isotope_list: List[Tuple[str, str]], 
                       save_plot: bool = False) -> plt.Figure:
        """
        Compare multiple isotope spectra on the same plot.
        
        Args:
            isotope_list: List of (category, isotope_name) tuples
            save_plot: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(isotope_list)))
        max_counts = 0
        
        for i, (category, isotope_name) in enumerate(isotope_list):
            data = self.load_isotope_data(category, isotope_name)
            spectrum = np.array(data['isotope_data']['spectrum_counts'])
            max_counts = max(max_counts, max(spectrum))
            
            # Normalize for comparison
            normalized_spectrum = spectrum / max(spectrum) if max(spectrum) > 0 else spectrum
            
            ax1.plot(self.energy_axis, spectrum, linewidth=1.5, alpha=0.8, 
                    color=colors[i], label=f"{isotope_name} ({category})")
            
            ax2.plot(self.energy_axis, normalized_spectrum, linewidth=1.5, alpha=0.8,
                    color=colors[i], label=f"{isotope_name} (normalized)")
        
        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel('Counts')
        ax1.set_title('Spectrum Comparison - Absolute Counts')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax2.set_xlabel('Energy (keV)')
        ax2.set_ylabel('Normalized Counts')
        ax2.set_title('Spectrum Comparison - Normalized')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self.base_path / 'plots'
            output_dir.mkdir(exist_ok=True)
            filename = f"comparison_{len(isotope_list)}_isotopes.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved: {output_dir / filename}")
        
        return fig
    
    def category_overview(self, category: str, save_plot: bool = False) -> plt.Figure:
        """
        Create an overview plot of all isotopes in a category.
        
        Args:
            category: Category to visualize
            save_plot: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        isotopes = self.get_available_isotopes().get(category, [])
        
        if not isotopes:
            raise ValueError(f"No isotopes found in category: {category}")
        
        # Calculate grid size
        n_isotopes = len(isotopes)
        cols = min(3, n_isotopes)
        rows = (n_isotopes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_isotopes == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, isotope_name in enumerate(isotopes):
            data = self.load_isotope_data(category, isotope_name)
            spectrum = np.array(data['isotope_data']['spectrum_counts'])
            
            axes[i].plot(self.energy_axis, spectrum, linewidth=1.0)
            axes[i].fill_between(self.energy_axis, spectrum, alpha=0.3)
            axes[i].set_title(f"{isotope_name}")
            axes[i].set_xlabel('Energy (keV)')
            axes[i].set_ylabel('Counts')
            axes[i].grid(True, alpha=0.3)
            
            # Add half-life info
            half_life = data['isotope_data'].get('half_life', 'N/A')
            axes[i].text(0.02, 0.98, f"t½: {half_life}", transform=axes[i].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                        fontsize=8)
        
        # Hide unused subplots
        for i in range(n_isotopes, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{category.title()} Isotopes Overview', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_plot:
            output_dir = self.base_path / 'plots'
            output_dir.mkdir(exist_ok=True)
            filename = f"{category}_overview.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Category overview saved: {output_dir / filename}")
        
        return fig
    
    def energy_resolution_analysis(self, save_plot: bool = False) -> plt.Figure:
        """
        Analyze and visualize the detector energy resolution model.
        
        Args:
            save_plot: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        # Load detector config if not already loaded
        if self.detector_config is None:
            # Load any isotope to get detector config
            isotopes = self.get_available_isotopes()
            category = list(isotopes.keys())[0]
            isotope = isotopes[category][0]
            self.load_isotope_data(category, isotope)
        
        # Create energy range for resolution calculation
        energies = np.linspace(50, 3000, 100)  # keV
        a_res = self.detector_config['a_res']
        
        # Calculate FWHM using the model: FWHM(E) = a*sqrt(E)
        fwhm = a_res * np.sqrt(energies)
        resolution_percent = (fwhm / energies) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # FWHM vs Energy
        ax1.plot(energies, fwhm, 'b-', linewidth=2, label='FWHM(E) = a√E')
        ax1.axhline(y=a_res * np.sqrt(662), color='red', linestyle='--', 
                   label=f'FWHM at 662 keV: {a_res * np.sqrt(662):.1f} keV')
        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel('FWHM (keV)')
        ax1.set_title('Detector Energy Resolution (FWHM)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Resolution percentage vs Energy
        ax2.plot(energies, resolution_percent, 'g-', linewidth=2)
        ax2.axhline(y=(a_res * np.sqrt(662) / 662) * 100, color='red', linestyle='--',
                   label=f'Resolution at 662 keV: {(a_res * np.sqrt(662) / 662) * 100:.1f}%')
        ax2.set_xlabel('Energy (keV)')
        ax2.set_ylabel('Resolution (%)')
        ax2.set_title('Detector Resolution Percentage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add detector info
        info_text = f"""Detector Configuration:
Type: {self.detector_config['type']}
Resolution model: {self.detector_config['energy_resolution_model']}
Parameter a = {a_res:.3f}"""
        
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = self.base_path / 'plots'
            output_dir.mkdir(exist_ok=True)
            filename = "detector_resolution_analysis.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Resolution analysis saved: {output_dir / filename}")
        
        return fig
    
    def list_available_isotopes(self):
        """Print a formatted list of all available isotopes."""
        isotopes = self.get_available_isotopes()
        
        print("📋 AVAILABLE ISOTOPES BY CATEGORY:")
        print("="*50)
        
        total_count = 0
        for category, isotope_list in isotopes.items():
            print(f"\n🔹 {category.upper()} ({len(isotope_list)} isotopes):")
            for isotope in isotope_list:
                print(f"   • {isotope}")
            total_count += len(isotope_list)
        
        print(f"\n📊 Total: {total_count} isotopes across {len(isotopes)} categories")
        print("="*50)


def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize gamma spectroscopy data responsibly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_spectra.py --list
  python visualize_spectra.py --single medical Tc-99m
  python visualize_spectra.py --category medical --save
  python visualize_spectra.py --compare medical Tc-99m medical I-131
  python visualize_spectra.py --resolution
        """
    )
    
    parser.add_argument('--base-path', default='master_spectra',
                      help='Path to master_spectra directory (default: master_spectra)')
    parser.add_argument('--list', action='store_true',
                      help='List all available isotopes')
    parser.add_argument('--single', nargs=2, metavar=('CATEGORY', 'ISOTOPE'),
                      help='Plot single isotope spectrum')
    parser.add_argument('--category', metavar='CATEGORY',
                      help='Plot overview of all isotopes in category')
    parser.add_argument('--compare', nargs='+', metavar='CAT ISOTOPE',
                      help='Compare multiple isotopes (pairs of category isotope)')
    parser.add_argument('--resolution', action='store_true',
                      help='Analyze detector energy resolution')
    parser.add_argument('--save', action='store_true',
                      help='Save plots to files')
    parser.add_argument('--no-peaks', action='store_true',
                      help='Don\'t show gamma ray peak annotations')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = SpectrumVisualizer(args.base_path)
    
    try:
        if args.list:
            visualizer.list_available_isotopes()
        
        elif args.single:
            category, isotope = args.single
            fig = visualizer.plot_single_spectrum(category, isotope, 
                                                args.save, not args.no_peaks)
            plt.show()
        
        elif args.category:
            fig = visualizer.category_overview(args.category, args.save)
            plt.show()
        
        elif args.compare:
            if len(args.compare) % 2 != 0:
                print("Error: --compare requires pairs of category isotope arguments")
                return
            
            isotope_pairs = []
            for i in range(0, len(args.compare), 2):
                isotope_pairs.append((args.compare[i], args.compare[i+1]))
            
            fig = visualizer.compare_spectra(isotope_pairs, args.save)
            plt.show()
        
        elif args.resolution:
            fig = visualizer.energy_resolution_analysis(args.save)
            plt.show()
        
        else:
            # Interactive mode
            print("🎯 INTERACTIVE MODE")
            print("Choose an option:")
            print("1. List available isotopes")
            print("2. Plot single isotope")
            print("3. Category overview")
            print("4. Compare isotopes")
            print("5. Detector resolution analysis")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                visualizer.list_available_isotopes()
            elif choice == '2':
                category = input("Enter category: ").strip()
                isotope = input("Enter isotope name: ").strip()
                fig = visualizer.plot_single_spectrum(category, isotope, show_peaks=True)
                plt.show()
            elif choice == '3':
                category = input("Enter category: ").strip()
                fig = visualizer.category_overview(category)
                plt.show()
            elif choice == '4':
                print("Enter isotopes to compare (category isotope pairs):")
                isotope_pairs = []
                while True:
                    category = input("Category (or 'done'): ").strip()
                    if category.lower() == 'done':
                        break
                    isotope = input("Isotope: ").strip()
                    isotope_pairs.append((category, isotope))
                
                if isotope_pairs:
                    fig = visualizer.compare_spectra(isotope_pairs)
                    plt.show()
            elif choice == '5':
                fig = visualizer.energy_resolution_analysis()
                plt.show()
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

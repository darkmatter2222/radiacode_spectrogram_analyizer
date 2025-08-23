#!/usr/bin/env python3
"""
Synthetic Gamma Spectrum Generator for Machine Learning Training

This script generates synthetic blended isotope spectra by combining individual
isotope signatures at various ratios. The goal is to create training data for
machine learning models that will learn to identify constituent isotopes from
mixed spectra.

Key Features:
- Blends 1-5 isotopes per synthetic spectrum
- Variable mixing ratios (including background considerations)
- Realistic spectral blending with proper scaling
- Comprehensive labeling for ML training
- Configurable output for 100,000+ synthetic spectra
- Proper handling of background vs. signal isotopes

IMPORTANT: This generates SIMULATED training data for educational/research AI models.
Real radioactive material identification requires proper training and licensing.
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import logging
from collections import defaultdict

class SyntheticSpectrumGenerator:
    """
    Generates synthetic blended gamma spectra for machine learning training.
    """
    
    def __init__(self, master_spectra_path: str, output_path: str):
        """
        Initialize the synthetic spectrum generator.
        
        Args:
            master_spectra_path (str): Path to the master_spectra directory
            output_path (str): Path where synthetic spectra will be saved
        """
        self.master_path = Path(master_spectra_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load all isotope data
        self.isotope_database = {}
        self.background_isotopes = []
        self.signal_isotopes = []  # calibration, industrial, medical
        self.detector_config = None
        self.energy_axis = None
        
        self._load_isotope_database()
        self._setup_logging()
        
        print("🔬 SYNTHETIC SPECTRUM GENERATOR")
        print("="*50)
        print("⚠️  IMPORTANT: This generates SIMULATED training data")
        print("   Real radioactive material identification requires")
        print("   proper training, licensing, and safety protocols.")
        print("="*50)
        print(f"📊 Loaded {len(self.isotope_database)} isotopes:")
        print(f"   • Background: {len(self.background_isotopes)} isotopes")
        print(f"   • Signal: {len(self.signal_isotopes)} isotopes")
        print(f"💾 Output directory: {self.output_path}")
        print()
    
    def _setup_logging(self):
        """Set up logging for the generation process."""
        log_file = self.output_path / 'generation.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_isotope_database(self):
        """Load all isotope data from the master_spectra directory."""
        categories = ['background', 'calibration', 'industrial', 'medical']
        
        for category in categories:
            category_path = self.master_path / category
            if not category_path.exists():
                continue
                
            for isotope_file in category_path.glob("*.json"):
                isotope_name = isotope_file.stem
                
                with open(isotope_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Store isotope data
                self.isotope_database[isotope_name] = {
                    'category': category,
                    'file_path': isotope_file,
                    'isotope_data': data['isotope_data'],
                    'spectrum_counts': np.array(data['isotope_data']['spectrum_counts'])
                }
                
                # Cache detector config and energy axis (same for all)
                if self.detector_config is None:
                    self.detector_config = data['detector']
                    self.energy_axis = np.array(data['energy_axis_keV'])
                
                # Categorize isotopes for blending strategy
                if category == 'background':
                    self.background_isotopes.append(isotope_name)
                else:
                    self.signal_isotopes.append(isotope_name)
    
    def _normalize_spectrum(self, spectrum: np.ndarray, target_counts: float) -> np.ndarray:
        """
        Normalize a spectrum to a target total count level.
        
        Args:
            spectrum: Input spectrum counts
            target_counts: Target total counts
            
        Returns:
            Normalized spectrum
        """
        current_total = np.sum(spectrum)
        if current_total > 0:
            scaling_factor = target_counts / current_total
            return spectrum * scaling_factor
        return spectrum
    
    def _add_poisson_noise(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Add realistic Poisson counting statistics noise.
        
        Args:
            spectrum: Input spectrum (can have float values)
            
        Returns:
            Spectrum with Poisson noise applied
        """
        # Ensure non-negative values for Poisson sampling
        spectrum_positive = np.maximum(spectrum, 0.1)
        
        # Apply Poisson noise
        noisy_spectrum = np.random.poisson(spectrum_positive)
        
        return noisy_spectrum.astype(float)
    
    def _select_isotopes_for_blend(self, include_background: bool = True) -> List[str]:
        """
        Select 1-5 isotopes for blending with intelligent selection logic.
        
        Args:
            include_background: Whether to include background isotopes
            
        Returns:
            List of selected isotope names
        """
        # Determine number of isotopes (1-5)
        num_isotopes = random.randint(1, 5)
        selected_isotopes = []
        
        # Strategy for isotope selection
        if include_background and random.random() < 0.7:  # 70% chance to include background
            # Select 1 background isotope
            background_isotope = random.choice(self.background_isotopes)
            selected_isotopes.append(background_isotope)
            num_isotopes -= 1
        
        # Fill remaining slots with signal isotopes
        if num_isotopes > 0:
            available_signals = [iso for iso in self.signal_isotopes if iso not in selected_isotopes]
            if len(available_signals) >= num_isotopes:
                signal_isotopes = random.sample(available_signals, num_isotopes)
                selected_isotopes.extend(signal_isotopes)
            else:
                selected_isotopes.extend(available_signals)
        
        # Ensure we have at least one isotope
        if not selected_isotopes:
            selected_isotopes = [random.choice(self.signal_isotopes)]
        
        return selected_isotopes
    
    def _generate_mixing_ratios(self, num_isotopes: int) -> List[float]:
        """
        Generate realistic mixing ratios for isotopes.
        
        Args:
            num_isotopes: Number of isotopes to mix
            
        Returns:
            List of mixing ratios (sum to 1.0)
        """
        if num_isotopes == 1:
            return [1.0]
        
        # Generate random weights
        weights = []
        for _ in range(num_isotopes):
            # Use exponential distribution to favor some isotopes over others
            weight = random.expovariate(1.0)
            weights.append(weight)
        
        # Normalize to sum to 1.0
        total_weight = sum(weights)
        ratios = [w / total_weight for w in weights]
        
        # Ensure ratios are reasonable (no isotope below 1% unless it's background)
        min_ratio = 0.005  # 0.5%
        for i, ratio in enumerate(ratios):
            if ratio < min_ratio:
                ratios[i] = min_ratio
        
        # Renormalize
        total_ratio = sum(ratios)
        ratios = [r / total_ratio for r in ratios]
        
        return ratios
    
    def _blend_spectra(self, isotope_names: List[str], mixing_ratios: List[float], 
                      target_total_counts: float) -> Tuple[np.ndarray, Dict]:
        """
        Blend multiple isotope spectra according to mixing ratios.
        
        Args:
            isotope_names: List of isotopes to blend
            mixing_ratios: Corresponding mixing ratios
            target_total_counts: Target total counts for the blended spectrum
            
        Returns:
            Tuple of (blended_spectrum, metadata)
        """
        if len(isotope_names) != len(mixing_ratios):
            raise ValueError("Number of isotopes must match number of mixing ratios")
        
        # Initialize blended spectrum
        blended_spectrum = np.zeros(len(self.energy_axis))
        blend_metadata = {
            'constituent_isotopes': [],
            'mixing_ratios': [],
            'categories': [],
            'total_counts_by_isotope': [],
            'gamma_energies_all': [],
            'has_background': False
        }
        
        # Blend each isotope contribution
        for isotope_name, ratio in zip(isotope_names, mixing_ratios):
            isotope_data = self.isotope_database[isotope_name]
            spectrum = isotope_data['spectrum_counts'].copy()
            category = isotope_data['category']
            
            # Calculate target counts for this isotope
            isotope_target_counts = target_total_counts * ratio
            
            # Normalize spectrum to target counts
            normalized_spectrum = self._normalize_spectrum(spectrum, isotope_target_counts)
            
            # Add to blended spectrum
            blended_spectrum += normalized_spectrum
            
            # Record metadata
            blend_metadata['constituent_isotopes'].append(isotope_name)
            blend_metadata['mixing_ratios'].append(ratio)
            blend_metadata['categories'].append(category)
            blend_metadata['total_counts_by_isotope'].append(float(np.sum(normalized_spectrum)))
            
            # Collect gamma energies
            gamma_energies = isotope_data['isotope_data'].get('gamma_energies_keV', [])
            blend_metadata['gamma_energies_all'].extend(gamma_energies)
            
            # Check for background
            if category == 'background':
                blend_metadata['has_background'] = True
        
        return blended_spectrum, blend_metadata
    
    def _create_ml_labels(self, blend_metadata: Dict, isotope_names: List[str], 
                         mixing_ratios: List[float]) -> Dict:
        """
        Create comprehensive labels for machine learning training.
        
        Args:
            blend_metadata: Metadata from spectrum blending
            isotope_names: List of constituent isotopes
            mixing_ratios: Corresponding mixing ratios
            
        Returns:
            Dictionary of ML-ready labels
        """
        # Binary presence labels for all possible isotopes
        isotope_presence = {}
        for isotope_name in self.isotope_database.keys():
            isotope_presence[isotope_name] = isotope_name in isotope_names
        
        # Concentration labels (0.0 if not present)
        isotope_concentrations = {}
        for isotope_name in self.isotope_database.keys():
            if isotope_name in isotope_names:
                idx = isotope_names.index(isotope_name)
                isotope_concentrations[isotope_name] = mixing_ratios[idx]
            else:
                isotope_concentrations[isotope_name] = 0.0
        
        # Category presence
        category_presence = {
            'has_background': False,
            'has_calibration': False,
            'has_industrial': False,
            'has_medical': False
        }
        
        for isotope_name in isotope_names:
            category = self.isotope_database[isotope_name]['category']
            category_presence[f'has_{category}'] = True
        
        # Multi-hot encoding for categories
        categories_present = list(set(blend_metadata['categories']))
        
        # Complexity metrics
        complexity_metrics = {
            'num_isotopes': len(isotope_names),
            'max_concentration': max(mixing_ratios),
            'min_concentration': min(mixing_ratios),
            'concentration_entropy': -sum(r * np.log(r + 1e-10) for r in mixing_ratios),
            'has_multiple_categories': len(categories_present) > 1
        }
        
        return {
            'isotope_presence': isotope_presence,
            'isotope_concentrations': isotope_concentrations,
            'category_presence': category_presence,
            'categories_present': categories_present,
            'complexity_metrics': complexity_metrics,
            'primary_isotope': isotope_names[np.argmax(mixing_ratios)],
            'secondary_isotopes': [iso for iso in isotope_names if iso != isotope_names[np.argmax(mixing_ratios)]]
        }
    
    def generate_synthetic_spectrum(self, spectrum_id: int, 
                                  target_counts_range: Tuple[float, float] = (5000, 50000),
                                  include_background_prob: float = 0.7) -> Dict:
        """
        Generate a single synthetic spectrum with complete metadata.
        
        Args:
            spectrum_id: Unique identifier for this spectrum
            target_counts_range: Range of total counts for the spectrum
            include_background_prob: Probability of including background isotopes
            
        Returns:
            Complete synthetic spectrum data structure
        """
        # Select isotopes for blending
        include_background = random.random() < include_background_prob
        isotope_names = self._select_isotopes_for_blend(include_background)
        
        # Generate mixing ratios
        mixing_ratios = self._generate_mixing_ratios(len(isotope_names))
        
        # Select target total counts
        target_counts = random.uniform(*target_counts_range)
        
        # Blend spectra
        blended_spectrum, blend_metadata = self._blend_spectra(
            isotope_names, mixing_ratios, target_counts
        )
        
        # Add realistic counting noise
        noisy_spectrum = self._add_poisson_noise(blended_spectrum)
        
        # Create ML labels
        ml_labels = self._create_ml_labels(blend_metadata, isotope_names, mixing_ratios)
        
        # Compile complete synthetic spectrum data
        synthetic_data = {
            'spectrum_id': spectrum_id,
            'generation_timestamp': datetime.now().isoformat(),
            'detector_config': self.detector_config,
            'energy_axis_keV': self.energy_axis.tolist(),
            'spectrum_counts': noisy_spectrum.tolist(),
            'synthesis_info': {
                'constituent_isotopes': isotope_names,
                'mixing_ratios': mixing_ratios,
                'target_total_counts': target_counts,
                'actual_total_counts': float(np.sum(noisy_spectrum)),
                'include_background': include_background,
                'num_isotopes': len(isotope_names)
            },
            'blend_metadata': blend_metadata,
            'ml_labels': ml_labels,
            'generation_parameters': {
                'target_counts_range': target_counts_range,
                'include_background_prob': include_background_prob,
                'poisson_noise_applied': True
            }
        }
        
        return synthetic_data
    
    def generate_dataset(self, num_spectra: int = 100000, 
                        batch_size: int = 1000,
                        save_individual_files: bool = False) -> None:
        """
        Generate a large dataset of synthetic spectra.
        
        Args:
            num_spectra: Total number of synthetic spectra to generate
            batch_size: Number of spectra to process in each batch
            save_individual_files: Whether to save individual spectrum files
        """
        self.logger.info(f"Starting generation of {num_spectra} synthetic spectra")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Save individual files: {save_individual_files}")
        
        # Statistics tracking
        stats = {
            'total_generated': 0,
            'isotope_usage': defaultdict(int),
            'category_combinations': defaultdict(int),
            'complexity_distribution': defaultdict(int)
        }
        
        # Generate in batches
        for batch_start in range(0, num_spectra, batch_size):
            batch_end = min(batch_start + batch_size, num_spectra)
            current_batch_size = batch_end - batch_start
            
            self.logger.info(f"Generating batch {batch_start//batch_size + 1}: "
                           f"spectra {batch_start+1}-{batch_end}")
            
            batch_data = []
            
            for i in range(current_batch_size):
                spectrum_id = batch_start + i
                
                # Generate synthetic spectrum
                synthetic_spectrum = self.generate_synthetic_spectrum(spectrum_id)
                
                # Update statistics
                stats['total_generated'] += 1
                for isotope in synthetic_spectrum['synthesis_info']['constituent_isotopes']:
                    stats['isotope_usage'][isotope] += 1
                
                categories = tuple(sorted(synthetic_spectrum['blend_metadata']['categories']))
                stats['category_combinations'][categories] += 1
                
                num_isotopes = synthetic_spectrum['synthesis_info']['num_isotopes']
                stats['complexity_distribution'][num_isotopes] += 1
                
                # Save individual file if requested
                if save_individual_files:
                    individual_file = self.output_path / f"synthetic_spectrum_{spectrum_id:06d}.json"
                    with open(individual_file, 'w', encoding='utf-8') as f:
                        json.dump(synthetic_spectrum, f, indent=2, ensure_ascii=False)
                
                batch_data.append(synthetic_spectrum)
            
            # Save batch file
            batch_file = self.output_path / f"synthetic_batch_{batch_start//batch_size:04d}.json"
            batch_info = {
                'batch_info': {
                    'batch_number': batch_start // batch_size,
                    'batch_size': current_batch_size,
                    'spectrum_id_range': [batch_start, batch_end - 1],
                    'generation_timestamp': datetime.now().isoformat()
                },
                'spectra': batch_data
            }
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved batch to: {batch_file}")
            
            # Progress update
            progress = (batch_end / num_spectra) * 100
            print(f"Progress: {progress:.1f}% ({batch_end}/{num_spectra})")
        
        # Save final statistics and metadata
        self._save_generation_summary(stats, num_spectra)
        
        self.logger.info("Dataset generation completed!")
        self.logger.info(f"Generated {stats['total_generated']} synthetic spectra")
        print(f"\n🎉 GENERATION COMPLETED!")
        print(f"📊 Generated {stats['total_generated']} synthetic spectra")
        print(f"💾 Output directory: {self.output_path}")
    
    def _save_generation_summary(self, stats: Dict, num_spectra: int) -> None:
        """Save comprehensive generation summary and statistics."""
        summary = {
            'generation_summary': {
                'total_spectra_requested': num_spectra,
                'total_spectra_generated': stats['total_generated'],
                'generation_timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_path),
                'generator_version': '1.0'
            },
            'dataset_statistics': {
                'isotope_usage_counts': dict(stats['isotope_usage']),
                'category_combinations': {str(k): v for k, v in stats['category_combinations'].items()},
                'complexity_distribution': dict(stats['complexity_distribution'])
            },
            'available_isotopes': {
                'background_isotopes': self.background_isotopes,
                'signal_isotopes': self.signal_isotopes,
                'total_isotopes': len(self.isotope_database)
            },
            'detector_configuration': self.detector_config,
            'ml_training_info': {
                'label_types': [
                    'isotope_presence (binary for each isotope)',
                    'isotope_concentrations (continuous 0-1)',
                    'category_presence (binary for each category)',
                    'complexity_metrics (various numerical features)'
                ],
                'recommended_models': [
                    'Multi-label classification for isotope presence',
                    'Regression for concentration estimation',
                    'Multi-task learning combining both approaches'
                ]
            }
        }
        
        summary_file = self.output_path / 'generation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generation summary saved: {summary_file}")


def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic gamma spectra for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_synthetic_spectra.py --num-spectra 100000
  python generate_synthetic_spectra.py --num-spectra 10000 --batch-size 500 --individual-files
  python generate_synthetic_spectra.py --output synthetic_data --num-spectra 50000
        """
    )
    
    parser.add_argument('--master-path', default='master_spectra',
                      help='Path to master_spectra directory (default: master_spectra)')
    parser.add_argument('--output', default='O:/master_data_collection/isotope',
                      help='Output directory for synthetic spectra')
    parser.add_argument('--num-spectra', type=int, default=1000000,
                      help='Number of synthetic spectra to generate (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Batch size for processing (default: 1000)')
    parser.add_argument('--individual-files', action='store_true',
                      help='Save individual spectrum files (in addition to batches)')
    parser.add_argument('--background-prob', type=float, default=0.7,
                      help='Probability of including background isotopes (default: 0.7)')
    parser.add_argument('--min-counts', type=float, default=5000,
                      help='Minimum total counts per spectrum (default: 5000)')
    parser.add_argument('--max-counts', type=float, default=50000,
                      help='Maximum total counts per spectrum (default: 50000)')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = SyntheticSpectrumGenerator(args.master_path, args.output)
        
        # Generate dataset
        generator.generate_dataset(
            num_spectra=args.num_spectra,
            batch_size=args.batch_size,
            save_individual_files=args.individual_files
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

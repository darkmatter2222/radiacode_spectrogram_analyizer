# Synthetic Gamma Spectrum Dataset

This document describes the comprehensive synthetic gamma spectrum dataset generated for machine learning training. The dataset contains 100,000 simulated gamma spectroscopy measurements with realistic isotope blending and proper labeling for AI model training.

## ⚠️ IMPORTANT SAFETY NOTICE

- **This dataset contains SIMULATED spectroscopy data only**
- **Real radioactive material identification requires proper licensing & training**
- **Always follow radiation safety protocols and regulations**
- **For educational and research purposes only**

## 🎯 Dataset Overview

### Generation Parameters
- **Total Spectra**: 100,000 synthetic measurements
- **Batch Size**: 1,000 spectra per batch file
- **Isotope Blending**: 1-5 isotopes per spectrum
- **Mixing Ratios**: Variable (including single isotope and multi-isotope blends)
- **Background Inclusion**: 70% probability of including background isotopes
- **Count Range**: 5,000-50,000 total counts per spectrum
- **Poisson Noise**: Applied for realistic counting statistics

### Isotope Database Used
- **Background Isotopes (4)**: K-40, U-238_series_Bi-214, Th-232_series_Tl-208, U-235_peak
- **Calibration Isotopes (3)**: Cs-137, Ba-133, Co-57
- **Industrial Isotopes (6)**: Cs-134, Co-60, Ir-192, Se-75, Yb-169, Am-241
- **Medical Isotopes (8)**: Tc-99m, I-131, I-123, F-18, Ga-67, In-111, Tl-201, Xe-133

## 📁 File Structure

```
O:/master_data_collection/isotope/
├── synthetic_batch_0000.json    # Batch 0: Spectra 0-999
├── synthetic_batch_0001.json    # Batch 1: Spectra 1000-1999
├── ...
├── synthetic_batch_0099.json    # Batch 99: Spectra 99000-99999
├── generation_summary.json      # Dataset statistics and metadata
└── generation.log              # Generation process log
```

## 📊 Data Structure

### Individual Spectrum Format

Each synthetic spectrum contains the following structure:

```json
{
  "spectrum_id": 0,
  "generation_timestamp": "2025-08-21T18:51:58.913798",
  
  "detector_config": {
    "type": "NaI(Tl)",
    "channels": 1024,
    "energy_range_keV": [0.0, 3000.0],
    "energy_resolution_model": "FWHM(E)=a*sqrt(E), ~7% at 662 keV",
    "a_res": 1.8010552462376053
  },
  
  "energy_axis_keV": [0.0, 2.933, 5.865, ...], // 1024 energy bins
  "spectrum_counts": [4123, 3890, 3654, ...],   // 1024 count values
  
  "synthesis_info": {
    "constituent_isotopes": ["Th-232_series_Tl-208", "Tc-99m", "F-18"],
    "mixing_ratios": [0.485, 0.187, 0.328],
    "target_total_counts": 37220.0,
    "actual_total_counts": 37500.0,
    "include_background": true,
    "num_isotopes": 3
  },
  
  "ml_labels": {
    "isotope_presence": {
      "K-40": false,
      "Tc-99m": true,
      "F-18": true,
      // ... all 21 isotopes
    },
    "isotope_concentrations": {
      "K-40": 0.0,
      "Tc-99m": 0.187,
      "F-18": 0.328,
      // ... all 21 isotopes
    },
    "category_presence": {
      "has_background": true,
      "has_calibration": false,
      "has_industrial": false,
      "has_medical": true
    },
    "complexity_metrics": {
      "num_isotopes": 3,
      "max_concentration": 0.485,
      "min_concentration": 0.187,
      "concentration_entropy": 1.045,
      "has_multiple_categories": true
    }
  }
}
```

## 🤖 Machine Learning Applications

### Recommended Model Types

1. **Multi-Label Classification**
   - **Task**: Predict presence/absence of each isotope
   - **Labels**: `ml_labels.isotope_presence`
   - **Architecture**: Binary classification for each of 21 isotopes

2. **Regression for Concentration Estimation**
   - **Task**: Predict mixing ratios of detected isotopes
   - **Labels**: `ml_labels.isotope_concentrations`
   - **Architecture**: Multi-output regression (21 outputs, 0-1 range)

3. **Multi-Task Learning**
   - **Task**: Combined classification and regression
   - **Labels**: Both presence and concentration
   - **Architecture**: Shared encoder with separate heads

4. **Category Classification**
   - **Task**: Predict isotope categories present
   - **Labels**: `ml_labels.category_presence`
   - **Architecture**: Multi-label classification (4 categories)

### Input Features

#### Spectral Data
- **Primary Input**: `spectrum_counts` (1024-dimensional vector)
- **Energy Axis**: `energy_axis_keV` (for energy-based features)
- **Preprocessing**: Normalization, background subtraction, peak detection

#### Derived Features
- **Peak Positions**: Energy locations of prominent peaks
- **Peak Areas**: Integrated counts under peaks
- **Spectral Statistics**: Total counts, max/min ratios, entropy
- **Energy Moments**: Centroid, width, skewness of energy distribution

## 📈 Dataset Statistics

### Complexity Distribution
- **Single Isotope**: ~20% of spectra
- **Two Isotopes**: ~20% of spectra
- **Three Isotopes**: ~30% of spectra
- **Four Isotopes**: ~20% of spectra
- **Five Isotopes**: ~10% of spectra

### Category Combinations
- **Background Only**: ~21% of spectra
- **Medical + Background**: ~25% of spectra
- **Industrial + Background**: ~18% of spectra
- **Mixed Categories**: ~36% of spectra

### Isotope Usage Frequency
Each isotope appears in roughly 15,000-25,000 spectra, ensuring balanced representation for training.

## 💡 Training Strategies

### Data Splitting
- **Training**: 70,000 spectra (batches 0-69)
- **Validation**: 15,000 spectra (batches 70-84)
- **Testing**: 15,000 spectra (batches 85-99)

### Data Augmentation
- **Count Scaling**: Simulate different measurement times
- **Poisson Resampling**: Additional noise variations
- **Energy Calibration Shifts**: Simulate detector drift
- **Background Subtraction**: Various background estimation methods

### Loss Functions
- **Binary Cross-Entropy**: For isotope presence prediction
- **Mean Squared Error**: For concentration regression
- **Focal Loss**: For handling class imbalance
- **Multi-Task Loss**: Weighted combination of classification and regression

## 🔬 Physical Realism

### Detector Simulation
- **NaI(Tl) Response**: Realistic energy resolution modeling
- **Peak Shapes**: Gaussian broadening with energy-dependent width
- **Background**: Compton continuum and detector response effects

### Counting Statistics
- **Poisson Noise**: Applied to all count values
- **Realistic Count Rates**: Based on typical measurement scenarios
- **Collection Times**: Variable from 120s to 1800s

### Isotope Physics
- **Gamma Energies**: Accurate nuclear data
- **Branching Ratios**: Realistic relative intensities
- **Half-Lives**: Physical decay constants
- **Category Classification**: Based on real-world applications

## 📚 Usage Examples

### Loading Data (Python)

```python
import json
import numpy as np

# Load a batch of spectra
with open('synthetic_batch_0000.json', 'r') as f:
    batch_data = json.load(f)

spectra = []
labels = []

for spectrum in batch_data['spectra']:
    # Extract spectral data
    counts = np.array(spectrum['spectrum_counts'])
    energy_axis = np.array(spectrum['energy_axis_keV'])
    
    # Extract labels
    isotope_presence = spectrum['ml_labels']['isotope_presence']
    concentrations = spectrum['ml_labels']['isotope_concentrations']
    
    spectra.append(counts)
    labels.append(isotope_presence)  # or concentrations

# Convert to numpy arrays
X = np.array(spectra)  # Shape: (1000, 1024)
y = np.array(labels)   # Shape depends on label type
```

### Data Preprocessing

```python
# Normalize spectra
X_normalized = X / X.sum(axis=1, keepdims=True)

# Log transformation for better dynamic range
X_log = np.log(X + 1)

# Background subtraction (estimate from low-energy region)
background = X[:, :50].mean(axis=1, keepdims=True)
X_background_subtracted = X - background
```

## 🎯 Validation and Testing

### Ground Truth Verification
- **Known Blending**: All mixing ratios are exactly known
- **Synthetic Validation**: Test on known single-isotope spectra
- **Cross-Validation**: Ensure consistent performance across batches

### Performance Metrics
- **Classification**: Precision, Recall, F1-score per isotope
- **Regression**: MAE, RMSE for concentration prediction
- **Multi-Label**: Hamming loss, subset accuracy
- **Physical Constraints**: Concentration sum validation

### Real-World Validation
- **Calibration Sources**: Test on known reference spectra
- **Mixed Standards**: Validate on laboratory-prepared mixtures
- **Blind Testing**: Unknown sample identification challenges

## 🔐 Responsible Use Guidelines

### Educational Applications
- **Training Material**: Spectrum analysis education
- **Algorithm Development**: Method validation and comparison
- **Research Tools**: Academic and industrial research

### Safety Considerations
- **Simulation Only**: Never use for real radioactive material identification without proper training
- **Professional Oversight**: Consult radiation safety experts for real applications
- **Regulatory Compliance**: Follow all local and international regulations

### Attribution and Ethics
- **Data Citation**: Acknowledge synthetic nature in publications
- **Open Science**: Share methods and validation results
- **Collaboration**: Work with domain experts for validation

## 📞 Support and Documentation

For questions about this dataset:
1. **Technical Issues**: Check generation logs and validation scripts
2. **Scientific Questions**: Consult nuclear physics and detector literature
3. **Safety Concerns**: Contact radiation safety professionals
4. **Usage Examples**: Refer to provided demonstration scripts

---

**Remember**: This synthetic dataset is a powerful educational and research tool, but real radioactive material identification requires proper training, licensing, and safety protocols. Always consult with radiation safety professionals when working with actual radioactive sources.

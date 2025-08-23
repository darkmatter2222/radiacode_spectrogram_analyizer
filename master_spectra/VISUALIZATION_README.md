# Gamma Spectrum Visualization Tool

A comprehensive and responsible tool for visualizing simulated gamma spectroscopy data from radioactive isotopes.

## ⚠️ IMPORTANT SAFETY NOTICE

- **This tool displays SIMULATED detector responses only**
- **Real radioactive materials require proper licensing & training**
- **Always follow radiation safety protocols and regulations**
- **For educational and research purposes only**

## 🎯 Features

### Core Visualization Capabilities
- **Single Isotope Analysis**: Detailed spectrum plots with gamma ray peak annotations
- **Category Overviews**: Grid layouts showing all isotopes in a category
- **Multi-Isotope Comparisons**: Side-by-side spectrum comparisons
- **Detector Analysis**: Energy resolution and performance characteristics
- **Interactive & Command-Line**: Both interactive mode and CLI options

### Scientific Accuracy
- Proper energy calibration (0-3000 keV range)
- NaI(Tl) detector response modeling
- Realistic peak broadening and backgrounds
- Nuclear data integration (half-lives, gamma energies)
- Both linear and logarithmic scale visualization

### Responsible Presentation
- Prominent safety warnings on all outputs
- Educational context and scientific information
- Professional scientific plotting standards
- Clear isotope categorization by application

## 📁 Database Structure

The tool works with the organized isotope database in `master_spectra/`:

```
master_spectra/
├── background/        # Natural background sources (4 isotopes)
│   ├── K-40.json
│   ├── U-238_series_Bi-214.json
│   ├── Th-232_series_Tl-208.json
│   └── U-235_peak.json
├── calibration/       # Standard calibration sources (3 isotopes)
│   ├── Cs-137.json
│   ├── Ba-133.json
│   └── Co-57.json
├── industrial/        # Industrial applications (6 isotopes)
│   ├── Co-60.json
│   ├── Cs-134.json
│   ├── Ir-192.json
│   ├── Se-75.json
│   ├── Yb-169.json
│   └── Am-241.json
└── medical/          # Medical isotopes (8 isotopes)
    ├── Tc-99m.json
    ├── I-131.json
    ├── I-123.json
    ├── F-18.json
    ├── Ga-67.json
    ├── In-111.json
    ├── Tl-201.json
    └── Xe-133.json
```

## 🚀 Usage

### Command Line Interface

```bash
# List all available isotopes
python visualize_spectra.py --list

# Plot a single isotope spectrum
python visualize_spectra.py --single medical Tc-99m

# Create category overview
python visualize_spectra.py --category medical --save

# Compare multiple isotopes
python visualize_spectra.py --compare calibration Cs-137 calibration Co-57

# Analyze detector resolution
python visualize_spectra.py --resolution --save

# Save plots to files (add --save to any command)
python visualize_spectra.py --single calibration Cs-137 --save
```

### Interactive Mode

```bash
# Run without arguments for interactive menu
python visualize_spectra.py
```

### Demo Script

```bash
# Run comprehensive demonstration
python demo_visualization.py
```

## 📊 Visualization Types

### 1. Single Isotope Analysis
- Linear and logarithmic scale plots
- Gamma ray peak annotations with energies
- Nuclear properties (half-life, category)
- Detector configuration details
- Total count statistics

### 2. Category Overviews
- Grid layout of all isotopes in a category
- Half-life information for each isotope
- Standardized energy scale for comparison
- Professional subplot organization

### 3. Multi-Isotope Comparisons
- Absolute and normalized count comparisons
- Color-coded isotope identification
- Side-by-side spectrum analysis
- Quantitative comparison capabilities

### 4. Detector Resolution Analysis
- FWHM vs Energy relationship
- Resolution percentage across energy range
- NaI(Tl) detector characteristics
- Model parameter visualization

## 🔬 Scientific Information Included

### Nuclear Data
- **Half-lives**: Physical decay time constants
- **Gamma Energies**: Primary emission lines in keV
- **Relative Intensities**: Branching ratios for gamma rays
- **Categories**: Background, calibration, industrial, medical

### Detector Physics
- **Type**: NaI(Tl) scintillation detector
- **Energy Range**: 0-3000 keV with 1024 channels
- **Resolution Model**: FWHM(E) = a√E (~7% at 662 keV)
- **Response Function**: Realistic peak shapes and backgrounds

### Measurement Parameters
- **Collection Times**: Typical measurement durations
- **Count Statistics**: Simulated detector responses
- **Energy Calibration**: Precise keV mapping

## 📈 Output Options

### Plot Types
- **PNG Files**: High-resolution (300 DPI) scientific plots
- **Interactive Display**: Real-time matplotlib visualization
- **Multiple Formats**: Easy export to other formats

### File Organization
- Plots saved to `master_spectra/plots/`
- Systematic naming convention
- Category-based organization maintained

## 🎓 Educational Applications

### Radiation Safety Training
- Proper isotope identification techniques
- Understanding detector responses
- Recognition of common gamma signatures
- Safety protocol awareness

### Nuclear Physics Education
- Gamma spectroscopy principles
- Detector physics and resolution
- Nuclear decay processes
- Energy calibration methods

### Research Applications
- Isotope identification algorithms
- Machine learning training data
- Detector performance analysis
- Spectroscopic method development

## 📋 Requirements

```bash
pip install matplotlib seaborn numpy
```

## 🏃‍♂️ Quick Start

1. **List available isotopes**:
   ```bash
   python visualize_spectra.py --list
   ```

2. **Plot a medical isotope**:
   ```bash
   python visualize_spectra.py --single medical Tc-99m
   ```

3. **Compare calibration sources**:
   ```bash
   python visualize_spectra.py --compare calibration Cs-137 calibration Co-57
   ```

4. **Run the demo**:
   ```bash
   python demo_visualization.py
   ```

## 🔐 Responsible Use Guidelines

1. **Educational Context**: Always emphasize the educational and research nature
2. **Safety First**: Include radiation safety information in presentations
3. **Proper Attribution**: Credit simulation methods and data sources
4. **Regulatory Compliance**: Follow local radiation safety regulations
5. **Professional Standards**: Maintain scientific accuracy and responsibility

## 📝 File Formats

### Input Files (JSON)
- Complete isotope database with detector configuration
- Energy axis calibration data
- Nuclear property information
- Simulated spectrum counts

### Output Files (PNG)
- High-resolution scientific plots
- Professional formatting and annotation
- Consistent styling and color schemes
- Print-ready quality

## 🤝 Contributing

When contributing to this visualization tool:
- Maintain safety warnings and educational context
- Follow scientific plotting standards
- Include proper documentation
- Test with various isotope types
- Ensure responsible presentation

## 📚 Further Reading

- Gamma-ray spectroscopy fundamentals
- NaI(Tl) detector characteristics
- Radiation safety regulations
- Nuclear decay physics
- Spectroscopic analysis methods

---

**Remember**: This tool is designed for education and research. Real radioactive materials require proper training, licensing, and safety protocols. Always consult with radiation safety professionals when working with actual radioactive sources.

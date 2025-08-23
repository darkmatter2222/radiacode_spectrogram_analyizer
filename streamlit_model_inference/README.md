# RadiaCode Isotope Detector App

## Overview
Real-time gamma spectroscopy isotope detection using trained CNN models. Upload RadiaCode XML spectrum files and get instant analysis results for 21 different isotopes.

## Features

### 🔬 **Comprehensive Isotope Detection**
- **21 Isotope Models**: Covers background, calibration, industrial, and medical isotopes
- **Real-time Analysis**: Instant results with confidence levels
- **Binary Classification**: Clear "Detected" or "Not Detected" results

### 📊 **Advanced Visualization**
- **Interactive Spectrum Plot**: Energy-calibrated or channel-based display
- **Confidence Distribution**: Visual comparison of all isotope confidences
- **Color-coded Results**: Green for detected, red for not detected

### 🎯 **User-Friendly Interface**
- **Drag & Drop Upload**: Simple XML file upload
- **Metadata Display**: Sample info, measurement time, total counts
- **Organized Results**: Grouped by isotope category
- **Real-time Progress**: Loading indicators and status updates

## Supported Isotopes

### Background Isotopes
- K-40 (Potassium-40)
- U-238 series (Bi-214)
- U-235 peak
- Th-232 series (Tl-208)

### Calibration Sources
- Cs-137 (Cesium-137)
- Co-60 (Cobalt-60)
- Am-241 (Americium-241)

### Industrial Isotopes
- Ir-192, Co-57, Se-75, Yb-169, Ba-133, Cs-134

### Medical Isotopes
- Tc-99m, I-131, I-123, F-18, Tl-201, In-111, Ga-67, Xe-133

## Installation & Usage

### Prerequisites
- Python 3.8+
- Trained isotope models in `../models/` directory
- RadiaCode XML spectrum files

### Quick Start
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run streamlit_isotope_detector.py
   ```
   
   Or use the batch file:
   ```bash
   run_app.bat
   ```

3. **Open browser**: Navigate to `http://localhost:8501`

### Usage Steps
1. **Upload Spectrum**: Drag & drop RadiaCode XML file
2. **View Metadata**: Check measurement details and spectrum plot
3. **Analyze**: Click "Analyze Isotopes" button
4. **Review Results**: See detection results with confidence levels

## Technical Details

### Model Architecture
- **CNN Design**: 4 Conv1D layers + 3 FC layers
- **Input**: 1024-channel gamma spectrum
- **Output**: Binary classification (0-1 confidence)
- **Threshold**: 50% confidence for positive detection

### File Format Support
- **RadiaCode XML**: Native XML export format
- **1024 Channels**: Standard RadiaCode resolution
- **Energy Calibration**: Automatic energy axis conversion
- **Metadata Extraction**: Sample info, timing, calibration

### Performance
- **Real-time Processing**: < 1 second per spectrum
- **Memory Efficient**: Models loaded once at startup
- **Batch Processing**: Multiple isotopes analyzed simultaneously

## Model Information

Each isotope model was trained on:
- **100,000+ synthetic spectra**
- **Balanced datasets** (50% positive/negative)
- **Realistic noise modeling**
- **Energy resolution effects**

Typical accuracies:
- **K-40**: 97.9%
- **Cs-137**: 96.5%
- **Co-60**: 96.3%
- **Average**: 95%+

## Output Features

### Detection Results
- ✅ **Clear Status**: "DETECTED" or "NOT DETECTED"
- 📊 **Confidence Level**: Percentage confidence (0-100%)
- 🎨 **Color Coding**: Visual status indicators
- 📈 **Distribution Plot**: Confidence comparison chart

### Spectrum Analysis
- **Interactive Plot**: Zoom, pan, hover for details
- **Energy Calibration**: Automatic keV conversion
- **Count Statistics**: Total counts, peak values
- **Measurement Info**: Duration, start/end times

## Troubleshooting

### Common Issues
1. **Models not loading**: Ensure `../models/` directory contains trained models
2. **XML parsing error**: Verify file is valid RadiaCode XML export
3. **Performance issues**: Check available system memory

### System Requirements
- **RAM**: 4GB+ recommended (for model loading)
- **GPU**: Not required (CPU inference)
- **Storage**: 500MB+ for all models

## Future Enhancements
- Multi-file batch processing
- Export results to PDF/CSV
- Historical analysis tracking
- Custom threshold settings
- Additional file format support

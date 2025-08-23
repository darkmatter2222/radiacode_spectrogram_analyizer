#!/usr/bin/env python3
"""
Test XML parsing for RadiaCode files
"""

import xml.etree.ElementTree as ET
import numpy as np

def test_xml_parsing():
    try:
        xml_file = r"sample_real_spectra\RadiaCode_ Spectrum_1.xml"
        print(f"Testing XML parsing on: {xml_file}")
        
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print(f"Root element: {root.tag}")
        
        # Find the spectrum data
        spectrum_element = root.find('.//Spectrum')
        if spectrum_element is None:
            print("No spectrum element found")
            return
        
        print(f"Found spectrum element")
        
        # Extract spectrum counts
        data_points = spectrum_element.findall('DataPoint')
        spectrum_counts = [int(point.text) for point in data_points]
        
        print(f"Extracted {len(spectrum_counts)} data points")
        print(f"First 10 counts: {spectrum_counts[:10]}")
        print(f"Last 10 counts: {spectrum_counts[-10:]}")
        print(f"Total counts: {sum(spectrum_counts)}")
        print(f"Max count: {max(spectrum_counts)}")
        
        # Test energy calibration
        calibration = root.find('.//EnergyCalibration')
        if calibration is not None:
            coeffs = calibration.findall('.//Coefficient')
            if len(coeffs) >= 2:
                cal_coeffs = [float(c.text) for c in coeffs]
                print(f"Energy calibration coefficients: {cal_coeffs}")
        
        print("XML parsing test successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_xml_parsing()

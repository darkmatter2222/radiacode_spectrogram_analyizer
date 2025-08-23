"""
RadiaCode Isotope Detection App
Minimal, clean Streamlit UI with model loading, XML parsing, and peak annotations.
"""

import os
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn


# ---------- Page config ----------
st.set_page_config(
    page_title="RadiaCode Spectrum Analyzer",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Model ----------
class IsotopeCNN(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc_input_size = 256 * 64  # 1024 -> 64 after 4 pools
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x.squeeze()


# ---------- Data ----------
@st.cache_data
def load_isotope_database():
    try:
        with open('isotope_peak_energies_database.json', 'r') as f:
            data = json.load(f)
        if 'isotope_gamma_energies_database' in data:
            isotopes_data = data['isotope_gamma_energies_database'].get('isotopes', {})
            categorized = {"background": [], "calibration": [], "industrial": [], "medical": []}
            for isotope_name, isotope_info in isotopes_data.items():
                model_name = isotope_name.replace('-', '_')
                category = isotope_info.get('category', 'background')
                if category in categorized:
                    categorized[category].append(model_name)
            return categorized
        return data
    except Exception as e:
        print(f"DATABASE ERROR: {e}")
        return {
            "background": ["K_40", "U_238_series_Bi_214", "U_235_peak", "Th_232_series_Tl_208"],
            "calibration": ["Cs_137", "Co_60", "Am_241"],
            "industrial": ["Ir_192", "Co_57", "Se_75", "Yb_169", "Ba_133", "Cs_134"],
            "medical": ["Tc_99m", "I_131", "I_123", "F_18", "Tl_201", "In_111", "Ga_67", "Xe_133"],
        }


@st.cache_data
def load_isotope_peak_database():
    try:
        db_path = (
            "../isotope_peak_energies_database.json"
            if os.path.exists("../isotope_peak_energies_database.json")
            else "isotope_peak_energies_database.json"
        )
        if not os.path.exists(db_path):
            db_path = os.path.abspath("isotope_peak_energies_database.json")
        print(f"PEAKS: Loading peak database from {db_path}")
        with open(db_path, 'r') as f:
            peak_db = json.load(f)
        print(
            f"PEAKS: Loaded database with {len(peak_db['isotope_gamma_energies_database']['isotopes'])} isotopes"
        )
        return peak_db['isotope_gamma_energies_database']
    except Exception as e:
        print(f"PEAKS: Failed to load peak database: {e}")
        return None


def channel_to_energy(channel, max_energy_keV=3000, channels=1024):
    return (channel / channels) * max_energy_keV


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average for display smoothing; window>=1.
    Returns same-length array using 'same' convolution.
    """
    try:
        w = int(window)
    except Exception:
        w = 1
    if w <= 1:
        return values
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(values, kernel, mode='same')


@st.cache_resource
def load_all_models():
    models = {}
    model_info = {}

    current_dir = os.getcwd()
    models_dir = "models"
    print(f"MODEL LOADING: Current working directory: {current_dir}")

    isotope_db = load_isotope_database()
    all_isotopes = []
    for category in isotope_db.values():
        if isinstance(category, list):
            all_isotopes.extend(category)
    print(
        f"MODEL LOADING: Attempting to load {len(all_isotopes)} isotopes: {all_isotopes}"
    )

    if os.path.exists(models_dir):
        print(f"MODEL LOADING: Contents of models directory: {os.listdir(models_dir)}")

    with st.spinner("Loading trained models..."):
        progress_bar = st.progress(0)
        for i, isotope in enumerate(all_isotopes):
            model_path = os.path.join(models_dir, isotope, f"{isotope}_model.pth")
            print(f"MODEL LOADING: Looking for {isotope} at: {model_path}")
            if os.path.exists(model_path):
                try:
                    model = IsotopeCNN()
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                    models[isotope] = model
                    print(f"MODEL LOADING: Successfully loaded {isotope}")
                except Exception as e:
                    print(
                        f"MODEL LOADING ERROR: Could not load model for {isotope}: {e}"
                    )
                    st.warning(f"Could not load model for {isotope}: {e}")
            else:
                print(
                    f"MODEL LOADING: Model file not found for {isotope} at {model_path}"
                )
            progress_bar.progress((i + 1) / len(all_isotopes))

    print(
        f"MODEL LOADING: Successfully loaded {len(models)} models: {list(models.keys())}"
    )
    return models, model_info


def parse_radiacode_xml(uploaded_file):
    try:
        print(f"XML PARSING: Starting to parse {uploaded_file.name}")
        if hasattr(uploaded_file, 'getvalue'):
            file_content = uploaded_file.getvalue()
            print(f"XML PARSING: File size: {len(file_content)} bytes")
            try:
                content_str = file_content.decode('utf-8')
                root = ET.fromstring(content_str)
            except UnicodeDecodeError:
                for encoding in ['utf-16', 'latin-1', 'cp1252']:
                    try:
                        content_str = file_content.decode(encoding)
                        root = ET.fromstring(content_str)
                        print(f"XML PARSING: Successfully decoded with {encoding}")
                        break
                    except Exception:
                        continue
                else:
                    raise ValueError(
                        "Could not decode file with any known encoding"
                    )
        else:
            uploaded_file.seek(0)
            root = ET.parse(uploaded_file).getroot()

        print(f"XML PARSING: Root element: {root.tag}")
        spectrum_element = root.find('.//Spectrum')
        if spectrum_element is None:
            for ns in ['', '{http://www.example.com}']:
                spectrum_element = root.find(f'.//{ns}Spectrum')
                if spectrum_element is not None:
                    break
        if spectrum_element is None:
            raise ValueError("No spectrum data found in XML file")

        data_points = spectrum_element.findall('DataPoint') or spectrum_element.findall(
            './/DataPoint'
        )
        spectrum_counts = [int(point.text) for point in data_points]
        print(f"XML PARSING: Extracted {len(spectrum_counts)} data points")
        if len(spectrum_counts) != 1024:
            raise ValueError(f"Expected 1024 channels, got {len(spectrum_counts)}")

        metadata = {}
        sample_name = root.find('.//SampleInfo/Name')
        if sample_name is not None and sample_name.text:
            metadata['sample_name'] = sample_name.text.strip()
        meas_time = root.find('.//MeasurementTime')
        if meas_time is not None and meas_time.text:
            metadata['measurement_time'] = float(meas_time.text)
        start_time = root.find('.//StartTime')
        if start_time is not None and start_time.text:
            metadata['start_time'] = start_time.text.strip()
        calibration = root.find('.//EnergyCalibration')
        if calibration is not None:
            coeffs = calibration.findall('.//Coefficient')
            if len(coeffs) >= 2:
                metadata['energy_calibration'] = [float(c.text) for c in coeffs if c.text]
                print(
                    f"XML PARSING: Energy calibration coefficients: {metadata['energy_calibration']}"
                )

        spectrum_array = np.array(spectrum_counts)
        spectrum_array[-1] = 0
        print(
            f"XML PARSING: Final spectrum - shape: {spectrum_array.shape}, sum: {spectrum_array.sum()}, max: {spectrum_array.max()}"
        )
        return spectrum_array, metadata
    except Exception as e:
        print(f"XML PARSING ERROR: {str(e)}")
        raise


def run_isotope_detection(spectrum_counts, models):
    results = {}
    print(f"INFERENCE: Starting analysis with {len(models)} models")
    current_total = spectrum_counts.sum()
    training_range = (5000, 50000)
    if current_total > training_range[1] * 1.5:
        target_total = (training_range[0] + training_range[1]) / 2
        scaling_factor = target_total / current_total
        spectrum_counts = spectrum_counts * scaling_factor
        print(
            f"INFERENCE: Applied scaling - factor: {scaling_factor:.4f}, new total: {spectrum_counts.sum():.0f}"
        )

    spectrum_tensor = torch.FloatTensor(spectrum_counts.astype(np.float32)).unsqueeze(0)
    print(f"INFERENCE: Final tensor shape: {spectrum_tensor.shape}")

    with torch.no_grad():
        for isotope_name, model in models.items():
            output = model(spectrum_tensor)
            print(f"INFERENCE: Model {isotope_name} output: {output}")
            confidence = (
                float(output.squeeze().item()) if hasattr(output, 'item') else float(output)
            )
            confidence = max(0.0, min(1.0, confidence))
            absence_confidence = 1.0 - confidence
            detected = confidence > 0.5
            status = (
                "Strong Detection"
                if confidence > 0.75
                else "Moderate Detection"
                if confidence > 0.5
                else "Weak/Uncertain"
                if confidence > 0.25
                else "Strong Absence"
            )
            results[isotope_name] = {
                'detected': detected,
                'confidence': confidence,
                'confidence_percent': confidence * 100,
                'absence_confidence': absence_confidence,
                'absence_confidence_percent': absence_confidence * 100,
                'status': status,
            }
    print(
        f"INFERENCE: Completed analysis. Detected: {sum(1 for r in results.values() if r['detected'])}/{len(results)}"
    )
    return results


def create_spectrum_plot(spectrum_counts, metadata, *, display_counts: np.ndarray | None = None, log_y: bool = False):
    channels = np.arange(len(spectrum_counts))
    if 'energy_calibration' in metadata and len(metadata['energy_calibration']) >= 2:
        coeffs = metadata['energy_calibration']
        energies = (
            coeffs[0] + coeffs[1] * channels
            if len(coeffs) == 2
            else coeffs[0] + coeffs[1] * channels + coeffs[2] * channels**2
        )
        x_axis = energies
        x_label = "Energy (keV)"
    else:
        x_axis = channels
        x_label = "Channel"
    # choose y to display
    y_vals = display_counts if display_counts is not None else spectrum_counts
    if log_y:
        y_plot = (np.asarray(y_vals, dtype=float) + 1.0).tolist()
    else:
        y_plot = y_vals.tolist() if isinstance(y_vals, np.ndarray) else y_vals
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_plot,
            mode='lines',
            name='Spectrum',
            line=dict(color='#4fc3f7', width=1.5),
        )
    )
    fig.update_layout(
        template='plotly_dark',
        title="Gamma Ray Spectrum",
        xaxis_title=x_label,
        yaxis_title="Counts",
        height=400,
        showlegend=False,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    if log_y:
        fig.update_yaxes(type='log')
    return fig


def create_spectrum_plot_with_peaks(
    spectrum_counts,
    detection_results,
    peak_database,
    energy_calibration_coeffs=None,
    *,
    display_counts: np.ndarray | None = None,
    show_annotations: bool = True,
    threshold: float = 0.75,
    peaks_per_isotope: int = 1,
    log_y: bool = False,
):
    channels = np.arange(len(spectrum_counts))
    if energy_calibration_coeffs and len(energy_calibration_coeffs) >= 2:
        coeffs = energy_calibration_coeffs
        energies = (
            coeffs[0] + coeffs[1] * channels
            if len(coeffs) == 2
            else coeffs[0] + coeffs[1] * channels + coeffs[2] * channels**2
        )
        x_label = "Energy (keV)"
        max_energy = energies[-1]
    else:
        energies = channel_to_energy(channels, 3000, len(spectrum_counts))
        x_label = "Energy (keV) [estimated]"
        max_energy = 3000
    # choose y to display
    y_vals = display_counts if display_counts is not None else spectrum_counts
    if log_y:
        y_plot = (np.asarray(y_vals, dtype=float) + 1.0).tolist()
    else:
        y_plot = y_vals.tolist() if isinstance(y_vals, np.ndarray) else y_vals
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=energies,
            y=y_plot,
            mode='lines',
            name='Spectrum',
            line=dict(color='#4fc3f7', width=1.5),
        )
    )
    if peak_database and detection_results and show_annotations:
        # Gather strong detections and their main peak energies
        labels: list[dict] = []
        for isotope_name, result in detection_results.items():
            if result.get('detected') and result.get('confidence', 0) >= float(threshold):
                peak_isotope_name = isotope_name.replace('_', '-')
                iso = peak_database['isotopes'].get(peak_isotope_name)
                if not iso:
                    continue
                primary_peaks = [p for p in iso.get('primary_peaks', []) if p.get('significance') == 'primary']
                if not primary_peaks:
                    continue
                # sort by intensity desc and take top N
                sorted_peaks = sorted(primary_peaks, key=lambda x: x.get('intensity_percent', 0), reverse=True)
                for j, peak in enumerate(sorted_peaks[: max(1, int(peaks_per_isotope)) ]):
                    energy = peak.get('energy_keV')
                    if energy and energy <= max_energy:
                        labels.append({'isotope': peak_isotope_name, 'energy': float(energy), 'rank': j})

        # Sort by energy to lay out labels left-to-right
        labels.sort(key=lambda x: x['energy'])
        y_source = np.asarray(y_plot, dtype=float)
        y_max = float(np.max(y_source)) if y_source.size else 1.0
        # Stagger label rows to reduce overlap
        levels = [0.92, 0.84, 0.76]

        for idx, item in enumerate(labels):
            energy = item['energy']
            row = levels[idx % len(levels)]
            y_pos = y_max * row
            # Vertical line
            fig.add_vline(
                x=energy,
                line=dict(color='rgba(255,255,255,0.6)', width=2),
                opacity=0.9,
            )
            # Clean label box
            label_text = (
                f"<b>{item['isotope']}</b><br>{int(round(energy))} keV"
                if item.get('rank', 0) == 0 else f"{int(round(energy))} keV"
            )
            fig.add_annotation(
                x=energy,
                y=y_pos,
                xref='x',
                yref='y',
                showarrow=False,
                xanchor='center',
                yanchor='bottom',
                align='center',
                text=label_text,
                font=dict(size=11 if item.get('rank', 0) == 0 else 10, color='#e0e0e0'),
                bgcolor='rgba(0,0,0,0.6)' if item.get('rank', 0) == 0 else 'rgba(0,0,0,0.4)',
                bordercolor='#4fc3f7' if item.get('rank', 0) == 0 else 'rgba(79,195,247,0.6)',
                borderwidth=1,
                borderpad=4,
            )
    fig.update_layout(
        template='plotly_dark',
        title="Gamma Ray Spectrum",
        xaxis_title=x_label,
        yaxis_title="Counts",
        height=500,
        showlegend=False,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=30, t=50, b=50),
        dragmode='zoom',
        font=dict(color='#e0e0e0'),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    if log_y:
        fig.update_yaxes(type='log')
    return fig


# ---------- App ----------
def main():
    st.title("RadiaCode Spectrum Analyzer")
    st.caption("AI-powered isotope detection for RadiaCode spectra")

    models, _ = load_all_models()
    peak_db = load_isotope_peak_database()

    with st.expander(f"Model status ({len(models)}/21 loaded)", expanded=False):
        if models:
            st.success(f"Loaded {len(models)} models")
            iso_db = load_isotope_database()
            for category, isotopes in iso_db.items():
                loaded = [i for i in isotopes if i in models]
                if loaded:
                    st.write(
                        f"{category.title()}: {', '.join(i.replace('_','-') for i in loaded)}"
                    )
        else:
            st.error("No models loaded. Check models directory.")
        if peak_db:
            st.info(f"Peak DB loaded with {len(peak_db['isotopes'])} isotopes")

    st.divider()

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Upload spectrum file")
        uploaded_file = st.file_uploader(
            "Upload RadiaCode XML spectrum file", type=['xml']
        )
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            try:
                spectrum_counts, metadata = parse_radiacode_xml(uploaded_file)
                st.subheader("Spectrum info")
                if 'sample_name' in metadata:
                    st.write(f"Sample: {metadata['sample_name']}")
                if 'measurement_time' in metadata:
                    d = metadata['measurement_time']
                    st.write(
                        f"Duration: {int(d//3600):02d}:{int((d%3600)//60):02d}:{int(d%60):02d}"
                    )
                st.write(f"Total counts: {int(np.sum(spectrum_counts)):,}")
                st.write(f"Peak count: {int(np.max(spectrum_counts)):,}")
                st.write(f"Channels: {len(spectrum_counts)}")
                if st.button("Analyze isotopes", type="primary", use_container_width=True):
                    if not models:
                        st.error("No models loaded.")
                    else:
                        with st.spinner("Running isotope detection..."):
                            results = run_isotope_detection(spectrum_counts, models)
                        st.session_state['detection_results'] = results
                        st.session_state['spectrum_data'] = (
                            spectrum_counts,
                            metadata,
                        )
                        st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Error parsing spectrum file: {e}")

    with col2:
        st.subheader("Spectrum")
        # Chart controls
        with st.expander("Chart controls", expanded=False):
            colc1, colc2, colc3 = st.columns([1,1,1])
            with colc1:
                show_annotations = st.checkbox("Show annotations", True)
                log_y = st.checkbox("Log Y-axis (counts+1)", False)
            with colc2:
                threshold_pct = st.slider("Annotation threshold (%)", 50, 99, 75, step=1)
                peaks_per_iso = st.number_input("Peaks per isotope", min_value=1, max_value=3, value=1, step=1)
            with colc3:
                smooth_on = st.checkbox("Smooth display", False)
                window = st.slider("Smooth window", 1, 21, 5, step=2)
        # Prepare display spectrum (only affects visualization)
        display_counts = None
        if 'uploaded_file' in locals() and uploaded_file is not None:
            # no-op, defined below during plotting
            pass
        if uploaded_file is not None:
            try:
                spectrum_counts, metadata = parse_radiacode_xml(uploaded_file)
                display_counts = moving_average(spectrum_counts, window) if smooth_on else spectrum_counts
                if 'detection_results' in st.session_state and peak_db:
                    results = st.session_state['detection_results']
                    energy_cal = metadata.get('energy_calibration')
                    fig = create_spectrum_plot_with_peaks(
                        spectrum_counts,
                        results,
                        peak_db,
                        energy_cal,
                        display_counts=display_counts,
                        show_annotations=show_annotations,
                        threshold=threshold_pct/100.0,
                        peaks_per_isotope=int(peaks_per_iso),
                        log_y=log_y,
                    )
                else:
                    fig = create_spectrum_plot(spectrum_counts, metadata, display_counts=display_counts, log_y=log_y)
                st.plotly_chart(fig, use_container_width=True, theme=None)
            except Exception as e:
                st.error(f"Could not display spectrum plot: {e}")
        else:
            st.info("Upload a spectrum file to see the visualization")

    if 'detection_results' in st.session_state:
        st.subheader("Results")
        results = st.session_state['detection_results']
        total_isotopes = len(results)
        detected_isotopes = sum(1 for r in results.values() if r.get('detected'))
        strong_detections = sum(
            1
            for r in results.values()
            if r.get('detected') and r.get('confidence_percent', 0) > 75
        )
        rate = (detected_isotopes / total_isotopes * 100) if total_isotopes else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Isotopes analyzed", total_isotopes)
        c2.metric("Detected", f"{detected_isotopes} ({strong_detections} strong)")
        c3.metric("Detection rate", f"{rate:.1f}%")

        strong_rows = [
            {
                'Isotope': iso.replace('_', '-'),
                'Confidence %': round(res.get('confidence_percent', 0), 1),
            }
            for iso, res in results.items()
            if res.get('detected') and res.get('confidence_percent', 0) > 75 and 'error' not in res
        ]
        if strong_rows:
            st.markdown("### Strong detections (>75%)")
            st.dataframe(
                pd.DataFrame(sorted(strong_rows, key=lambda r: -r['Confidence %'])),
                use_container_width=True,
            )
        with st.expander("All results", expanded=False):
            rows = []
            for iso, res in results.items():
                print(iso, res)
                rows.append(
                    {
                        'Isotope': iso.replace('_', '-'),
                        'Detected': 'Yes' if res.get('detected') else 'No',
                        'Confidence %': round(res.get('confidence_percent', 0), 1),
                        'Absence %': round(res.get('absence_confidence_percent', 0), 1),
                        'Status': res.get('status', 'Unknown'),
                    }
                )
            df = pd.DataFrame(rows).sort_values(
                by=['Detected', 'Confidence %'], ascending=[False, False]
            )
            st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()

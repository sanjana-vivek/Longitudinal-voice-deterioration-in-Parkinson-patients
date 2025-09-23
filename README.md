# Longitudinal voice deterioration in Parkinson's patients

A comprehensive Python pipeline for extracting and analyzing voice biomarkers from audio recordings to detect vocal deterioration patterns over time. This tool is designed for longitudinal voice analysis in clinical and research settings.

## Overview

> "Voice analysis represents a non-invasive, cost-effective approach to monitoring neurological health and treatment outcomes in clinical populations."

This pipeline processes audio recordings to identify subtle changes in voice characteristics that may indicate health deterioration or treatment progress. By combining traditional acoustic analysis with modern deep learning approaches, it provides a robust framework for voice-based health monitoring across 277+ distinct voice features.

## Key Features

| Feature Category | Description | Clinical Impact |
|------------------|-------------|-----------------|
| **Multi-Domain Analysis** | 6 distinct feature extraction domains | Comprehensive voice characterization |
| **Longitudinal Tracking** | Session-by-session trend analysis with slope calculation | Disease progression monitoring |
| **Statistical Process Control** | CUSUM-based change point detection | Early deterioration warning system |
| **Automated Reporting** | Visual summaries with baseline normalization | Clinical decision support |

## Feature Extraction Architecture

| Feature Category | Description | Clinical Relevance |
|------------------|-------------|-------------------|
| **Multi-Domain Analysis** | Extracts 277+ voice features across 6 domains | Comprehensive voice characterization |
| **Longitudinal Tracking** | Session-by-session trend analysis | Disease progression monitoring |
| **Baseline Normalization** | Patient-specific reference establishment | Personalized health baselines |
| **Real-time Detection** | CUSUM statistical process control | Early deterioration warning system |
| **Automated Reporting** | Visual summaries and statistical insights | Clinical decision support |

## Feature Extraction Domains

### Acoustic Features
| Feature Type | Parameters | Clinical Application |
|--------------|------------|---------------------|
| **Sound Pressure Level (SPL)** | Mean, Peak, Standard Deviation | Voice intensity and control assessment |
| **Spectral Statistics** | Centroid, Bandwidth, Rolloff | Frequency content analysis |
| **Temporal Dynamics** | Frame-level variations | Voice stability measurement |

### Classical Voice Metrics
| Feature | Description | Pathology Indicator |
|---------|-------------|-------------------|
| **Jitter** | Fundamental frequency variation | Vocal fold irregularity |
| **Shimmer** | Amplitude perturbation | Voice quality degradation |
| **Harmonics-to-Noise Ratio** | Signal clarity measure | Breathiness and roughness |

### Advanced Analysis
| Domain | Features | Research Applications |
|--------|----------|---------------------|
| **Nonlinear Dynamics** | RPDE, DFA, Correlation Dimension | Neurological voice changes |
| **Wavelet Transform** | Multi-scale energy decomposition | Time-frequency analysis |
| **Scattering Transform** | Invariant representations | Robust feature extraction |
| **Deep Embeddings** | wav2vec2 statistics | Modern representation learning |


## Technical Specifications

### System Requirements
| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Python Version** | 3.7+ | Core runtime |
| **Memory** | 4GB+ RAM recommended | Large feature matrices |
| **Storage** | Variable (depends on dataset) | Audio file storage |
| **Processing** | Multi-core CPU beneficial | Parallel feature extraction |

### Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | Latest | Numerical computations |
| `pandas` | Latest | Data manipulation |
| `scikit-learn` | Latest | Machine learning algorithms |
| `matplotlib` | Latest | Data visualization |
| `librosa` | Latest | Audio signal processing |
| `transformers` | Latest (optional) | Deep learning embeddings |


## Core Functions

### Feature Extraction Pipeline

```python
def extract_features_for_file(path):
    """
    Comprehensive feature extraction from single audio file
    
    Returns:
        dict: 277+ features across 6 domains plus metadata
    """
    # Load and preprocess audio
    y, sr = load_resample(path)
    y_trim = vad_trim(y, sr)
    
    # Extract multi-domain features
    features = {}
    features.update(session_spl_stats(y_trim, sr))
    features.update(classical_features(path))
    features.update(nonlinear_features(path))
    # ... additional domains
    
    return features
```

**Pro Tip**: Ensure consistent recording conditions across sessions for optimal feature reliability. Variations in microphone distance or background noise can introduce artifacts.

### Patient Analysis Workflow

```python
def analyze_patient_fixed(patient_group, patient_subfolder=None):
    """
    Complete longitudinal analysis pipeline
    
    Process:
    1. Batch feature extraction
    2. Baseline normalization
    3. Trend analysis
    4. Statistical monitoring
    5. Visualization and reporting
    """
```

The analysis follows this systematic approach:

1. **Data Collection**: Recursive discovery of WAV files in patient directories
2. **Baseline Establishment**: Uses first 2 sessions as patient-specific reference
3. **Feature Normalization**: Converts absolute values to relative changes from baseline
4. **Trend Analysis**: Linear regression across session indices for each feature
5. **Deterioration Scoring**: Weighted composite index using clinically relevant features
6. **Change Detection**: CUSUM algorithm for statistical process control

## Clinical Applications

### Primary Use Cases

**Neurological Monitoring**
- Parkinson's disease progression tracking
- ALS vocal symptom detection
- Multiple sclerosis speech changes

**Speech-Language Pathology**
- Treatment efficacy measurement
- Vocal rehabilitation progress
- Voice therapy outcomes

**Population Health Research**
- Aging voice studies
- Biomarker validation
- Preventive screening programs

### Research Integration

```python
# Example research workflow
patients = ['PD001', 'PD002', 'PD003']
results = {}

for patient_id in patients:
    df, slopes = analyze_patient_fixed(patient_id)
    results[patient_id] = {
        'deterioration_trend': df['deterioration_score'].tolist(),
        'top_features': dict(sorted(slopes.items(), 
                                  key=lambda x: abs(x[1]), 
                                  reverse=True)[:10])
    }
```

## Technical Implementation

### System Requirements

```bash
# Minimum requirements
Python >= 3.7
RAM >= 4GB (8GB+ recommended for large datasets)
Storage: Variable based on audio collection size

# Recommended for optimal performance
Multi-core CPU for parallel processing
SSD storage for faster I/O operations
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/voice-biomarker-pipeline.git
cd voice-biomarker-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install transformers for deep learning features
pip install transformers torch
```

### Configuration

Create a configuration file for your environment:

```python
# config.py
BASE_DIR = "/path/to/your/audio/data"
BASELINE_SESSIONS = 2
CUSUM_THRESHOLD = 2.0
DETERIORATION_WEIGHTS = {
    "spl_mean_db": -1.0,      # Lower volume indicates deterioration
    "jitter_local": 1.0,      # Higher jitter indicates deterioration
    "shimmer_local": 1.0,     # Higher shimmer indicates deterioration
    "rpde_approx": 1.0        # Higher RPDE may indicate deterioration
}
```

## Data Organization

### Directory Structure

```
audio_data/
├── patient_groups/
│   ├── PD_cohort/
│   │   ├── PD001/
│   │   │   ├── session_001.wav
│   │   │   ├── session_001.txt  # Optional transcript
│   │   │   ├── session_002.wav
│   │   │   └── ...
│   │   └── PD002/
│   └── control_group/
└── metadata/
    ├── patient_demographics.csv
    └── session_logs.json
```

### File Naming Conventions

**Best Practices:**
- Use consistent naming patterns: `session_XXX.wav`
- Include date stamps when possible: `20240315_session_001.wav`
- Maintain matching transcript files: `session_001.txt`
- Organize by patient ID for easy batch processing

### Audio Requirements

| Specification | Requirement | Rationale |
|---------------|-------------|-----------|
| **Sample Rate** | 16kHz minimum | Adequate for speech analysis |
| **Bit Depth** | 16-bit minimum | Sufficient dynamic range |
| **Duration** | 30+ seconds recommended | Statistical reliability |
| **Format** | WAV (uncompressed) | Lossless audio quality |

## Usage Examples

### Basic Patient Analysis

```python
# Analyze single patient group
df, slopes = analyze_patient_fixed("PD_patient_001")

# View top deteriorating features
top_features = sorted(slopes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
for feature, slope in top_features:
    print(f"{feature}: {slope:.5f} change per session")
```

### Batch Processing

```python
# Process multiple patients
import os
patients = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

results_summary = {}
for patient in patients:
    try:
        df, slopes = analyze_patient_fixed(patient)
        results_summary[patient] = {
            'sessions': len(df),
            'deterioration_score': df['deterioration_score'].iloc[-1],
            'cusum_alarms': df['cusum_flag'].value_counts().to_dict()
        }
    except Exception as e:
        print(f"Error processing {patient}: {e}")
```

## Performance Optimization Tips

### Processing Efficiency
- **Parallel Processing**: Use multiprocessing for large patient cohorts
- **Memory Management**: Process patients individually to avoid memory issues
- **Caching**: Store intermediate results to avoid recomputation

### Quality Assurance
```python
# Validate audio quality before processing
def validate_audio(file_path):
    y, sr = librosa.load(file_path)
    
    # Check duration
    if len(y) / sr < 10:  # Less than 10 seconds
        warnings.warn(f"Short audio file: {file_path}")
    
    # Check for silence
    if np.max(np.abs(y)) < 0.01:
        warnings.warn(f"Very quiet audio: {file_path}")
    
    return True
```

## Output Interpretation

### Deterioration Score Interpretation

> The deterioration score is a weighted composite metric where higher values indicate greater deviation from baseline voice characteristics.

- **Score < 0**: Generally indicates improvement or stable condition
- **Score 0-2**: Within normal variation range
- **Score > 2**: Potential deterioration warranting clinical attention
- **Score > 5**: Significant changes requiring immediate review

### CUSUM Alert System

The CUSUM (Cumulative Sum) control chart provides three status levels:
- **"ok"**: Voice parameters within expected variation
- **"pos_alarm"**: Positive drift detected (potential deterioration)
- **"neg_alarm"**: Negative drift detected (potential improvement)

## Contributing

We welcome contributions from the research and clinical communities. Please review our contribution guidelines:

**Development Setup:**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black voice_pipeline/
flake8 voice_pipeline/
```

**Areas for Contribution:**
- Additional feature extraction methods
- Alternative statistical monitoring approaches
- Clinical validation studies
- Performance optimizations

## References and Citations

Key publications supporting this methodology:

1. Tsanas, A., et al. (2012). "Accurate telemonitoring of Parkinson's disease symptom severity using nonlinear speech signal processing and statistical machine learning." *IEEE Transactions on Biomedical Engineering*
2. Rusz, J., et al. (2015). "Quantitative acoustic measurements for characterization of speech and voice disorders in early untreated Parkinson's disease." *Journal of the Acoustical Society of America*

**Citation Format:**
```bibtex
@software{voice_biomarker_pipeline,
  title={Voice Biomarker Analysis Pipeline},
  author={[Author Names]},
  year={2024},
  url={https://github.com/your-org/voice-biomarker-pipeline}
}
```

## Support and Documentation

- **Issues**: [GitHub Issues](https://github.com/sanjana-vivek/Longitudinal-voice-deterioration-in-Parkinson-s-patients-/issues)
- **Documentation**: Coming soon
- **Discussions**: [GitHub Discussions](https://github.com/sanjana-vivek/Longitudinal-voice-deterioration-in-Parkinson-s-patients-)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This software is intended for research purposes. Clinical applications require appropriate validation and should be supervised by qualified healthcare professionals.

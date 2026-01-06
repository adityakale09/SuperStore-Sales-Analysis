# Multimodal Sentiment Analysis Data Preparation

This script processes multimodal data from EEG, TIVA, and PSY files to create a complete dataset for sentiment analysis.

## Features

### Data Processing
- **Timestamp Alignment**: Properly aligns timestamps between PSY.csv, EEG.csv, and TIVA.csv files
- **Trial Window Extraction**: Extracts features during specific time windows (routineStart to routineEnd)
- **Multi-participant Support**: Processes data for all 38 participants automatically

### Feature Extraction

#### EEG Features
- **Frequency Band Analysis**: Extracts power in 5 frequency bands:
  - Delta (0.5-4 Hz)
  - Theta (4-8 Hz) 
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-100 Hz)
- **Channel Statistics**: Mean, std, max, min for each EEG channel
- **14 EEG Channels**: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

#### Facial Expression Features (TIVA)
- **8 Expression Types**: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
- **Statistical Measures**: Mean, std, max, min for each expression during trial windows

#### Sentiment Labels (PSY)
- **Sentiment Classification**: positive, negative, neutral
- **Valence and Arousal**: Continuous emotion dimensions
- **Trial Information**: Links features to specific experimental trials

## Usage

### Basic Usage
```bash
python multimodal_sentiment_preparation.py
```

### Expected Data Structure
The script expects data in the following structure:
```
data/
├── 1/
│   ├── 1_PSY.csv
│   ├── 1_EEG.csv
│   ├── 1_TIVA.csv
│   └── 1_GSR.csv
├── 2/
│   ├── 2_PSY.csv
│   ├── 2_EEG.csv
│   ├── 2_TIVA.csv
│   └── 2_GSR.csv
...
└── 38/
    ├── 38_PSY.csv
    ├── 38_EEG.csv
    ├── 38_TIVA.csv
    └── 38_GSR.csv
```

### File Formats

#### PSY.csv
- `timestamp`: Timestamp for each sample
- `routineStart`: Start time offset (seconds) for trial
- `routineEnd`: End time offset (seconds) for trial
- `valence`: Valence score (-1 to 1)
- `arousal`: Arousal score (-1 to 1)
- `sentiment_label`: Sentiment classification (positive/negative/neutral)

#### EEG.csv
- `timestamp`: Timestamp for each sample
- `[channel_name]`: EEG signal values for each channel (AF3, F7, F3, etc.)

#### TIVA.csv
- `timestamp`: Timestamp for each sample
- `[expression_name]`: Probability values for each facial expression

## Output

The script generates `multimodal_sentiment_dataset.csv` with 163 features:
- **70 EEG frequency band features** (14 channels × 5 bands)
- **56 EEG statistical features** (14 channels × 4 statistics)
- **32 facial expression features** (8 expressions × 4 statistics)
- **5 metadata features** (participant_id, trial_idx, sentiment_label, valence, arousal)

## Sample Data Generation

If no data directory exists, the script automatically creates sample data for demonstration purposes. This includes:
- Simulated EEG signals with realistic noise characteristics
- Random facial expression probabilities
- Synthetic sentiment labels and emotion ratings
- Proper timestamp alignment across all modalities

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

## Validation

The script includes built-in validation that checks:
- Total number of samples and features
- Distribution of sentiment labels
- Presence of all expected feature types
- Missing value counts
- Data integrity across participants

## Customization

### Modifying Parameters
```python
# Initialize with custom parameters
prep = MultimodalSentimentPreparation(
    data_dir="custom_data_path",
    output_file="custom_output.csv"
)

# Modify EEG frequency bands
prep.eeg_frequency_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}

# Modify facial expressions
prep.facial_expressions = [
    'Anger', 'Contempt', 'Disgust', 'Fear', 
    'Happiness', 'Neutral', 'Sadness', 'Surprise'
]
```

## Error Handling

The script includes comprehensive error handling for:
- Missing data files
- Corrupted or malformed data
- Timestamp alignment issues
- Feature extraction failures
- File I/O errors

## Performance

- Processes 38 participants with 4 trials each (152 total samples)
- Generates 163 features per sample
- Runs in approximately 30-60 seconds on standard hardware
- Memory usage scales with data size (typically < 1GB for full dataset)
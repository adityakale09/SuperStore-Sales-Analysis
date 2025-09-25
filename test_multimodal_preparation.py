#!/usr/bin/env python3
"""
Test script for multimodal sentiment analysis data preparation
"""

import pandas as pd
import numpy as np
import os
import sys
from multimodal_sentiment_preparation import MultimodalSentimentPreparation


def test_dataset_structure():
    """Test that the generated dataset has the correct structure"""
    print("Testing dataset structure...")
    
    # Load the generated dataset
    if not os.path.exists('multimodal_sentiment_dataset.csv'):
        print("❌ Dataset file not found!")
        return False
    
    df = pd.read_csv('multimodal_sentiment_dataset.csv')
    
    # Check basic structure
    expected_participants = 38
    expected_min_samples = 38 * 3  # At least 3 trials per participant
    expected_min_features = 150    # Minimum expected features
    
    if len(df) < expected_min_samples:
        print(f"❌ Insufficient samples: {len(df)} < {expected_min_samples}")
        return False
        
    if len(df.columns) < expected_min_features:
        print(f"❌ Insufficient features: {len(df.columns)} < {expected_min_features}")
        return False
        
    if df['participant_id'].nunique() != expected_participants:
        print(f"❌ Wrong number of participants: {df['participant_id'].nunique()} != {expected_participants}")
        return False
    
    print(f"✅ Dataset structure correct: {len(df)} samples, {len(df.columns)} features, {df['participant_id'].nunique()} participants")
    return True


def test_eeg_features():
    """Test that EEG frequency band features are present"""
    print("Testing EEG features...")
    
    df = pd.read_csv('multimodal_sentiment_dataset.csv')
    
    # Check for frequency band features
    frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    missing_features = []
    
    for channel in eeg_channels:
        for band in frequency_bands:
            feature_name = f"{channel}_{band}_power"
            if feature_name not in df.columns:
                missing_features.append(feature_name)
    
    if missing_features:
        print(f"❌ Missing EEG frequency band features: {missing_features[:5]}...")
        return False
    
    # Check that features have non-zero values
    eeg_band_features = [col for col in df.columns if any(band in col for band in frequency_bands)]
    if df[eeg_band_features].sum().sum() == 0:
        print("❌ All EEG frequency band features are zero")
        return False
    
    print(f"✅ EEG frequency band features present: {len(eeg_band_features)} features")
    return True


def test_facial_expression_features():
    """Test that facial expression features are present"""
    print("Testing facial expression features...")
    
    df = pd.read_csv('multimodal_sentiment_dataset.csv')
    
    expressions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    statistics = ['mean', 'std', 'max', 'min']
    
    missing_features = []
    
    for expression in expressions:
        for stat in statistics:
            feature_name = f"{expression}_{stat}"
            if feature_name not in df.columns:
                missing_features.append(feature_name)
    
    if missing_features:
        print(f"❌ Missing facial expression features: {missing_features[:5]}...")
        return False
    
    # Check that features have reasonable values (0-1 range)
    expression_features = [col for col in df.columns if any(expr in col for expr in expressions)]
    max_value = df[expression_features].max().max()
    min_value = df[expression_features].min().min()
    
    if max_value > 2 or min_value < -1:  # Allow some margin for statistics
        print(f"❌ Facial expression features out of expected range: {min_value} to {max_value}")
        return False
    
    print(f"✅ Facial expression features present: {len(expression_features)} features")
    return True


def test_sentiment_labels():
    """Test that sentiment labels are properly assigned"""
    print("Testing sentiment labels...")
    
    df = pd.read_csv('multimodal_sentiment_dataset.csv')
    
    required_columns = ['sentiment_label', 'valence', 'arousal', 'participant_id', 'trial_idx']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    # Check sentiment label values
    valid_sentiments = {'positive', 'negative', 'neutral'}
    unique_sentiments = set(df['sentiment_label'].unique())
    
    if not unique_sentiments.issubset(valid_sentiments):
        print(f"❌ Invalid sentiment labels: {unique_sentiments - valid_sentiments}")
        return False
    
    # Check valence and arousal ranges
    if df['valence'].abs().max() > 2 or df['arousal'].abs().max() > 2:
        print("❌ Valence or arousal values out of expected range")
        return False
    
    print(f"✅ Sentiment labels valid: {unique_sentiments}")
    return True


def test_no_missing_values():
    """Test that there are no missing values in the dataset"""
    print("Testing for missing values...")
    
    df = pd.read_csv('multimodal_sentiment_dataset.csv')
    
    missing_count = df.isnull().sum().sum()
    
    if missing_count > 0:
        print(f"❌ Found {missing_count} missing values")
        return False
    
    print("✅ No missing values found")
    return True


def test_feature_extraction_consistency():
    """Test that features are consistently extracted across participants"""
    print("Testing feature extraction consistency...")
    
    df = pd.read_csv('multimodal_sentiment_dataset.csv')
    
    # Check that all participants have similar number of trials
    trial_counts = df.groupby('participant_id')['trial_idx'].count()
    
    if trial_counts.std() > 2:  # Allow some variation but not too much
        print(f"❌ Inconsistent trial counts across participants: {trial_counts.min()} to {trial_counts.max()}")
        return False
    
    # Check that feature distributions are reasonable (not all identical)
    numeric_features = df.select_dtypes(include=[np.number]).columns
    std_values = df[numeric_features].std()
    
    zero_std_count = (std_values == 0).sum()
    if zero_std_count > len(numeric_features) * 0.1:  # More than 10% features with zero std
        print(f"❌ Too many features with zero standard deviation: {zero_std_count}")
        return False
    
    print(f"✅ Feature extraction consistent across participants")
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("MULTIMODAL SENTIMENT PREPARATION TESTS")
    print("=" * 50)
    
    tests = [
        test_dataset_structure,
        test_eeg_features,
        test_facial_expression_features,
        test_sentiment_labels,
        test_no_missing_values,
        test_feature_extraction_consistency
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        try:
            if test():
                passed_tests += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 50)
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The multimodal sentiment preparation script is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
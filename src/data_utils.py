import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_dir, window_size=100, overlap=0.5):
    """
    Load and preprocess the MotionSense dataset.

    Args:
        data_dir: Path to the A_DeviceMotion_data folder
        window_size: Size of the sliding window for segmentation
        overlap: Overlap between consecutive windows (0-1)

    Returns:
        X: Segmented sensor data
        y: Activity labels
        subject_ids: Subject IDs for each segment
        preprocessing_info: Dictionary with preprocessing information
    """
    from tqdm import tqdm
    import time

    start_time = time.time()

    # Activity mapping
    activity_map = {
        'dws': 0, 'ups': 1, 'sit': 2, 'std': 3, 'wlk': 4, 'jog': 5
    }
    activity_names = {
        0: 'Downstairs', 1: 'Upstairs', 2: 'Sitting',
        3: 'Standing', 4: 'Walking', 5: 'Jogging'
    }

    print("="*80)
    print("DATA PREPROCESSING DETAILS")
    print("="*80)
    print(f"Window size: {window_size} samples")
    print(f"Window overlap: {overlap*100:.1f}%")
    print(f"Stride: {int(window_size * (1 - overlap))} samples")

    # Load subject information
    subjects_info = pd.read_csv(os.path.join(os.path.dirname(data_dir), "data_subjects_info.csv"))
    print(f"Loaded information for {len(subjects_info)} subjects")

    # Initialize counters for detailed statistics
    activity_counts = {act: 0 for act in activity_map.keys()}
    subject_counts = {sub_id: 0 for sub_id in subjects_info['code']}
    total_raw_samples = 0

    X_data = []
    y_data = []
    subject_ids = []

    # Iterate through all activity folders
    print("\nProcessing data folders:")
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and not f.startswith('__')]

    for folder in tqdm(folders, desc="Activity folders"):
        # Extract activity and trial
        if '_' not in folder:
            print(f"  Skipping invalid folder: {folder}")
            continue

        activity = folder.split('_')[0]
        trial = folder.split('_')[1] if len(folder.split('_')) > 1 else "unknown"

        # Skip if not a valid activity
        if activity not in activity_map:
            print(f"  Skipping unknown activity: {activity}")
            continue

        activity_id = activity_map[activity]

        # Process each subject file
        files = [f for f in os.listdir(os.path.join(data_dir, folder)) if f.startswith('sub_') and f.endswith('.csv')]

        for file in tqdm(files, desc=f"  {folder} files", leave=False):
            # Extract subject ID
            subject_id = int(file.split('_')[1].split('.')[0])

            # Load data
            data_path = os.path.join(data_dir, folder, file)
            df = pd.read_csv(data_path)

            # Extract features (12 columns)
            sensor_data = df.iloc[:, 1:13].values  # Skip the first column (index)
            total_raw_samples += len(sensor_data)

            # Apply sliding window
            stride = int(window_size * (1 - overlap))
            segments_count = 0

            for i in range(0, len(sensor_data) - window_size + 1, stride):
                segment = sensor_data[i:i+window_size]
                X_data.append(segment)
                y_data.append(activity_id)
                subject_ids.append(subject_id)
                segments_count += 1

            # Update counters
            activity_counts[activity] += segments_count
            subject_counts[subject_id] += segments_count

    X = np.array(X_data)
    y = np.array(y_data)
    subject_ids = np.array(subject_ids)

    # Calculate class distribution
    class_distribution = {}
    for act_id, act_name in activity_names.items():
        count = np.sum(y == act_id)
        percentage = (count / len(y)) * 100
        class_distribution[act_name] = (count, percentage)

    # Print detailed statistics
    print("\nPreprocessing Summary:")
    print(f"  Total raw samples: {total_raw_samples}")
    print(f"  Total segments after windowing: {len(X)}")
    print(f"  Segment shape: {X[0].shape} (window_size × features)")
    print(f"  Features: attitude (roll, pitch, yaw), gravity (x, y, z), rotationRate (x, y, z), userAcceleration (x, y, z)")

    print("\nClass Distribution:")
    for act_name, (count, percentage) in class_distribution.items():
        print(f"  {act_name}: {count} segments ({percentage:.1f}%)")

    print("\nSubject Distribution:")
    for sub_id, count in sorted([(sub_id, count) for sub_id, count in subject_counts.items() if count > 0]):
        sub_info = subjects_info[subjects_info['code'] == sub_id].iloc[0]
        gender = "Male" if sub_info['gender'] == 1 else "Female"
        print(f"  Subject {sub_id} ({gender}, {sub_info['age']} years): {count} segments")

    print(f"\nPreprocessing completed in {time.time() - start_time:.2f} seconds")
    print("="*80)

    # Create preprocessing info dictionary
    preprocessing_info = {
        'window_size': window_size,
        'overlap': overlap,
        'stride': int(window_size * (1 - overlap)),
        'total_raw_samples': total_raw_samples,
        'total_segments': len(X),
        'segment_shape': X[0].shape,
        'class_distribution': class_distribution,
        'subject_counts': {k: v for k, v in subject_counts.items() if v > 0},
        'activity_counts': activity_counts
    }

    return X, y, subject_ids, preprocessing_info

# Custom Dataset class
class MotionSenseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Function to split data ensuring no information leakage
def subject_wise_split(X, y, subject_ids, test_size=0.2, val_size=0.1):
    """
    Split data by subjects to ensure no information leakage
    """
    print("\n" + "="*80)
    print("DATA SPLITTING (SUBJECT-WISE)")
    print("="*80)

    # Get unique subjects
    unique_subjects = np.unique(subject_ids)
    print(f"Total number of subjects: {len(unique_subjects)}")
    print(f"Test size: {test_size*100:.1f}% of subjects")
    print(f"Validation size: {val_size*100:.1f}% of subjects")

    # Split subjects into train and test
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=42
    )

    # Further split train subjects into train and validation
    if val_size > 0:
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=val_size/(1-test_size), random_state=42
        )

        print("\nSubject allocation:")
        print(f"  Training: {len(train_subjects)} subjects - {sorted(train_subjects)}")
        print(f"  Validation: {len(val_subjects)} subjects - {sorted(val_subjects)}")
        print(f"  Testing: {len(test_subjects)} subjects - {sorted(test_subjects)}")

        # Create masks
        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)
        test_mask = np.isin(subject_ids, test_subjects)

        # Split data
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print("\nData distribution:")
        print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

        # Check class distribution in each split
        print("\nClass distribution:")
        for split_name, split_y in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
            print(f"  {split_name} set:")
            for class_id in range(6):  # Assuming 6 classes
                class_count = np.sum(split_y == class_id)
                class_percent = (class_count / len(split_y)) * 100
                print(f"    Class {class_id}: {class_count} samples ({class_percent:.1f}%)")

        print("\nIMPORTANT: Data is split by subjects to prevent information leakage")
        print("This ensures that data from the same subject doesn't appear in different splits")
        print("="*80)

        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        print("\nSubject allocation:")
        print(f"  Training: {len(train_subjects)} subjects - {sorted(train_subjects)}")
        print(f"  Testing: {len(test_subjects)} subjects - {sorted(test_subjects)}")

        # Create masks
        train_mask = np.isin(subject_ids, train_subjects)
        test_mask = np.isin(subject_ids, test_subjects)

        # Split data
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print("\nData distribution:")
        print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

        # Check class distribution in each split
        print("\nClass distribution:")
        for split_name, split_y in [("Training", y_train), ("Test", y_test)]:
            print(f"  {split_name} set:")
            for class_id in range(6):  # Assuming 6 classes
                class_count = np.sum(split_y == class_id)
                class_percent = (class_count / len(split_y)) * 100
                print(f"    Class {class_id}: {class_count} samples ({class_percent:.1f}%)")

        print("\nIMPORTANT: Data is split by subjects to prevent information leakage")
        print("This ensures that data from the same subject doesn't appear in different splits")
        print("="*80)

        return X_train, y_train, X_test, y_test

# Function to normalize data
def normalize_data(X_train, X_val, X_test):
    """
    Normalize data using StandardScaler
    """
    import time

    print("\n" + "="*80)
    print("DATA NORMALIZATION")
    print("="*80)

    start_time = time.time()

    # Reshape data for scaling
    n_train, w, f = X_train.shape
    n_val, _, _ = X_val.shape
    n_test, _, _ = X_test.shape

    print(f"Normalizing data with StandardScaler:")
    print(f"  Training set: {n_train} samples")
    print(f"  Validation set: {n_val} samples")
    print(f"  Test set: {n_test} samples")

    X_train_reshaped = X_train.reshape(n_train * w, f)
    X_val_reshaped = X_val.reshape(n_val * w, f)
    X_test_reshaped = X_test.reshape(n_test * w, f)

    # Fit scaler on training data only
    print("\nFitting scaler on training data only (to prevent information leakage)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)

    # Print scaling parameters
    print("\nScaling parameters:")
    for i, feature_name in enumerate([
        "attitude.roll", "attitude.pitch", "attitude.yaw",
        "gravity.x", "gravity.y", "gravity.z",
        "rotationRate.x", "rotationRate.y", "rotationRate.z",
        "userAcceleration.x", "userAcceleration.y", "userAcceleration.z"
    ]):
        print(f"  {feature_name}: mean = {scaler.mean_[i]:.4f}, std = {scaler.scale_[i]:.4f}")

    # Apply same transformation to validation and test data
    print("\nApplying transformation to validation and test sets")
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    # Reshape back
    X_train = X_train_scaled.reshape(n_train, w, f)
    X_val = X_val_scaled.reshape(n_val, w, f)
    X_test = X_test_scaled.reshape(n_test, w, f)

    print(f"\nNormalization completed in {time.time() - start_time:.2f} seconds")
    print("="*80)

    return X_train, X_val, X_test, scaler

# """
# File Check and Management Utility
# Manages training_set_addition.txt and testing_set_addition.txt files
# """

# import os
# from datetime import datetime

# TRAINING_DIR = "training_set"
# TESTING_DIR = "testing_set"
# TRAINING_ADDITION_FILE = "training_set_addition.txt"
# TESTING_ADDITION_FILE = "testing_set_addition.txt"
# TRAINED_MODELS_DIR = "trained_models"


# def initialize_files():
#     """Initialize all required directories and files"""
#     # Create directories
#     os.makedirs(TRAINING_DIR, exist_ok=True)
#     os.makedirs(TESTING_DIR, exist_ok=True)
#     os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

#     # Create addition files if they don't exist
#     if not os.path.exists(TRAINING_ADDITION_FILE):
#         with open(TRAINING_ADDITION_FILE, 'w') as f:
#             f.write("# Training Set Addition Log\n")
#             f.write("# Format: speaker_name|filename|duration|timestamp\n")
#             f.write(f"# Created: {datetime.now().isoformat()}\n\n")

#     if not os.path.exists(TESTING_ADDITION_FILE):
#         with open(TESTING_ADDITION_FILE, 'w') as f:
#             f.write("# Testing Set Addition Log\n")
#             f.write("# Format: filename|duration|predicted_speaker|timestamp\n")
#             f.write(f"# Created: {datetime.now().isoformat()}\n\n")


# def get_training_summary():
#     """Get summary of training data"""
#     if not os.path.exists(TRAINING_ADDITION_FILE):
#         return {"total_samples": 0, "speakers": {}}

#     summary = {"total_samples": 0, "speakers": {}}

#     try:
#         with open(TRAINING_ADDITION_FILE, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or line.startswith("#"):
#                     continue

#                 try:
#                     parts = line.split("|")
#                     if len(parts) >= 3:
#                         speaker = parts[0].strip()
#                         filename = parts[1].strip()

#                         if speaker not in summary["speakers"]:
#                             summary["speakers"][speaker] = {
#                                 "count": 0, "total_duration": 0.0}

#                         summary["speakers"][speaker]["count"] += 1
#                         summary["speakers"][speaker]["total_duration"] += float(
#                             parts[2].strip())
#                         summary["total_samples"] += 1
#                 except Exception as e:
#                     continue
#     except Exception as e:
#         print(f"Error reading training_set_addition.txt: {e}")

#     return summary


# def get_testing_summary():
#     """Get summary of testing data"""
#     if not os.path.exists(TESTING_ADDITION_FILE):
#         return {"total_tests": 0, "predictions": {}}

#     summary = {"total_tests": 0, "predictions": {}}

#     try:
#         with open(TESTING_ADDITION_FILE, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or line.startswith("#"):
#                     continue

#                 try:
#                     parts = line.split("|")
#                     if len(parts) >= 3:
#                         predicted_speaker = parts[2].strip()

#                         if predicted_speaker not in summary["predictions"]:
#                             summary["predictions"][predicted_speaker] = 0

#                         summary["predictions"][predicted_speaker] += 1
#                         summary["total_tests"] += 1
#                 except Exception as e:
#                     continue
#     except Exception as e:
#         print(f"Error reading testing_set_addition.txt: {e}")

#     return summary


# def verify_data_integrity():
#     """Verify that referenced files exist in training_set and testing_set directories"""
#     issues = []

#     # Check training_set_addition.txt
#     if os.path.exists(TRAINING_ADDITION_FILE):
#         with open(TRAINING_ADDITION_FILE, 'r') as f:
#             for idx, line in enumerate(f, 1):
#                 line = line.strip()
#                 if not line or line.startswith("#"):
#                     continue

#                 try:
#                     parts = line.split("|")
#                     if len(parts) >= 2:
#                         speaker = parts[0].strip()
#                         filename = parts[1].strip()
#                         filepath = os.path.join(
#                             TRAINING_DIR, speaker, filename)

#                         if not os.path.exists(filepath):
#                             issues.append(
#                                 f"training_set_addition.txt:{idx} - Missing file: {filepath}")
#                 except Exception as e:
#                     continue

#     # Check testing_set_addition.txt
#     if os.path.exists(TESTING_ADDITION_FILE):
#         with open(TESTING_ADDITION_FILE, 'r') as f:
#             for idx, line in enumerate(f, 1):
#                 line = line.strip()
#                 if not line or line.startswith("#"):
#                     continue

#                 try:
#                     parts = line.split("|")
#                     if len(parts) >= 1:
#                         filename = parts[0].strip()
#                         filepath = os.path.join(TESTING_DIR, filename)

#                         if not os.path.exists(filepath):
#                             issues.append(
#                                 f"testing_set_addition.txt:{idx} - Missing file: {filepath}")
#                 except Exception as e:
#                     continue

#     return issues


# def cleanup_unused_files():
#     """Remove files that are not referenced in addition files"""
#     referenced_files = set()

#     # Get all referenced files from training_set_addition.txt
#     if os.path.exists(TRAINING_ADDITION_FILE):
#         with open(TRAINING_ADDITION_FILE, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or line.startswith("#"):
#                     continue

#                 try:
#                     parts = line.split("|")
#                     if len(parts) >= 2:
#                         speaker = parts[0].strip()
#                         filename = parts[1].strip()
#                         filepath = os.path.join(
#                             TRAINING_DIR, speaker, filename)
#                         referenced_files.add(filepath)
#                 except Exception as e:
#                     continue

#     # Get all referenced files from testing_set_addition.txt
#     if os.path.exists(TESTING_ADDITION_FILE):
#         with open(TESTING_ADDITION_FILE, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or line.startswith("#"):
#                     continue

#                 try:
#                     parts = line.split("|")
#                     if len(parts) >= 1:
#                         filename = parts[0].strip()
#                         filepath = os.path.join(TESTING_DIR, filename)
#                         referenced_files.add(filepath)
#                 except Exception as e:
#                     continue

#     # Find and remove unreferenced files
#     removed = []

#     # Clean training_set directory
#     if os.path.exists(TRAINING_DIR):
#         for root, dirs, files in os.walk(TRAINING_DIR):
#             for file in files:
#                 filepath = os.path.join(root, file)
#                 if filepath not in referenced_files:
#                     try:
#                         os.remove(filepath)
#                         removed.append(filepath)
#                     except Exception as e:
#                         print(f"Error removing {filepath}: {e}")

#     # Clean testing_set directory
#     if os.path.exists(TESTING_DIR):
#         for root, dirs, files in os.walk(TESTING_DIR):
#             for file in files:
#                 filepath = os.path.join(root, file)
#                 if filepath not in referenced_files:
#                     try:
#                         os.remove(filepath)
#                         removed.append(filepath)
#                     except Exception as e:
#                         print(f"Error removing {filepath}: {e}")

#     return removed


# def get_file_structure_info():
#     """Get information about the current file structure"""
#     info = {
#         'training_dir_exists': os.path.exists(TRAINING_DIR),
#         'testing_dir_exists': os.path.exists(TESTING_DIR),
#         'training_addition_exists': os.path.exists(TRAINING_ADDITION_FILE),
#         'testing_addition_exists': os.path.exists(TESTING_ADDITION_FILE),
#         'trained_models_dir_exists': os.path.exists(TRAINED_MODELS_DIR),
#         'training_samples_count': 0,
#         'testing_samples_count': 0,
#         'trained_models_count': 0
#     }

#     # Count training samples
#     if os.path.exists(TRAINING_DIR):
#         for root, dirs, files in os.walk(TRAINING_DIR):
#             info['training_samples_count'] += len(files)

#     # Count testing samples
#     if os.path.exists(TESTING_DIR):
#         for root, dirs, files in os.walk(TESTING_DIR):
#             info['testing_samples_count'] += len(files)

#     # Count trained models
#     if os.path.exists(TRAINED_MODELS_DIR):
#         files = os.listdir(TRAINED_MODELS_DIR)
#         info['trained_models_count'] = len(
#             [f for f in files if f.endswith('_model.pkl')])

#     return info


# if __name__ == "__main__":
#     print("=== File Check Utility ===\n")

#     # Initialize
#     print("Initializing files...")
#     initialize_files()
#     print("✓ Files initialized\n")

#     # File structure info
#     print("File Structure Information:")
#     info = get_file_structure_info()
#     for key, value in info.items():
#         print(f"  {key}: {value}")
#     print()

#     # Training summary
#     train_summary = get_training_summary()
#     print(f"Training Summary:")
#     print(f"  Total samples: {train_summary['total_samples']}")
#     for speaker, data in train_summary['speakers'].items():
#         print(
#             f"  - {speaker}: {data['count']} samples ({data['total_duration']:.2f}s)")
#     print()

#     # Testing summary
#     test_summary = get_testing_summary()
#     print(f"Testing Summary:")
#     print(f"  Total tests: {test_summary['total_tests']}")
#     for speaker, count in test_summary['predictions'].items():
#         print(f"  - {speaker}: {count} predictions")
#     print()

#     # Integrity check
#     print("Checking data integrity...")
#     issues = verify_data_integrity()
#     if issues:
#         print(f"⚠️  Found {len(issues)} issues:")
#         for issue in issues:
#             print(f"  - {issue}")
#     else:
#         print("✓ No integrity issues found")
#     print()

#     # Cleanup
#     print("Cleaning up unreferenced files...")
#     removed = cleanup_unused_files()
#     if removed:
#         print(f"Removed {len(removed)} unreferenced files:")
#         for file in removed:
#             print(f"  - {file}")
#     else:
#         print("✓ No cleanup needed")
import os

folder_path = 'training_set'  # or 'testing_set'
manifest_path = 'training_set_addition.txt'  # or 'testing_set_addition.txt'

# Get all wav files in the folder (case-sensitive)
files_in_folder = set(f for f in os.listdir(
    folder_path) if f.lower().endswith('.wav'))

cleaned_lines = []
with open(manifest_path, 'r') as f:
    for line in f:
        fname = line.strip()
        if fname in files_in_folder:
            cleaned_lines.append(fname)
        else:
            print(f'File missing: {fname} (removed from manifest)')

with open(manifest_path, 'w') as f:
    for fname in cleaned_lines:
        f.write(fname + '\n')

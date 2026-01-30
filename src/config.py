# -*- coding: utf-8 -*-
"""
Project Configuration File
"""

import os
from pathlib import Path

# ============================================
# Path Configuration
# ============================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset root path (Update this to your local path)
DATASET_ROOT = r"C:\Users\Echo\Desktop\Modbus Dataset\Modbus Dataset"

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"

# Output paths
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

# PCAP file paths
BENIGN_PCAP_DIR = os.path.join(DATASET_ROOT, "benign", "network-wide-pcap-capture", "network-wide")
EXTERNAL_PCAP_DIR = os.path.join(DATASET_ROOT, "attack", "external", "external-attacker", "external-attacker-network-capture")
IED_PCAP_DIR = os.path.join(DATASET_ROOT, "attack", "compromised-ied", "ied1b", "ied1b-network-captures")
SCADA_PCAP_DIR = os.path.join(DATASET_ROOT, "attack", "compromised-scada", "substation-wide-capture")

# CSV label paths
EXTERNAL_LABELS_DIR = os.path.join(DATASET_ROOT, "attack", "external", "external-attacker", "attacker logs")
IED_LABELS_DIR = os.path.join(DATASET_ROOT, "attack", "compromised-ied", "attack logs")
SCADA_LABELS_DIR = os.path.join(DATASET_ROOT, "attack", "compromised-scada", "attack logs")

# ============================================
# Sampling Configuration
# ============================================

SAMPLING_CONFIG = {
    'benign': 200000,
    'external': -1,  # -1 means use all
    'ied': -1,
    'scada': 200000,
    'ied_oversample_target': 5000
}

# ============================================
# Model Configuration
# ============================================

RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# ============================================
# Feature Engineering Configuration
# ============================================

# Time window configuration
TIME_WINDOW_MAIN = 15  # seconds, for main experiments
TIME_WINDOW_SENSITIVITY = [10, 30]  # for sensitivity analysis

# Legacy window configurations for compatibility
TIME_WINDOWS = {
    'short': '1S',
    'medium': '5S',
    'long': '10S',
    'session': '60S'
}

# ============================================
# Labeling Strategy Configuration
# ============================================

# Attacker IP
ATTACKER_IP = '185.175.0.7'

# Scenario definitions
SCENARIOS = {
    'benign': 'benign',
    'external': 'external', 
    'scada': 'compromised-scada',
    'ied': 'compromised-ied'
}

# ============================================
# Data Split Configuration
# ============================================

SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15
}

RANDOM_SEED = 42

# ============================================
# Rule Engine Configuration
# ============================================

RULE_THRESHOLDS = {
    'packet_rate_max': 100,  # packets/sec
    'write_ratio_max': 0.8,
    'entropy_min': 0.5,
    'duplicate_count_max': 10
}

# ============================================
# Device IP Mapping
# ============================================

DEVICE_IPS = {
    'scada_secure': '185.175.0.2',
    'scada_normal': '185.175.0.3',
    'ied1a': '185.175.0.4',
    'ied1b': '185.175.0.5',
    'central_agent': '185.175.0.6',
    'external_attacker': '185.175.0.7',
    'ied4c': '185.175.0.8'
}

IED_IPS = ['185.175.0.4', '185.175.0.5', '185.175.0.8']
SCADA_IPS = ['185.175.0.2', '185.175.0.3']
EXTERNAL_IPS = ['185.175.0.7']

# ============================================
# Modbus Protocol Configuration
# ============================================

VALID_FUNCTION_CODES = [
    0x01,  # Read Coils
    0x02,  # Read Discrete Inputs
    0x03,  # Read Holding Registers
    0x04,  # Read Input Registers
    0x05,  # Write Single Coil
    0x06,  # Write Single Register
    0x0F,  # Write Multiple Coils (15)
    0x10,  # Write Multiple Registers (16)
    0x17   # Read/Write Multiple Registers (23)
]

READ_FUNCTION_CODES = [0x01, 0x02, 0x03, 0x04]
WRITE_FUNCTION_CODES = [0x05, 0x06, 0x0F, 0x10]

# Sensitive register ranges (examples)
SENSITIVE_REGISTERS = [
    (1000, 1050, "Temperature Setpoint"),
    (2000, 2020, "Pressure Threshold"),
    (5000, 5100, "Safety Interlock Parameters"),
    (8000, 8050, "Critical Control Parameters")
]

# ============================================
# Utility Functions
# ============================================

def ensure_dirs():
    """Ensure all necessary directories exist"""
    dirs = [
        DATA_RAW, DATA_PROCESSED, DATA_SPLITS,
        MODELS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✓ All directories created")

if __name__ == "__main__":
    # Configuration test
    print("=" * 60)
    print("Project Configuration Test")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"\nChecking paths...")
    
    # Check dataset path
    if os.path.exists(DATASET_ROOT):
        print(f"✓ Dataset path exists")
    else:
        print(f"✗ Dataset path does not exist! Please check the path")
    
    # Create necessary directories
    ensure_dirs()
    
    print(f"\n✓ Configuration loaded successfully!")

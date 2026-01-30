# -*- coding: utf-8 -*-
"""
Feature Extraction Module - Phase 1 Step 3

Function: Extract window-level features (44 features) from packet-level data.
Author: AI Assistant
Date: 2026-01-09

Feature Categories:
- Protocol layer features (~13)
- Temporal layer features (~9)
- DCS business logic features (~22)
"""

import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# Constant Definitions
# ============================================

# Device IP mappings
SCADA_IPS = ['185.175.0.2', '185.175.0.3']
IED_IPS = ['185.175.0.4', '185.175.0.5', '185.175.0.8']
AGENT_IPS = ['185.175.0.6']
ATTACKER_IP = '185.175.0.7'
ALL_KNOWN_IPS = SCADA_IPS + IED_IPS + AGENT_IPS + [ATTACKER_IP]

# Function code classifications
READ_FCS = [1, 2, 3, 4]
WRITE_FCS = [5, 6, 15, 16]
VALID_FCS = [1, 2, 3, 4, 5, 6, 15, 16, 23]


# ============================================
# Helper Functions
# ============================================

def safe_mean(arr):
    """Safely calculate mean, returns 0.0 for empty arrays"""
    if len(arr) == 0:
        return 0.0
    return np.mean(arr)

def safe_std(arr):
    """Safely calculate standard deviation, returns 0.0 for empty arrays"""
    if len(arr) == 0:
        return 0.0
    return np.std(arr)

def safe_min(arr):
    """Safely calculate minimum, returns 0.0 for empty arrays"""
    if len(arr) == 0:
        return 0.0
    return np.min(arr)

def safe_max(arr):
    """Safely calculate maximum, returns 0.0 for empty arrays"""
    if len(arr) == 0:
        return 0.0
    return np.max(arr)

def calculate_entropy(values):
    """Calculate Shannon entropy"""
    if len(values) == 0:
        return 0.0
    counts = Counter(values)
    probs = np.array(list(counts.values())) / len(values)
    return entropy(probs, base=2)


# ============================================
# Protocol Layer Features (13 features)
# ============================================

def extract_protocol_features(window_df: pd.DataFrame) -> Dict:
    """
    Extract protocol layer features
    
    Args:
        window_df: Packet data within the window
        
    Returns:
        Dictionary of protocol layer features
    """
    n_packets = len(window_df)
    if n_packets == 0:
        return _empty_protocol_features()
    
    fcs = window_df['function_code'].values
    txids = window_df['transaction_id'].values
    uids = window_df['unit_id'].values
    payload_sizes = window_df['payload_size'].values
    addresses = window_df['start_address'].dropna().values
    
    # Function code statistics
    fc_counts = Counter(fcs)
    read_count = sum(fc_counts.get(fc, 0) for fc in READ_FCS)
    write_count = sum(fc_counts.get(fc, 0) for fc in WRITE_FCS)
    
    features = {
        # Function code distribution
        'fc_distribution_entropy': calculate_entropy(fcs),
        'fc_diversity': len(set(fcs)),
        'fc_read_ratio': read_count / n_packets if n_packets > 0 else 0,
        'fc_write_ratio': write_count / n_packets if n_packets > 0 else 0,
        
        # Transaction ID statistics
        'txid_mean': safe_mean(txids),
        'txid_std': safe_std(txids),
        'txid_unique_ratio': len(set(txids)) / n_packets if n_packets > 0 else 0,
        
        # Unit ID
        'unit_id_diversity': len(set(uids)),
        
        # Packet size
        'packet_size_mean': safe_mean(payload_sizes),
        'packet_size_std': safe_std(payload_sizes),
        
        # Address range
        'address_range': (safe_max(addresses) - safe_min(addresses)) if len(addresses) > 0 else 0,
        
        # Write value statistics (using quantity as a proxy)
        'value_mean': safe_mean(window_df['quantity'].dropna().values),
        'value_std': safe_std(window_df['quantity'].dropna().values),
    }
    
    return features

def _empty_protocol_features() -> Dict:
    """Return protocol features for an empty window"""
    return {
        'fc_distribution_entropy': 0.0,
        'fc_diversity': 0,
        'fc_read_ratio': 0.0,
        'fc_write_ratio': 0.0,
        'txid_mean': 0.0,
        'txid_std': 0.0,
        'txid_unique_ratio': 0.0,
        'unit_id_diversity': 0,
        'packet_size_mean': 0.0,
        'packet_size_std': 0.0,
        'address_range': 0.0,
        'value_mean': 0.0,
        'value_std': 0.0,
    }


# ============================================
# Temporal Layer Features (9 features)
# ============================================

def extract_temporal_features(window_df: pd.DataFrame, window_size: float = 15.0) -> Dict:
    """
    Extract temporal layer features
    
    Args:
        window_df: Packet data within the window
        window_size: Window size in seconds
        
    Returns:
        Dictionary of temporal layer features
    """
    n_packets = len(window_df)
    if n_packets == 0:
        return _empty_temporal_features()
    
    timestamps = pd.to_datetime(window_df['timestamp'])
    
    # Calculate inter-packet intervals
    if n_packets > 1:
        intervals = timestamps.diff().dt.total_seconds().dropna().values
    else:
        intervals = np.array([])
    
    # Packet rate
    packet_rate = n_packets / window_size if window_size > 0 else 0
    
    # Burst detection (sequences of packets with interval < 0.1s)
    burst_threshold = 0.1  # 100ms
    burst_count = 0
    burst_lengths = []
    current_burst = 0
    
    for interval in intervals:
        if interval < burst_threshold:
            current_burst += 1
        else:
            if current_burst > 0:
                burst_count += 1
                burst_lengths.append(current_burst)
            current_burst = 0
    if current_burst > 0:
        burst_count += 1
        burst_lengths.append(current_burst)
    
    # Session count (based on src_ip-dst_ip pairs)
    if 'src_ip' in window_df.columns and 'dst_ip' in window_df.columns:
        sessions = window_df.groupby(['src_ip', 'dst_ip']).ngroups
    else:
        sessions = 0
    
    features = {
        'packet_count': n_packets,
        'packet_rate': packet_rate,
        'interval_mean': safe_mean(intervals),
        'interval_std': safe_std(intervals),
        'interval_min': safe_min(intervals),
        'interval_max': safe_max(intervals),
        'burst_count': burst_count,
        'burst_intensity': safe_mean(burst_lengths) if burst_lengths else 0,
        'session_count': sessions,
    }
    
    return features

def _empty_temporal_features() -> Dict:
    """Return temporal features for an empty window"""
    return {
        'packet_count': 0,
        'packet_rate': 0.0,
        'interval_mean': 0.0,
        'interval_std': 0.0,
        'interval_min': 0.0,
        'interval_max': 0.0,
        'burst_count': 0,
        'burst_intensity': 0.0,
        'session_count': 0,
    }


# ============================================
# DCS Business Logic Features (22 features)
# ============================================

def extract_dcs_features(window_df: pd.DataFrame) -> Dict:
    """
    Extract DCS business logic features (Core Innovation)
    
    Args:
        window_df: Packet data within the window
        
    Returns:
        Dictionary of DCS business logic features
    """
    n_packets = len(window_df)
    if n_packets == 0:
        return _empty_dcs_features()
    
    features = {}
    
    # A. Device Role Features (6)
    features.update(_extract_device_role_features(window_df, n_packets))
    
    # B. Communication Topology Features (5)
    features.update(_extract_topology_features(window_df, n_packets))
    
    # C. Operation Pattern Features (6)
    features.update(_extract_operation_features(window_df, n_packets))
    
    # D. Anomaly Indicator Features (5)
    features.update(_extract_anomaly_features(window_df, n_packets))
    
    return features


def _extract_device_role_features(window_df: pd.DataFrame, n_packets: int) -> Dict:
    """Extract device role features (6)"""
    src_ips = window_df['src_ip'].values
    dst_ips = window_df['dst_ip'].values
    
    # Statistics for each role as source/destination
    scada_src_count = sum(1 for ip in src_ips if ip in SCADA_IPS)
    scada_dst_count = sum(1 for ip in dst_ips if ip in SCADA_IPS)
    ied_src_count = sum(1 for ip in src_ips if ip in IED_IPS)
    ied_dst_count = sum(1 for ip in dst_ips if ip in IED_IPS)
    
    # Typical SCADA->IED pattern: SCADA initiates request, IED responds
    typical_pattern_count = 0
    for src, dst in zip(src_ips, dst_ips):
        if src in SCADA_IPS and dst in IED_IPS:
            typical_pattern_count += 1
        elif src in IED_IPS and dst in SCADA_IPS:
            typical_pattern_count += 1
    
    # Device role distribution entropy
    role_counts = {
        'scada': scada_src_count + scada_dst_count,
        'ied': ied_src_count + ied_dst_count,
    }
    role_values = [v for v in role_counts.values() if v > 0]
    role_entropy = calculate_entropy(role_values) if role_values else 0
    
    return {
        'scada_src_ratio': scada_src_count / n_packets,
        'scada_dst_ratio': scada_dst_count / n_packets,
        'ied_src_ratio': ied_src_count / n_packets,
        'ied_dst_ratio': ied_dst_count / n_packets,
        'device_role_entropy': role_entropy,
        'is_typical_scada_to_ied': typical_pattern_count / n_packets,
    }


def _extract_topology_features(window_df: pd.DataFrame, n_packets: int) -> Dict:
    """Extract communication topology features (5)"""
    src_ips = window_df['src_ip'].values
    dst_ips = window_df['dst_ip'].values
    
    # Unique IP counts
    unique_src = len(set(src_ips))
    unique_dst = len(set(dst_ips))
    
    # Communication pairs
    comm_pairs = list(zip(src_ips, dst_ips))
    unique_pairs = len(set(comm_pairs))
    
    # Dominant communication pair ratio
    pair_counts = Counter(comm_pairs)
    max_pair_count = max(pair_counts.values()) if pair_counts else 0
    dominant_ratio = max_pair_count / n_packets
    
    # Multi-target operations (one source IP to multiple destinations)
    src_to_dst = {}
    for src, dst in comm_pairs:
        if src not in src_to_dst:
            src_to_dst[src] = set()
        src_to_dst[src].add(dst)
    
    multi_target_count = sum(1 for dsts in src_to_dst.values() if len(dsts) > 1)
    multi_target_ratio = multi_target_count / len(src_to_dst) if src_to_dst else 0
    
    return {
        'src_ip_count': unique_src,
        'dst_ip_count': unique_dst,
        'comm_pair_count': unique_pairs,
        'dominant_pair_ratio': dominant_ratio,
        'multi_target_ratio': multi_target_ratio,
    }


def _extract_operation_features(window_df: pd.DataFrame, n_packets: int) -> Dict:
    """Extract operation pattern features (6) - Core Features"""
    fcs = window_df['function_code'].values
    
    # Consecutive write analysis
    write_sequences = []
    current_write_seq = 0
    
    for fc in fcs:
        if fc in WRITE_FCS:
            current_write_seq += 1
        else:
            if current_write_seq > 0:
                write_sequences.append(current_write_seq)
            current_write_seq = 0
    if current_write_seq > 0:
        write_sequences.append(current_write_seq)
    
    # Write bursts (consecutive writes > 3)
    write_burst_threshold = 3
    write_burst_count = sum(1 for seq in write_sequences if seq > write_burst_threshold)
    
    # Read-Write alternation frequency
    alternation_count = 0
    prev_is_write = None
    for fc in fcs:
        is_write = fc in WRITE_FCS
        if prev_is_write is not None and is_write != prev_is_write:
            alternation_count += 1
        prev_is_write = is_write
    
    # Operation sequence entropy
    op_sequence = ['W' if fc in WRITE_FCS else 'R' for fc in fcs]
    op_entropy = calculate_entropy(op_sequence)
    
    # Ratio of writes without corresponding reads (isolated writes)
    write_count = sum(1 for fc in fcs if fc in WRITE_FCS)
    read_count = sum(1 for fc in fcs if fc in READ_FCS)
    write_without_read_ratio = 1.0 if (write_count > 0 and read_count == 0) else 0.0
    
    return {
        'consecutive_write_max': max(write_sequences) if write_sequences else 0,
        'consecutive_write_mean': safe_mean(write_sequences),
        'write_burst_count': write_burst_count,
        'read_write_alternation': alternation_count / n_packets if n_packets > 0 else 0,
        'operation_sequence_entropy': op_entropy,
        'write_without_read_ratio': write_without_read_ratio,
    }


def _extract_anomaly_features(window_df: pd.DataFrame, n_packets: int) -> Dict:
    """Extract anomaly indicator features (5)"""
    src_ips = set(window_df['src_ip'].values)
    dst_ips = set(window_df['dst_ip'].values)
    all_ips = src_ips | dst_ips
    fcs = window_df['function_code'].values
    addresses = window_df['start_address'].dropna().values
    
    # External IP detection
    external_present = 1 if ATTACKER_IP in all_ips else 0
    
    # External IP packet ratio
    external_packet_count = sum(1 for _, row in window_df.iterrows() 
                                 if row['src_ip'] == ATTACKER_IP or row['dst_ip'] == ATTACKER_IP)
    external_ratio = external_packet_count / n_packets
    
    # Unknown IP count
    unknown_ips = [ip for ip in all_ips if ip not in ALL_KNOWN_IPS]
    unknown_ip_count = len(unknown_ips)
    
    # Abnormal function code ratio
    abnormal_fc_count = sum(1 for fc in fcs if fc not in VALID_FCS)
    abnormal_fc_ratio = abnormal_fc_count / n_packets
    
    # Address range anomaly (beyond typical range 0-65535)
    address_exceeded_count = sum(1 for addr in addresses if addr > 10000)  # Assuming >10000 is abnormal
    address_exceeded_ratio = address_exceeded_count / len(addresses) if len(addresses) > 0 else 0
    
    return {
        'external_ip_present': external_present,
        'external_ip_packet_ratio': external_ratio,
        'unknown_ip_count': unknown_ip_count,
        'abnormal_fc_ratio': abnormal_fc_ratio,
        'address_range_exceeded': address_exceeded_ratio,
    }


def _empty_dcs_features() -> Dict:
    """Return DCS business logic features for an empty window"""
    return {
        # Device role features
        'scada_src_ratio': 0.0,
        'scada_dst_ratio': 0.0,
        'ied_src_ratio': 0.0,
        'ied_dst_ratio': 0.0,
        'device_role_entropy': 0.0,
        'is_typical_scada_to_ied': 0.0,
        # Communication topology features
        'src_ip_count': 0,
        'dst_ip_count': 0,
        'comm_pair_count': 0,
        'dominant_pair_ratio': 0.0,
        'multi_target_ratio': 0.0,
        # Operation pattern features
        'consecutive_write_max': 0,
        'consecutive_write_mean': 0.0,
        'write_burst_count': 0,
        'read_write_alternation': 0.0,
        'operation_sequence_entropy': 0.0,
        'write_without_read_ratio': 0.0,
        # Anomaly indicator features
        'external_ip_present': 0,
        'external_ip_packet_ratio': 0.0,
        'unknown_ip_count': 0,
        'abnormal_fc_ratio': 0.0,
        'address_range_exceeded': 0.0,
    }


# ============================================
# Window-level Feature Extraction
# ============================================

def extract_window_features(window_df: pd.DataFrame, window_size: float = 15.0) -> Dict:
    """
    Extract all features for a window (44 features)
    
    Args:
        window_df: Packet data within the window
        window_size: Window size in seconds
        
    Returns:
        Dictionary containing all features
    """
    features = {}
    
    # Protocol layer features (13)
    features.update(extract_protocol_features(window_df))
    
    # Temporal layer features (9)
    features.update(extract_temporal_features(window_df, window_size))
    
    # DCS business logic features (22)
    features.update(extract_dcs_features(window_df))
    
    return features


def get_feature_names() -> List[str]:
    """Get list of all feature names (44)"""
    # Merge empty window features to get all keys
    all_features = {}
    all_features.update(_empty_protocol_features())
    all_features.update(_empty_temporal_features())
    all_features.update(_empty_dcs_features())
    return list(all_features.keys())


def get_feature_groups() -> Dict[str, List[str]]:
    """Get feature groupings"""
    return {
        'protocol': list(_empty_protocol_features().keys()),
        'temporal': list(_empty_temporal_features().keys()),
        'dcs_device_role': ['scada_src_ratio', 'scada_dst_ratio', 'ied_src_ratio', 
                           'ied_dst_ratio', 'device_role_entropy', 'is_typical_scada_to_ied'],
        'dcs_topology': ['src_ip_count', 'dst_ip_count', 'comm_pair_count',
                        'dominant_pair_ratio', 'multi_target_ratio'],
        'dcs_operation': ['consecutive_write_max', 'consecutive_write_mean', 'write_burst_count',
                         'read_write_alternation', 'operation_sequence_entropy', 'write_without_read_ratio'],
        'dcs_anomaly': ['external_ip_present', 'external_ip_packet_ratio', 'unknown_ip_count',
                       'abnormal_fc_ratio', 'address_range_exceeded'],
    }


# ============================================
# Test Code
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Feature Extraction Module Test")
    print("=" * 60)
    
    # Get feature information
    feature_names = get_feature_names()
    feature_groups = get_feature_groups()
    
    print(f"\nTotal Features: {len(feature_names)}")
    print("\nFeature Groupings:")
    for group, features in feature_groups.items():
        print(f"  - {group}: {len(features)}")
    
    print("\nAll Features List:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")

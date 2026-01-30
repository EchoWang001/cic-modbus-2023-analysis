#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Data Sampling Script - Overnight Run Version
Estimated time: 4-6 hours
"""

import pandas as pd
import numpy as np
from scapy.all import rdpcap, TCP
import os
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
import pickle
import logging

# Configure logging
log_file = r'C:\Users\Echo\Desktop\modbus-detection\results\logs\sampling_log.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

sys.path.append(r'C:\Users\Echo\Desktop\modbus-detection\src')
import config

logging.info("=" * 60)
logging.info("Full Data Sampling Started")
logging.info("=" * 60)

# ============================================
# Function Definitions
# ============================================

def parse_pcap_modbus(pcap_path, is_attack=False):
    """
    Parse a single PCAP file and extract Modbus packets.
    Strict mode: Skip any invalid data, no default values used.
    
    Args:
        pcap_path: Path to the PCAP file
        is_attack: Whether it's attack data (requires timezone conversion)
    
    Returns:
        DataFrame containing parsed Modbus packets
    """
    packets_data = []
    total_packets = 0
    skipped_packets = 0
    
    try:
        packets = rdpcap(pcap_path)
        total_packets = len(packets)
        
        for pkt in packets:
            # Only process Modbus packets (TCP port 502)
            if TCP not in pkt:
                skipped_packets += 1
                continue
            
            if pkt[TCP].sport != 502 and pkt[TCP].dport != 502:
                skipped_packets += 1
                continue
            
            try:
                # Extract timestamp (Strict mode: skip if invalid)
                try:
                    timestamp = datetime.fromtimestamp(float(pkt.time))
                except (ValueError, OSError, OverflowError, TypeError):
                    skipped_packets += 1
                    continue
                
                # Validate timestamp (Dataset should be between 2020-2025)
                if timestamp.year < 2020 or timestamp.year > 2025:
                    skipped_packets += 1
                    continue
                
                # Timezone conversion (Attack data: ADT -> UTC, +3 hours)
                if is_attack:
                    timestamp = timestamp + timedelta(hours=3)
                
                # Extract network layer info (Strict mode: skip if missing)
                try:
                    src_ip = pkt['IP'].src
                    dst_ip = pkt['IP'].dst
                    src_port = pkt[TCP].sport
                    dst_port = pkt[TCP].dport
                except (KeyError, AttributeError, IndexError):
                    skipped_packets += 1
                    continue
                
                # Extract Modbus application layer data
                payload = bytes(pkt[TCP].payload)
                
                # MBAP Header is at least 7 bytes + 1 byte Function Code
                if len(payload) < 8:
                    skipped_packets += 1
                    continue
                
                # Parse MBAP Header (Strict mode: skip if parsing fails)
                try:
                    transaction_id = int.from_bytes(payload[0:2], byteorder='big')
                    protocol_id = int.from_bytes(payload[2:4], byteorder='big')
                    length = int.from_bytes(payload[4:6], byteorder='big')
                    unit_id = payload[6]
                    function_code = payload[7]
                except (IndexError, ValueError, TypeError):
                    skipped_packets += 1
                    continue
                
                # Validate Modbus protocol ID (should be 0)
                if protocol_id != 0:
                    skipped_packets += 1
                    continue
                
                # Validate function code
                valid_function_codes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x0F, 0x10, 0x17]
                if function_code not in valid_function_codes:
                    skipped_packets += 1
                    continue
                
                # Parse specific fields based on function code
                start_address = None
                quantity = None
                data_length = None
                
                try:
                    if function_code in [0x01, 0x02, 0x03, 0x04]:  # Read operations
                        if len(payload) >= 12:
                            start_address = int.from_bytes(payload[8:10], byteorder='big')
                            quantity = int.from_bytes(payload[10:12], byteorder='big')
                            data_length = length - 2
                    
                    elif function_code in [0x05, 0x06]:  # Write Single
                        if len(payload) >= 12:
                            start_address = int.from_bytes(payload[8:10], byteorder='big')
                            data_length = length - 2
                    
                    elif function_code in [0x0F, 0x10]:  # Write Multiple
                        if len(payload) >= 13:
                            start_address = int.from_bytes(payload[8:10], byteorder='big')
                            quantity = int.from_bytes(payload[10:12], byteorder='big')
                            data_length = payload[12]
                except (IndexError, ValueError, TypeError):
                    # Field parsing failed, use None (these fields are optional)
                    pass
                
                # Build data record
                packet_info = {
                    'timestamp': timestamp,
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'transaction_id': transaction_id,
                    'protocol_id': protocol_id,
                    'length': length,
                    'unit_id': unit_id,
                    'function_code': function_code,
                    'start_address': start_address,
                    'quantity': quantity,
                    'data_length': data_length,
                    'payload_size': len(payload)
                }
                
                packets_data.append(packet_info)
                
            except Exception as e:
                # Unexpected error, skip this packet
                skipped_packets += 1
                continue
        
        # Log statistics
        if total_packets > 0:
            valid_rate = (total_packets - skipped_packets) / total_packets * 100
            if valid_rate < 50:
                logging.warning(f"File {os.path.basename(pcap_path)}: Low efficiency {valid_rate:.1f}% ({total_packets - skipped_packets}/{total_packets})")
            else:
                logging.info(f"File {os.path.basename(pcap_path)}: Valid packets {total_packets - skipped_packets}/{total_packets} ({valid_rate:.1f}%)")
                
    except Exception as e:
        logging.error(f"Failed to read PCAP file: {pcap_path} - {e}")
        return pd.DataFrame()
    
    return pd.DataFrame(packets_data)


def align_labels(df_packets, csv_labels, scenario='unknown', time_window=1.0):
    """
    Align CSV labels to PCAP packets.
    Uses different strategies based on the scenario.
    
    Args:
        df_packets: PCAP data
        csv_labels: List of CSV label files
        scenario: 'external', 'ied', 'scada'
        time_window: Time window in seconds
    """
    
    if len(df_packets) == 0:
        logging.warning("PCAP data is empty, skipping label alignment")
        return df_packets
    
    df_packets = df_packets.copy()
    df_packets['attack_type'] = 'unknown'
    
    # Read all CSV labels
    all_labels = []
    for csv_file in csv_labels:
        try:
            df_label = pd.read_csv(csv_file)
            all_labels.append(df_label)
        except Exception as e:
            logging.warning(f"Failed to read CSV: {os.path.basename(csv_file)} - {e}")
            continue
    
    if not all_labels:
        logging.warning("No valid CSV label files found")
        return df_packets
    
    labels_df = pd.concat(all_labels, ignore_index=True)
    logging.info(f"Merged CSV labels: {len(labels_df)} entries")
    
    # Parse timestamps
    try:
        labels_df['Timestamp'] = pd.to_datetime(labels_df['Timestamp'], format='mixed', errors='coerce')
        invalid_count = labels_df['Timestamp'].isna().sum()
        
        if invalid_count > 0:
            logging.warning(f"Skipped {invalid_count} invalid timestamps in CSV")
            labels_df = labels_df.dropna(subset=['Timestamp'])
        
        logging.info(f"CSV timestamps parsed successfully: {len(labels_df)} valid entries")
        
    except Exception as e:
        logging.error(f"Failed to parse CSV timestamps: {e}")
        return df_packets
    
    # Convert PCAP timestamps
    try:
        df_packets['timestamp'] = pd.to_datetime(df_packets['timestamp'], errors='coerce')
        df_packets = df_packets.dropna(subset=['timestamp'])
    except Exception as e:
        logging.error(f"Failed to convert PCAP timestamps: {e}")
        return df_packets
    
    logging.info(f"Starting label alignment ({scenario}): {len(labels_df)} labels -> {len(df_packets)} PCAP packets")
    
    # ===== Select matching strategy based on scenario =====
    
    if scenario == 'external':
        # External: Only timestamps available, use a wider time window
        logging.info("Using strategy: Time window matching only (External)")
        time_window = 5.0  # Expand to 5 seconds
        
        matched_packets = 0
        matched_labels = 0
        
        for idx, label_row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Aligning External"):
            try:
                label_time = label_row['Timestamp']
                attack_type = label_row['Attack']
                
                if pd.isna(label_time):
                    continue
                
                # Time window matching only
                time_mask = abs((df_packets['timestamp'] - label_time).dt.total_seconds()) <= time_window
                
                if time_mask.sum() > 0:
                    df_packets.loc[time_mask, 'attack_type'] = attack_type
                    matched_packets += time_mask.sum()
                    matched_labels += 1
            except Exception as e:
                continue
        
        logging.info(f"External matches: {matched_labels}/{len(labels_df)} labels, {matched_packets} packets")
    
    elif scenario in ['ied', 'scada']:
        # IED/SCADA: TransactionID and TargetIP available, use triplet matching
        logging.info(f"Using strategy: Triplet matching ({scenario.upper()})")
        
        # Validate required columns
        if 'TransactionID' not in labels_df.columns or 'TargetIP' not in labels_df.columns:
            logging.error(f"CSV for {scenario} is missing TransactionID or TargetIP columns")
            return df_packets
        
        matched_packets = 0
        matched_labels = 0
        
        for idx, label_row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Aligning {scenario.upper()}"):
            try:
                label_time = label_row['Timestamp']
                target_ip = label_row['TargetIP']
                txid = label_row['TransactionID']
                attack_type = label_row['Attack']
                
                # Check data validity
                if pd.isna(label_time) or pd.isna(target_ip) or pd.isna(txid):
                    continue
                
                # Triplet matching
                time_mask = abs((df_packets['timestamp'] - label_time).dt.total_seconds()) <= time_window
                ip_mask = df_packets['dst_ip'] == target_ip
                txid_mask = df_packets['transaction_id'] == txid
                
                match_mask = time_mask & ip_mask & txid_mask
                
                if match_mask.sum() > 0:
                    df_packets.loc[match_mask, 'attack_type'] = attack_type
                    matched_packets += match_mask.sum()
                    matched_labels += 1
            except Exception as e:
                continue
        
        logging.info(f"{scenario.upper()} matches: {matched_labels}/{len(labels_df)} labels, {matched_packets} packets")
    
    else:
        logging.warning(f"Unknown scenario: {scenario}, skipping label alignment")
    
    # Statistics
    attack_count = (df_packets['attack_type'] != 'unknown').sum()
    logging.info(f"Label alignment complete: {attack_count}/{len(df_packets)} packets labeled as attack ({attack_count/len(df_packets)*100:.1f}%)")
    
    return df_packets


# ============================================
# Main Process
# ============================================

try:
    # Load Phase 1 results
    with open(config.DATA_PROCESSED / 'stage1_results.pkl', 'rb') as f:
        stage1 = pickle.load(f)
    
    # ========== 1. Benign Data Sampling ==========
    logging.info("\n" + "=" * 60)
    logging.info("1. Processing Benign Data")
    logging.info("=" * 60)
    
    benign_pcaps = glob.glob(os.path.join(config.BENIGN_PCAP_DIR, "*.pcap"))
    logging.info(f"Number of Benign PCAP files: {len(benign_pcaps)}")
    
    # Target 200,000 entries, estimate number of files to process
    target_benign = 200000
    avg_packets_per_file = 1000000  # Estimate ~1M packets per file
    files_needed = max(1, int(target_benign / avg_packets_per_file * 1.2))  # Sample 20% extra
    
    sampled_benign_pcaps = np.random.choice(benign_pcaps, min(files_needed, len(benign_pcaps)), replace=False)
    
    benign_data = []
    for pcap_file in tqdm(sampled_benign_pcaps, desc="Benign"):
        df = parse_pcap_modbus(pcap_file, is_attack=False)
        if len(df) > 0:
            df['source'] = 'benign'
            df['label'] = 'normal'
            df['attack_type'] = 'normal'
            benign_data.append(df)
    
    if benign_data:
        df_benign_all = pd.concat(benign_data, ignore_index=True)
        # Randomly sample to target count
        if len(df_benign_all) > target_benign:
            df_benign = df_benign_all.sample(n=target_benign, random_state=42)
        else:
            df_benign = df_benign_all
        
        logging.info(f"✓ Benign data: {len(df_benign):,} entries")
    else:
        df_benign = pd.DataFrame()
        logging.warning("Benign data is empty")
    
    # ========== 2. External Attack Data ==========
    logging.info("\n" + "=" * 60)
    logging.info("2. Processing External Attack Data")
    logging.info("=" * 60)
    
    external_pcaps = glob.glob(os.path.join(config.EXTERNAL_PCAP_DIR, "*.pcap"))
    logging.info(f"Number of External PCAP files: {len(external_pcaps)}")
    
    external_data = []
    for pcap_file in tqdm(external_pcaps, desc="External"):
        df = parse_pcap_modbus(pcap_file, is_attack=True)
        if len(df) > 0:
            df['source'] = 'external'
            df['label'] = 'attack'
            external_data.append(df)
    
    if external_data:
        df_external_all = pd.concat(external_data, ignore_index=True)
        logging.info(f"External PCAP data: {len(df_external_all):,} entries")
        
        # Align labels
        df_external = align_labels(df_external_all, stage1['external_labels'], scenario='external')
        logging.info(f"✓ External data (after alignment): {len(df_external):,} entries")
    else:
        df_external = pd.DataFrame()
    
    # ========== 3. Compromised-IED Data ==========
    logging.info("\n" + "=" * 60)
    logging.info("3. Processing Compromised-IED Data")
    logging.info("=" * 60)
    
    ied_pcaps = glob.glob(os.path.join(config.IED_PCAP_DIR, "*.pcap"))
    logging.info(f"Number of IED PCAP files: {len(ied_pcaps)}")
    
    ied_data = []
    for pcap_file in tqdm(ied_pcaps, desc="IED"):
        df = parse_pcap_modbus(pcap_file, is_attack=True)
        if len(df) > 0:
            df['source'] = 'compromised-ied'
            df['label'] = 'attack'
            ied_data.append(df)
    
    if ied_data:
        df_ied_all = pd.concat(ied_data, ignore_index=True)
        # Align labels
        df_ied = align_labels(df_ied_all, stage1['ied_labels'], scenario='ied')
        logging.info(f"✓ IED data: {len(df_ied):,} entries")
    else:
        df_ied = pd.DataFrame()
    
    # ========== 4. Compromised-SCADA Data ==========
    logging.info("\n" + "=" * 60)
    logging.info("4. Processing Compromised-SCADA Data")
    logging.info("=" * 60)
    
    scada_pcaps = glob.glob(os.path.join(config.SCADA_PCAP_DIR, "*.pcap"))
    logging.info(f"Number of SCADA PCAP files: {len(scada_pcaps)}")
    
    # Target 200,000 entries
    target_scada = 200000
    
    scada_data = []
    for pcap_file in tqdm(scada_pcaps, desc="SCADA"):
        df = parse_pcap_modbus(pcap_file, is_attack=True)
        if len(df) > 0:
            df['source'] = 'compromised-scada'
            df['label'] = 'attack'
            scada_data.append(df)
    
    if scada_data:
        df_scada_all = pd.concat(scada_data, ignore_index=True)
        logging.info(f"SCADA PCAP data: {len(df_scada_all):,} entries")
        
        # Align labels
        df_scada_aligned = align_labels(df_scada_all, stage1['scada_labels'], scenario='scada')
        
        # Stratified sampling: by attack type
        attack_types = df_scada_aligned['attack_type'].unique()
        sampled_scada = []
        
        for attack_type in attack_types:
            df_type = df_scada_aligned[df_scada_aligned['attack_type'] == attack_type]
            n_samples = min(len(df_type), target_scada // len(attack_types))
            sampled = df_type.sample(n=n_samples, random_state=42)
            sampled_scada.append(sampled)
        
        df_scada = pd.concat(sampled_scada, ignore_index=True)
        logging.info(f"✓ SCADA data (after sampling): {len(df_scada):,} entries")
    else:
        df_scada = pd.DataFrame()
    
    # ========== 5. Merge All Data ==========
    logging.info("\n" + "=" * 60)
    logging.info("5. Merging Data")
    logging.info("=" * 60)
    
    all_data = []
    if len(df_benign) > 0:
        all_data.append(df_benign)
    if len(df_external) > 0:
        all_data.append(df_external)
    if len(df_ied) > 0:
        all_data.append(df_ied)
    if len(df_scada) > 0:
        all_data.append(df_scada)
    
    if all_data:
        df_final = pd.concat(all_data, ignore_index=True)
        
        logging.info(f"Total data volume: {len(df_final):,} entries")
        logging.info(f"\nSource distribution:")
        logging.info(df_final['source'].value_counts())
        logging.info(f"\nAttack type distribution:")
        logging.info(df_final['attack_type'].value_counts())
        
        # Save
        output_path = config.DATA_PROCESSED / "modbus_sampled_full.csv"
        df_final.to_csv(output_path, index=False)
        
        logging.info(f"\n✓ Data saved to: {output_path}")
        logging.info(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    else:
        logging.error("No data to merge")
    
    logging.info("\n" + "=" * 60)
    logging.info("✓ Full sampling completed successfully!")
    logging.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise

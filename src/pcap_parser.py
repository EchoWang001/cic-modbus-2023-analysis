# -*- coding: utf-8 -*-
"""
PCAP Parsing Module - Phase 1 Step 1

Function: Parse Modbus PCAP files and extract packet-level data.
Author: AI Assistant
Date: 2026-01-09
"""

import os
import glob
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scapy.all import rdpcap, TCP, IP
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModbusPacketParser:
    """Modbus PCAP file parser"""
    
    # Modbus TCP port
    MODBUS_PORT = 502
    
    # Valid function codes
    VALID_FUNCTION_CODES = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x0F, 0x10, 0x17]
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the parser
        
        Args:
            strict_mode: If True, skip invalid data instead of using default values.
        """
        self.strict_mode = strict_mode
        self.stats = {
            'total_packets': 0,
            'valid_packets': 0,
            'skipped_packets': 0,
            'error_packets': 0
        }
    
    def parse_single_pcap(self, pcap_path: str, scenario: str) -> pd.DataFrame:
        """
        Parse a single PCAP file
        
        Args:
            pcap_path: Path to the PCAP file
            scenario: Scenario identifier (benign/external/scada/ied)
            
        Returns:
            DataFrame containing parsed Modbus packets
        """
        packets_data = []
        pcap_filename = os.path.basename(pcap_path)
        
        try:
            packets = rdpcap(pcap_path)
            self.stats['total_packets'] += len(packets)
            
            for pkt in packets:
                packet_info = self._parse_packet(pkt, pcap_filename, scenario)
                if packet_info is not None:
                    packets_data.append(packet_info)
                    self.stats['valid_packets'] += 1
                else:
                    self.stats['skipped_packets'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to read PCAP file: {pcap_path} - {e}")
            self.stats['error_packets'] += 1
            return pd.DataFrame()
        
        return pd.DataFrame(packets_data)
    
    def _parse_packet(self, pkt, pcap_filename: str, scenario: str) -> Optional[Dict]:
        """
        Parse a single packet
        
        Args:
            pkt: Scapy packet object
            pcap_filename: Source filename
            scenario: Scenario identifier
            
        Returns:
            Dictionary of parsed fields, or None if invalid
        """
        # Check TCP layer
        if TCP not in pkt:
            return None
        
        # Check Modbus port
        if pkt[TCP].sport != self.MODBUS_PORT and pkt[TCP].dport != self.MODBUS_PORT:
            return None
        
        # Check IP layer
        if IP not in pkt:
            return None
        
        try:
            # Extract timestamp
            try:
                timestamp = datetime.fromtimestamp(float(pkt.time))
                # Validate timestamp (dataset should be between 2020-2026)
                if timestamp.year < 2020 or timestamp.year > 2026:
                    return None
            except (ValueError, OSError, OverflowError, TypeError):
                return None
            
            # Extract network layer information
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            
            # Extract Modbus application layer data
            payload = bytes(pkt[TCP].payload)
            
            # MBAP Header is at least 7 bytes + 1 byte Function Code
            if len(payload) < 8:
                return None
            
            # Parse MBAP Header
            transaction_id = int.from_bytes(payload[0:2], byteorder='big')
            protocol_id = int.from_bytes(payload[2:4], byteorder='big')
            length = int.from_bytes(payload[4:6], byteorder='big')
            unit_id = payload[6]
            function_code = payload[7]
            
            # Validate Modbus protocol ID (should be 0)
            if protocol_id != 0:
                return None
            
            # Validate function code (strict mode)
            if self.strict_mode and function_code not in self.VALID_FUNCTION_CODES:
                return None
            
            # Parse function code specific fields
            start_address = None
            quantity = None
            
            try:
                if function_code in [0x01, 0x02, 0x03, 0x04]:  # Read operations
                    if len(payload) >= 12:
                        start_address = int.from_bytes(payload[8:10], byteorder='big')
                        quantity = int.from_bytes(payload[10:12], byteorder='big')
                
                elif function_code in [0x05, 0x06]:  # Write Single
                    if len(payload) >= 12:
                        start_address = int.from_bytes(payload[8:10], byteorder='big')
                        # For single write, quantity is 1
                        quantity = 1
                
                elif function_code in [0x0F, 0x10]:  # Write Multiple
                    if len(payload) >= 12:
                        start_address = int.from_bytes(payload[8:10], byteorder='big')
                        quantity = int.from_bytes(payload[10:12], byteorder='big')
            except (IndexError, ValueError):
                pass  # These fields are optional, failure doesn't affect overall parsing
            
            # Build data record
            packet_info = {
                'timestamp': timestamp,
                'pcap_file': pcap_filename,
                'scenario': scenario,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'transaction_id': transaction_id,
                'unit_id': unit_id,
                'function_code': function_code,
                'start_address': start_address,
                'quantity': quantity,
                'payload_size': len(payload)
            }
            
            return packet_info
            
        except Exception as e:
            return None
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_packets': 0,
            'valid_packets': 0,
            'skipped_packets': 0,
            'error_packets': 0
        }
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return self.stats.copy()


def scan_pcap_files(base_path: str, scenario: str) -> List[Tuple[str, str]]:
    """
    Scan a directory for PCAP files
    
    Args:
        base_path: Directory containing PCAP files
        scenario: Scenario identifier
        
    Returns:
        List of tuples (file_path, scenario)
    """
    if not os.path.exists(base_path):
        logger.warning(f"Path does not exist: {base_path}")
        return []
    
    pcap_files = glob.glob(os.path.join(base_path, "*.pcap"))
    logger.info(f"Found {len(pcap_files)} PCAP files in {scenario} directory")
    
    return [(f, scenario) for f in pcap_files]


def parse_all_pcaps(pcap_list: List[Tuple[str, str]], 
                    parser: ModbusPacketParser = None,
                    show_progress: bool = True) -> pd.DataFrame:
    """
    Parse multiple PCAP files in batch
    
    Args:
        pcap_list: List of (file_path, scenario) tuples
        parser: Parser instance, creates a new one if None
        show_progress: Whether to show a progress bar
        
    Returns:
        Merged DataFrame
    """
    if parser is None:
        parser = ModbusPacketParser()
    
    parser.reset_stats()
    all_data = []
    
    iterator = tqdm(pcap_list, desc="Parsing PCAP files") if show_progress else pcap_list
    
    for pcap_path, scenario in iterator:
        df = parser.parse_single_pcap(pcap_path, scenario)
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        logger.warning("No data parsed")
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    # Output statistics
    stats = parser.get_stats()
    logger.info(f"Parsing complete:")
    logger.info(f"  - Total packets: {stats['total_packets']:,}")
    logger.info(f"  - Valid Modbus packets: {stats['valid_packets']:,}")
    logger.info(f"  - Skipped packets: {stats['skipped_packets']:,}")
    logger.info(f"  - Efficiency: {stats['valid_packets']/max(1,stats['total_packets'])*100:.1f}%")
    
    return result


def save_packets_to_parquet(df: pd.DataFrame, output_path: str):
    """
    Save data to Parquet format
    
    Args:
        df: Data DataFrame
        output_path: Output path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    # Output file info
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    logger.info(f"Data saved to: {output_path}")
    logger.info(f"  - Record count: {len(df):,}")
    logger.info(f"  - File size: {file_size:.2f} MB")


def load_packets_from_parquet(input_path: str) -> pd.DataFrame:
    """
    Load data from Parquet format
    
    Args:
        input_path: Input path
        
    Returns:
        DataFrame
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = pd.read_parquet(input_path, engine='pyarrow')
    logger.info(f"Data loaded from: {input_path}, Record count: {len(df):,}")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("PCAP Parsing Module Test")
    print("=" * 60)
    
    # Import configuration
    import config
    
    # Scan files
    benign_files = scan_pcap_files(config.BENIGN_PCAP_DIR, 'benign')
    external_files = scan_pcap_files(config.EXTERNAL_PCAP_DIR, 'external')
    scada_files = scan_pcap_files(config.SCADA_PCAP_DIR, 'scada')
    ied_files = scan_pcap_files(config.IED_PCAP_DIR, 'ied')
    
    print(f"\nPCAP File Statistics:")
    print(f"  - Benign: {len(benign_files)}")
    print(f"  - External: {len(external_files)}")
    print(f"  - SCADA: {len(scada_files)}")
    print(f"  - IED: {len(ied_files)}")
    print(f"  - Total: {len(benign_files) + len(external_files) + len(scada_files) + len(ied_files)}")

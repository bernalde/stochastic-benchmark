#!/usr/bin/env python3
"""
Transfer simulation data files from QAOA-Parameter-Setting repository to R3R/heavy_hex subdirectories.
Organizes files by method (F, FA, I, TQA, TS) into corresponding topology folders.
"""

import os
import shutil
import re
from pathlib import Path
from collections import defaultdict
import json
import argparse

def extract_info_from_filename(filename):
    """
    Extract instance, N, method from simulation filename.
    
    Supports both patterns:
    - R3R: YYYYMMDD_HHMMSS_###N##R#R_MC_METHOD_..._.json
    - heavy_hex: YYYYMMDD_HHMMSS_###N##HH##_MC_METHOD_..._.json
    """
    basename = os.path.basename(filename)
    
    # Try R3R pattern first: ###N##R#R
    match = re.search(r'(\d+)N(\d+)R(\d+)R', basename)
    if match:
        instance_id = match.group(1)
        n_nodes = int(match.group(2))
        regularity = int(match.group(3))
        
        # Extract method
        method_match = re.search(r'_(?:MC|MIS)_([A-Z]+)_', basename)
        method = method_match.group(1) if method_match else None
        
        return instance_id, n_nodes, regularity, method
    
    # Try heavy_hex pattern: ###N##HH##
    match = re.search(r'(\d+)N(\d+)HH\d+', basename)
    if match:
        instance_id = match.group(1)
        n_nodes = int(match.group(2))
        regularity = None  # Heavy hex doesn't have regularity
        
        # Extract method
        method_match = re.search(r'_(?:MC|MIS)_([A-Z]+)_', basename)
        method = method_match.group(1) if method_match else None
        
        return instance_id, n_nodes, regularity, method
    
    return None, None, None, None

def transfer_simulation_files(source_dir, dest_base='R3R', topology='R3R', n_filter=None, skip_existing=True):
    """
    Transfer simulation files from source to topology subdirectories.
    
    Args:
        source_dir: Source directory containing simulation JSON files
        dest_base: Base directory for structure (default: topology name)
        topology: Topology type - 'R3R' or 'heavy_hex'
        n_filter: If specified, only transfer files for this N value (e.g., 40)
        skip_existing: If True, skip files that already exist in destination
    """
    # Methods we care about
    target_methods = ['F', 'FA', 'I', 'TQA', 'TS', 'RTS']
    
    # Statistics
    stats = defaultdict(lambda: {'transferred': 0, 'skipped': 0, 'errors': 0})
    
    # Find all JSON files
    source_path = Path(source_dir)
    json_files = list(source_path.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files in {source_dir}")
    
    # Filter and organize
    for json_file in json_files:
        instance_id, n_nodes, regularity, method = extract_info_from_filename(str(json_file))
        
        # Skip if extraction failed
        if not all([instance_id, n_nodes, method]):
            continue
        
        # Filter by N if specified
        if n_filter is not None and n_nodes != n_filter:
            continue
        
        # For R3R, filter by regularity (only 3-regular)
        # For heavy_hex, regularity is None so this check is skipped
        if topology == 'R3R' and regularity != 3:
            continue
        
        # Filter by method
        if method not in target_methods:
            continue
        
        # Determine destination directory
        dest_dir = os.path.join(dest_base, method)
        
        # Create directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        
        # Destination file path
        dest_file = os.path.join(dest_dir, json_file.name)
        
        # Check if file already exists
        if skip_existing and os.path.exists(dest_file):
            stats[method]['skipped'] += 1
            continue
        
        # Copy file
        try:
            shutil.copy2(json_file, dest_file)
            stats[method]['transferred'] += 1
        except Exception as e:
            print(f"Error copying {json_file.name}: {e}")
            stats[method]['errors'] += 1
    
    return stats

def print_summary(stats, n_filter=None):
    """Print transfer summary statistics."""
    print("\n" + "="*80)
    print("Simulation Data Transfer Summary")
    if n_filter:
        print(f"(Filtered to N={n_filter})")
    print("="*80)
    
    total_transferred = 0
    total_skipped = 0
    total_errors = 0
    
    for method in sorted(stats.keys()):
        method_stats = stats[method]
        transferred = method_stats['transferred']
        skipped = method_stats['skipped']
        errors = method_stats['errors']
        
        total_transferred += transferred
        total_skipped += skipped
        total_errors += errors
        
        print(f"\n{method}:")
        print(f"  Transferred: {transferred}")
        print(f"  Skipped:     {skipped} (already exist)")
        if errors > 0:
            print(f"  Errors:      {errors}")
    
    print(f"\n{'='*80}")
    print(f"Total:")
    print(f"  Transferred: {total_transferred}")
    print(f"  Skipped:     {total_skipped}")
    if total_errors > 0:
        print(f"  Errors:      {total_errors}")
    print("="*80)

def verify_hamiltonians(dest_base='R3R', n_nodes=40):
    """
    Verify that transferred simulation files contain Hamiltonians.
    Check a few random files from each method.
    """
    print(f"\n{'='*80}")
    print(f"Verifying Hamiltonians in Transferred Files (N={n_nodes})")
    print("="*80)
    
    target_methods = ['F', 'FA', 'I', 'TQA', 'TS']
    
    for method in target_methods:
        method_dir = os.path.join(dest_base, method)
        if not os.path.exists(method_dir):
            print(f"\n{method}: Directory not found")
            continue
        
        # Find N40 files in this directory
        json_files = [f for f in os.listdir(method_dir) 
                     if f.endswith('.json') and f'N{n_nodes}R3R' in f]
        
        if not json_files:
            print(f"\n{method}: No N{n_nodes} files found")
            continue
        
        # Check first file
        sample_file = os.path.join(method_dir, json_files[0])
        try:
            with open(sample_file, 'r') as f:
                data = json.load(f)
            
            has_cost_op = 'cost_operator' in data
            has_pauli_strings = False
            if has_cost_op and isinstance(data['cost_operator'], list):
                has_pauli_strings = len(data['cost_operator']) > 0
            
            print(f"\n{method}: {len(json_files)} files")
            print(f"  Sample: {json_files[0]}")
            print(f"  Has cost_operator: {has_cost_op}")
            print(f"  Has Pauli strings: {has_pauli_strings}")
            if has_pauli_strings:
                print(f"  Number of terms: {len(data['cost_operator'])}")
        except Exception as e:
            print(f"\n{method}: Error reading {json_files[0]}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transfer simulation data from QAOA-Parameter-Setting repository',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--topology',
        choices=['R3R', 'heavy_hex'],
        default='R3R',
        help='Topology type: R3R (random 3-regular) or heavy_hex'
    )
    parser.add_argument(
        '--n-filter',
        type=int,
        default=None,
        help='Only transfer files for this N value (e.g., --n-filter 40)'
    )
    
    args = parser.parse_args()
    
    # Set paths based on topology
    if args.topology == 'R3R':
        source_dir = '/mnt/c/Users/rames102/Desktop/QAOA-Parameter-Setting/data/training/random_regular'
    else:  # heavy_hex
        source_dir = '/mnt/c/Users/rames102/Desktop/QAOA-Parameter-Setting/data/training/heavy_hex'
    
    dest_base = args.topology
    
    print("="*80)
    print(f"SIMULATION DATA TRANSFER - {args.topology.upper()}")
    print("="*80)
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_base}/{{F,FA,I,TQA,TS,RTS}}/")
    if args.n_filter:
        print(f"Filter: N={args.n_filter}")
    if args.topology == 'R3R':
        print("Filter: 3-regular graphs only")
    print(f"Target methods: F, FA, I, TQA, TS, RTS")
    print("="*80)
    
    # Confirm before proceeding
    print("\nThis will copy simulation files to subdirectories.")
    print("Existing files will be skipped (not overwritten).")
    response = input("\nProceed? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Transfer cancelled.")
        exit(0)
    
    # Execute transfer
    print("\nTransferring files...")
    stats = transfer_simulation_files(
        source_dir, 
        dest_base=dest_base,
        topology=args.topology,
        n_filter=args.n_filter
    )
    
    # Print summary
    print_summary(stats, n_filter=args.n_filter)
    
    # Verify
    if args.n_filter:
        verify_hamiltonians(dest_base, n_nodes=args.n_filter)
    
    print("\n✓ Transfer complete!")
    print(f"\nSimulation files available in: {dest_base}/{{F,FA,I,TQA,TS,RTS}}/")


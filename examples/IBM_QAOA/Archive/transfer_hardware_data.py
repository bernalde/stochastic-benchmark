#!/usr/bin/env python3
"""
Transfer and organize hardware QAOA data and minmax cuts from QAOA-Parameter-Setting repository.
Organizes files by node count (N) into subdirectories.
Skips duplicate files that already exist in destination.
"""

import os
import shutil
import re
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
import argparse

# Source directories will be set based on topology
HARDWARE_SOURCE = None
MINMAX_SOURCE = None

# Destination directories (will be set based on topology)
DEST_BASE = None
HARDWARE_DEST = None
MINMAX_DEST = None


def set_topology_paths(topology: str):
    """Set source and destination paths based on topology."""
    global HARDWARE_SOURCE, MINMAX_SOURCE, DEST_BASE, HARDWARE_DEST, MINMAX_DEST
    
    base_path = "/mnt/c/Users/rames102/Desktop/QAOA-Parameter-Setting/data"
    
    if topology == "R3R":
        HARDWARE_SOURCE = f"{base_path}/hardware/random_regular"
        MINMAX_SOURCE = f"{base_path}/minmax_cuts/random_regular"
        DEST_BASE = "R3R"
    elif topology == "heavy_hex":
        HARDWARE_SOURCE = f"{base_path}/hardware/heavy_hex"
        MINMAX_SOURCE = f"{base_path}/minmax_cuts/heavy_hex"
        DEST_BASE = "heavy_hex"
    else:
        raise ValueError(f"Unknown topology: {topology}. Use 'R3R' or 'heavy_hex'")
    
    HARDWARE_DEST = f"{DEST_BASE}/Hardware"
    MINMAX_DEST = f"{DEST_BASE}/minmax_cuts"


def create_backup():
    """Create timestamped backup of destination directory."""
    if not os.path.exists(DEST_BASE):
        print(f"⚠️  Destination directory {DEST_BASE}/ not found, skipping backup")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{DEST_BASE}_backup_{timestamp}"
    
    print(f"📦 Creating backup: {backup_dir}/")
    shutil.copytree(DEST_BASE, backup_dir)
    print(f"✓ Backup created successfully")
    return backup_dir


def extract_n_from_hardware_filename(filename: str) -> Tuple[str, int]:
    """Extract instance ID and N value from hardware filename.
    
    Supports both patterns:
    - R3R: 000N40R3R_10_hash.json
    - heavy_hex: 000N105HH72_10_hash.json
    """
    # Try R3R pattern first: 000N40R3R_...
    match = re.match(r'(\d+)N(\d+)R\dR', filename)
    if match:
        instance_id = match.group(1)
        n_value = int(match.group(2))
        return instance_id, n_value
    
    # Try heavy_hex pattern: 000N105HH72_...
    match = re.match(r'(\d+)N(\d+)HH\d+', filename)
    if match:
        instance_id = match.group(1)
        n_value = int(match.group(2))
        return instance_id, n_value
    
    return None, None


def extract_n_from_minmax_filename(filename: str) -> Tuple[str, int]:
    """Extract instance ID and N value from minmax cuts filename.
    
    Supports both patterns:
    - R3R: 000_40nodes_random3regular_maxmin_cut.json
    - heavy_hex: 000_7_3_heavyhex_144nodes_weighted_maxmin_cut.json
    """
    # Try R3R pattern: 000_40nodes_random3regular_maxmin_cut.json
    match = re.match(r'(\d+)_(\d+)nodes_random(\d)regular', filename)
    if match:
        instance_id = match.group(1)
        n_value = int(match.group(2))
        regularity = int(match.group(3))
        # Only process 3-regular graphs
        if regularity == 3:
            return instance_id, n_value
    
    # Try heavy_hex pattern: 000_7_3_heavyhex_144nodes_weighted_maxmin_cut.json
    match = re.match(r'(\d+)_\d+_\d+_heavyhex_(\d+)nodes', filename)
    if match:
        instance_id = match.group(1)
        n_value = int(match.group(2))
        return instance_id, n_value
    
    return None, None


def transfer_hardware_files(skip_duplicates: bool = True) -> Dict[int, Dict[str, int]]:
    """
    Transfer hardware files organized by N value.
    
    Returns:
        Dictionary mapping N value to stats: {N: {'transferred': count, 'skipped': count}}
    """
    print("\n" + "="*80)
    print("HARDWARE FILES TRANSFER")
    print("="*80)
    
    if not os.path.exists(HARDWARE_SOURCE):
        print(f"❌ Source directory not found: {HARDWARE_SOURCE}")
        return {}
    
    # Create base hardware directory
    os.makedirs(HARDWARE_DEST, exist_ok=True)
    
    # Get all hardware JSON files
    source_files = glob.glob(os.path.join(HARDWARE_SOURCE, "*.json"))
    print(f"\n📂 Found {len(source_files)} files in source")
    
    # Group files by N value
    files_by_n: Dict[int, List[str]] = {}
    for filepath in source_files:
        filename = os.path.basename(filepath)
        instance_id, n_value = extract_n_from_hardware_filename(filename)
        
        if n_value is not None:
            if n_value not in files_by_n:
                files_by_n[n_value] = []
            files_by_n[n_value].append(filepath)
    
    print(f"\n📊 Files grouped by N value:")
    for n in sorted(files_by_n.keys()):
        print(f"  N{n}: {len(files_by_n[n])} files")
    
    # Transfer files for each N value
    stats = {}
    for n_value in sorted(files_by_n.keys()):
        n_dest_dir = os.path.join(HARDWARE_DEST, f"N{n_value}")
        os.makedirs(n_dest_dir, exist_ok=True)
        
        transferred = 0
        skipped = 0
        
        print(f"\n📁 Processing N{n_value}/ ({len(files_by_n[n_value])} files)")
        
        for source_file in files_by_n[n_value]:
            filename = os.path.basename(source_file)
            dest_file = os.path.join(n_dest_dir, filename)
            
            if skip_duplicates and os.path.exists(dest_file):
                skipped += 1
                print(f"  ⊘ SKIP: {filename} (already exists)")
            else:
                shutil.copy2(source_file, dest_file)
                transferred += 1
                print(f"  ✓ COPY: {filename}")
        
        stats[n_value] = {'transferred': transferred, 'skipped': skipped}
        print(f"  📊 N{n_value} summary: {transferred} transferred, {skipped} skipped")
    
    return stats


def transfer_minmax_files(skip_duplicates: bool = True) -> Dict[int, Dict[str, int]]:
    """
    Transfer minmax cuts files organized by N value.
    
    Returns:
        Dictionary mapping N value to stats: {N: {'transferred': count, 'skipped': count}}
    """
    print("\n" + "="*80)
    print("MINMAX CUTS TRANSFER")
    print("="*80)
    
    if not os.path.exists(MINMAX_SOURCE):
        print(f"❌ Source directory not found: {MINMAX_SOURCE}")
        return {}
    
    # Create base minmax directory
    os.makedirs(MINMAX_DEST, exist_ok=True)
    
    # Get all minmax JSON files
    source_files = glob.glob(os.path.join(MINMAX_SOURCE, "*.json"))
    print(f"\n📂 Found {len(source_files)} files in source")
    
    # Group files by N value (only 3-regular graphs)
    files_by_n: Dict[int, List[str]] = {}
    for filepath in source_files:
        filename = os.path.basename(filepath)
        instance_id, n_value = extract_n_from_minmax_filename(filename)
        
        if n_value is not None:
            if n_value not in files_by_n:
                files_by_n[n_value] = []
            files_by_n[n_value].append(filepath)
    
    print(f"\n📊 Files grouped by N value (3-regular only):")
    for n in sorted(files_by_n.keys()):
        print(f"  N{n}: {len(files_by_n[n])} files")
    
    # Transfer files for each N value
    stats = {}
    for n_value in sorted(files_by_n.keys()):
        n_dest_dir = os.path.join(MINMAX_DEST, f"N{n_value}")
        os.makedirs(n_dest_dir, exist_ok=True)
        
        transferred = 0
        skipped = 0
        
        print(f"\n📁 Processing N{n_value}/ ({len(files_by_n[n_value])} files)")
        
        for source_file in files_by_n[n_value]:
            filename = os.path.basename(source_file)
            dest_file = os.path.join(n_dest_dir, filename)
            
            if skip_duplicates and os.path.exists(dest_file):
                skipped += 1
                print(f"  ⊘ SKIP: {filename} (already exists)")
            else:
                shutil.copy2(source_file, dest_file)
                transferred += 1
                print(f"  ✓ COPY: {filename}")
        
        stats[n_value] = {'transferred': transferred, 'skipped': skipped}
        print(f"  📊 N{n_value} summary: {transferred} transferred, {skipped} skipped")
    
    return stats


def validate_instance_consistency() -> Dict[int, Dict[str, Set[str]]]:
    """
    Validate that hardware and minmax cuts have consistent instance IDs for each N value.
    
    Returns:
        Dictionary mapping N value to instance ID sets: {N: {'hardware': set, 'minmax': set}}
    """
    print("\n" + "="*80)
    print("INSTANCE ID VALIDATION")
    print("="*80)
    
    validation_results = {}
    
    # Get all N-based subdirectories from hardware
    hardware_n_dirs = glob.glob(os.path.join(HARDWARE_DEST, "N*"))
    hardware_n_values = sorted([int(os.path.basename(d)[1:]) for d in hardware_n_dirs if os.path.isdir(d)])
    
    # Get all N-based subdirectories from minmax_cuts
    minmax_n_dirs = glob.glob(os.path.join(MINMAX_DEST, "N*"))
    minmax_n_values = sorted([int(os.path.basename(d)[1:]) for d in minmax_n_dirs if os.path.isdir(d)])
    
    # Check all N values present in either hardware or minmax
    all_n_values = sorted(set(hardware_n_values + minmax_n_values))
    
    for n_value in all_n_values:
        print(f"\n📊 Validating N{n_value}:")
        
        # Get hardware instances for this N
        hardware_instances = set()
        hardware_dir = os.path.join(HARDWARE_DEST, f"N{n_value}")
        if os.path.exists(hardware_dir):
            for filename in os.listdir(hardware_dir):
                if filename.endswith('.json'):
                    instance_id, _ = extract_n_from_hardware_filename(filename)
                    if instance_id:
                        hardware_instances.add(instance_id)
        
        # Get minmax instances for this N
        minmax_instances = set()
        minmax_dir = os.path.join(MINMAX_DEST, f"N{n_value}")
        if os.path.exists(minmax_dir):
            for filename in os.listdir(minmax_dir):
                if filename.endswith('.json'):
                    instance_id, _ = extract_n_from_minmax_filename(filename)
                    if instance_id:
                        minmax_instances.add(instance_id)
        
        validation_results[n_value] = {
            'hardware': hardware_instances,
            'minmax': minmax_instances
        }
        
        # Report findings
        print(f"  Hardware instances: {sorted(hardware_instances)}")
        print(f"  Minmax instances:   {sorted(minmax_instances)}")
        
        # Check for missing minmax files
        missing_minmax = hardware_instances - minmax_instances
        if missing_minmax:
            print(f"  ⚠️  WARNING: Hardware instances missing minmax cuts: {sorted(missing_minmax)}")
        
        # Check for unused minmax files
        unused_minmax = minmax_instances - hardware_instances
        if unused_minmax:
            print(f"  ℹ️  INFO: Minmax cuts without hardware data: {sorted(unused_minmax)}")
        
        if hardware_instances == minmax_instances:
            print(f"  ✓ Perfect match: {len(hardware_instances)} instances")
    
    return validation_results


def print_final_summary(hardware_stats, minmax_stats, validation_results):
    """Print final transfer summary."""
    print("\n" + "="*80)
    print("TRANSFER SUMMARY")
    print("="*80)
    
    # Hardware summary
    print("\n📁 Hardware Files:")
    total_hw_transferred = sum(s['transferred'] for s in hardware_stats.values())
    total_hw_skipped = sum(s['skipped'] for s in hardware_stats.values())
    for n_value in sorted(hardware_stats.keys()):
        stats = hardware_stats[n_value]
        print(f"  N{n_value}: {stats['transferred']} transferred, {stats['skipped']} skipped")
    print(f"  TOTAL: {total_hw_transferred} transferred, {total_hw_skipped} skipped")
    
    # Minmax summary
    print("\n📁 Minmax Cuts:")
    total_mm_transferred = sum(s['transferred'] for s in minmax_stats.values())
    total_mm_skipped = sum(s['skipped'] for s in minmax_stats.values())
    for n_value in sorted(minmax_stats.keys()):
        stats = minmax_stats[n_value]
        print(f"  N{n_value}: {stats['transferred']} transferred, {stats['skipped']} skipped")
    print(f"  TOTAL: {total_mm_transferred} transferred, {total_mm_skipped} skipped")
    
    # Validation summary
    print("\n✓ Instance Validation:")
    all_valid = True
    for n_value in sorted(validation_results.keys()):
        hw_instances = validation_results[n_value]['hardware']
        mm_instances = validation_results[n_value]['minmax']
        missing = hw_instances - mm_instances
        if missing:
            print(f"  N{n_value}: ⚠️  {len(missing)} hardware instances missing minmax cuts")
            all_valid = False
        else:
            print(f"  N{n_value}: ✓ All {len(hw_instances)} hardware instances have minmax cuts")
    
    if all_valid:
        print("\n✅ All validation checks passed!")
    else:
        print("\n⚠️  Some validation warnings - review above")


def main():
    """Main transfer execution."""
    parser = argparse.ArgumentParser(
        description='Transfer hardware data and minmax cuts from QAOA-Parameter-Setting repository',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--topology',
        choices=['R3R', 'heavy_hex'],
        default='R3R',
        help='Topology type: R3R (random 3-regular) or heavy_hex'
    )
    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip creating backup before transfer'
    )
    
    args = parser.parse_args()
    
    # Set paths based on topology
    set_topology_paths(args.topology)
    
    print("="*80)
    print(f"HARDWARE DATA & MINMAX CUTS TRANSFER - {args.topology.upper()}")
    print("="*80)
    print(f"Source (Hardware): {HARDWARE_SOURCE}")
    print(f"Source (Minmax):   {MINMAX_SOURCE}")
    print(f"Destination:       {DEST_BASE}/")
    print("="*80)
    
    print("\nThis script will:")
    if not args.skip_backup:
        print(f"  1. Create backup of existing {DEST_BASE}/ directory")
    else:
        print("  1. Skip backup (--skip-backup)")
    print("  2. Transfer hardware files organized by N value")
    print("  3. Transfer minmax cuts files organized by N value")
    print("  4. Skip duplicate files (preserve existing data)")
    print("  5. Validate instance ID consistency")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Transfer cancelled.")
        return
    
    # Step 1: Backup
    backup_dir = None
    if not args.skip_backup:
        backup_dir = create_backup()
    
    # Step 2: Transfer hardware files
    hardware_stats = transfer_hardware_files(skip_duplicates=True)
    
    # Step 3: Transfer minmax cuts
    minmax_stats = transfer_minmax_files(skip_duplicates=True)
    
    # Step 4: Validate consistency
    validation_results = validate_instance_consistency()
    
    # Step 5: Summary
    print_final_summary(hardware_stats, minmax_stats, validation_results)
    
    if backup_dir:
        print(f"\n💾 Backup saved to: {backup_dir}/")
    
    print(f"\n📂 Hardware files: {HARDWARE_DEST}/N*/")
    print(f"📂 Minmax cuts:    {MINMAX_DEST}/N*/")
    print("\n✅ Transfer complete!")


if __name__ == "__main__":
    main()

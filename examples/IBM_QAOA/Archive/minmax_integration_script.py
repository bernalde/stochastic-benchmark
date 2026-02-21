#!/usr/bin/env python
"""
Quick test script to verify minmax cuts integration works correctly.
"""
import sys
sys.path.append('../../src')
import ibm_qaoa_processing

# Test loading minmax cuts
print("=" * 60)
print("Testing minmax cuts loading...")
print("=" * 60)

minmax_data = ibm_qaoa_processing.load_minmax_cuts('R3R/minmax_cuts')
print(f"\nLoaded minmax data for {len(minmax_data)} instances")

# Show sample data
if minmax_data:
    sample_id = list(minmax_data.keys())[0]
    print(f"\nSample instance {sample_id}:")
    print(f"  min_cut: {minmax_data[sample_id]['min_cut']}")
    print(f"  max_cut: {minmax_data[sample_id]['max_cut']}")
    print(f"  sum_weights: {minmax_data[sample_id]['sum_weights']}")
    
    # Test approximation ratio calculation
    print(f"\n{'=' * 60}")
    print("Testing approximation ratio calculation...")
    print("=" * 60)
    
    # Test with sample energy value
    energy = -6.5  # Sample energy
    ratio = ibm_qaoa_processing.maxcut_approximation_ratio(
        energy,
        minmax_data[sample_id]['min_cut'],
        minmax_data[sample_id]['max_cut'],
        minmax_data[sample_id]['sum_weights']
    )
    print(f"\nFor instance {sample_id} with energy {energy}:")
    print(f"  cut_val = {energy} + 0.5 * {minmax_data[sample_id]['sum_weights']} = {energy + 0.5 * minmax_data[sample_id]['sum_weights']}")
    print(f"  Approximation ratio = {ratio:.4f}")

print(f"\n{'=' * 60}")
print("Testing FA data processing with minmax cuts...")
print("=" * 60)

# Process a small subset of FA data to verify integration
sb, agg_df = ibm_qaoa_processing.process_qaoa_data(
    json_pattern="R3R/FA/*_000N10R3R_*.json",  # Just instance 000
    output_dir="exp_raw"
)

print(f"\nProcessed {len(agg_df)} rows")
print(f"\nApproximation Ratio statistics:")
print(f"  Non-NaN values: {agg_df['Approximation_Ratio'].notna().sum()}/{len(agg_df)}")
print(f"  Mean: {agg_df['Approximation_Ratio'].mean():.4f}")
print(f"  Min: {agg_df['Approximation_Ratio'].min():.4f}")
print(f"  Max: {agg_df['Approximation_Ratio'].max():.4f}")

print("\nSample rows:")
print(agg_df[['instance', 'p', 'Energy', 'Approximation_Ratio', 'MeanTime', 'optimized']].head(10))

print(f"\n{'=' * 60}")
print("✓ Integration test completed successfully!")
print("=" * 60)

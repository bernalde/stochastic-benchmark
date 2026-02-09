#!/usr/bin/env python3
"""
Simple runner script to test IBM QAOA data processing
"""

import os
import sys

# Add the script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from ibm_qaoa_processing import main

if __name__ == '__main__':
    print("Starting IBM QAOA data processing...")
    try:
        sb = main()
        print("✅ Processing completed successfully!")
        print(f"Results saved to: {sb.here}")
        
        # Quick summary
        if hasattr(sb, 'bs_results'):
            print(f"Bootstrap results shape: {sb.bs_results.shape if hasattr(sb.bs_results, 'shape') else 'N/A'}")
        if hasattr(sb, 'interp_results'):
            print(f"Interpolation results shape: {sb.interp_results.shape if hasattr(sb.interp_results, 'shape') else 'N/A'}")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

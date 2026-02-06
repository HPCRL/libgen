#!/usr/bin/env python3
"""Test script to discover nvMatmulHeuristics API"""

try:
    from nvMatmulHeuristics import (
        NvMatmulHeuristicsInterface,
        NvMatmulHeuristicsNvidiaGpu,
    )
    
    print("Successfully imported nvMatmulHeuristics")
    print("\nAvailable methods in NvMatmulHeuristicsInterface:")
    interface = NvMatmulHeuristicsInterface()
    methods = [m for m in dir(interface) if not m.startswith('_')]
    for m in sorted(methods):
        print(f"  - {m}")
    
    print("\nAvailable GPU types:")
    gpus = [g for g in dir(NvMatmulHeuristicsNvidiaGpu) if not g.startswith('_')]
    for g in sorted(gpus):
        print(f"  - {g}")
    
    print("\nTrying to query a simple configuration...")
    
    # Try without hw_descriptor first
    try:
        layout = 1  # NT
        interface.loadInternalDiscoverySet(layout)
        print("  ✓ loadInternalDiscoverySet(layout) works")
        
        # Create hardware descriptor
        hw_desc = interface.createHardwareDescriptor()
        interface.setHardwarePredefinedGpu(hw_desc, NvMatmulHeuristicsNvidiaGpu.RTX_3090)
        print("  ✓ Hardware descriptor created for RTX_3090")
        
        # Try get_with_mnk with proper signature: (m, n, k, layout, count, hw_descriptor)
        configs = interface.get_with_mnk(
            64, 64, 4096,  # m, n, k
            layout,         # matmulLayout
            5,             # count
            hw_desc,       # hardware_descriptor
        )
        print(f"  ✓ get_with_mnk works, got {len(configs)} configs")
        
        if configs:
            cfg = configs[0]
            print(f"\nFirst config:")
            print(f"  Type: {type(cfg)}")
            if isinstance(cfg, dict):
                print(f"  Keys: {list(cfg.keys())}")
                for k, v in cfg.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  CTA: {cfg.cta[0]}x{cfg.cta[1]}x{cfg.cta[2]}")
                print(f"  Stages: {cfg.loadStages}")
                print(f"  Split-K: {cfg.splitK}")
            
            # Try estimating runtime
            try:
                runtime = interface.estimateSiliconMetric(
                    64, 64, 4096,
                    "HSH",  # precision string
                    layout,
                    cfg,
                    hw_desc,
                )
                print(f"  Estimated runtime: {runtime:.3f} ms")
            except Exception as e:
                print(f"  Runtime estimation failed: {e}")
        
        # Clean up
        interface.destroyHardwareDescriptor(hw_desc)
        print("  ✓ Hardware descriptor destroyed")
    
    except Exception as e:
        import traceback
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
    
except ImportError as e:
    print(f"Failed to import nvMatmulHeuristics: {e}")

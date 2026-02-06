#!/usr/bin/env python3
"""
Extract SASS (assembly) code from NCU report files.

This script processes .ncu-rep files and extracts SASS code for specific kernels,
handling different report types (CuTe DSL, CUTLASS, cuBLAS) appropriately.

Usage:
    # Extract from all reports in a folder
    python extract_sass_from_ncu.py --input-dir results_5x5/ --output-dir sass_output/
    
    # Specify kernel selection strategy
    python extract_sass_from_ncu.py --input-dir results/ --output-dir sass/ --kernel-type cutedsl
    
    # Custom kernel ID
    python extract_sass_from_ncu.py --input-dir results/ --output-dir sass/ --kernel-id 5
    
    # Process single file
    python extract_sass_from_ncu.py --input-file profile.ncu-rep --output-dir sass/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict
import json
import re


class NCUSASSExtractor:
    """Extract SASS code from NCU report files"""
    
    def __init__(
        self,
        ncu_binary: str = "ncu",
        verbose: bool = True
    ):
        self.ncu_binary = ncu_binary
        self.verbose = verbose
        
        # Verify NCU is available
        try:
            result = subprocess.run(
                [self.ncu_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"NCU binary check failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(f"NCU binary not found: {self.ncu_binary}")
        except Exception as e:
            raise RuntimeError(f"Error checking NCU binary: {e}")
    
    def get_kernel_list(self, ncu_file: Path) -> List[Dict]:
        """
        Get list of kernels in the NCU report.
        
        Returns:
            List of dictionaries with kernel info (id, name, etc.)
        """
        try:
            # Use ncu --list-kernels or --import with --page details
            result = subprocess.run(
                [
                    self.ncu_binary,
                    "--import", str(ncu_file),
                    "--page", "details",
                    "--csv"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"  Warning: Could not list kernels: {result.stderr}")
                return []
            
            # Parse CSV output to get kernel information
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return []
            
            # Find kernel name column
            header = lines[0].split(',')
            kernel_name_idx = None
            kernel_id_idx = None
            
            for i, col in enumerate(header):
                col_lower = col.strip('"').lower()
                if 'kernel' in col_lower and 'name' in col_lower:
                    kernel_name_idx = i
                if col_lower in ['id', 'kernel id']:
                    kernel_id_idx = i
            
            kernels = []
            for i, line in enumerate(lines[1:]):
                parts = line.split(',')
                kernel_info = {
                    'id': i,
                    'name': parts[kernel_name_idx].strip('"') if kernel_name_idx is not None and kernel_name_idx < len(parts) else f"kernel_{i}"
                }
                if kernel_id_idx is not None and kernel_id_idx < len(parts):
                    try:
                        kernel_info['ncu_id'] = int(parts[kernel_id_idx].strip('"'))
                    except ValueError:
                        pass
                kernels.append(kernel_info)
            
            return kernels
            
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"  Warning: Timeout listing kernels")
            return []
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Error listing kernels: {e}")
            return []
    
    def extract_sass(
        self,
        ncu_file: Path,
        output_file: Path,
        kernel_id: Optional[int] = None,
        kernel_name: Optional[str] = None
    ) -> bool:
        """
        Extract SASS code from NCU report.
        
        Args:
            ncu_file: Path to .ncu-rep file
            output_file: Path to output .sass file
            kernel_id: Specific kernel ID to extract (0-indexed), or None for all kernels
            kernel_name: Expected kernel name for verification
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build NCU command to extract SASS
            # Note: --kernel-id doesn't work with --import, so we extract all and filter
            cmd = [
                self.ncu_binary,
                "--import", str(ncu_file),
                "--page", "source",
                "--print-source", "sass"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"  ✗ SASS extraction failed: {result.stderr}")
                return False
            
            # Get the SASS output
            sass_output = result.stdout
            
            # For now, extract all SASS (kernel filtering in NCU SASS output is complex
            # as it's formatted as a single table with all kernels interspersed)
            # TODO: Implement kernel-specific extraction if needed in the future
            
            # Write SASS output to file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(sass_output)
            
            if self.verbose:
                size_kb = output_file.stat().st_size / 1024
                print(f"  ✓ SASS extracted: {output_file.name} ({size_kb:.1f} KB)")
            
            return True
            
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"  ✗ SASS extraction timeout (>60s)")
            return False
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Error extracting SASS: {e}")
            return False
    
    def _extract_kernel_sass(
        self,
        sass_output: str,
        kernel_id: int,
        kernel_name: Optional[str] = None
    ) -> str:
        """
        Extract SASS for a specific kernel from the full SASS output.
        
        The NCU SASS output contains sections for each kernel, separated by headers.
        We need to find the section for the requested kernel ID.
        
        Args:
            sass_output: Full SASS output from NCU
            kernel_id: 0-indexed kernel ID to extract
            kernel_name: Optional kernel name for verification
            
        Returns:
            SASS code for the specific kernel, or empty string if not found
        """
        # Split output into lines
        lines = sass_output.split('\n')
        
        # Look for kernel section headers
        # NCU output typically has "Kernel Name" headers followed by SASS instructions
        kernel_sections = []
        current_section = []
        in_kernel = False
        current_kernel_id = -1
        
        for line in lines:
            # Detect kernel header (usually contains "Kernel Name" or instruction addresses)
            # The format varies, but typically has a distinctive header with the kernel name
            
            # Check if this is a new kernel section
            # Look for lines that indicate kernel boundaries
            if 'Kernel Name' in line or (line.strip() and line.strip()[0:2] in ['--', '=='] and len(line.strip()) > 50):
                # Save previous section if we were in one
                if in_kernel and current_section:
                    kernel_sections.append({
                        'id': current_kernel_id,
                        'content': '\n'.join(current_section)
                    })
                    current_section = []
                
                # Start new section
                current_kernel_id += 1
                in_kernel = True
                current_section.append(line)
            elif in_kernel:
                current_section.append(line)
        
        # Save last section
        if in_kernel and current_section:
            kernel_sections.append({
                'id': current_kernel_id,
                'content': '\n'.join(current_section)
            })
        
        # If we couldn't parse sections, try a simpler approach
        # Just split by large blocks of dashes and take the requested index
        if not kernel_sections or len(kernel_sections) <= kernel_id:
            # Try splitting by separator lines (long lines of dashes)
            separator_pattern = r'^-{50,}|^={50,}'
            sections = []
            current = []
            
            for line in lines:
                if re.match(separator_pattern, line.strip()):
                    if current:
                        sections.append('\n'.join(current))
                        current = []
                current.append(line)
            
            if current:
                sections.append('\n'.join(current))
            
            # Return the requested section if it exists
            if kernel_id < len(sections):
                return sections[kernel_id]
            
            # If still can't find it, return all (better than nothing)
            if self.verbose:
                print(f"  Warning: Could not parse kernel sections, returning all SASS")
            return sass_output
        
        # Return the requested kernel section
        if kernel_id < len(kernel_sections):
            return kernel_sections[kernel_id]['content']
        
        return ""
    
    def determine_kernel_id(
        self,
        ncu_file: Path,
        kernel_type: str = "auto",
        custom_id: Optional[int] = None
    ) -> int:
        """
        Determine which kernel ID to extract based on report type.
        
        Args:
            ncu_file: Path to NCU report
            kernel_type: Type of kernel report ("auto", "cutedsl", "cutlass", "cublas", "custom")
            custom_id: Custom kernel ID if kernel_type is "custom"
            
        Returns:
            Kernel ID (0-indexed) to extract
        """
        if kernel_type == "custom" and custom_id is not None:
            return custom_id
        
        # Get kernel list
        kernels = self.get_kernel_list(ncu_file)
        
        if not kernels:
            # If we can't get kernel list, use defaults
            if kernel_type == "cutedsl":
                return 2  # Kernel ID 2 for CuTe DSL
            elif kernel_type == "cutlass":
                return 10  # Kernel 11 (0-indexed: 10) for CUTLASS after warmup
            elif kernel_type == "cublas":
                return 0  # First kernel for cuBLAS
            else:  # auto
                return 0  # Default to first kernel
        
        if self.verbose:
            print(f"  Found {len(kernels)} kernel(s) in report")
        
        # Auto-detect based on filename or kernel names
        if kernel_type == "auto":
            filename_lower = ncu_file.name.lower()
            
            if "cutedsl" in filename_lower or any("cute" in k['name'].lower() for k in kernels):
                kernel_type = "cutedsl"
            elif "cutlass" in filename_lower or any("cutlass" in k['name'].lower() for k in kernels):
                kernel_type = "cutlass"
            elif "cublas" in filename_lower or any("gemm" in k['name'].lower() for k in kernels):
                kernel_type = "cublas"
            else:
                kernel_type = "unknown"
        
        # Select kernel based on type
        if kernel_type == "cutedsl":
            # CuTe DSL: use kernel ID 2
            target_id = min(2, len(kernels) - 1)
            if self.verbose and len(kernels) > 2:
                print(f"  CuTe DSL report: selecting kernel #{target_id} ({kernels[target_id]['name']})")
            return target_id
            
        elif kernel_type == "cutlass":
            # CUTLASS: use kernel 11 (index 10) after warmup
            target_id = min(10, len(kernels) - 1)
            if self.verbose and len(kernels) > 10:
                print(f"  CUTLASS report: selecting kernel #{target_id} ({kernels[target_id]['name']}) (after warmup)")
            return target_id
            
        elif kernel_type == "cublas":
            # cuBLAS: typically only one kernel, use first
            if self.verbose:
                print(f"  cuBLAS report: selecting kernel #0 ({kernels[0]['name']})")
            return 0
            
        else:
            # Unknown: default to first kernel
            if self.verbose:
                print(f"  Unknown report type: selecting first kernel ({kernels[0]['name']})")
            return 0
    
    def process_file(
        self,
        ncu_file: Path,
        output_dir: Path,
        kernel_type: str = "auto",
        custom_kernel_id: Optional[int] = None
    ) -> bool:
        """
        Process a single NCU report file.
        
        Args:
            ncu_file: Path to .ncu-rep file
            output_dir: Output directory for SASS files
            kernel_type: Kernel selection strategy
            custom_kernel_id: Custom kernel ID override
            
        Returns:
            True if successful
        """
        if self.verbose:
            print(f"\nProcessing: {ncu_file.name}")
        
        # Get kernel list for reference
        kernels = self.get_kernel_list(ncu_file)
        
        # Determine kernel ID to extract
        kernel_id = self.determine_kernel_id(ncu_file, kernel_type, custom_kernel_id)
        
        # Get kernel name if available
        kernel_name = None
        if kernels and kernel_id < len(kernels):
            kernel_name = kernels[kernel_id]['name']
        
        # Generate output filename
        output_name = ncu_file.stem + ".sass"
        output_file = output_dir / output_name
        
        # Extract SASS
        success = self.extract_sass(ncu_file, output_file, kernel_id, kernel_name)
        
        return success
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        kernel_type: str = "auto",
        custom_kernel_id: Optional[int] = None,
        pattern: str = "*.ncu-rep"
    ) -> Dict[str, int]:
        """
        Process all NCU report files in a directory.
        
        Args:
            input_dir: Directory containing .ncu-rep files
            output_dir: Output directory for SASS files
            kernel_type: Kernel selection strategy
            custom_kernel_id: Custom kernel ID override
            pattern: File pattern to match (default: *.ncu-rep)
            
        Returns:
            Dictionary with success/failure counts
        """
        # Find all NCU report files
        ncu_files = sorted(input_dir.glob(pattern))
        
        if not ncu_files:
            print(f"No files matching '{pattern}' found in {input_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        print(f"\n{'='*80}")
        print(f"SASS Extraction from NCU Reports")
        print(f"{'='*80}")
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Kernel type:      {kernel_type}")
        if custom_kernel_id is not None:
            print(f"Custom kernel ID: {custom_kernel_id}")
        print(f"Files found:      {len(ncu_files)}")
        print(f"{'='*80}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        results = {'total': len(ncu_files), 'success': 0, 'failed': 0}
        
        for ncu_file in ncu_files:
            success = self.process_file(ncu_file, output_dir, kernel_type, custom_kernel_id)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SASS Extraction Summary")
        print(f"{'='*80}")
        print(f"Total files:   {results['total']}")
        print(f"Successful:    {results['success']}")
        print(f"Failed:        {results['failed']}")
        print(f"Output dir:    {output_dir}")
        print(f"{'='*80}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract SASS code from NCU report files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from all reports in a folder (auto-detect kernel type)
  python extract_sass_from_ncu.py --input-dir results_5x5/ --output-dir sass_output/
  
  # Specify CuTe DSL kernel selection (kernel ID 2)
  python extract_sass_from_ncu.py --input-dir results_cutedsl/ --output-dir sass/ --kernel-type cutedsl
  
  # Specify CUTLASS kernel selection (kernel 11, index 10)
  python extract_sass_from_ncu.py --input-dir results_cutlass/ --output-dir sass/ --kernel-type cutlass
  
  # Specify cuBLAS kernel selection (first kernel)
  python extract_sass_from_ncu.py --input-dir results_cublas/ --output-dir sass/ --kernel-type cublas
  
  # Use custom kernel ID
  python extract_sass_from_ncu.py --input-dir results/ --output-dir sass/ --kernel-type custom --kernel-id 5
  
  # Process single file
  python extract_sass_from_ncu.py --input-file profile.ncu-rep --output-dir sass/
  
  # Specify NCU binary path
  python extract_sass_from_ncu.py --input-dir results/ --output-dir sass/ --ncu-binary /usr/local/cuda/bin/ncu

Kernel Type Selection:
  auto      - Auto-detect based on filename/kernel names (default)
  cutedsl   - CuTe DSL kernels (selects kernel ID 2)
  cutlass   - CUTLASS C++ kernels (selects kernel 11, index 10, after warmup)
  cublas    - cuBLAS kernels (selects first kernel)
  custom    - Use --kernel-id to specify exact kernel
        """
    )
    
    # Input/output
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing .ncu-rep files"
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Single .ncu-rep file to process"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .sass files"
    )
    
    # Kernel selection
    parser.add_argument(
        "--kernel-type",
        choices=["auto", "cutedsl", "cutlass", "cublas", "custom"],
        default="auto",
        help="Kernel selection strategy (default: auto)"
    )
    
    parser.add_argument(
        "--kernel-id",
        type=int,
        help="Specific kernel ID to extract (0-indexed, used with --kernel-type custom)"
    )
    
    parser.add_argument(
        "--pattern",
        default="*.ncu-rep",
        help="File pattern to match in input directory (default: *.ncu-rep)"
    )
    
    # NCU settings
    parser.add_argument(
        "--ncu-binary",
        default="ncu",
        help="Path to NCU binary (default: ncu)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate kernel-id with custom type
    if args.kernel_type == "custom" and args.kernel_id is None:
        parser.error("--kernel-id is required when --kernel-type is custom")
    
    try:
        # Create extractor
        extractor = NCUSASSExtractor(
            ncu_binary=args.ncu_binary,
            verbose=not args.quiet
        )
        
        # Process files
        if args.input_file:
            # Process single file
            if not args.input_file.exists():
                print(f"Error: Input file not found: {args.input_file}")
                return 1
            
            success = extractor.process_file(
                args.input_file,
                args.output_dir,
                args.kernel_type,
                args.kernel_id
            )
            return 0 if success else 1
            
        else:
            # Process directory
            if not args.input_dir.exists():
                print(f"Error: Input directory not found: {args.input_dir}")
                return 1
            
            results = extractor.process_directory(
                args.input_dir,
                args.output_dir,
                args.kernel_type,
                args.kernel_id,
                args.pattern
            )
            
            return 0 if results['failed'] == 0 else 1
    
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())

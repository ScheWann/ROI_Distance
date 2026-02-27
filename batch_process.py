#!/usr/bin/env python3
"""
Batch ROI Distance Processing - Command-line script for processing multiple patient folders.
Runs on remote servers without GUI (no PyQt5 required).

Usage:
  python batch_process.py /path/to/parent_folder [options]

  Example:
  python batch_process.py /Users/data/Folder_7_ABAS --skip-existing
  nohup python batch_process.py /data/patients/ --cores 8 > batch.log 2>&1 &
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from roi_processing_core import (
    scan_folder_sync,
    merge_rtstruct_dicom,
    generate_csv_sync,
)


def is_valid_patient_folder(folder_path):
    """Check if folder looks like a patient folder (has CT or DICOM)"""
    p = Path(folder_path)
    if not p.is_dir():
        return False
    ct_folders = list(p.glob('**/CT'))
    if ct_folders:
        return True
    dicom_count = len(list(p.glob('**/*.dcm'))) + len(list(p.glob('**/*.DCM')))
    return dicom_count > 0


def get_patient_folders(parent_path):
    """Get list of patient subfolders (direct children that look valid)"""
    parent = Path(parent_path)
    if not parent.exists() or not parent.is_dir():
        return []
    folders = []
    for item in sorted(parent.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            if is_valid_patient_folder(item):
                folders.append(str(item))
    return folders


def process_one_patient(patient_path, merge_if_possible=True, num_cores=4, skip_existing=False,
                       progress_callback=None):
    """Process a single patient folder: scan -> merge (if applicable) -> generate CSV."""
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"  {msg}")

    results = {'path': patient_path, 'success': False, 'error': None, 'merged': False, 'csv_generated': False}

    try:
        # Scan
        log("Scanning...")
        scan = scan_folder_sync(patient_path)
        ct_folder = scan.get('ct_folder')
        abas_set = set(scan.get('abas_rtstruct_files', []))
        normal_paths = [f for f in scan.get('rtstruct_files', []) if f not in abas_set]
        abas_paths = list(abas_set)
        merged_path = scan.get('merged_rtstruct')

        if not ct_folder or not os.path.exists(ct_folder):
            results['error'] = "No CT folder found"
            return results

        # Determine RTSTRUCT to use
        rtstruct_to_use = None

        if merged_path and os.path.exists(merged_path):
            rtstruct_to_use = merged_path
            results['merged'] = True
            log(f"Using existing MergedSS.dcm")
        elif merge_if_possible and normal_paths and abas_paths:
            # Merge
            output_folder = Path(normal_paths[0]).parent
            merge_output = str(output_folder / 'MergedSS.dcm')
            log(f"Merging {len(normal_paths)} Normal + {len(abas_paths)} ABAS -> MergedSS.dcm")
            merge_result = merge_rtstruct_dicom(normal_paths, abas_paths, merge_output)
            rtstruct_to_use = merge_output
            results['merged'] = True
            log(f"  Merged: {merge_result['normal_rois']} normal + {merge_result['abas_rois']} ABAS ROIs")
        elif normal_paths:
            rtstruct_to_use = normal_paths[0]
            log(f"Using Normal RTSTRUCT: {Path(rtstruct_to_use).name}")
        elif abas_paths:
            rtstruct_to_use = abas_paths[0]
            log(f"Using ABAS RTSTRUCT: {Path(rtstruct_to_use).name}")

        if not rtstruct_to_use or not os.path.exists(rtstruct_to_use):
            results['error'] = "No RTSTRUCT file found (need Normal, ABAS, or Merged)"
            return results

        # output_folder = Path(rtstruct_to_use).parent
        # centroid_path = output_folder / 'CT_centroid.csv'
        # distance_path = output_folder / 'CT_distances.csv'

        # if skip_existing and centroid_path.exists() and distance_path.exists():
        #     log("CSV files already exist, skipping generation")
        #     results['success'] = True
        #     results['csv_generated'] = True
        #     return results

        # Generate CSV
        log("Generating CSV files...")

        def csv_progress(pct, msg):
            if progress_callback:
                progress_callback(f"  [{pct}%] {msg}")
                
                
        output_folder = Path(rtstruct_to_use).parent

        centroid_final = output_folder / 'CT_centroid.csv'
        distance_final = output_folder / 'CT_distances.csv'
        centroid_tmp = output_folder / 'CT_centroid.csv.tmp'
        distance_tmp = output_folder / 'CT_distances.csv.tmp'

        # If previous run died, remove old tmp
        for p in (centroid_tmp, distance_tmp):
            if p.exists():
                p.unlink()

        # Skip only if FINAL files exist
        if skip_existing and centroid_final.exists() and distance_final.exists():
            log("CSV files already exist, skipping generation")
            results['success'] = True
            results['csv_generated'] = True
            return results

        # Generate CSV (WRITE TO TMP)
        log("Generating CSV files...")

        def csv_progress(pct, msg):
            if progress_callback:
                progress_callback(f"  [{pct}%] {msg}")

        generate_csv_sync(
            ct_folder,
            rtstruct_to_use,
            str(output_folder),
            num_cores=num_cores,
            progress_callback=csv_progress,
            centroid_filename="CT_centroid.csv.tmp",
            distance_filename="CT_distances.csv.tmp",
        )

        # Publish TMP -> FINAL (atomic)
        centroid_tmp.replace(centroid_final)
        distance_tmp.replace(distance_final)

        results['success'] = True
        results['csv_generated'] = True
        log("Done: CT_centroid.csv, CT_distances.csv")
    except Exception as e:
        results['error'] = str(e)
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple patient folders for ROI distance analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'parent_folder',
        type=str,
        help='Parent folder containing patient subfolders (e.g. Folder_7_ABAS)',
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Do not merge Normal+ABAS; use first available RTSTRUCT only',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip CSV generation if CT_centroid.csv and CT_distances.csv already exist',
    )
    parser.add_argument(
        '--cores',
        type=int,
        default=4,
        metavar='N',
        help='Number of CPU cores for distance calculation (default: 4)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )
    args = parser.parse_args()

    parent = Path(args.parent_folder).resolve()
    if not parent.exists():
        print(f"Error: Parent folder does not exist: {parent}")
        sys.exit(1)
    if not parent.is_dir():
        print(f"Error: Not a directory: {parent}")
        sys.exit(1)

    patient_folders = get_patient_folders(parent)
    if not patient_folders:
        print(f"No valid patient folders found under {parent}")
        print("A valid folder should contain CT or DICOM files.")
        sys.exit(1)

    print(f"Found {len(patient_folders)} patient folder(s)")
    print(f"Options: merge={'on' if not args.no_merge else 'off'}, skip_existing={args.skip_existing}, cores={args.cores}")
    print("-" * 60)

    success_count = 0
    fail_count = 0
    start_time = datetime.now()

    for i, pf in enumerate(patient_folders):
        name = Path(pf).name
        print(f"\n[{i + 1}/{len(patient_folders)}] {name}")

        def cb(msg):
            if args.verbose:
                print(msg)

        result = process_one_patient(
            pf,
            merge_if_possible=not args.no_merge,
            num_cores=args.cores,
            skip_existing=args.skip_existing,
            progress_callback=cb if args.verbose else None,
        )

        if result['success']:
            success_count += 1
        else:
            fail_count += 1
            print(f"  FAILED: {result.get('error', 'Unknown error')}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print(f"Batch complete: {success_count} succeeded, {fail_count} failed")
    print(f"Time: {elapsed:.1f} seconds")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()

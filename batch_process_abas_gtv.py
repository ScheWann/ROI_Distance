#!/usr/bin/env python3
"""
Batch ROI Distance Processing for ABAS organ ROIs plus Normal-derived GTV ROIs.

This script does not merge all Normal and ABAS structures. It uses ABAS as the
base RTSTRUCT, then copies only the patient-specific GTV_p/GTV_n source ROIs
from the Normal RTSTRUCT according to tumor_roi_selection_20260421.csv.

Usage:
  python batch_process_abas_gtv.py /path/to/parent_folder [options]

Example:
  python batch_process_abas_gtv.py /Users/data/Folder_7_ABAS --cores 8 --skip-existing
"""

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydicom

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from roi_processing_core import generate_csv_sync, scan_folder_sync


excluded_gtv_missing_id = excluded_gtv_missing_id = ['105025355210', '139123597330', '140881477949', '154270148296', '156320095591', '160908779974', '161120525775', '162338411679', '167800999244', '175393242095', '175858867340', '176268482619', '184285931319', '195672393332', '197965012329', '203362125714', '205876857238', '214335244635', '226926195577', '230439881947', '235393299667', '240295756907', '247377995066', '276772052044', '277089814380', '277431880233', '280489084994', '288910275196', '289499607380', '290806098388', '293974098976', '303572115007', '310026611461', '319274969366', '319292636320', '320500592855', '332157140963', '344254882250', '344968892802', '369407392063', '381696011115', '462557850342', '484006968728', '511051809968', '707454175961', '709317957738', '753408672849', '764961562475', '875433292830', '884514874829', '899667420050', '910319911165', '931251318367', '962607992104', '987810235978', '131129643906']
excluded_organ_missing_id = ['224012867216', '217361683229', '555557827139']


def is_valid_patient_folder(folder_path):
    """Check if folder looks like a patient folder (has CT or DICOM)."""
    p = Path(folder_path)
    if not p.is_dir():
        return False
    ct_folders = list(p.glob('**/CT'))
    if ct_folders:
        return True
    dicom_count = len(list(p.glob('**/*.dcm'))) + len(list(p.glob('**/*.DCM')))
    return dicom_count > 0


def get_patient_folders(parent_path):
    """Get direct child folders that look like patient folders."""
    parent = Path(parent_path)
    if not parent.exists() or not parent.is_dir():
        return []

    skipped_ids = set(excluded_gtv_missing_id) | set(excluded_organ_missing_id)
    folders = []
    for item in sorted(parent.iterdir()):
        if not item.is_dir() or item.name.startswith('.'):
            continue
        patient_id = extract_patient_id_from_name(item.name)
        if patient_id in skipped_ids:
            continue
        if is_valid_patient_folder(item):
            folders.append(str(item))
    return folders


def extract_patient_id_from_name(name):
    """Extract a patient id from folder/file name when possible."""
    import re

    match = re.search(r'\d{6,}', str(name))
    return match.group() if match else str(name)


def load_tumor_roi_selection(csv_path):
    """Load mapping: patient_id -> {'GTV_p': source_roi, 'GTV_n': source_roi_or_none}."""
    df = pd.read_csv(csv_path, dtype=str).fillna('')
    required = {'patient_id', 'GTV_p', 'GTV_n'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    mapping = {}
    for _, row in df.iterrows():
        patient_id = str(row['patient_id']).strip()
        if not patient_id:
            continue
        mapping[patient_id] = {
            'GTV_p': str(row['GTV_p']).strip(),
            'GTV_n': str(row['GTV_n']).strip(),
        }
    return mapping


def normalize_roi_name(name):
    return str(name).strip().lower()


def find_roi_by_name(ds, roi_name):
    """Return the ROI item and matching contour item for roi_name."""
    if not roi_name or not hasattr(ds, 'StructureSetROISequence'):
        return None, None

    wanted = normalize_roi_name(roi_name)
    for roi in ds.StructureSetROISequence:
        if normalize_roi_name(roi.ROIName) != wanted:
            continue

        roi_contours = getattr(ds, 'ROIContourSequence', None) or []
        contour = next(
            (
                rc for rc in roi_contours
                if hasattr(rc, 'ReferencedROINumber')
                and int(rc.ReferencedROINumber) == int(roi.ROINumber)
            ),
            None,
        )
        return roi, contour
    return None, None


def remove_roi_by_name(ds, roi_name):
    """Remove an ROI and its contour by name, if it already exists in the ABAS base."""
    if not hasattr(ds, 'StructureSetROISequence'):
        return

    wanted = normalize_roi_name(roi_name)
    removed_numbers = {
        int(roi.ROINumber)
        for roi in ds.StructureSetROISequence
        if normalize_roi_name(roi.ROIName) == wanted
    }
    if not removed_numbers:
        return

    ds.StructureSetROISequence = pydicom.sequence.Sequence(
        roi for roi in ds.StructureSetROISequence
        if int(roi.ROINumber) not in removed_numbers
    )
    if hasattr(ds, 'ROIContourSequence') and ds.ROIContourSequence is not None:
        ds.ROIContourSequence = pydicom.sequence.Sequence(
            rc for rc in ds.ROIContourSequence
            if not (
                hasattr(rc, 'ReferencedROINumber')
                and int(rc.ReferencedROINumber) in removed_numbers
            )
        )


def add_normal_roi_as_gtv(ds_base, ds_normal, source_roi_name, output_roi_name):
    """Copy one ROI from Normal RTSTRUCT into the ABAS base under a standard GTV name."""
    source_roi, source_contour = find_roi_by_name(ds_normal, source_roi_name)
    if source_roi is None:
        raise ValueError(f"Normal RTSTRUCT missing source ROI '{source_roi_name}' for {output_roi_name}")
    if source_contour is None or not hasattr(source_contour, 'ContourSequence'):
        raise ValueError(f"Normal source ROI '{source_roi_name}' has no contour data")

    remove_roi_by_name(ds_base, output_roi_name)

    existing_numbers = [int(roi.ROINumber) for roi in ds_base.StructureSetROISequence]
    new_number = max(existing_numbers, default=0) + 1

    new_roi = deepcopy(source_roi)
    new_roi.ROINumber = new_number
    new_roi.ROIName = output_roi_name
    ds_base.StructureSetROISequence.append(new_roi)

    if not hasattr(ds_base, 'ROIContourSequence') or ds_base.ROIContourSequence is None:
        ds_base.ROIContourSequence = pydicom.sequence.Sequence()

    new_contour = deepcopy(source_contour)
    new_contour.ReferencedROINumber = new_number
    ds_base.ROIContourSequence.append(new_contour)


def build_abas_with_gtv(abas_path, normal_path, tumor_selection, output_path):
    """Create ABAS_with_GTV.dcm from ABAS plus selected Normal GTV_p/GTV_n ROIs."""
    ds_base = pydicom.dcmread(abas_path)
    ds_normal = pydicom.dcmread(normal_path)

    if not hasattr(ds_base, 'StructureSetROISequence'):
        raise ValueError("ABAS RTSTRUCT has no StructureSetROISequence")
    if not hasattr(ds_normal, 'StructureSetROISequence'):
        raise ValueError("Normal RTSTRUCT has no StructureSetROISequence")

    gtv_p_source = tumor_selection.get('GTV_p', '').strip()
    gtv_n_source = tumor_selection.get('GTV_n', '').strip()
    if not gtv_p_source:
        raise ValueError("GTV_p source ROI is empty in tumor ROI selection CSV")

    add_normal_roi_as_gtv(ds_base, ds_normal, gtv_p_source, 'GTV_p')
    added = ['GTV_p']

    if gtv_n_source:
        add_normal_roi_as_gtv(ds_base, ds_normal, gtv_n_source, 'GTV_n')
        added.append('GTV_n')
    else:
        remove_roi_by_name(ds_base, 'GTV_n')

    ds_base.save_as(output_path)
    return {'output_path': output_path, 'added_rois': added}


def process_one_patient(patient_path, tumor_mapping, num_cores=4, skip_existing=False,
                       progress_callback=None):
    """Process one patient: scan -> ABAS_with_GTV -> generate CSV."""
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"  {msg}")

    results = {
        'path': patient_path,
        'success': False,
        'error': None,
        'augmented_rtstruct': None,
        'csv_generated': False,
    }

    try:
        patient_name = Path(patient_path).name
        folder_patient_id = extract_patient_id_from_name(patient_name)

        log("Scanning...")
        scan = scan_folder_sync(patient_path)
        ct_folder = scan.get('ct_folder')
        scan_patient_id = str(scan.get('mrn') or '').strip()
        candidate_patient_ids = [
            pid for pid in [folder_patient_id, scan_patient_id]
            if pid and pid != patient_name
        ]
        patient_id = next((pid for pid in candidate_patient_ids if pid in tumor_mapping), None)
        patient_id = patient_id or (candidate_patient_ids[0] if candidate_patient_ids else folder_patient_id)

        skipped_ids = set(excluded_gtv_missing_id) | set(excluded_organ_missing_id)
        if any(pid in skipped_ids for pid in candidate_patient_ids + [patient_id]):
            results['success'] = True
            results['error'] = 'Skipped excluded patient'
            log(f"Skipping excluded patient: {patient_id}")
            return results

        if not ct_folder or not os.path.exists(ct_folder):
            results['error'] = "No CT folder found"
            return results

        tumor_selection = tumor_mapping.get(patient_id)
        if tumor_selection is None:
            results['error'] = f"No tumor ROI selection found for patient_id {patient_id}"
            return results

        normal_paths = sorted(scan.get('rtstruct_files', []))
        abas_paths = sorted(scan.get('abas_rtstruct_files', []))
        if not abas_paths:
            results['error'] = "No ABAS RTSTRUCT file found"
            return results
        if not normal_paths:
            results['error'] = "No Normal RTSTRUCT file found for GTV_p/GTV_n"
            return results

        abas_path = abas_paths[0]
        normal_path = normal_paths[0]
        output_folder = Path(abas_path).parent
        augmented_path = output_folder / 'ABAS_with_GTV.dcm'
        centroid_path = output_folder / 'CT_centroid.csv'
        distance_path = output_folder / 'CT_distances.csv'

        if skip_existing and centroid_path.exists() and distance_path.exists():
            log("CSV files already exist, skipping generation")
            results['success'] = True
            results['csv_generated'] = True
            return results

        log(
            "Creating ABAS_with_GTV.dcm "
            f"from ABAS + Normal ROI GTV_p='{tumor_selection.get('GTV_p', '')}', "
            f"GTV_n='{tumor_selection.get('GTV_n', '')}'"
        )
        augment_result = build_abas_with_gtv(
            abas_path,
            normal_path,
            tumor_selection,
            str(augmented_path),
        )
        results['augmented_rtstruct'] = str(augmented_path)
        log(f"Added ROI(s): {', '.join(augment_result['added_rois'])}")

        log("Generating CSV files...")

        def csv_progress(pct, msg):
            if progress_callback:
                progress_callback(f"  [{pct}%] {msg}")

        generate_csv_sync(
            ct_folder,
            str(augmented_path),
            str(output_folder),
            num_cores=num_cores,
            progress_callback=csv_progress,
        )

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
    default_csv = Path(__file__).resolve().parent / 'tumor_roi_selection_20260421.csv'
    parser = argparse.ArgumentParser(
        description='Batch process ABAS organ ROIs plus Normal-derived GTV_p/GTV_n.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'parent_folder',
        type=str,
        help='Parent folder containing patient subfolders',
    )
    parser.add_argument(
        '--tumor-selection-csv',
        type=str,
        default=str(default_csv),
        help=f'Tumor ROI selection CSV (default: {default_csv})',
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

    tumor_mapping = load_tumor_roi_selection(args.tumor_selection_csv)
    patient_folders = get_patient_folders(parent)
    if not patient_folders:
        print(f"No valid patient folders found under {parent}")
        print("A valid folder should contain CT or DICOM files and not be in the excluded id lists.")
        sys.exit(1)

    print(f"Found {len(patient_folders)} patient folder(s)")
    print(
        "Options: "
        f"skip_existing={args.skip_existing}, cores={args.cores}, "
        f"tumor_selection_csv={args.tumor_selection_csv}"
    )
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
            tumor_mapping,
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

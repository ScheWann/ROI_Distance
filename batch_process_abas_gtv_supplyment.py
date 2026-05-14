#!/usr/bin/env python3
"""
Batch ROI Distance Processing for ABAS organ ROIs plus Normal-derived GTV ROIs
(supplement run — processed layout + ROI fallbacks).

Expects a *processed* root (e.g. /Users/data/processed) containing batch folders
named like Batch_1, Batch_2, … Each batch folder holds patient
subfolders. Only patients in REMAINING_IDS are processed.

GTV_p / GTV_n source names come from tumor_roi_selection_20260421.csv first; if
the named ROI has no contour on the Normal RTSTRUCT, candidates are tried in
order using the same primary/nodal heuristics as tumor ROI selection (regex).

Usage:
  python batch_process_abas_gtv_supplyment.py /Users/data/processed [options]

Example:
  python batch_process_abas_gtv_supplyment.py /Users/data/processed --cores 8 --skip-existing
"""

import argparse
import os
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydicom

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from roi_processing_core import generate_csv_sync, scan_folder_sync


# Only these patient IDs are collected under Batch_* batch folders beneath the processed root.
REMAINING_IDS = frozenset({
    '110301574943',
    '187905891341',
    '214443426645',
    '217361683229',
    '555557827139',
    '150155589586',
    '165031314713',
    '224012867216',
    '274959696549',
    '486258600183',
    '663081172442',
    '796419334154',
    '939908042928',
})

# Matches Batch_1, Batch_2, … or Batch_1_ABAS if you use that suffix.
_BATCH_FOLDER_RE = re.compile(r'^Batch_\d+(?:_ABAS)?$', re.IGNORECASE)


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


def get_patient_folders_processed_root(processed_root):
    """
    Under processed_root, scan immediate children Batch_* (see _BATCH_FOLDER_RE); within each,
    collect patient subfolders whose id is in REMAINING_IDS and looks valid.
    """
    root = Path(processed_root)
    if not root.exists() or not root.is_dir():
        return []

    folders = []
    for batch_dir in sorted(root.iterdir()):
        if not batch_dir.is_dir() or batch_dir.name.startswith('.'):
            continue
        if not _BATCH_FOLDER_RE.match(batch_dir.name):
            continue
        for patient_dir in sorted(batch_dir.iterdir()):
            if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
                continue
            patient_id = extract_patient_id_from_name(patient_dir.name)
            if patient_id not in REMAINING_IDS:
                continue
            if is_valid_patient_folder(patient_dir):
                folders.append(str(patient_dir))
    return folders


def extract_patient_id_from_name(name):
    """Extract a patient id from folder/file name when possible."""
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


def _get_prefix_rois(names, prefix):
    """Return ROI names that contain the given prefix (case-insensitive)."""
    return [n for n in names if prefix in n.lower()]


def _is_gtv_ctv_ptv_related(name):
    """True if ROI name plausibly refers to a target volume (substring gtv / ctv / ptv)."""
    lower = str(name).lower()
    return any(p in lower for p in ('gtv', 'ctv', 'ptv'))


def _is_primary_pattern(name, prefix):
    """
    True if `name` looks like a primary-tumour ROI for the given prefix.
    Matches: exact base (e.g. 'GTV'), or prefix + optional separator + p/P/primary
    e.g. GTVp, GTV_p, GTV-P, GTV P, GTV primary, GTVp_5600, GTV_P_5600
    """
    lower = name.lower().strip()
    if lower == prefix:
        return True
    pattern = rf'{re.escape(prefix)}[_\-\s]*(?:p(?:[^a-z]|$)|primary)'
    return bool(re.search(pattern, lower))


def _is_nodal_pattern(name, prefix):
    """
    True if `name` looks like a nodal ROI for the given prefix.
    Matches patterns like: GTVn, GTV-N, GTV N, GTVn1, GTV_N1, GTV-NR, GTV-NL, GTV node(s)
    """
    lower = name.lower()
    pattern = rf'{re.escape(prefix)}[_\-\s]*n(?:[rl\d]|ode[s]?|[_\-\s]|$)'
    return bool(re.search(pattern, lower))


def find_primary_roi(matched_rois):
    """
    Select the primary tumour ROI (GTV_p) from a patient's matched ROI list.
    Priority order: GTV > CTV > PTV
    Within each prefix:
      1. Exact base name ('GTV', 'CTV', 'PTV')
      2. Primary-specific pattern  (e.g. GTVp, GTV_P, GTV primary)
    Fallback: first GTV/CTV/PTV-related name that is not nodal-shaped; else None.
    """
    for prefix in ["gtv", "ctv", "ptv"]:
        subset = _get_prefix_rois(matched_rois, prefix)
        if not subset:
            continue
        exact = [n for n in subset if n.lower().strip() == prefix]
        if exact:
            return exact[0]
        primary = [n for n in subset if _is_primary_pattern(n, prefix)]
        if primary:
            return primary[0]
    related = [n for n in matched_rois if _is_gtv_ctv_ptv_related(n)]
    non_nodal_related = [
        n for n in related
        if not any(_is_nodal_pattern(n, p) for p in ["gtv", "ctv", "ptv"])
    ]
    if non_nodal_related:
        return non_nodal_related[0]
    return None


def find_nodal_roi(matched_rois):
    """
    Select the nodal tumour ROI (GTV_n) from a patient's matched ROI list.
    Priority order: GTV > CTV > PTV
    Returns None if no nodal ROI is found.
    """
    for prefix in ["gtv", "ctv", "ptv"]:
        subset = _get_prefix_rois(matched_rois, prefix)
        nodal = [n for n in subset if _is_nodal_pattern(n, prefix)]
        if nodal:
            return nodal[0]
    return None


def normal_roi_has_valid_contour(ds_normal, roi_name):
    """True if Normal RTSTRUCT has ContourSequence for this ROI name."""
    _, contour = find_roi_by_name(ds_normal, roi_name)
    return (
        contour is not None
        and hasattr(contour, 'ContourSequence')
        and contour.ContourSequence
    )


def _candidate_append_unique(ordered, seen, name):
    n = str(name).strip() if name else ''
    if not n or n in seen:
        return
    seen.add(n)
    ordered.append(n)


def ordered_gtv_p_candidates(matched_rois, csv_gtv_p):
    """
    Ordered GTV_p sources: CSV first, then GTV/CTV/PTV-related names only.
    Never fall back to unrelated OARs (e.g. Mandible).
    """
    seen = set()
    out = []
    _candidate_append_unique(out, seen, csv_gtv_p)
    fp = find_primary_roi(matched_rois)
    _candidate_append_unique(out, seen, fp)
    for prefix in ["gtv", "ctv", "ptv"]:
        subset = _get_prefix_rois(matched_rois, prefix)
        exact = [n for n in subset if n.lower().strip() == prefix]
        for n in exact:
            _candidate_append_unique(out, seen, n)
        primary_like = [n for n in subset if _is_primary_pattern(n, prefix)]
        for n in primary_like:
            _candidate_append_unique(out, seen, n)
    for n in matched_rois:
        if not _is_gtv_ctv_ptv_related(n):
            continue
        if any(_is_nodal_pattern(n, p) for p in ["gtv", "ctv", "ptv"]):
            continue
        _candidate_append_unique(out, seen, n)
    return out


def ordered_gtv_n_candidates(matched_rois, csv_gtv_n):
    """CSV first, then nodal structures whose names relate to GTV/CTV/PTV only."""
    seen = set()
    out = []
    _candidate_append_unique(out, seen, csv_gtv_n)
    fn = find_nodal_roi(matched_rois)
    _candidate_append_unique(out, seen, fn)
    for prefix in ["gtv", "ctv", "ptv"]:
        subset = _get_prefix_rois(matched_rois, prefix)
        nodal = [n for n in subset if _is_nodal_pattern(n, prefix)]
        for n in nodal:
            _candidate_append_unique(out, seen, n)
    for n in matched_rois:
        if not _is_gtv_ctv_ptv_related(n):
            continue
        if not any(_is_nodal_pattern(n, p) for p in ["gtv", "ctv", "ptv"]):
            continue
        _candidate_append_unique(out, seen, n)
    return out


def resolve_gtv_sources_with_contours(ds_normal, tumor_selection):
    """
    Pick Normal ROI names for GTV_p / GTV_n that have contour data, trying CSV
    labels first then regex/heuristic candidate lists.
    """
    matched_rois = [str(roi.ROIName) for roi in ds_normal.StructureSetROISequence]
    csv_p = tumor_selection.get('GTV_p', '').strip()
    csv_n = tumor_selection.get('GTV_n', '').strip()

    p_candidates = ordered_gtv_p_candidates(matched_rois, csv_p)
    gtv_p_source = None
    for cand in p_candidates:
        if normal_roi_has_valid_contour(ds_normal, cand):
            gtv_p_source = cand
            break
    if not gtv_p_source:
        raise ValueError(
            "Could not find any Normal ROI with contour data for GTV_p "
            f"(tried {len(p_candidates)} candidate(s) from CSV + heuristics)"
        )

    gtv_n_source = ''
    want_nodal = bool(csv_n) or (find_nodal_roi(matched_rois) is not None)
    if want_nodal:
        n_candidates = ordered_gtv_n_candidates(matched_rois, csv_n)
        for cand in n_candidates:
            if normal_roi_has_valid_contour(ds_normal, cand):
                gtv_n_source = cand
                break

    return {
        'GTV_p': gtv_p_source,
        'GTV_n': gtv_n_source,
        'matched_rois': matched_rois,
    }


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

    resolved = resolve_gtv_sources_with_contours(ds_normal, tumor_selection)
    gtv_p_source = resolved['GTV_p']
    gtv_n_source = str(resolved.get('GTV_n', '') or '').strip()

    add_normal_roi_as_gtv(ds_base, ds_normal, gtv_p_source, 'GTV_p')
    added = ['GTV_p']

    if gtv_n_source:
        add_normal_roi_as_gtv(ds_base, ds_normal, gtv_n_source, 'GTV_n')
        added.append('GTV_n')
    else:
        remove_roi_by_name(ds_base, 'GTV_n')

    ds_base.save_as(output_path)
    return {
        'output_path': output_path,
        'added_rois': added,
        'resolved_gtv_p': gtv_p_source,
        'resolved_gtv_n': gtv_n_source or None,
        'csv_gtv_p': tumor_selection.get('GTV_p', '').strip(),
        'csv_gtv_n': tumor_selection.get('GTV_n', '').strip(),
    }


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
            f"(CSV GTV_p={tumor_selection.get('GTV_p', '')!r}, "
            f"GTV_n={tumor_selection.get('GTV_n', '')!r})"
        )
        augment_result = build_abas_with_gtv(
            abas_path,
            normal_path,
            tumor_selection,
            str(augmented_path),
        )
        results['augmented_rtstruct'] = str(augmented_path)
        log(
            f"Resolved Normal sources → GTV_p={augment_result.get('resolved_gtv_p')!r}, "
            f"GTV_n={augment_result.get('resolved_gtv_n')!r}"
        )
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
        'processed_root',
        type=str,
        help='Processed root containing Batch_* batch folders (see REMAINING_IDS)',
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

    parent = Path(args.processed_root).resolve()
    if not parent.exists():
        print(f"Error: Parent folder does not exist: {parent}")
        sys.exit(1)
    if not parent.is_dir():
        print(f"Error: Not a directory: {parent}")
        sys.exit(1)

    tumor_mapping = load_tumor_roi_selection(args.tumor_selection_csv)
    patient_folders = get_patient_folders_processed_root(parent)
    if not patient_folders:
        print(f"No matching patient folders under {parent}")
        print(
            "Expected: child dirs named like Batch_1 (or Batch_1_ABAS) with patient subfolders "
            f"for IDs in REMAINING_IDS ({len(REMAINING_IDS)} ids), each with CT/DICOM."
        )
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

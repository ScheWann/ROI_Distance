"""
ROI Processing Core - Headless processing logic for scan, merge, and CSV generation.
No PyQt5 dependency - can run on remote servers.
"""

import os
import re
import warnings
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label
from skimage.morphology import closing

warnings.filterwarnings('ignore', message='.*Camel case attribute.*', category=UserWarning)

IGNORE_ROI_NAMES = {
    "external",
    "fsbody",
    "vms igrtct 6mv d1.05",
    "brainlab couch 6mv d1.05",
    "fs nt",
    "fsnormaltissue",
    "fs_contracted",
    "ext-0.3",
    "couch",
    "marked iso_1",
    "final iso_1",
    "marked iso",
    "final iso",
}

# GPU support (optional)
GPU_AVAILABLE = False
try:
    import cupy
    GPU_AVAILABLE = True
except ImportError:
    pass


def scan_folder_sync(folder_path):
    """Scan folder for DICOM files. Returns dict with ct_folder, rtstruct_files, abas_rtstruct_files, merged_rtstruct, mrn."""
    results = {
        'ct_folder': None,
        'rtstruct_files': [],
        'abas_rtstruct_files': [],
        'merged_rtstruct': None,
        'mrn': None,
    }
    folder = Path(folder_path)

    results['mrn'] = _extract_mrn(folder)

    ct_folders = list(folder.glob('**/CT'))
    if ct_folders:
        results['ct_folder'] = str(ct_folders[0])
    else:
        dicom_files = list(folder.glob('**/*.dcm'))
        if dicom_files:
            results['ct_folder'] = str(dicom_files[0].parent)

    all_dicom = list(folder.glob('**/*.dcm')) + list(folder.glob('**/*.DCM'))
    all_rtstructs = []
    for dicom_file in all_dicom:
        try:
            ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
            if hasattr(ds, 'Modality') and ds.Modality == 'RTSTRUCT':
                all_rtstructs.append((str(dicom_file), ds))
        except Exception:
            continue

    abas_files_set = set()
    normal_files_set = set()
    for dicom_file, ds in all_rtstructs:
        if _is_abas_rtstruct(dicom_file, ds):
            abas_files_set.add(dicom_file)
        else:
            normal_files_set.add(dicom_file)

    results['abas_rtstruct_files'] = list(abas_files_set)
    results['rtstruct_files'] = list(normal_files_set)

    merged_files = list(folder.glob('**/MergedSS.dcm'))
    if merged_files:
        results['merged_rtstruct'] = str(merged_files[0])

    return results


def _is_abas_rtstruct(file_path, dicom_dataset):
    """Determine if RTSTRUCT is ABAS"""
    file_str = str(file_path).lower()
    abas_keywords = ['abas', 'atlas', 'auto', 'autoseg', 'automatic']
    if any(kw in file_str for kw in abas_keywords):
        return True
    if hasattr(dicom_dataset, 'SeriesDescription'):
        desc = str(dicom_dataset.SeriesDescription).lower()
        if any(kw in desc for kw in abas_keywords):
            return True
    if hasattr(dicom_dataset, 'StructureSetROISequence'):
        total = len(dicom_dataset.StructureSetROISequence)
        abas_count = sum(1 for roi in dicom_dataset.StructureSetROISequence
                         if any(kw in str(roi.ROIName).lower() for kw in abas_keywords))
        if total > 0 and (abas_count / total) > 0.3:
            return True
    return False


def _extract_mrn(folder):
    """Extract MRN from folder name or DICOM"""
    mrn_match = re.search(r'\d{6,}', folder.name)
    if mrn_match:
        return mrn_match.group()
    for dicom_file in list(folder.glob('**/*.dcm'))[:5]:
        try:
            ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
            if hasattr(ds, 'PatientID'):
                return str(ds.PatientID)
        except Exception:
            continue
    return folder.name


def merge_rtstruct_dicom(normal_paths, abas_paths, output_path):
    """Chain-merge RTSTRUCT files.

    ABAS has priority: use first ABAS RTSTRUCT as base if available,
    then merge remaining ABAS files, then merge Normal files.
    Returns dict with output_path, normal_rois, abas_rois.
    """
    if not normal_paths and not abas_paths:
        raise ValueError("Need at least one Normal or ABAS RTSTRUCT file")

    # Choose base: prefer ABAS if present
    if abas_paths:
        base_path = abas_paths[0]
        base_is_abas = True
        remaining_abas = abas_paths[1:]
        remaining_normals = normal_paths
    else:
        base_path = normal_paths[0]
        base_is_abas = False
        remaining_abas = []
        remaining_normals = normal_paths[1:]

    ds_base = pydicom.dcmread(base_path)
    if not hasattr(ds_base, 'StructureSetROISequence'):
        raise ValueError("Base RTSTRUCT has no StructureSetROISequence")

    existing_names = {str(roi.ROIName).strip().lower()
                      for roi in ds_base.StructureSetROISequence}

    max_roi_num = max(int(roi.ROINumber) for roi in ds_base.StructureSetROISequence)
    if not hasattr(ds_base, 'ROIContourSequence') or ds_base.ROIContourSequence is None:
        ds_base.ROIContourSequence = pydicom.sequence.Sequence()

    # Count how many ROIs come from Normal vs ABAS
    if base_is_abas:
        abas_count = len(ds_base.StructureSetROISequence)
        normal_count = 0
    else:
        normal_count = len(ds_base.StructureSetROISequence)
        abas_count = 0

    def add_from_file(ds_src, is_abas: bool):
        nonlocal max_roi_num, normal_count, abas_count
        roi_contour_src = getattr(ds_src, 'ROIContourSequence', None) or []

        for roi in ds_src.StructureSetROISequence:
            key = str(roi.ROIName).strip().lower()

            # Skip ignored ROI names entirely
            if key in IGNORE_ROI_NAMES:
                continue

            # Skip duplicate names
            if key in existing_names:
                continue

            existing_names.add(key)

            max_roi_num += 1
            new_number = max_roi_num
            orig_number = roi.ROINumber

            new_roi = deepcopy(roi)
            new_roi.ROINumber = new_number
            ds_base.StructureSetROISequence.append(new_roi)

            for rc in roi_contour_src:
                if hasattr(rc, 'ReferencedROINumber') and int(rc.ReferencedROINumber) == int(orig_number):
                    new_contour = deepcopy(rc)
                    new_contour.ReferencedROINumber = new_number
                    ds_base.ROIContourSequence.append(new_contour)
                    break

            if is_abas:
                abas_count += 1
            else:
                normal_count += 1

    # 1) merge remaining ABAS files
    for path in remaining_abas:
        ds_src = pydicom.dcmread(path)
        if hasattr(ds_src, 'StructureSetROISequence'):
            add_from_file(ds_src, True)

    # 2) then merge Normal files
    for path in remaining_normals:
        ds_src = pydicom.dcmread(path)
        if hasattr(ds_src, 'StructureSetROISequence'):
            add_from_file(ds_src, False)

    ds_base.save_as(output_path)
    return {'output_path': output_path, 'normal_rois': normal_count, 'abas_rois': abas_count}


def generate_csv_sync(
    ct_folder,
    rtstruct_path,
    output_folder,
    num_cores=4,
    progress_callback=None,
    centroid_filename="CT_centroid.csv",
    distance_filename="CT_distances.csv",
):
    """Generate centroid + distance CSVs. Writes to filenames provided (defaults to CT_centroid.csv / CT_distances.csv).
    progress_callback(percent, message) optional.
    """
    def log(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    log(10, "Loading DICOM files...")
    image_data, spatial_data = _extract_image_spatial_data(ct_folder)

    log(40, "Loading RTSTRUCT contours...")
    contours = _rtstruct_to_mask(rtstruct_path, spatial_data)
    if not contours:
        raise ValueError("No contours found in RTSTRUCT")

    contour_list = [c['label'] for c in contours]
    dx = spatial_data[0]['xImageDim']
    dy = spatial_data[0]['yImageDim']
    if len(spatial_data) >= 2:
        dz = abs(spatial_data[0]['zImagePosition'] - spatial_data[1]['zImagePosition'])
    else:
        dz = float(spatial_data[0].get('sliceThickness', 1.0))

    log(50, "Calculating ROI centroids...")
    centroid_path = os.path.join(output_folder, centroid_filename)
    _generate_centroid_csv(contours, contour_list, centroid_path)

    log(70, "Calculating distances...")
    distance_path = os.path.join(output_folder, distance_filename)
    _generate_distance_csv_parallel(contours, contour_list, dx, dy, dz, distance_path, num_cores, log)

    log(100, "CSV generation complete!")
    return centroid_path, distance_path


def _extract_image_spatial_data(image_directory):
    """Extract spatial data from CT DICOM. Returns (image_data, spatial_data)."""
    image_dir = Path(image_directory)
    dicom_files = sorted(list(image_dir.glob('*.dcm')) + list(image_dir.glob('*.DCM')))
    if not dicom_files:
        raise ValueError(f"No DICOM files in {image_directory}")

    spatial_data = []
    for dcm in dicom_files:
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            if hasattr(ds, 'Modality') and ds.Modality in ['RTSTRUCT', 'RTDOSE', 'RTPLAN', 'RTIMAGE']:
                continue
            info = {
                'filename': str(dcm),
                'xImageVoxels': int(ds.Width) if hasattr(ds, 'Width') else 512,
                'yImageVoxels': int(ds.Height) if hasattr(ds, 'Height') else 512,
                'xImageDim': float(ds.PixelSpacing[0]) if hasattr(ds, 'PixelSpacing') else 1.0,
                'yImageDim': float(ds.PixelSpacing[1]) if hasattr(ds, 'PixelSpacing') else 1.0,
                'xImagePosition': float(ds.ImagePositionPatient[0]) if hasattr(ds, 'ImagePositionPatient') else 0.0,
                'yImagePosition': float(ds.ImagePositionPatient[1]) if hasattr(ds, 'ImagePositionPatient') else 0.0,
                'zImagePosition': float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else 0.0,
            }
            if hasattr(ds, 'SliceThickness'):
                info['sliceThickness'] = float(ds.SliceThickness)
            spatial_data.append(info)
        except Exception:
            continue

    if not spatial_data:
        raise ValueError("No valid CT DICOM found")
    spatial_data.sort(key=lambda x: x['zImagePosition'], reverse=True)
    return None, spatial_data


def _rtstruct_to_mask(struct_file, spatial_data):
    """Convert RTSTRUCT to binary masks"""
    from skimage.draw import polygon

    ds = pydicom.dcmread(struct_file)
    if not hasattr(ds, 'StructureSetROISequence'):
        raise ValueError("No StructureSetROISequence")
    roi_list = ds.StructureSetROISequence
    roi_contour_seq = getattr(ds, 'ROIContourSequence', None) or []
    image_x = spatial_data[0]['xImageVoxels']
    image_y = spatial_data[0]['yImageVoxels']
    image_z = len(spatial_data)
    structure_array = []

    for roi in roi_list:
        roi_number = roi.ROINumber
        roi_label = roi.ROIName

        # Skip ignored ROI names
        name_norm = str(roi_label).strip().lower()
        if name_norm in IGNORE_ROI_NAMES:
            continue

        contour_data = next((rc for rc in roi_contour_seq
                            if hasattr(rc, 'ReferencedROINumber') and
                            int(rc.ReferencedROINumber) == int(roi_number)), None)
        if not contour_data or not hasattr(contour_data, 'ContourSequence'):
            continue

        binary_map = np.zeros((image_y, image_x, image_z), dtype=bool)
        for contour_seq in contour_data.ContourSequence:
            if not hasattr(contour_seq, 'ContourData'):
                continue
            pts = contour_seq.ContourData
            z_pos = pts[2]
            slice_idx = min(range(len(spatial_data)),
                           key=lambda i: abs(spatial_data[i]['zImagePosition'] - z_pos))
            if abs(spatial_data[slice_idx]['zImagePosition'] - z_pos) > 0.01:
                continue
            x_coords = np.array(pts[0::3])
            y_coords = np.array(pts[1::3])
            x_pos = spatial_data[slice_idx]['xImagePosition']
            y_pos = spatial_data[slice_idx]['yImagePosition']
            x_dim = spatial_data[slice_idx]['xImageDim']
            y_dim = spatial_data[slice_idx]['yImageDim']
            pixel_x = np.floor((x_coords - x_pos) / x_dim).astype(int)
            pixel_y = np.floor((y_coords - y_pos) / y_dim).astype(int)
            try:
                rr, cc = polygon(pixel_y, pixel_x, shape=(image_y, image_x))
                binary_map[rr, cc, slice_idx] = True
            except Exception:
                pass

        for z in range(image_z):
            if np.any(binary_map[:, :, z]):
                binary_map[:, :, z] = binary_fill_holes(binary_map[:, :, z])
                binary_map[:, :, z] = closing(binary_map[:, :, z])

        structure_array.append({'voxels': int(np.sum(binary_map)), 'label': roi_label, 'dat': binary_map})
    return structure_array


def _generate_centroid_csv(contours, contour_list, output_path):
    """Write centroid CSV"""
    with open(output_path, 'w') as f:
        f.write('ROI,x coordinate,y coordinate,z coordinate\n')
        for i, contour in enumerate(contours):
            mask = contour['dat']
            labeled = label(mask)
            props = regionprops(labeled)
            if props:
                c = props[0].centroid
                f.write(f"{contour_list[i]},{round(c[1], 2)},{round(c[0], 2)},{round(c[2], 2)}\n")
            else:
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    f.write(f"{contour_list[i]},{round(np.mean(coords[1]), 2)},{round(np.mean(coords[0]), 2)},{round(np.mean(coords[2]), 2)}\n")


def _distance_reference_worker(args):
    """Worker for one reference ROI: compute EDT once, then all pairs (i, j) for j > i.
    Module-level so it can be used with ThreadPoolExecutor (shared memory, no large serialization).
    args: (i, contours, contour_list, dx, dy, dz)
    Returns: list of result dicts for this reference."""
    from scipy.ndimage import distance_transform_edt

    i, contours, contour_list, dx, dy, dz = args
    n = len(contour_list)
    ref_mask = contours[i]['dat'].astype(bool)
    ref_name = contour_list[i]
    ref_coords = np.where(ref_mask)
    out = []

    if len(ref_coords[0]) == 0:
        for j in range(i + 1, n):
            out.append({
                'Reference ROI': ref_name,
                'Target ROI': contour_list[j],
                'Eucledian Distance (mm)': 0.0,
                'Phi (degrees)': 0.0,
                'Theta (degrees)': 0.0,
                '% of Target Overlap': 0.0,
                'Eucledian Distance (mm) 5th Percentile': 0.0,
            })
        return out

    aspect = [dy, dx, dz]
    dist_ref = distance_transform_edt(~ref_mask, sampling=aspect).astype(np.float32)
    dist_not_ref = distance_transform_edt(ref_mask, sampling=aspect).astype(np.float32)
    ref_c = np.array([
        np.mean(ref_coords[1]),
        np.mean(ref_coords[0]),
        np.mean(ref_coords[2]),
    ])

    for j in range(i + 1, n):
        target_mask = contours[j]['dat'].astype(bool)
        target_name = contour_list[j]
        target_coords = np.where(target_mask)
        if len(target_coords[0]) == 0:
            out.append({
                'Reference ROI': ref_name,
                'Target ROI': target_name,
                'Eucledian Distance (mm)': 0.0,
                'Phi (degrees)': 0.0,
                'Theta (degrees)': 0.0,
                '% of Target Overlap': 0.0,
                'Eucledian Distance (mm) 5th Percentile': 0.0,
            })
            continue

        target_voxels = target_mask
        distances_all = np.where(
            target_voxels,
            np.where(ref_mask, dist_not_ref, dist_ref),
            0.0,
        )[target_voxels]
        distances = distances_all[distances_all > 0]

        if len(distances) == 0:
            min_dist = 0.0
            r5 = 0.0
        else:
            min_dist = float(np.min(distances))
            r5 = float(np.percentile(distances, 5)) if len(distances) > 1 else min_dist

        tgt_c = np.array([
            np.mean(target_coords[1]),
            np.mean(target_coords[0]),
            np.mean(target_coords[2]),
        ])
        vec = (tgt_c - ref_c) * np.array([dx, dy, dz])
        r = np.linalg.norm(vec)
        if r > 0:
            theta = np.arctan2(vec[0], -vec[1]) * 180 / np.pi
            phi = np.arcsin(vec[2] / r) * 180 / np.pi
        else:
            theta = 0.0
            phi = 0.0
        overlap = np.sum(ref_mask & target_mask) / max(1, np.sum(target_mask))

        out.append({
            'Reference ROI': ref_name,
            'Target ROI': target_name,
            'Eucledian Distance (mm)': round(min_dist, 2),
            'Phi (degrees)': round(float(phi), 2),
            'Theta (degrees)': round(float(theta), 2),
            '% of Target Overlap': round(float(overlap), 2),
            'Eucledian Distance (mm) 5th Percentile': round(r5, 2),
        })
    return out


def _generate_distance_csv_parallel(contours, contour_list, dx, dy, dz, output_path, num_cores, log_fn):
    """Generate distance CSV by reusing distance transforms per reference ROI, in parallel over references."""
    n = len(contour_list)
    total_pairs = n * (n - 1) // 2 if n > 1 else 0
    num_workers = max(1, min(num_cores, n, mp.cpu_count()))

    if n <= 0:
        pd.DataFrame(columns=[
            'Reference ROI', 'Target ROI', 'Eucledian Distance (mm)',
            'Phi (degrees)', 'Theta (degrees)', '% of Target Overlap',
            'Eucledian Distance (mm) 5th Percentile'
        ]).to_csv(output_path, index=False)
        return

    task_args = [(i, contours, contour_list, dx, dy, dz) for i in range(n)]
    results = []

    # ThreadPoolExecutor avoids Windows multiprocessing pipe size limits when passing large contours.
    # SciPy/Numpy release the GIL during EDT, so threads still get real CPU parallelism.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = list(executor.submit(_distance_reference_worker, a) for a in task_args)
        done = 0
        for future in futures:
            results.extend(future.result())
            done += 1
            if log_fn and total_pairs > 0 and done % max(1, n // 10 or 1) == 0:
                pct = 70 + int(30 * done / n)
                pairs_approx = done * n - (done * (done + 1) // 2) if done <= n else total_pairs
                log_fn(pct, f"Processed {pairs_approx}/{total_pairs} pairs...")

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, float_format='%.2f')

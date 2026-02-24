"""
Enhanced ROI Distance Generator and Visualizer
Complete workflow tool for RTSTRUCT analysis, merging, and visualization
NO MATLAB REQUIRED - All processing done in Python
GPU ACCELERATED - Uses CUDA when available, falls back to CPU
PARALLEL PROCESSING - Multi-core CPU processing for faster computation
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path

# Suppress pydicom warnings about camel case attributes (common in RTSTRUCT files)
# This warning occurs when RTSTRUCT files contain non-standard DICOM attributes
warnings.filterwarnings('ignore', message='.*Camel case attribute.*', category=UserWarning)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
                             QLabel, QFileDialog, QComboBox, QLineEdit, QGroupBox,
                             QSplitter, QMessageBox, QHeaderView, QCheckBox, QSpinBox,
                             QDoubleSpinBox, QProgressBar, QTextEdit, QTabWidget,
                             QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
                             QDialogButtonBox, QDialog, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
import matplotlib.transforms as mtransforms
# Optional seaborn import - can hang on some systems, so skip it entirely
# Seaborn is only used for color palette, which is not critical
# If needed, seaborn can be imported later on-demand
SEABORN_AVAILABLE = False
sns = None

import pydicom
from pydicom.errors import InvalidDicomError
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist
from skimage.measure import regionprops, label, marching_cubes
from skimage.morphology import binary_closing
from skimage import filters, morphology
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Windows-specific: Set multiprocessing start method to 'spawn' if not already set
# This prevents issues with multiprocessing on Windows
# Only set if we're in the main module (not during import in subprocess)
if sys.platform == 'win32' and __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=False)
    except (RuntimeError, ValueError):
        # Start method already set or cannot be set, ignore
        pass

# GPU Support - Try to import CuPy for CUDA acceleration
# Automatically detect and add PyTorch lib directory to DLL search path
# This helps CuPy find CUDA runtime DLLs (like nvrtc64_112_0.dll) that come with PyTorch
def _setup_cuda_dll_path():
    """Automatically detect and add PyTorch lib directory to DLL search path"""
    if sys.platform != 'win32':
        return  # Only needed on Windows
    
    try:
        import torch
        # Get PyTorch installation path
        torch_path = os.path.dirname(torch.__file__)
        torch_lib_path = os.path.join(torch_path, 'lib')
        
        # Check if the lib directory exists and contains CUDA DLLs
        if os.path.exists(torch_lib_path):
            # Add to PATH environment variable (for subprocess calls)
            current_path = os.environ.get('PATH', '')
            if torch_lib_path not in current_path:
                os.environ['PATH'] = torch_lib_path + os.pathsep + current_path
            
            # Don't set CUDA_PATH to torch_lib_path - CuPy expects CUDA_PATH to point to
            # the CUDA toolkit root (which has a bin/ subdirectory), not the lib directory.
            # The PATH addition above is sufficient for finding DLLs.
            
            # Also add to DLL search path for current process (Python 3.8+)
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(torch_lib_path)
                except (OSError, ValueError):
                    pass  # Directory might already be added or invalid
    except ImportError:
        pass  # PyTorch not installed, skip
    except Exception as e:
        # Log the error for debugging but don't fail
        print(f"[_setup_cuda_dll_path] Warning: {e}")
        pass

# Suppress CuPy CUDA path warning since we're handling it ourselves
warnings.filterwarnings('ignore', message='.*CUDA path could not be detected.*', category=UserWarning)

# Setup CUDA DLL path before importing CuPy
_setup_cuda_dll_path()

GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
except Exception as e:
    # CuPy might be installed but fail to load due to missing DLLs
    # This will be handled gracefully with CPU fallback
    GPU_AVAILABLE = False

# Set modern style - avoid seaborn styles that might trigger seaborn import
try:
    plt.style.use('dark_background')
except:
    try:
        plt.style.use('default')
    except:
        pass  # Use matplotlib default if style setting fails

# Set color palette if seaborn is available (optional)
if SEABORN_AVAILABLE and sns is not None:
    try:
        sns.set_palette("husl")
    except:
        pass  # Ignore if palette setting fails


def check_gpu():
    """Check if GPU is available and working - simplified to avoid hanging"""
    if not GPU_AVAILABLE:
        return False, "CuPy not installed"
    
    try:
        # Simple check - just try to import and create a small array
        # Avoid accessing CUDA runtime directly which can hang
        test_array = cp.array([1, 2, 3])
        # Don't access memGetInfo as it can hang on some systems
        return True, "GPU (CuPy) available"
    except Exception as e:
        return False, f"GPU error: {str(e)}"


class PythonCSVGeneratorV3(QThread):
    """Thread for generating CSV files using Python with GPU/Parallel support"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str, str)  # centroid_path, distance_path
    error = pyqtSignal(str)
    
    def __init__(self, ct_folder, rtstruct_path, output_folder, use_gpu=True, use_parallel=True, num_cores=4):
        super().__init__()
        self.ct_folder = ct_folder
        self.rtstruct_path = rtstruct_path
        self.output_folder = output_folder
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_parallel = use_parallel
        self.num_cores = num_cores  # Number of CPU cores to use
    
    def run(self):
        try:
            self.progress.emit(10, "Loading DICOM files...")
            
            # Extract spatial data from CT
            self.progress.emit(20, "Extracting CT spatial information...")
            image_data, spatial_data = self.extract_image_spatial_data(self.ct_folder)
            
            self.progress.emit(40, "Loading RTSTRUCT contours...")
            # Convert RTSTRUCT to masks
            primary_contours = self.rtstruct_to_mask(self.rtstruct_path, spatial_data)
            
            if not primary_contours or len(primary_contours) == 0:
                raise Exception("No contours found in RTSTRUCT file")
            
            self.progress.emit(50, "Calculating ROI centroids...")
            # Get contour list and calculate centroids
            contour_list = [contour['label'] for contour in primary_contours]
            
            # Get spatial dimensions
            dx = spatial_data[0]['xImageDim']
            dy = spatial_data[0]['yImageDim']
            dz = abs(spatial_data[0]['zImagePosition'] - spatial_data[1]['zImagePosition'])
            
            # Generate centroid CSV
            centroid_path = os.path.join(self.output_folder, 'CT_centroid.csv')
            self.generate_centroid_csv(primary_contours, contour_list, centroid_path)
            
            self.progress.emit(70, "Calculating distances between ROIs...")
            # Generate distance CSV with GPU/Parallel support
            distance_path = os.path.join(self.output_folder, 'CT_distances.csv')
            
            # Always use parallel processing - GPU if available and requested, otherwise parallel CPU
            if self.use_gpu and GPU_AVAILABLE:
                self.progress.emit(71, "Using GPU acceleration with parallel processing...")
                self.generate_distance_csv_gpu(primary_contours, contour_list, dx, dy, dz, distance_path)
            else:
                # Use parallel CPU processing (default and fallback)
                self.progress.emit(71, "Using parallel CPU processing...")
                self.generate_distance_csv_parallel(primary_contours, contour_list, dx, dy, dz, distance_path)
            
            self.progress.emit(100, "CSV generation complete!")
            self.finished.emit(centroid_path, distance_path)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def extract_image_spatial_data(self, image_directory):
        """Extract spatial data from DICOM CT images (Python version)"""
        image_dir = Path(image_directory)
        dicom_files = sorted(list(image_dir.glob('*.dcm')) + list(image_dir.glob('*.DCM')))
        
        if not dicom_files:
            raise Exception(f"No DICOM files found in {image_directory}")
        
        spatial_data = []
        image_slices = []
        
        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
                
                # Skip non-CT files
                if hasattr(ds, 'Modality') and ds.Modality in ['RTSTRUCT', 'RTDOSE', 'RTPLAN', 'RTIMAGE']:
                    continue
                
                # Extract spatial information
                spatial_info = {
                    'filename': str(dicom_file),
                    'xImageVoxels': int(ds.Width) if hasattr(ds, 'Width') else 512,
                    'yImageVoxels': int(ds.Height) if hasattr(ds, 'Height') else 512,
                    'xImageDim': float(ds.PixelSpacing[0]) if hasattr(ds, 'PixelSpacing') else 1.0,
                    'yImageDim': float(ds.PixelSpacing[1]) if hasattr(ds, 'PixelSpacing') else 1.0,
                    'xImagePosition': float(ds.ImagePositionPatient[0]) if hasattr(ds, 'ImagePositionPatient') else 0.0,
                    'yImagePosition': float(ds.ImagePositionPatient[1]) if hasattr(ds, 'ImagePositionPatient') else 0.0,
                    'zImagePosition': float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else 0.0,
                }
                
                if hasattr(ds, 'SliceThickness'):
                    spatial_info['sliceThickness'] = float(ds.SliceThickness)
                
                spatial_data.append(spatial_info)
                image_slices.append(dicom_file)
                
            except Exception as e:
                continue
        
        if not spatial_data:
            raise Exception("No valid CT DICOM files found")
        
        # Sort by z position
        spatial_data.sort(key=lambda x: x['zImagePosition'], reverse=True)
        
        # Read image data
        image_data = []
        for info in spatial_data:
            ds = pydicom.dcmread(info['filename'])
            img = ds.pixel_array
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            image_data.append(img)
        
        # Stack into 3D array
        if image_data:
            image_3d = np.stack(image_data, axis=2)
        else:
            image_3d = np.array([])
        
        return image_3d, spatial_data
    
    def rtstruct_to_mask(self, struct_file, spatial_data):
        """Convert RTSTRUCT to binary masks (Python version)"""
        ds = pydicom.dcmread(struct_file)
        
        if not hasattr(ds, 'StructureSetROISequence'):
            raise Exception("No StructureSetROISequence found in RTSTRUCT")
        
        image_x = spatial_data[0]['xImageVoxels']
        image_y = spatial_data[0]['yImageVoxels']
        image_z = len(spatial_data)
        
        structure_array = []
        
        roi_list = ds.StructureSetROISequence
        roi_contour_sequence = ds.ROIContourSequence if hasattr(ds, 'ROIContourSequence') else None
        
        for i, roi in enumerate(roi_list):
            roi_number = roi.ROINumber
            roi_label = roi.ROIName
            
            # Find corresponding contour
            contour_data = None
            if roi_contour_sequence:
                for roi_contour in roi_contour_sequence:
                    if hasattr(roi_contour, 'ReferencedROINumber') and roi_contour.ReferencedROINumber == roi_number:
                        contour_data = roi_contour
                        break
            
            if not contour_data or not hasattr(contour_data, 'ContourSequence'):
                continue
            
            binary_map = np.zeros((image_y, image_x, image_z), dtype=bool)
            
            for contour_seq in contour_data.ContourSequence:
                if not hasattr(contour_seq, 'ContourData'):
                    continue
                
                contour_points = contour_seq.ContourData
                z_pos = contour_points[2]  # Z coordinate
                
                # Find matching slice
                slice_idx = None
                min_diff = float('inf')
                for idx, spatial in enumerate(spatial_data):
                    diff = abs(z_pos - spatial['zImagePosition'])
                    if diff < min_diff and diff < 0.01:  # 0.01mm tolerance
                        min_diff = diff
                        slice_idx = idx
                
                if slice_idx is None:
                    continue
                
                # Extract x, y coordinates
                x_coords = contour_points[0::3]
                y_coords = contour_points[1::3]
                
                # Convert to pixel coordinates
                x_pos = spatial_data[slice_idx]['xImagePosition']
                y_pos = spatial_data[slice_idx]['yImagePosition']
                x_dim = spatial_data[slice_idx]['xImageDim']
                y_dim = spatial_data[slice_idx]['yImageDim']
                
                pixel_x = np.floor((np.array(x_coords) - x_pos) / x_dim).astype(int)
                pixel_y = np.floor((np.array(y_coords) - y_pos) / y_dim).astype(int)
                
                # Create mask using polygon fill
                from skimage.draw import polygon
                try:
                    rr, cc = polygon(pixel_y, pixel_x, shape=(image_y, image_x))
                    binary_map[rr, cc, slice_idx] = True
                except:
                    # Fallback: use simpler method
                    pass
            
            # Fill holes and clean up
            for z in range(image_z):
                if np.any(binary_map[:, :, z]):
                    binary_map[:, :, z] = binary_fill_holes(binary_map[:, :, z])
                    binary_map[:, :, z] = binary_closing(binary_map[:, :, z])
            
            structure_info = {
                'voxels': int(np.sum(binary_map)),
                'label': roi_label,
                'dat': binary_map,
                'show': 0
            }
            structure_array.append(structure_info)
        
        return structure_array
    
    def generate_centroid_csv(self, contours, contour_list, output_path):
        """Generate centroid CSV file"""
        with open(output_path, 'w') as f:
            f.write('ROI,x coordinate,y coordinate,z coordinate\n')
            
            for i, contour in enumerate(contours):
                mask = contour['dat']
                # Calculate centroid using regionprops
                labeled = label(mask)
                props = regionprops(labeled)
                
                if props:
                    # Use first (largest) region
                    centroid = props[0].centroid
                    # Note: regionprops returns (row, col, slice) which is (y, x, z)
                    f.write(f"{contour_list[i]},{round(centroid[1], 2)},{round(centroid[0], 2)},{round(centroid[2], 2)}\n")
                else:
                    # Fallback: calculate manually
                    coords = np.where(mask)
                    if len(coords[0]) > 0:
                        centroid_y = np.mean(coords[0])
                        centroid_x = np.mean(coords[1])
                        centroid_z = np.mean(coords[2])
                        f.write(f"{contour_list[i]},{round(centroid_x, 2)},{round(centroid_y, 2)},{round(centroid_z, 2)}\n")
    
    def generate_distance_csv(self, contours, contour_list, dx, dy, dz, output_path):
        """Generate distance CSV file (CPU, single-threaded)"""
        from scipy.ndimage import distance_transform_edt
        
        # Generate all pairs
        n = len(contour_list)
        combo = []
        for i in range(n):
            for j in range(i + 1, n):
                combo.append((i, j))
        
        results = []
        
        for idx, (i, j) in enumerate(combo):
            if (idx + 1) % 10 == 0:
                progress = 70 + int(30 * (idx + 1) / len(combo))
                self.progress.emit(progress, f"Processing pair {idx + 1}/{len(combo)}...")
            
            reference = contours[i]['dat']
            target = contours[j]['dat']
            
            # Calculate distances
            distance, phi, theta, overlap, r5 = self.calculate_distances_cpu(
                reference, target, dx, dy, dz
            )
            
            results.append({
                'Reference ROI': contour_list[i],
                'Target ROI': contour_list[j],
                'Eucledian Distance (mm)': round(distance, 2),
                'Phi (degrees)': round(phi, 2),
                'Theta (degrees)': round(theta, 2),
                '% of Target Overlap': round(overlap, 2),
                'Eucledian Distance (mm) 5th Percentile': round(r5, 2)
            })
        
        # Write to CSV with rounded values
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, float_format='%.2f')
    
    def generate_distance_csv_parallel(self, contours, contour_list, dx, dy, dz, output_path):
        """Generate distance CSV file using parallel CPU processing"""
        from scipy.ndimage import distance_transform_edt
        
        # Generate all pairs
        n = len(contour_list)
        combo = []
        for i in range(n):
            for j in range(i + 1, n):
                combo.append((i, j))

        # Enable lightweight parallelism with small batches to balance speed and memory
        # Allow up to the selected/available cores (no artificial cap)
        num_workers = max(1, min(self.num_cores, len(combo), mp.cpu_count()))
        # Dynamic batch size: larger batches for more cores to reduce overhead
        # For many cores, use larger batches to keep all cores busy
        if num_workers >= 24:
            batch_size = num_workers  # One batch per core for high core counts
        elif num_workers >= 12:
            batch_size = max(6, num_workers // 2)  # Medium batches
        else:
            batch_size = 3  # Small batches for low core counts
        
        # Note about memory bandwidth limitations
        if num_workers >= 24:
            note = " (Note: Performance may be limited by memory bandwidth, not CPU cores)"
        else:
            note = ""
        self.progress.emit(72, f"Using parallel CPU processing ({num_workers} cores, batch size {batch_size}){note}...")

        results = []
        completed = 0
        errors = 0

        executor = None
        failed_pairs = []
        try:
            executor = ProcessPoolExecutor(max_workers=num_workers)

            for batch_start in range(0, len(combo), batch_size):
                batch_end = min(batch_start + batch_size, len(combo))
                batch = combo[batch_start:batch_end]

                future_to_pair = {}
                for i, j in batch:
                    try:
                        reference = contours[i]['dat']
                        target = contours[j]['dat']
                        future = executor.submit(
                            calculate_distances_worker,
                            reference, target, dx, dy, dz,
                            contour_list[i], contour_list[j]
                        )
                        future_to_pair[future] = (i, j)
                    except Exception:
                        errors += 1
                        failed_pairs.append((i, j))

                for future in as_completed(future_to_pair):
                    try:
                        result = future.result(timeout=300)
                        results.append(result)
                        completed += 1
                        # Progress reporting: more frequent for low core counts, less for high core counts
                        report_interval = max(1, 20 // max(1, num_workers // 6)) if num_workers >= 12 else 5
                        if completed % report_interval == 0 or completed == 1:
                            progress_pct = 70 + (30 * completed / len(combo))
                            progress = min(99, int(progress_pct))
                            self.progress.emit(progress, f"Processed {completed}/{len(combo)} pairs... (Errors: {errors}) [{progress}%]")
                    except Exception:
                        errors += 1
                        pair_info = future_to_pair.get(future, None)
                        if pair_info:
                            failed_pairs.append(pair_info)
                            completed += 1
                            if errors <= 10:
                                progress_pct = 70 + (30 * completed / len(combo))
                                progress = min(99, int(progress_pct))
                                self.progress.emit(progress, f"Error on pair {completed}/{len(combo)} - retrying sequentially [{progress}%]")

                # GC between batches (less frequent for high core counts to reduce overhead)
                import gc
                if num_workers < 12:
                    # For lower core counts, GC more frequently to manage memory
                    gc.collect()
                    import time
                    time.sleep(0.1)
                elif completed % (batch_size * 5) == 0:
                    # For high core counts, GC less frequently to reduce overhead
                    gc.collect()

        except Exception:
            # Fallback to sequential for remaining pairs if pool breaks
            remaining_pairs = combo[completed:]
            import gc
            gc.collect()
            for i, j in remaining_pairs:
                try:
                    reference = contours[i]['dat']
                    target = contours[j]['dat']
                    result = calculate_distances_worker(reference, target, dx, dy, dz,
                                                       contour_list[i], contour_list[j])
                    results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        progress_pct = 70 + (30 * completed / len(combo))
                        progress = min(99, int(progress_pct))
                        self.progress.emit(progress, f"Sequential: {completed}/{len(combo)} pairs... [{progress}%]")
                    if completed % 50 == 0:
                        gc.collect()
                except Exception:
                    errors += 1
                    results.append({
                        'Reference ROI': contour_list[i],
                        'Target ROI': contour_list[j],
                        'Eucledian Distance (mm)': 0.0,
                        'Phi (degrees)': 0.0,
                        'Theta (degrees)': 0.0,
                        '% of Target Overlap': 0.0,
                        'Eucledian Distance (mm) 5th Percentile': 0.0
                    })
                    completed += 1
        finally:
            if executor:
                executor.shutdown(wait=True)

        # Retry failed pairs sequentially to avoid losing distances
        if failed_pairs:
            import gc
            gc.collect()
            for idx, (i, j) in enumerate(failed_pairs, 1):
                try:
                    reference = contours[i]['dat']
                    target = contours[j]['dat']
                    result = calculate_distances_worker(reference, target, dx, dy, dz,
                                                       contour_list[i], contour_list[j])
                    results.append(result)
                except Exception:
                    # On repeated failure, return default zeros
                    results.append({
                        'Reference ROI': contour_list[i],
                        'Target ROI': contour_list[j],
                        'Eucledian Distance (mm)': 0.0,
                        'Phi (degrees)': 0.0,
                        'Theta (degrees)': 0.0,
                        '% of Target Overlap': 0.0,
                        'Eucledian Distance (mm) 5th Percentile': 0.0
                    })
                if idx % 10 == 0 or idx == len(failed_pairs):
                    progress_pct = 70 + (30 * (completed + idx) / len(combo))
                    progress = min(99, int(progress_pct))
                    self.progress.emit(progress, f"Sequential retry: {idx}/{len(failed_pairs)} failed pairs... [{progress}%]")
                if idx % 20 == 0:
                    gc.collect()

        # Final progress update before writing CSV
        self.progress.emit(99, f"Writing results to CSV... ({len(results)} pairs)")

        # Write to CSV with rounded values
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, float_format='%.2f')
    
    def generate_distance_csv_gpu(self, contours, contour_list, dx, dy, dz, output_path):
        """Generate distance CSV file using GPU acceleration with parallel processing"""
        if not GPU_AVAILABLE:
            # Fallback to parallel CPU
            self.progress.emit(71, "GPU not available - falling back to parallel CPU processing...")
            self.generate_distance_csv_parallel(contours, contour_list, dx, dy, dz, output_path)
            return
        
        # Log GPU device information
        try:
            import cupy as cp
            device_id = cp.cuda.Device().id
            device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mem_gb = mem_info[0] / (1024**3)
            total_mem_gb = mem_info[1] / (1024**3)
            self.progress.emit(71, f"[GPU] Using: {device_name} (Device {device_id}, {free_mem_gb:.1f}/{total_mem_gb:.1f} GB free)")
            self.progress.emit(72, f"[GPU] Note: Distance transform uses CPU (CuPy limitation), GPU used for array operations")
        except Exception as e:
            self.progress.emit(71, f"[GPU] Detected but device info unavailable: {e}")
        
        # Generate all pairs
        n = len(contour_list)
        combo = []
        for i in range(n):
            for j in range(i + 1, n):
                combo.append((i, j))
        
        # Use parallel processing for GPU - batch process multiple pairs
        # GPU can handle multiple operations in parallel through CuPy's parallel execution
        results = []
        batch_size = 50  # Process in batches for better GPU utilization
        
        for batch_start in range(0, len(combo), batch_size):
            batch_end = min(batch_start + batch_size, len(combo))
            batch = combo[batch_start:batch_end]
            
            for idx, (i, j) in enumerate(batch):
                global_idx = batch_start + idx
                if (global_idx + 1) % 10 == 0:
                    progress = 70 + int(30 * (global_idx + 1) / len(combo))
                    self.progress.emit(progress, f"[GPU] Processing pair {global_idx + 1}/{len(combo)}...")
                
                reference = contours[i]['dat']
                target = contours[j]['dat']
                
                # Calculate distances on GPU
                distance, phi, theta, overlap, r5 = self.calculate_distances_gpu(
                    reference, target, dx, dy, dz
                )
                
                results.append({
                    'Reference ROI': contour_list[i],
                    'Target ROI': contour_list[j],
                    'Eucledian Distance (mm)': round(distance, 2),
                    'Phi (degrees)': round(phi, 2),
                    'Theta (degrees)': round(theta, 2),
                    '% of Target Overlap': round(overlap, 2),
                    'Eucledian Distance (mm) 5th Percentile': round(r5, 2)
                })
        
        # Write to CSV with rounded values
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, float_format='%.2f')
    
    def calculate_distances_cpu(self, reference, target, xdim, ydim, zdim):
        """Calculate distances using CPU (scipy)"""
        from scipy.ndimage import distance_transform_edt
        
        # Memory optimization: crop to bounding box
        ref_coords = np.where(reference)
        target_coords = np.where(target)
        
        if len(ref_coords[0]) == 0 or len(target_coords[0]) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Get bounding box with padding
        all_y = np.concatenate([ref_coords[0], target_coords[0]])
        all_x = np.concatenate([ref_coords[1], target_coords[1]])
        all_z = np.concatenate([ref_coords[2], target_coords[2]])
        
        y_min, y_max = max(0, all_y.min() - 10), min(reference.shape[0], all_y.max() + 10)
        x_min, x_max = max(0, all_x.min() - 10), min(reference.shape[1], all_x.max() + 10)
        z_min, z_max = max(0, all_z.min() - 2), min(reference.shape[2], all_z.max() + 2)
        
        # Crop to bounding box
        ref_cropped = reference[y_min:y_max, x_min:x_max, z_min:z_max]
        target_cropped = target[y_min:y_max, x_min:x_max, z_min:z_max]
        
        # Calculate distance transform
        aspect = [ydim, xdim, zdim]
        
        # Distance from each voxel to the nearest boundary of reference ROI
        # ~ref_cropped = outside reference ROI, so this gives distance from outside to boundary
        dist_ref = distance_transform_edt(~ref_cropped, sampling=aspect).astype(np.float32)
        
        # Distance from inside reference ROI to boundary
        dist_not_ref = distance_transform_edt(ref_cropped, sampling=aspect).astype(np.float32)
        
        # Get distances at target voxels
        target_voxels = target_cropped == 1
        
        # For target voxels outside reference: use dist_ref (distance to boundary)
        # For target voxels inside reference: use -dist_not_ref (negative = inside)
        # This gives us the signed distance: positive = outside, negative = inside
        signed_dist = dist_not_ref - dist_ref
        distances = signed_dist[target_voxels]
        
        # If all distances are negative (target is inside reference), use absolute value
        # Otherwise, we want the minimum positive distance (surface-to-surface)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Filter to only positive distances (target outside reference)
        positive_distances = distances[distances > 0]
        
        if len(positive_distances) > 0:
            # Target is outside reference - use minimum positive distance
            min_dist = np.min(positive_distances)
            r5 = np.percentile(positive_distances, 5) if len(positive_distances) > 0 else min_dist
        else:
            # Target is completely inside reference - use absolute value of minimum
            min_dist = abs(np.min(distances))
            r5 = abs(np.percentile(distances, 5))
        
        # Calculate angles
        ref_coords = np.where(reference)
        target_coords = np.where(target)
        
        if len(ref_coords[0]) > 0 and len(target_coords[0]) > 0:
            ref_centroid = np.array([np.mean(ref_coords[1]), np.mean(ref_coords[0]), np.mean(ref_coords[2])])
            target_centroid = np.array([np.mean(target_coords[1]), np.mean(target_coords[0]), np.mean(target_coords[2])])
            
            vec = target_centroid - ref_centroid
            vec_scaled = vec * np.array([xdim, ydim, zdim])
            
            r = np.linalg.norm(vec_scaled)
            if r > 0:
                theta = np.arctan2(vec_scaled[0], -vec_scaled[1]) * 180 / np.pi
                phi = np.arcsin(vec_scaled[2] / r) * 180 / np.pi
            else:
                theta = 0.0
                phi = 0.0
        else:
            theta = 0.0
            phi = 0.0
        
        # Calculate overlap
        overlap_mask = (reference == 1) & (target == 1)
        overlap_volume = np.sum(target == 1)
        if overlap_volume > 0:
            overlap_percent = np.sum(overlap_mask) / overlap_volume
        else:
            overlap_percent = 0.0
        
        return float(min_dist), float(phi), float(theta), float(overlap_percent), float(r5)
    
    def calculate_distances_gpu(self, reference, target, xdim, ydim, zdim):
        """Calculate distances using GPU (CuPy)"""
        if not GPU_AVAILABLE:
            return self.calculate_distances_cpu(reference, target, xdim, ydim, zdim)
        
        try:
            # Memory optimization: crop to bounding box
            ref_coords = np.where(reference)
            target_coords = np.where(target)
            
            if len(ref_coords[0]) == 0 or len(target_coords[0]) == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Get bounding box with padding
            all_y = np.concatenate([ref_coords[0], target_coords[0]])
            all_x = np.concatenate([ref_coords[1], target_coords[1]])
            all_z = np.concatenate([ref_coords[2], target_coords[2]])
            
            y_min, y_max = max(0, all_y.min() - 10), min(reference.shape[0], all_y.max() + 10)
            x_min, x_max = max(0, all_x.min() - 10), min(reference.shape[1], all_x.max() + 10)
            z_min, z_max = max(0, all_z.min() - 2), min(reference.shape[2], all_z.max() + 2)
            
            # Crop to bounding box
            ref_cropped = reference[y_min:y_max, x_min:x_max, z_min:z_max]
            target_cropped = target[y_min:y_max, x_min:x_max, z_min:z_max]
            
            # Transfer to GPU
            ref_gpu = cp.asarray(ref_cropped.astype(np.float32))
            target_gpu = cp.asarray(target_cropped.astype(np.float32))
            
            # Calculate distance transform on GPU using CuPy
            # Note: CuPy doesn't have distance_transform_edt, so we use a workaround
            # For now, fallback to CPU for distance transform, but do other operations on GPU
            # Transfer back to CPU for distance transform
            ref_cpu = cp.asnumpy(ref_gpu)
            target_cpu = cp.asnumpy(target_gpu)
            
            from scipy.ndimage import distance_transform_edt
            aspect = [ydim, xdim, zdim]
            
            dist_ref = distance_transform_edt(~ref_cpu.astype(bool), sampling=aspect).astype(np.float32)
            dist_not_ref = distance_transform_edt(ref_cpu.astype(bool), sampling=aspect).astype(np.float32)
            
            # Transfer to GPU for computation
            dist_ref_gpu = cp.asarray(dist_ref)
            dist_not_ref_gpu = cp.asarray(dist_not_ref)
            
            # Signed distance on GPU
            signed_dist_gpu = dist_not_ref_gpu - dist_ref_gpu
            
            # Get distances at target voxels
            target_voxels_gpu = target_gpu.astype(bool)
            distances_gpu = signed_dist_gpu[target_voxels_gpu]
            
            # Transfer back to CPU for statistics
            distances = cp.asnumpy(distances_gpu)
            
            if len(distances) == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            
            min_dist = float(np.min(distances))
            r5 = float(np.percentile(distances, 5))
            
            # Calculate angles (on CPU)
            ref_coords = np.where(reference)
            target_coords = np.where(target)
            
            if len(ref_coords[0]) > 0 and len(target_coords[0]) > 0:
                ref_centroid = np.array([np.mean(ref_coords[1]), np.mean(ref_coords[0]), np.mean(ref_coords[2])])
                target_centroid = np.array([np.mean(target_coords[1]), np.mean(target_coords[0]), np.mean(target_coords[2])])
                
                vec = target_centroid - ref_centroid
                vec_scaled = vec * np.array([xdim, ydim, zdim])
                
                r = np.linalg.norm(vec_scaled)
                if r > 0:
                    theta = np.arctan2(vec_scaled[0], -vec_scaled[1]) * 180 / np.pi
                    phi = np.arcsin(vec_scaled[2] / r) * 180 / np.pi
                else:
                    theta = 0.0
                    phi = 0.0
            else:
                theta = 0.0
                phi = 0.0
            
            # Calculate overlap (on CPU)
            overlap_mask = (reference == 1) & (target == 1)
            overlap_volume = np.sum(target == 1)
            if overlap_volume > 0:
                overlap_percent = np.sum(overlap_mask) / overlap_volume
            else:
                overlap_percent = 0.0
            
            return float(min_dist), float(phi), float(theta), float(overlap_percent), float(r5)
            
        except Exception as e:
            # Fallback to CPU if GPU fails - suppress repeated error messages
            # Only print once to avoid spam
            if not hasattr(self, '_gpu_error_logged'):
                print(f"GPU calculation failed, falling back to CPU: {e}")
                self._gpu_error_logged = True
            return self.calculate_distances_cpu(reference, target, xdim, ydim, zdim)


# Worker function for parallel processing (must be at module level for pickling)
def calculate_distances_worker(reference, target, xdim, ydim, zdim, ref_name, target_name):
    """Worker function for parallel distance calculation"""
    from scipy.ndimage import distance_transform_edt
    
    # Memory optimization: crop to bounding box
    ref_coords = np.where(reference)
    target_coords = np.where(target)
    
    if len(ref_coords[0]) == 0 or len(target_coords[0]) == 0:
        return {
            'Reference ROI': ref_name,
            'Target ROI': target_name,
            'Eucledian Distance (mm)': 0.0,
            'Phi (degrees)': 0.0,
            'Theta (degrees)': 0.0,
            '% of Target Overlap': 0.0,
            'Eucledian Distance (mm) 5th Percentile': 0.0
        }
    
    # Get bounding box with padding
    all_y = np.concatenate([ref_coords[0], target_coords[0]])
    all_x = np.concatenate([ref_coords[1], target_coords[1]])
    all_z = np.concatenate([ref_coords[2], target_coords[2]])
    
    y_min, y_max = max(0, all_y.min() - 10), min(reference.shape[0], all_y.max() + 10)
    x_min, x_max = max(0, all_x.min() - 10), min(reference.shape[1], all_x.max() + 10)
    z_min, z_max = max(0, all_z.min() - 2), min(reference.shape[2], all_z.max() + 2)
    
    # Crop to bounding box
    ref_cropped = reference[y_min:y_max, x_min:x_max, z_min:z_max]
    target_cropped = target[y_min:y_max, x_min:x_max, z_min:z_max]
    
    # Calculate distance transform
    aspect = [ydim, xdim, zdim]
    
    # Distance from each voxel to the nearest boundary of reference ROI
    # ~ref_cropped = outside reference ROI, so this gives distance from outside to boundary
    dist_ref = distance_transform_edt(~ref_cropped, sampling=aspect).astype(np.float32)
    
    # Distance from inside reference ROI to boundary
    dist_not_ref = distance_transform_edt(ref_cropped, sampling=aspect).astype(np.float32)
    
    # Get distances at target voxels
    target_voxels = target_cropped == 1
    
    # For surface-to-surface distance:
    # - If target voxel is outside reference: use dist_ref (distance to boundary)
    # - If target voxel is inside reference: use dist_not_ref (distance from inside to boundary)
    # We want the minimum distance from target to reference boundary
    distances = np.where(target_voxels, 
                        np.where(ref_cropped == 1, dist_not_ref, dist_ref),
                        0.0)[target_voxels]
    
    # Filter out zero values (shouldn't happen, but safety check)
    distances = distances[distances > 0]
    
    if len(distances) == 0:
        # If no positive distances, check if there's overlap (target inside reference)
        overlap_voxels = (ref_cropped == 1) & (target_cropped == 1)
        if np.any(overlap_voxels):
            # Target overlaps with reference - distance is 0 (they touch)
            min_dist = 0.0
            r5 = 0.0
        else:
            return {
                'Reference ROI': ref_name,
                'Target ROI': target_name,
                'Eucledian Distance (mm)': 0.0,
                'Phi (degrees)': 0.0,
                'Theta (degrees)': 0.0,
                '% of Target Overlap': 0.0,
                'Eucledian Distance (mm) 5th Percentile': 0.0
            }
    else:
        min_dist = float(np.min(distances))
        r5 = float(np.percentile(distances, 5)) if len(distances) > 1 else min_dist
    
    # Calculate angles
    ref_coords = np.where(reference)
    target_coords = np.where(target)
    
    if len(ref_coords[0]) > 0 and len(target_coords[0]) > 0:
        ref_centroid = np.array([np.mean(ref_coords[1]), np.mean(ref_coords[0]), np.mean(ref_coords[2])])
        target_centroid = np.array([np.mean(target_coords[1]), np.mean(target_coords[0]), np.mean(target_coords[2])])
        
        vec = target_centroid - ref_centroid
        vec_scaled = vec * np.array([xdim, ydim, zdim])
        
        r = np.linalg.norm(vec_scaled)
        if r > 0:
            theta = np.arctan2(vec_scaled[0], -vec_scaled[1]) * 180 / np.pi
            phi = np.arcsin(vec_scaled[2] / r) * 180 / np.pi
        else:
            theta = 0.0
            phi = 0.0
    else:
        theta = 0.0
        phi = 0.0
    
    # Calculate overlap
    overlap_mask = (reference == 1) & (target == 1)
    overlap_volume = np.sum(target == 1)
    if overlap_volume > 0:
        overlap_percent = np.sum(overlap_mask) / overlap_volume
    else:
        overlap_percent = 0.0
    
    return {
        'Reference ROI': ref_name,
        'Target ROI': target_name,
        'Eucledian Distance (mm)': round(min_dist, 2),
        'Phi (degrees)': round(phi, 2),
        'Theta (degrees)': round(theta, 2),
        '% of Target Overlap': round(overlap_percent, 2),
        'Eucledian Distance (mm) 5th Percentile': round(r5, 2)
    }


# DICOMScanner class (same as V2)
class HeadMaskLoader(QThread):
    """Thread for loading head/neck mask to prevent GUI freezing"""
    finished = pyqtSignal(object, object, object)  # external_mask, spatial_data, external_name
    error = pyqtSignal(str)
    
    def __init__(self, ct_folder, rtstruct_path=None):
        super().__init__()
        self.ct_folder = ct_folder
        self.rtstruct_path = rtstruct_path  # Optional - only needed if we want to try RTSTRUCT first
    
    def run(self):
        try:
            print(f"[HeadMaskLoader] Starting... CT folder: {self.ct_folder}, RTSTRUCT: {self.rtstruct_path}")
            # Use the same extraction methods from PythonCSVGeneratorV3
            # Create processor - rtstruct_path can be None if we're only using CT
            output_folder = os.path.dirname(self.ct_folder) if self.rtstruct_path is None else os.path.dirname(self.rtstruct_path)
            processor = PythonCSVGeneratorV3(self.ct_folder, self.rtstruct_path or '', output_folder)
            # extract_image_spatial_data returns (image_data, spatial_data) tuple
            print(f"[HeadMaskLoader] Extracting image spatial data from CT folder...")
            image_data, spatial_data = processor.extract_image_spatial_data(self.ct_folder)
            print(f"[HeadMaskLoader] Image data shape: {image_data.shape if image_data is not None else 'None'}, Spatial data entries: {len(spatial_data) if spatial_data else 0}")
            
            # OPTIMIZATION: Skip slow RTSTRUCT processing - use fast CT-based generation instead
            # RTSTRUCT processing can take minutes for large files, so we'll use CT-based generation
            # which is much faster (seconds instead of minutes)
            print(f"[HeadMaskLoader] Using fast CT-based mask generation (skipping RTSTRUCT for speed)...")
            
            # Generate mask directly from CT - much faster than RTSTRUCT processing
            external_mask = None
            external_name = "Body (CT-based)"
            
            if image_data is not None:
                external_mask = self._generate_body_mask_from_ct(image_data, spatial_data)
                if external_mask is not None:
                    print(f"[HeadMaskLoader] ✓ Successfully generated CT-based body mask: {external_mask.shape}, Volume: {np.sum(external_mask)} voxels")
                    self.finished.emit(external_mask, spatial_data, external_name)
                    return
                else:
                    self.error.emit("Failed to generate body mask from CT images")
                    return
            else:
                self.error.emit("Could not extract image data from CT folder")
                return
            
        except Exception as e:
            import traceback
            error_msg = f"[HeadMaskLoader] Error loading head mask: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(str(e))
    
    def _generate_body_mask_from_ct(self, image_3d, spatial_data):
        """Generate body mask from CT images using Stack Overflow approach: remove_small_objects method
        
        Strategy (from experimental_sagittal_mask_v4.py):
        1. Normalize image to 0-1 range
        2. Apply Otsu thresholding to find body/head region
        3. Remove small objects (CT bed, artifacts)
        4. Fill holes and smooth boundaries
        
        Args:
            image_3d: 3D numpy array of CT images (HU values)
            spatial_data: List of spatial information dictionaries
            
        Returns:
            3D binary mask of body outline, or None if generation fails
        """
        try:
            if image_3d is None or image_3d.size == 0:
                print("ERROR: CT image data is empty")
                return None
            
            print(f"Generating body mask from CT images using advanced method (shape: {image_3d.shape})...")
            
            # Step 1: Normalize image to 0-1 range for thresholding
            print("  Normalizing CT image...")
            image_min = image_3d.min()
            image_max = image_3d.max()
            if image_max > image_min:
                image_norm = (image_3d - image_min) / (image_max - image_min)
            else:
                image_norm = image_3d.astype(float)
            
            # Step 2: Apply Otsu thresholding to find body/head region
            print("  Applying Otsu thresholding...")
            from skimage import filters, morphology
            threshold_value = filters.threshold_otsu(image_norm)
            initial_mask = (image_norm > threshold_value).astype(np.uint8)
            
            print(f"  Otsu threshold: {threshold_value:.4f}")
            print(f"  Initial mask voxels: {np.sum(initial_mask):,}")
            
            # Step 3: Remove small objects (like CT bed lines, artifacts)
            print("  Removing small objects (CT bed, artifacts)...")
            total_voxels = initial_mask.size
            min_size = max(1000, int(total_voxels * 0.01))  # At least 1% of image or 1000 voxels
            print(f"  Minimum object size: {min_size:,} voxels")
            
            # Remove small objects (2D operation on each slice for better performance)
            mask_3d = np.zeros_like(initial_mask)
            for z_idx in range(initial_mask.shape[0]):
                slice_mask = initial_mask[z_idx, :, :]
                if np.any(slice_mask):
                    # Remove small objects in this slice
                    cleaned_slice = morphology.remove_small_objects(
                        slice_mask.astype(bool), 
                        min_size=min_size // initial_mask.shape[0]  # Adjust for 2D
                    )
                    # Remove small holes
                    cleaned_slice = morphology.remove_small_holes(
                        cleaned_slice, 
                        area_threshold=min_size // (initial_mask.shape[0] * 10)
                    )
                    mask_3d[z_idx, :, :] = cleaned_slice.astype(np.uint8)
            
            # Step 4: Final cleanup with morphological operations
            print("  Final cleanup with morphological operations...")
            for z_idx in range(mask_3d.shape[0]):
                if np.any(mask_3d[z_idx]):
                    # Fill remaining holes
                    mask_3d[z_idx] = binary_fill_holes(mask_3d[z_idx]).astype(np.uint8)
                    
                    # Smooth boundaries
                    try:
                        mask_3d[z_idx] = morphology.binary_closing(
                            mask_3d[z_idx], footprint=morphology.disk(3)
                        ).astype(np.uint8)
                    except TypeError:
                        mask_3d[z_idx] = morphology.binary_closing(
                            mask_3d[z_idx], selem=morphology.disk(3)
                        ).astype(np.uint8)
            
            print(f"  Final mask voxels: {np.sum(mask_3d):,}")
            
            if np.sum(mask_3d) == 0:
                print("ERROR: Generated body mask is empty")
                return None
            
            print(f"✓ Successfully generated body mask: {mask_3d.shape}, Volume: {np.sum(mask_3d)} voxels")
            return mask_3d.astype(np.uint8)
            
        except Exception as e:
            print(f"Error generating body mask from CT: {e}")
            import traceback
            traceback.print_exc()
            return None


class DICOMScanner(QThread):
    """Thread for scanning DICOM files"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
    
    def run(self):
        try:
            results = {
                'ct_folder': None,
                'rtstruct_files': [],
                'abas_rtstruct_files': [],
                'merged_rtstruct': None,
                'mrn': None,
                'patient_info': {}
            }
            
            self.progress.emit(10, "Scanning folder structure...")
            folder = Path(self.folder_path)
            
            self.progress.emit(20, "Extracting MRN...")
            results['mrn'] = self._extract_mrn(folder)
            
            self.progress.emit(30, "Looking for CT folder...")
            ct_folders = list(folder.glob('**/CT'))
            if ct_folders:
                results['ct_folder'] = str(ct_folders[0])
            else:
                dicom_files = list(folder.glob('**/*.dcm'))
                if dicom_files:
                    results['ct_folder'] = str(dicom_files[0].parent)
            
            self.progress.emit(50, "Scanning for RTSTRUCT files...")
            all_dicom = list(folder.glob('**/*.dcm')) + list(folder.glob('**/*.DCM'))
            
            all_rtstructs = []
            for dicom_file in all_dicom:
                try:
                    ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
                    if hasattr(ds, 'Modality'):
                        if ds.Modality == 'RTSTRUCT':
                            all_rtstructs.append((str(dicom_file), ds))
                except:
                    continue
            
            abas_files_set = set()
            normal_files_set = set()
            
            for dicom_file, ds in all_rtstructs:
                is_abas = self._is_abas_rtstruct(dicom_file, ds)
                if is_abas:
                    abas_files_set.add(dicom_file)
                else:
                    normal_files_set.add(dicom_file)
            
            results['abas_rtstruct_files'] = list(abas_files_set)
            results['rtstruct_files'] = list(normal_files_set)
            
            self.progress.emit(70, "Looking for merged structure set...")
            merged_files = list(folder.glob('**/MergedSS.dcm'))
            if merged_files:
                results['merged_rtstruct'] = str(merged_files[0])
            
            if results['rtstruct_files']:
                try:
                    ds = pydicom.dcmread(results['rtstruct_files'][0], stop_before_pixels=True)
                    if hasattr(ds, 'PatientName'):
                        results['patient_info']['name'] = str(ds.PatientName)
                    if hasattr(ds, 'PatientID'):
                        results['patient_info']['id'] = str(ds.PatientID)
                except:
                    pass
            
            self.progress.emit(100, "Scan complete!")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def _is_abas_rtstruct(self, file_path, dicom_dataset):
        """Determine if RTSTRUCT is ABAS"""
        file_str = str(file_path).lower()
        abas_keywords = ['abas', 'atlas', 'auto', 'autoseg', 'automatic']
        if any(keyword in file_str for keyword in abas_keywords):
            return True
        
        if hasattr(dicom_dataset, 'SeriesDescription'):
            series_desc = str(dicom_dataset.SeriesDescription).lower()
            if any(keyword in series_desc for keyword in abas_keywords):
                return True
        
        if hasattr(dicom_dataset, 'StructureSetROISequence'):
            abas_count = 0
            total = len(dicom_dataset.StructureSetROISequence)
            for roi in dicom_dataset.StructureSetROISequence:
                roi_name = str(roi.ROIName).lower()
                if any(keyword in roi_name for keyword in abas_keywords):
                    abas_count += 1
            if total > 0 and (abas_count / total) > 0.3:
                return True
        
        return False
    
    def _extract_mrn(self, folder):
        """Extract MRN from folder name or DICOM files"""
        folder_name = folder.name
        import re
        mrn_match = re.search(r'\d{6,}', folder_name)
        if mrn_match:
            return mrn_match.group()
        
        dicom_files = list(folder.glob('**/*.dcm'))[:5]
        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
                if hasattr(ds, 'PatientID'):
                    return str(ds.PatientID)
            except:
                continue
        
        return folder_name


# Centroid3DPlot class (same as V2)
class Centroid3DPlot(FigureCanvas):
    """3D visualization of ROI centroids"""
    
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(10, 8), facecolor='#2b2b2b')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1e1e1e')
        self.fig.patch.set_facecolor('#2b2b2b')
        
        # Set default plot dimensions
        self.fig.subplots_adjust(
            top=1.0,
            bottom=0.135,
            left=0.125,
            right=0.9,
            hspace=0.19,
            wspace=0.2
        )
        
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#404040')
        self.ax.yaxis.pane.set_edgecolor('#404040')
        self.ax.zaxis.pane.set_edgecolor('#404040')
        self.ax.grid(True, color='#404040', alpha=0.3)
        
        self.ax.set_xlabel('X (mm)', color='white', fontsize=10)
        self.ax.set_ylabel('Y (mm)', color='white', fontsize=10)
        self.ax.set_zlabel('Z (mm)', color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelsize=9)
        
        # Enable interactive 3D rotation (left-click drag to rotate, right-click to pan, scroll to zoom)
        self.ax.mouse_init()
        
        # Connect scroll wheel for zoom
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        self.centroid_data = None
        self.distance_data = None
        self.selected_roi = None
        self.gtv_roi = None
        self.scatter_plot = None
        self.line_plots = []
        self.head_mask_data = None  # Store head/neck mask data
        self.head_mask_spatial_data = None  # Store spatial data for mask
        self.head_mask_name = None  # Store mask name
        self.head_surface = None  # Store 3D surface plot
        self.show_head_mask = False  # Track head mask state
        self.scan_results = None  # Store scan results or user-selected files
        self.parent_window = None  # Reference to main window for QMessageBox
        self.head_mask_loader = None  # Thread for loading head mask
        self.loading_head_mask = False  # Track if head mask is being loaded
        self.mask_spatial_data = None  # Store spatial data for mask coordinate conversion
        
        # Rotation angles for alignment (temporary)
        self.mask_rotation_x = 0.0
        self.mask_rotation_y = 0.0
        self.mask_rotation_z = 0.0
        self.centroid_rotation_x = 0.0
        self.centroid_rotation_y = 0.0
        self.centroid_rotation_z = 0.0
        
        # Rotation angles for alignment (temporary)
        self.mask_rotation_x = 0.0
        self.mask_rotation_y = 0.0
        self.mask_rotation_z = 0.0
        self.centroid_rotation_x = 0.0
        self.centroid_rotation_y = 0.0
        self.centroid_rotation_z = 0.0
    
    def _load_head_mask_async(self, centroid_df):
        """Load head mask asynchronously using a background thread"""
        try:
            ct_folder = None
            rtstruct_path = None

            # First, check if we have scan_results from parent window (from MRN folder scan)
            if self.parent_window and hasattr(self.parent_window, 'scan_results') and self.parent_window.scan_results:
                scan_results = self.parent_window.scan_results
                # Use CT folder from scan results
                if scan_results.get('ct_folder') and os.path.exists(scan_results['ct_folder']):
                    ct_folder = scan_results['ct_folder']
                    print(f"Using CT folder from scan results: {ct_folder}")

                    # Find RTSTRUCT file from scan results (priority: MergedSS.dcm > RTSTRUCT.dcm > ABAS.dcm)
                    if scan_results.get('merged_rtstruct') and os.path.exists(scan_results['merged_rtstruct']):
                        rtstruct_path = scan_results['merged_rtstruct']
                    elif scan_results.get('rtstruct_files') and len(scan_results['rtstruct_files']) > 0:
                        # Prefer RTSTRUCT.dcm over ABAS if both exist
                        rtstruct_path = scan_results['rtstruct_files'][0]
                    elif scan_results.get('abas_rtstruct_files') and len(scan_results['abas_rtstruct_files']) > 0:
                        rtstruct_path = scan_results['abas_rtstruct_files'][0]

                    if rtstruct_path:
                        print(f"Using RTSTRUCT from scan results: {rtstruct_path}")
                    else:
                        print("RTSTRUCT not found in scan results - will generate mask from CT images only")

            # If no scan results, check if CT folder was previously stored
            if not ct_folder and self.parent_window and hasattr(self.parent_window, '_head_mask_ct_folder'):
                if self.parent_window._head_mask_ct_folder and os.path.exists(self.parent_window._head_mask_ct_folder):
                    ct_folder = self.parent_window._head_mask_ct_folder
                    rtstruct_path = getattr(self.parent_window, '_head_mask_rtstruct', None)
                    print(f"Using previously stored CT folder: {ct_folder}")

            # If still no CT folder, prompt user to select MRN folder
            if not ct_folder:
                if not self.parent_window:
                    print("ERROR: Cannot prompt for files - no parent window reference")
                    self.loading_head_mask = False
                    return

                from PyQt5.QtWidgets import QFileDialog, QMessageBox

                # Show folder selection dialog only if scan results are not available
                mrn_folder = QFileDialog.getExistingDirectory(
                    self.parent_window,
                    "Select MRN Folder for Head/Neck Mask\n\n"
                    "The tool will auto-detect:\n"
                    "- CT folder and DICOM files (required)\n\n"
                    "- The head/neck mask will be generated from CT images.\n"
                    "- RTSTRUCT files are not required for this feature.",
                    ""
                )

                if not mrn_folder:
                    self.loading_head_mask = False
                    return  # User cancelled

                # Auto-detect CT folder and RTSTRUCT file using same logic as DICOMScanner
                mrn_path = Path(mrn_folder)

                # Find CT folder (same logic as DICOMScanner)
                ct_folders = list(mrn_path.glob('**/CT'))
                if ct_folders:
                    ct_folder = str(ct_folders[0])
                else:
                    # Try to find DICOM files directly and check if they're CT
                    dicom_files = list(mrn_path.glob('**/*.dcm')) + list(mrn_path.glob('**/*.DCM'))
                    if dicom_files:
                        # Check if any are CT images (not RTSTRUCT)
                        for dicom_file in dicom_files[:20]:  # Check first 20
                            try:
                                ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
                                if hasattr(ds, 'Modality') and ds.Modality == 'CT':
                                    ct_folder = str(dicom_file.parent)
                                    break
                            except:
                                continue

                if not ct_folder:
                    QMessageBox.warning(self.parent_window, "CT Not Found",
                                      f"Could not find CT folder or CT DICOM files in:\n{mrn_folder}\n\n"
                                      f"Please ensure the folder contains a 'CT' subfolder or CT DICOM files.")
                    self.loading_head_mask = False
                    return

                # Find RTSTRUCT file (priority: MergedSS.dcm > RTSTRUCT.dcm > ABAS.dcm)
                # Priority 1: MergedSS.dcm
                merged_files = list(mrn_path.glob('**/MergedSS.dcm'))
                if merged_files:
                    rtstruct_path = str(merged_files[0])
                else:
                    # Priority 2: RTSTRUCT.dcm (preferred over ABAS)
                    rtstruct_files = list(mrn_path.glob('**/RTSTRUCT*.dcm')) + list(mrn_path.glob('**/RTSTRUCT*.DCM'))
                    abas_files = list(mrn_path.glob('**/ABAS*.dcm')) + list(mrn_path.glob('**/ABAS*.DCM'))

                    # If both exist, prefer RTSTRUCT.dcm
                    if rtstruct_files:
                        rtstruct_path = str(rtstruct_files[0])
                    elif abas_files:
                        rtstruct_path = str(abas_files[0])
                    else:
                        # Try to find any RTSTRUCT file by modality
                        all_dicom = list(mrn_path.glob('**/*.dcm')) + list(mrn_path.glob('**/*.DCM'))
                        for dicom_file in all_dicom:
                            try:
                                ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
                                if hasattr(ds, 'Modality') and ds.Modality == 'RTSTRUCT':
                                    rtstruct_path = str(dicom_file)
                                    break
                            except:
                                continue

                # RTSTRUCT is optional - if not found, we'll generate mask from CT only (no warning)
                if not rtstruct_path or not os.path.exists(rtstruct_path):
                    rtstruct_path = None  # This is okay - CT-based generation will be used
                    print(f"RTSTRUCT not found - will generate body mask from CT images only")

                # Store selections in parent window for future use
                if not hasattr(self.parent_window, '_head_mask_ct_folder'):
                    self.parent_window._head_mask_ct_folder = None
                    self.parent_window._head_mask_rtstruct = None
                self.parent_window._head_mask_ct_folder = ct_folder
                self.parent_window._head_mask_rtstruct = rtstruct_path
                print(f"Selected MRN folder: {mrn_folder}")
                print(f"Auto-detected CT folder: {ct_folder}")
                if rtstruct_path:
                    print(f"Auto-detected RTSTRUCT: {rtstruct_path} (optional - will try to use if available)")
                else:
                    print(f"RTSTRUCT not found - will generate mask from CT images only")

            # ct_folder is required, rtstruct_path is optional (will generate from CT if not found)
            if not ct_folder:
                self.loading_head_mask = False
                return  # No CT folder found

            # If mask is already loaded and we're just re-drawing, skip re-generation
            if self.head_mask_data is not None and self.head_mask_spatial_data is not None:
                # Just redraw with current data
                self._on_head_mask_loaded(self.head_mask_data, self.head_mask_spatial_data, self.head_mask_name or "Body (CT-based)", centroid_df)
                return

            # Start loading head mask in background thread
            # RTSTRUCT is optional - if not provided, will generate mask from CT only
            self.loading_head_mask = True
            self.head_mask_loader = HeadMaskLoader(ct_folder, rtstruct_path)  # rtstruct_path can be None
            self.head_mask_loader.finished.connect(lambda mask, spatial, name: self._on_head_mask_loaded(mask, spatial, name, centroid_df))
            self.head_mask_loader.error.connect(self._on_head_mask_error)
            self.head_mask_loader.start()

        except Exception as e:
            print(f"Error in _load_head_mask_async: {e}")
            import traceback
            traceback.print_exc()
            self.loading_head_mask = False
    
    def _generate_mask_from_scan_results(self, ct_folder, rtstruct_path=None):
        """Generate head/neck mask directly from scan results (no user prompt)"""
        try:
            if not ct_folder or not os.path.exists(ct_folder):
                print("CT folder not found in scan results")
                return
            
            # Skip if mask is already loaded
            if self.head_mask_data is not None and self.head_mask_spatial_data is not None:
                print("Mask already loaded, skipping re-generation")
                return
            
            # Start loading head mask in background thread
            # RTSTRUCT is optional - if not provided, will generate mask from CT only
            self.loading_head_mask = True
            self.head_mask_loader = HeadMaskLoader(ct_folder, rtstruct_path)  # rtstruct_path can be None
            # Pass None for centroid_df since we're generating mask before centroids are loaded
            self.head_mask_loader.finished.connect(lambda mask, spatial, name: self._on_head_mask_loaded(mask, spatial, name, None))
            self.head_mask_loader.error.connect(self._on_head_mask_error)
            self.head_mask_loader.start()
            
            # Store CT folder in parent window for future use
            if self.parent_window:
                if not hasattr(self.parent_window, '_head_mask_ct_folder'):
                    self.parent_window._head_mask_ct_folder = None
                    self.parent_window._head_mask_rtstruct = None
                self.parent_window._head_mask_ct_folder = ct_folder
                self.parent_window._head_mask_rtstruct = rtstruct_path
            
        except Exception as e:
            print(f"Error in _generate_mask_from_scan_results: {e}")
            import traceback
            traceback.print_exc()
            self.loading_head_mask = False
    
    def _on_head_mask_loaded(self, external_mask, spatial_data, external_name, centroid_df):
        """Called when head mask is loaded - draw it on the plot"""
        try:
            self.loading_head_mask = False
            
            # Skip if this is a duplicate call with the same mask data (prevent duplicate drawing)
            if (self.head_mask_data is not None and 
                self.head_mask_spatial_data is not None and
                external_mask is not None and
                np.array_equal(self.head_mask_data, external_mask)):
                print("Mask already loaded and drawn, skipping duplicate call")
                return
            
            print(f"Head mask loaded successfully: {external_name}")
            print(f"  Mask shape: {external_mask.shape if external_mask is not None else 'None'}")
            print(f"  Spatial data entries: {len(spatial_data) if spatial_data else 0}")
            
            # Update progress log in parent window if available (only once)
            if self.parent_window and hasattr(self.parent_window, 'progress_text'):
                # Check if we already logged this mask
                if not hasattr(self, '_mask_logged') or not self._mask_logged:
                    import pandas as pd
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    self.parent_window.progress_text.append(f"[{timestamp}] ✓ Head/neck mask generated successfully!")
                    self.parent_window.progress_text.verticalScrollBar().setValue(
                        self.parent_window.progress_text.verticalScrollBar().maximum()
                    )
                    self._mask_logged = True
            
            # Store mask data for re-drawing
            self.head_mask_data = external_mask
            self.head_mask_spatial_data = spatial_data
            self.head_mask_name = external_name
            
            # Automatically check "Show Head/Neck Mask" checkbox if available (mask is ready to display)
            if self.parent_window and hasattr(self.parent_window, 'show_head_mask_check'):
                self.parent_window.show_head_mask_check.setChecked(True)
                self.show_head_mask = True
            
            # EXACT from experimental: Draw mask first, then overlay ALL centroids
            # Clear axes and set up for mask visualization (EXACT from experimental)
            self.ax.clear()
            self.ax.set_facecolor('#1e1e1e')
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('#404040')
            self.ax.yaxis.pane.set_edgecolor('#404040')
            self.ax.zaxis.pane.set_edgecolor('#404040')
            # Remove grid lines (EXACT from experimental)
            self.ax.grid(False)
            # Remove axis lines (EXACT from experimental)
            self.ax.xaxis.line.set_visible(False)
            self.ax.yaxis.line.set_visible(False)
            self.ax.zaxis.line.set_visible(False)
            # Remove tick marks (EXACT from experimental)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_zticks([])
            self.ax.set_xlabel('X', color='white', fontsize=10)
            self.ax.set_ylabel('Y', color='white', fontsize=10)
            self.ax.set_zlabel('Z', color='white', fontsize=10)
            
            # Set initial view angle to show mask clearly (elevation, azimuth)
            # Elevation: 20 degrees (slightly above), Azimuth: 45 degrees (angled view)
            self.ax.view_init(elev=20, azim=45)
            
            # Draw mask first, then overlay centroids based on current selection
            # Only show all centroids if "Show All" is checked, otherwise show only selected ones
            # IMPORTANT: Do NOT show all centroids by default - only if "Show All" checkbox is checked
            if self.parent_window and hasattr(self.parent_window, 'show_all_check') and self.parent_window.show_all_check.isChecked():
                print("Drawing mask with ALL centroids overlay (Show All is checked)...")
                centroids_to_show = centroid_df if centroid_df is not None else None
            else:
                # Only show selected centroids (GTV and/or selected ROI) - do NOT show all by default
                print("Drawing mask with selected centroids only (Show All not checked)...")
                centroids_to_show = None
                if centroid_df is not None and len(centroid_df) > 0:
                    # Get current GTV and selected ROI from parent window
                    if self.parent_window:
                        gtv_name = self.parent_window.gtv_name if hasattr(self.parent_window, 'gtv_name') else None
                        selected_roi = self.parent_window.roi_combo.currentText() if hasattr(self.parent_window, 'roi_combo') else None
                        if selected_roi == "None":
                            selected_roi = None
                        
                        # Filter centroids to show only selected ones
                        if gtv_name or selected_roi:
                            possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
                            roi_col = next((col for col in possible_roi if col in centroid_df.columns), centroid_df.columns[0])
                            rois_to_show = []
                            if gtv_name:
                                rois_to_show.append(gtv_name)
                            if selected_roi:
                                rois_to_show.append(selected_roi)
                            if rois_to_show:
                                centroids_to_show = centroid_df[centroid_df[roi_col].isin(rois_to_show)]
                                print(f"  Showing only selected ROIs: {rois_to_show}")
                        else:
                            print("  No GTV or ROI selected - showing mask only (no centroids)")
            
            success = self._draw_head_mask(external_mask, spatial_data, external_name, centroids_to_show)
            
            # Force redraw to ensure mask is visible
            if success:
                self.draw()
            if not success:
                # Show warning to user if drawing failed
                if self.parent_window:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(self.parent_window, "Head Mask Display", 
                                      f"Found external contour '{external_name}' but could not display it.\n\n"
                                      f"This may be due to:\n"
                                      f"- Invalid mask data\n"
                                      f"- Missing spatial information\n"
                                      f"- Marching cubes algorithm failure\n\n"
                                      f"Check console for details.")
        except Exception as e:
            print(f"Error drawing head mask: {e}")
            import traceback
            traceback.print_exc()
            # Show error to user
            if self.parent_window:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self.parent_window, "Head Mask Error", 
                                  f"Error displaying head/neck mask:\n{str(e)}\n\n"
                                  f"Check console for details.")
    
    def _on_head_mask_error(self, error_msg):
        """Called when head mask loading fails"""
        self.loading_head_mask = False
        print(f"Head mask loading error: {error_msg}")
        if self.parent_window:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self.parent_window, "Head Mask Not Available", 
                              f"Could not load head/neck mask:\n\n{error_msg}\n\n"
                              f"This is not an error - the visualization will work without it.\n"
                              f"You can still view ROI centroids and distances.")
        
    def plot_centroids(self, centroid_df, distance_df=None, gtv_name=None, selected_roi=None, 
                      show_head_mask=False, scan_results=None):
        """Plot all centroids in 3D space"""
        self.centroid_data = centroid_df
        self.distance_data = distance_df
        self.gtv_roi = gtv_name
        self.selected_roi = selected_roi
        self.scan_results = scan_results
        
        # Clear axes - mask will be re-drawn after centroids if show_head_mask is True
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#404040')
        self.ax.yaxis.pane.set_edgecolor('#404040')
        self.ax.zaxis.pane.set_edgecolor('#404040')
        self.ax.grid(True, color='#404040', alpha=0.3)
        self.ax.set_xlabel('X (mm)', color='white', fontsize=10)
        self.ax.set_ylabel('Y (mm)', color='white', fontsize=10)
        self.ax.set_zlabel('Z (mm)', color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelsize=9)
        
        if centroid_df is None or len(centroid_df) == 0:
            self.draw()
            return
        
        possible_x = ['x coordinate', 'x_coordinate', 'X', 'x']
        possible_y = ['y coordinate', 'y_coordinate', 'Y', 'y']
        possible_z = ['z coordinate', 'z_coordinate', 'Z', 'z']
        possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
        
        x_col = next((col for col in possible_x if col in centroid_df.columns), centroid_df.columns[1] if len(centroid_df.columns) > 1 else centroid_df.columns[0])
        y_col = next((col for col in possible_y if col in centroid_df.columns), centroid_df.columns[2] if len(centroid_df.columns) > 2 else centroid_df.columns[0])
        z_col = next((col for col in possible_z if col in centroid_df.columns), centroid_df.columns[3] if len(centroid_df.columns) > 3 else centroid_df.columns[0])
        roi_col = next((col for col in possible_roi if col in centroid_df.columns), centroid_df.columns[0])
        
        # Show ALL centroids by default (like experimental) - don't filter by GTV/selected_roi
        # Store coordinates for potential conversion to voxel space if mask is loaded
        x = centroid_df[x_col].values
        y = centroid_df[y_col].values
        z = centroid_df[z_col].values
        roi_names = centroid_df[roi_col].values
        
        # Store raw coordinates for mask overlay (will be converted to voxel if mask exists)
        self.centroid_coords = {
            'x': x, 'y': y, 'z': z, 'roi_names': roi_names,
            'x_col': x_col, 'y_col': y_col, 'z_col': z_col, 'roi_col': roi_col
        }
        
        # If mask is loaded, centroids will be overlaid by _overlay_centroids_v4
        # Otherwise, show them in world coordinates with color coding
        if self.head_mask_data is None:
            # No mask - show centroids in world coordinates with color coding
            # Fix coordinate system orientation:
            if len(x) > 0:
                max_x = np.max(x) if len(x) > 0 else 0
                x = max_x - x  # Flip X coordinates (left/right)
            # Don't flip Z - keep it as is (Z axis is correct)
            
            colors = []
            sizes = []
            for i, roi in enumerate(roi_names):
                if gtv_name and roi == gtv_name:
                    colors.append('#FF6B6B')
                    sizes.append(30)  # Reduced from 200
                elif selected_roi and roi == selected_roi:
                    colors.append('#4ECDC4')
                    sizes.append(25)  # Reduced from 150
                else:
                    colors.append('#95E1D3')
                    sizes.append(20)  # Reduced from 100
            
            # Store plotted positions for distance line drawing
            self.plotted_centroid_positions = {
                'x': x.tolist() if hasattr(x, 'tolist') else list(x),
                'y': y.tolist() if hasattr(y, 'tolist') else list(y),
                'z': z.tolist() if hasattr(z, 'tolist') else list(z),
                'roi_names': roi_names.tolist() if hasattr(roi_names, 'tolist') else list(roi_names)
            }
            
            self.scatter_plot = self.ax.scatter(x, y, z, c=colors, s=sizes, 
                                                alpha=0.8, edgecolors='white', 
                                                linewidths=1, depthshade=True)  # Reduced linewidth from 1.5
            
            for i, roi in enumerate(roi_names):
                if gtv_name and roi == gtv_name:
                    text_obj = self.ax.text(x[i], y[i], z[i], f'  {roi} (GTV)', 
                               color='#FF6B6B', fontsize=9, fontweight='bold')
                    # GTV label: no rotation (rotation=0) - ensure it's visible
                    # No rotation needed for GTV
                elif selected_roi and roi == selected_roi:
                    # Use same approach as GTV ROI (ax.text with 3D coordinates)
                    # Apply rotation using transform
                    text_obj = self.ax.text(x[i], y[i], z[i], f'  {roi}', 
                               color='#4ECDC4', fontsize=8, zorder=11)
                    # Try to apply 90 degree clockwise rotation using transform
                    try:
                        from matplotlib import transforms
                        # Get the text's transform and apply rotation
                        t = text_obj.get_transform()
                        rot_transform = transforms.Affine2D().rotate_deg(90) + t
                        text_obj.set_transform(rot_transform)
                    except:
                        # If rotation fails, just use the text without rotation
                        pass
                else:
                    self.ax.text(x[i], y[i], z[i], f'  {roi}', 
                               color='#95E1D3', fontsize=7, alpha=0.7)
        
        # Draw head/neck mask if requested
        self.show_head_mask = show_head_mask
        if show_head_mask:
            # Only re-draw mask if it's already loaded (don't reload when ROI changes)
            if self.head_mask_data is not None:
                # Mask already loaded - re-draw it with current centroids (respecting Show All checkbox)
                print("Re-drawing existing head mask with current centroids...")
                # Determine which centroids to show based on Show All checkbox
                if self.parent_window and hasattr(self.parent_window, 'show_all_check') and self.parent_window.show_all_check.isChecked():
                    centroids_to_show = centroid_df
                else:
                    # Only show selected centroids (GTV and/or selected ROI)
                    centroids_to_show = None
                    if centroid_df is not None and len(centroid_df) > 0:
                        gtv_name = self.gtv_roi if hasattr(self, 'gtv_roi') else None
                        selected_roi = self.selected_roi if hasattr(self, 'selected_roi') else None
                        if gtv_name or selected_roi:
                            possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
                            roi_col = next((col for col in possible_roi if col in centroid_df.columns), centroid_df.columns[0])
                            rois_to_show = []
                            if gtv_name:
                                rois_to_show.append(gtv_name)
                            if selected_roi:
                                rois_to_show.append(selected_roi)
                            if rois_to_show:
                                centroids_to_show = centroid_df[centroid_df[roi_col].isin(rois_to_show)]
                
                self._draw_head_mask(self.head_mask_data, self.head_mask_spatial_data, 
                                   self.head_mask_name, centroids_to_show)
        
        # Draw distance line AFTER mask overlay (if mask is shown) so it uses the correct transformed coordinates
        if gtv_name and selected_roi and distance_df is not None:
            # Clear any existing distance lines first
            if hasattr(self, 'line_plots'):
                for line in self.line_plots:
                    try:
                        line.remove()
                    except:
                        pass
                self.line_plots = []
            else:
                self.line_plots = []
            # Draw the distance line using the current plotted positions (from mask overlay or regular plot)
            self._draw_distance_line(gtv_name, selected_roi, distance_df, centroid_df)
        
        # Handle case when mask checkbox is unchecked
        if not show_head_mask:
            # Remove head surface if checkbox is unchecked
            if self.head_surface:
                try:
                    self.head_surface.remove()
                except:
                    pass
                self.head_surface = None
            # Cancel any ongoing head mask loading
            if hasattr(self, 'head_mask_loader') and self.head_mask_loader and self.head_mask_loader.isRunning():
                self.head_mask_loader.terminate()
                self.head_mask_loader.wait()
                self.loading_head_mask = False
        
        self.draw()
    
    def _draw_distance_line(self, ref_roi, target_roi, distance_df, centroid_df):
        """Draw line connecting two ROIs with distance annotation"""
        mask = ((distance_df['Reference ROI'] == ref_roi) & 
                (distance_df['Target ROI'] == target_roi)) | \
               ((distance_df['Reference ROI'] == target_roi) & 
                (distance_df['Target ROI'] == ref_roi))
        
        if not mask.any():
            return
        
        dist_row = distance_df[mask].iloc[0]
        distance = dist_row['Eucledian Distance (mm)']
        
        possible_x = ['x coordinate', 'x_coordinate', 'X', 'x']
        possible_y = ['y coordinate', 'y_coordinate', 'Y', 'y']
        possible_z = ['z coordinate', 'z_coordinate', 'Z', 'z']
        possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
        
        x_col = next((col for col in possible_x if col in centroid_df.columns), centroid_df.columns[1] if len(centroid_df.columns) > 1 else centroid_df.columns[0])
        y_col = next((col for col in possible_y if col in centroid_df.columns), centroid_df.columns[2] if len(centroid_df.columns) > 2 else centroid_df.columns[0])
        z_col = next((col for col in possible_z if col in centroid_df.columns), centroid_df.columns[3] if len(centroid_df.columns) > 3 else centroid_df.columns[0])
        roi_col = next((col for col in possible_roi if col in centroid_df.columns), centroid_df.columns[0])
        
        # Get the actual plotted coordinates from the stored centroid positions
        # Check if we have transformed coordinates from mask overlay
        if hasattr(self, 'plotted_centroid_positions') and self.plotted_centroid_positions:
            # Use the stored plotted positions (from _overlay_centroids_v4 or plot_centroids)
            # Make ROI name comparison more robust (case-insensitive, strip whitespace)
            ref_roi_clean = str(ref_roi).strip().lower()
            target_roi_clean = str(target_roi).strip().lower()
            
            ref_idx = None
            target_idx = None
            for i, roi_name in enumerate(self.plotted_centroid_positions.get('roi_names', [])):
                roi_name_clean = str(roi_name).strip().lower()
                if roi_name_clean == ref_roi_clean:
                    ref_idx = i
                if roi_name_clean == target_roi_clean:
                    target_idx = i
            
            if ref_idx is not None and target_idx is not None:
                x_line = [self.plotted_centroid_positions['x'][ref_idx], 
                         self.plotted_centroid_positions['x'][target_idx]]
                y_line = [self.plotted_centroid_positions['y'][ref_idx], 
                         self.plotted_centroid_positions['y'][target_idx]]
                z_line = [self.plotted_centroid_positions['z'][ref_idx], 
                         self.plotted_centroid_positions['z'][target_idx]]
            else:
                # Fallback: use raw coordinates and apply same transformations as centroids
                ref_centroid = centroid_df[centroid_df[roi_col] == ref_roi].iloc[0]
                target_centroid = centroid_df[centroid_df[roi_col] == target_roi].iloc[0]
                if self.head_mask_data is None:
                    # No mask - use same flipping as plot_centroids
                    max_x = centroid_df[x_col].max() if len(centroid_df) > 0 else 0
                    x_line = [max_x - ref_centroid[x_col], max_x - target_centroid[x_col]]
                    y_line = [ref_centroid[y_col], target_centroid[y_col]]
                    z_line = [ref_centroid[z_col], target_centroid[z_col]]
                else:
                    # Mask loaded - need to apply same transformations as _overlay_centroids_v4
                    # Get mask dimensions
                    mask_z, mask_y, mask_x = self.head_mask_data.shape
                    
                    # Convert to voxel coordinates (assuming they're already in voxel space)
                    ref_vox_x, ref_vox_y, ref_vox_z = ref_centroid[x_col], ref_centroid[y_col], ref_centroid[z_col]
                    target_vox_x, target_vox_y, target_vox_z = target_centroid[x_col], target_centroid[y_col], target_centroid[z_col]
                    
                    # Apply same transformations as _overlay_centroids_v4
                    # Transform centroids to match mask coordinate system
                    ref_plot_x = (mask_z - 1) - ref_vox_z
                    ref_plot_y = ref_vox_y
                    ref_plot_z = (mask_x - 1) - ref_vox_x
                    ref_plot_z = mask_x - 1 - ref_plot_z
                    
                    target_plot_x = (mask_z - 1) - target_vox_z
                    target_plot_y = target_vox_y
                    target_plot_z = (mask_x - 1) - target_vox_x
                    target_plot_z = mask_x - 1 - target_plot_z
                    
                    # Apply all rotations (same as _overlay_centroids_v4)
                    # Rotate 90 degrees counter-clockwise around X axis
                    center_y = mask_y / 2
                    center_z = mask_x / 2
                    ref_y_centered = ref_plot_y - center_y
                    ref_z_centered = ref_plot_z - center_z
                    ref_plot_y = ref_z_centered + center_y
                    ref_plot_z = -ref_y_centered + center_z
                    
                    target_y_centered = target_plot_y - center_y
                    target_z_centered = target_plot_z - center_z
                    target_plot_y = target_z_centered + center_y
                    target_plot_z = -target_y_centered + center_z
                    
                    # Rotate 90 degrees clockwise around Y axis
                    center_x = mask_z / 2
                    ref_x_centered = ref_plot_x - center_x
                    ref_z_centered = ref_plot_z - center_z
                    ref_plot_x = ref_z_centered + center_x
                    ref_plot_z = -ref_x_centered + center_z
                    
                    target_x_centered = target_plot_x - center_x
                    target_z_centered = target_plot_z - center_z
                    target_plot_x = target_z_centered + center_x
                    target_plot_z = -target_x_centered + center_z
                    
                    # Rotate 180 degrees clockwise around Z axis
                    ref_x_centered = ref_plot_x - center_x
                    ref_y_centered = ref_plot_y - center_y
                    ref_plot_x = -ref_x_centered + center_x
                    ref_plot_y = -ref_y_centered + center_y
                    
                    target_x_centered = target_plot_x - center_x
                    target_y_centered = target_plot_y - center_y
                    target_plot_x = -target_x_centered + center_x
                    target_plot_y = -target_y_centered + center_y
                    
                    # Rotate 180 degrees around Y axis
                    ref_x_centered = ref_plot_x - center_x
                    ref_z_centered = ref_plot_z - center_z
                    ref_plot_x = -ref_x_centered + center_x
                    ref_plot_z = -ref_z_centered + center_z
                    
                    target_x_centered = target_plot_x - center_x
                    target_z_centered = target_plot_z - center_z
                    target_plot_x = -target_x_centered + center_x
                    target_plot_z = -target_z_centered + center_z
                    
                    # Final Y-axis flip
                    ref_plot_y = mask_y - 1 - ref_plot_y
                    target_plot_y = mask_y - 1 - target_plot_y
                    
                    # Apply centroid rotations from alignment knobs
                    if hasattr(self, 'centroid_rotation_x') or hasattr(self, 'centroid_rotation_y') or hasattr(self, 'centroid_rotation_z'):
                        rx = np.radians(getattr(self, 'centroid_rotation_x', 0))
                        ry = np.radians(getattr(self, 'centroid_rotation_y', 0))
                        rz = np.radians(getattr(self, 'centroid_rotation_z', 0))
                        
                        center_x = mask_z / 2
                        center_y = mask_y / 2
                        center_z = mask_x / 2
                        
                        # Apply rotations to ref
                        ref_x_centered = ref_plot_x - center_x
                        ref_y_centered = ref_plot_y - center_y
                        ref_z_centered = ref_plot_z - center_z
                        
                        if rx != 0:
                            y_temp = ref_y_centered * np.cos(rx) - ref_z_centered * np.sin(rx)
                            z_temp = ref_y_centered * np.sin(rx) + ref_z_centered * np.cos(rx)
                            ref_y_centered = y_temp
                            ref_z_centered = z_temp
                        if ry != 0:
                            x_temp = ref_x_centered * np.cos(ry) + ref_z_centered * np.sin(ry)
                            z_temp = -ref_x_centered * np.sin(ry) + ref_z_centered * np.cos(ry)
                            ref_x_centered = x_temp
                            ref_z_centered = z_temp
                        if rz != 0:
                            x_temp = ref_x_centered * np.cos(rz) - ref_y_centered * np.sin(rz)
                            y_temp = ref_x_centered * np.sin(rz) + ref_y_centered * np.cos(rz)
                            ref_x_centered = x_temp
                            ref_y_centered = y_temp
                        
                        ref_plot_x = ref_x_centered + center_x
                        ref_plot_y = ref_y_centered + center_y
                        ref_plot_z = ref_z_centered + center_z
                        
                        # Apply rotations to target
                        target_x_centered = target_plot_x - center_x
                        target_y_centered = target_plot_y - center_y
                        target_z_centered = target_plot_z - center_z
                        
                        if rx != 0:
                            y_temp = target_y_centered * np.cos(rx) - target_z_centered * np.sin(rx)
                            z_temp = target_y_centered * np.sin(rx) + target_z_centered * np.cos(rx)
                            target_y_centered = y_temp
                            target_z_centered = z_temp
                        if ry != 0:
                            x_temp = target_x_centered * np.cos(ry) + target_z_centered * np.sin(ry)
                            z_temp = -target_x_centered * np.sin(ry) + target_z_centered * np.cos(ry)
                            target_x_centered = x_temp
                            target_z_centered = z_temp
                        if rz != 0:
                            x_temp = target_x_centered * np.cos(rz) - target_y_centered * np.sin(rz)
                            y_temp = target_x_centered * np.sin(rz) + target_y_centered * np.cos(rz)
                            target_x_centered = x_temp
                            target_y_centered = y_temp
                        
                        target_plot_x = target_x_centered + center_x
                        target_plot_y = target_y_centered + center_y
                        target_plot_z = target_z_centered + center_z
                    
                    x_line = [ref_plot_x, target_plot_x]
                    y_line = [ref_plot_y, target_plot_y]
                    z_line = [ref_plot_z, target_plot_z]
        else:
            # No stored positions - use raw coordinates with appropriate transformations
            ref_centroid = centroid_df[centroid_df[roi_col] == ref_roi].iloc[0]
            target_centroid = centroid_df[centroid_df[roi_col] == target_roi].iloc[0]
            
            if self.head_mask_data is None:
                # No mask - use same flipping as plot_centroids
                max_x = centroid_df[x_col].max() if len(centroid_df) > 0 else 0
                x_line = [max_x - ref_centroid[x_col], max_x - target_centroid[x_col]]
                y_line = [ref_centroid[y_col], target_centroid[y_col]]
                z_line = [ref_centroid[z_col], target_centroid[z_col]]
            else:
                # Mask loaded - coordinates should match what's plotted
                # Use raw coordinates (they'll be transformed by the mask overlay)
                x_line = [ref_centroid[x_col], target_centroid[x_col]]
                y_line = [ref_centroid[y_col], target_centroid[y_col]]
                z_line = [ref_centroid[z_col], target_centroid[z_col]]
        
        # Draw a more visible line with thicker width and brighter color
        line = self.ax.plot(x_line, y_line, z_line, '-', 
                           color='#FFD93D', linewidth=4.0, alpha=1.0, 
                           label=f'Distance: {distance:.2f} mm')[0]
        
        # Add midpoint annotation with distance
        mid_x = (x_line[0] + x_line[1]) / 2
        mid_y = (y_line[0] + y_line[1]) / 2
        mid_z = (z_line[0] + z_line[1]) / 2
        
        # Calculate direction vector of the line for offset calculation
        dx = x_line[1] - x_line[0]
        dy = y_line[1] - y_line[0]
        dz = z_line[1] - z_line[0]
        line_length = np.sqrt(dx**2 + dy**2 + dz**2) if (dx != 0 or dy != 0 or dz != 0) else 1.0
        
        # Offset the annotation perpendicular to the line to avoid overlap with ROI labels
        # Use ~8% of the line length as offset in Y direction (perpendicular to typical line orientation)
        offset_factor = 0.08 * line_length if line_length > 0 else 15.0
        
        # Apply offset to midpoint (offset in Y direction to avoid overlapping with ROI names)
        annotation_x = mid_x
        annotation_y = mid_y + offset_factor
        annotation_z = mid_z
        
        # Use same approach as GTV ROI (ax.text with 3D coordinates)
        # Apply rotation using transform
        text_obj = self.ax.text(annotation_x, annotation_y, annotation_z, 
                    f'{distance:.1f} mm', 
                    color='#FFD93D', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='#FFD93D'),
                    zorder=12)
        # Try to apply 45 degree counter-clockwise rotation using transform
        try:
            from matplotlib import transforms
            # Get the text's transform and apply rotation
            t = text_obj.get_transform()
            rot_transform = transforms.Affine2D().rotate_deg(-45) + t
            text_obj.set_transform(rot_transform)
        except:
            # If rotation fails, just use the text without rotation
            pass
        self.line_plots.append(line)
    
    def _draw_head_mask(self, external_mask, spatial_data, external_name, centroid_df):
        """Draw semi-transparent head/neck mask for anatomical reference
        
        Returns:
            bool: True if mask was successfully drawn, False otherwise
        """
        try:
            # external_mask, spatial_data, and external_name are now provided as parameters
            if external_mask is None or spatial_data is None:
                print("WARNING: Cannot draw head mask - mask or spatial data is None")
                return False  # No valid mask found
            
            if len(spatial_data) == 0:
                print("WARNING: Cannot draw head mask - spatial data is empty")
                return False
            
            # IMPORTANT: Centroids in CSV are in VOXEL coordinates (indices), not world coordinates
            # So the mask vertices must also be in voxel coordinates to align properly
            # The mask shape is (y, x, z) = (rows, cols, slices)
            
            print(f"Mask shape: {external_mask.shape}")
            print(f"Mask dtype: {external_mask.dtype}")
            print(f"Mask non-zero voxels: {np.sum(external_mask)}")
            
            # Use 3D surface voxel visualization from experimental_sagittal_mask_v4.py
            # This provides better visualization with centroid overlay
            print("Using 3D surface voxel visualization method...")
            return self._draw_3d_mask_v4(external_mask, spatial_data, centroid_df)
        except Exception as e:
            # Log error for debugging
            print(f"Error in _draw_head_mask: {e}")
            import traceback
            traceback.print_exc()
            return False  # Failed
    
    def _draw_3d_mask_v4(self, mask, spatial_data, centroid_df=None):
        """Draw 3D mask using surface voxel visualization from experimental_sagittal_mask_v4.py
        Shows actual 3D shape with centroid overlay capability
        """
        try:
            if mask is None or spatial_data is None:
                print("ERROR: Cannot draw mask - mask or spatial data is None")
                return False
            
            print(f"[_draw_3d_mask_v4] Drawing 3D mask using surface voxel visualization...")
            print(f"[_draw_3d_mask_v4]   Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"[_draw_3d_mask_v4]   Mask value range: [{mask.min()}, {mask.max()}]")
            print(f"[_draw_3d_mask_v4]   Mask non-zero voxels: {np.sum(mask):,}")
            
            # Check mask shape - HeadMaskLoader returns (Z, Y, X) from _generate_body_mask_from_ct
            # But we need to verify and potentially transpose
            if len(mask.shape) == 3:
                dim0, dim1, dim2 = mask.shape
                # If first dimension is largest, it's likely (Z, Y, X)
                # If last dimension is largest, it's likely (Y, X, Z)
                if dim0 > dim1 and dim0 > dim2:
                    # Likely (Z, Y, X) - use as is
                    mask_z, mask_y, mask_x = mask.shape
                    print(f"  Detected shape: (Z, Y, X) = ({mask_z}, {mask_y}, {mask_x})")
                elif dim2 > dim0 and dim2 > dim1:
                    # Likely (Y, X, Z) - transpose to (Z, Y, X)
                    mask = np.transpose(mask, (2, 0, 1))
                    mask_z, mask_y, mask_x = mask.shape
                    print(f"  Transposed from (Y, X, Z) to (Z, Y, X) = ({mask_z}, {mask_y}, {mask_x})")
                else:
                    # Assume (Z, Y, X) by default
                    mask_z, mask_y, mask_x = mask.shape
                    print(f"  Assuming shape: (Z, Y, X) = ({mask_z}, {mask_y}, {mask_x})")
            else:
                print("ERROR: Mask must be 3D")
                return False
            
            # Limit Z range to 0-175 and use 15 bins (from experimental_sagittal_mask_v4.py)
            z_min = 0
            z_max = min(175, mask_z - 1)
            z_range = z_max - z_min + 1
            num_bins = 15
            slice_step = max(1, z_range // num_bins)
            
            print(f"  Z-axis range: {z_min} to {z_max} ({z_range} slices)")
            print(f"  Using {num_bins} bins, slice_step: {slice_step}")
            
            # Find surface voxels (edges) to show the 3D shape
            from scipy import ndimage
            print("  Finding surface voxels to visualize 3D mask shape...")
            
            # Sample every 4th pixel for performance
            pixel_step = 4
            
            all_x, all_y, all_z = [], [], []
            voxels_drawn = 0
            
            # Process ALL slices (not just 0-175) to avoid splitting the mask
            # FLIP Z coordinates to match centroids: centroids use flipped Z (max_z - z)
            for z_idx in range(0, mask_z, slice_step):
                slice_mask = mask[z_idx, ::pixel_step, ::pixel_step]  # Downsample
                
                if np.any(slice_mask):
                    # Find surface/edge pixels using erosion
                    eroded = ndimage.binary_erosion(slice_mask)
                    surface = slice_mask & (~eroded)  # Surface pixels
                    
                    # Get surface coordinates
                    y_surf, x_surf = np.where(surface)
                    
                    if len(y_surf) > 0:
                        # Scale back up to original coordinates
                        # Mask shape is (Z, Y, X) - we need to reorient:
                        # - Bottom should be on X plane -> use z_idx (slices)
                        # - Base should be on Y plane (width) -> use y_surf (rows)
                        # - Back should be on Z plane -> use x_surf (columns)
                        # So: X = z_idx (bottom) - FLIPPED, Y = y_surf (base), Z = x_surf (back) - FLIPPED
                        x_coords = np.full_like(y_surf, (mask_z - 1) - z_idx)  # Bottom (X) - use slice index, FLIPPED
                        y_coords = y_surf * pixel_step   # Base width (Y) - use row index
                        z_coords = (mask_x - 1) - (x_surf * pixel_step)  # Back (Z) - use column index, FLIPPED
                        
                        # Apply mask rotations from alignment knobs
                        if self.mask_rotation_x != 0 or self.mask_rotation_y != 0 or self.mask_rotation_z != 0:
                            # Convert to radians
                            rx = np.radians(self.mask_rotation_x)
                            ry = np.radians(self.mask_rotation_y)
                            rz = np.radians(self.mask_rotation_z)
                            
                            # Center of rotation
                            center_x = mask_z / 2
                            center_y = mask_y / 2
                            center_z = mask_x / 2
                            
                            # Translate to origin
                            x_centered = x_coords - center_x
                            y_centered = y_coords - center_y
                            z_centered = z_coords - center_z
                            
                            # Apply rotations (X, then Y, then Z)
                            # Rotation around X
                            if rx != 0:
                                y_temp = y_centered * np.cos(rx) - z_centered * np.sin(rx)
                                z_temp = y_centered * np.sin(rx) + z_centered * np.cos(rx)
                                y_centered = y_temp
                                z_centered = z_temp
                            
                            # Rotation around Y
                            if ry != 0:
                                x_temp = x_centered * np.cos(ry) + z_centered * np.sin(ry)
                                z_temp = -x_centered * np.sin(ry) + z_centered * np.cos(ry)
                                x_centered = x_temp
                                z_centered = z_temp
                            
                            # Rotation around Z
                            if rz != 0:
                                x_temp = x_centered * np.cos(rz) - y_centered * np.sin(rz)
                                y_temp = x_centered * np.sin(rz) + y_centered * np.cos(rz)
                                x_centered = x_temp
                                y_centered = y_temp
                            
                            # Translate back
                            x_coords = x_centered + center_x
                            y_coords = y_centered + center_y
                            z_coords = z_centered + center_z
                        
                        all_x.extend(x_coords)
                        all_y.extend(y_coords)
                        all_z.extend(z_coords)
                        voxels_drawn += len(x_coords)
            
            if voxels_drawn > 0:
                # Plot all surface voxels at once - this shows the 3D shape (EXACT from experimental)
                self.head_surface = self.ax.scatter(all_x, all_y, all_z, 
                                                   c='#808080', alpha=0.6, s=3, 
                                                   edgecolors='none')
                print(f"  Drew {voxels_drawn:,} surface voxels showing 3D mask shape")
                
                # Set axis limits - use full mask range
                # X = bottom (from Z slices), Y = base width (from Y rows), Z = back (from X columns)
                self.ax.set_xlim(0, mask_z)  # Bottom (from Z slices)
                self.ax.set_ylim(0, mask_y)  # Base width (from Y rows)
                self.ax.set_zlim(0, mask_x)  # Back (from X columns)
                
                # Set view angle to show mask clearly (elevation, azimuth)
                # Elevation: 20 degrees (slightly above), Azimuth: 45 degrees (angled view)
                self.ax.view_init(elev=20, azim=45)
                
                # Overlay centroids if available (using method from experimental_sagittal_mask_v4.py)
                if centroid_df is not None and len(centroid_df) > 0:
                    self._overlay_centroids_v4(centroid_df, mask, spatial_data)
                
                return True
            else:
                # Fallback: sample some mask pixels directly
                print("  No surface voxels found, sampling mask pixels...")
                all_x, all_y, all_z = [], [], []
                for z_idx in range(0, mask_z, slice_step * 2):
                    slice_mask = mask[z_idx, ::pixel_step * 2, ::pixel_step * 2]
                    y_coords_mask, x_coords_mask = np.where(slice_mask > 0)
                    if len(y_coords_mask) > 0:
                        x_coords = np.full_like(y_coords_mask, (mask_z - 1) - z_idx)  # Bottom (X) - use slice index, FLIPPED
                        y_coords = y_coords_mask * pixel_step * 2  # Base width (Y) - use row index
                        z_coords = (mask_x - 1) - (x_coords_mask * pixel_step * 2)  # Back (Z) - use column index, FLIPPED
                        all_x.extend(x_coords)
                        all_y.extend(y_coords)
                        all_z.extend(z_coords)
                
                if len(all_x) > 0:
                    # Sample for performance if too many
                    if len(all_x) > 50000:
                        step = len(all_x) // 50000
                        all_x = all_x[::step]
                        all_y = all_y[::step]
                        all_z = all_z[::step]
                    
                    self.head_surface = self.ax.scatter(all_x, all_y, all_z, 
                                                       c='#808080', alpha=0.5, s=2, 
                                                       edgecolors='none')
                    print(f"  Drew {len(all_x):,} mask voxels")
                    
                    # Overlay ALL centroids if available (EXACT from experimental)
                    if centroid_df is not None and len(centroid_df) > 0:
                        self._overlay_centroids_v4(centroid_df, mask, spatial_data)
                    
                    return True
                else:
                    print("ERROR: Could not generate any mask visualization")
                    return False
                    
        except Exception as e:
            print(f"Error in _draw_3d_mask_v4: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _overlay_centroids_v4(self, centroid_df, mask, spatial_data):
        """Overlay centroids on the 3D mask visualization - uses same visualization as regular plot"""
        try:
            if centroid_df is None or len(centroid_df) == 0:
                print("  No centroids loaded to overlay")
                return
            
            print("  Converting centroids from world to voxel coordinates...")
            
            if not spatial_data or len(spatial_data) == 0:
                print("  Warning: No spatial data available for coordinate conversion")
                return
            
            # Find coordinate columns
            possible_x = ['x coordinate', 'x_coordinate', 'X', 'x']
            possible_y = ['y coordinate', 'y_coordinate', 'Y', 'y']
            possible_z = ['z coordinate', 'z_coordinate', 'Z', 'z']
            possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name', 'ROI Name']
            
            x_col = next((col for col in possible_x if col in centroid_df.columns), None)
            y_col = next((col for col in possible_y if col in centroid_df.columns), None)
            z_col = next((col for col in possible_z if col in centroid_df.columns), None)
            roi_col = next((col for col in possible_roi if col in centroid_df.columns), None)
            
            if not all([x_col, y_col, z_col]):
                print(f"  Warning: Could not find coordinate columns in centroid CSV")
                print(f"  Available columns: {list(centroid_df.columns)}")
                return
            
            print(f"  Using columns: X={x_col}, Y={y_col}, Z={z_col}, ROI={roi_col}")
            
            # Get coordinates from CSV
            csv_x = centroid_df[x_col].values
            csv_y = centroid_df[y_col].values
            csv_z = centroid_df[z_col].values
            
            # Get GTV and selected ROI from stored values (same as regular plot)
            gtv_name = self.gtv_roi if hasattr(self, 'gtv_roi') else None
            selected_roi = self.selected_roi if hasattr(self, 'selected_roi') else None
            
            print(f"  CSV coordinate ranges: X=[{csv_x.min():.1f}, {csv_x.max():.1f}], "
                  f"Y=[{csv_y.min():.1f}, {csv_y.max():.1f}], "
                  f"Z=[{csv_z.min():.1f}, {csv_z.max():.1f}]")
            
            # Check if coordinates are already in voxel space (small values < 1000)
            # Centroids from CSV are stored as voxel coordinates (x=col, y=row, z=slice)
            is_voxel_coords = (csv_x.max() < 1000 and csv_y.max() < 1000 and csv_z.max() < 1000)
            
            mask_z, mask_y, mask_x = mask.shape
            print(f"  Mask dimensions: X={mask_x}, Y={mask_y}, Z={mask_z}")
            
            if is_voxel_coords:
                print("  Detected voxel coordinates in CSV - using directly")
            else:
                print("  Detected world coordinates in CSV - converting to voxel")
            
            # Convert to voxel coordinates and transform to match mask
            voxel_x_list, voxel_y_list, voxel_z_list = [], [], []
            roi_names = []
            colors = []
            sizes = []
            
            for i in range(len(csv_x)):
                if is_voxel_coords:
                    # Already in voxel coordinates (x=col, y=row, z=slice)
                    vox_x = csv_x[i]
                    vox_y = csv_y[i]
                    vox_z = csv_z[i]
                else:
                    # Convert from world coordinates to voxel
                    vox_x, vox_y, vox_z = self._world_to_voxel_v4(csv_x[i], csv_y[i], csv_z[i], spatial_data, mask)
                    if vox_x is None or np.isnan(vox_x) or np.isnan(vox_y) or np.isnan(vox_z):
                        continue
                
                # Ensure coordinates are within bounds
                vox_x = max(0, min(mask_x - 1, vox_x))
                vox_y = max(0, min(mask_y - 1, vox_y))
                vox_z = max(0, min(mask_z - 1, vox_z))
                
                # Transform centroids to match mask coordinate system:
                # Mask uses: X = (mask_z - 1) - z_idx (flipped slice), Y = y_surf (row), Z = (mask_x - 1) - x_surf (flipped column)
                # Centroids are in voxel coords: x=col, y=row, z=slice
                # So: X (plot) = (mask_z - 1) - vox_z (flip slice), Y (plot) = vox_y (row), Z (plot) = (mask_x - 1) - vox_x (flip column)
                plot_x = (mask_z - 1) - vox_z  # X = flipped slice (bottom)
                plot_y = vox_y  # Y = row (base width)
                plot_z = (mask_x - 1) - vox_x  # Z = flipped column (back)
                
                # Flip Z axis
                plot_z = mask_x - 1 - plot_z
                
                # Rotate 90 degrees counter-clockwise around X axis
                # Rotation matrix for 90° counter-clockwise around X: X' = X, Y' = Z, Z' = -Y
                # Center of rotation (middle of mask)
                center_y = mask_y / 2
                center_z = mask_x / 2
                # Translate to origin, rotate, translate back
                y_centered = plot_y - center_y
                z_centered = plot_z - center_z
                # 90° counter-clockwise around X: Y' = Z, Z' = -Y
                plot_y_rotated = z_centered + center_y
                plot_z_rotated = -y_centered + center_z
                
                # Rotate 90 degrees clockwise around Y axis
                # Rotation matrix for 90° clockwise around Y: X' = Z, Y' = Y, Z' = -X
                # Center of rotation (middle of mask)
                center_x = mask_z / 2
                center_z = mask_x / 2
                # Translate to origin, rotate, translate back
                x_centered = plot_x - center_x
                z_centered = plot_z_rotated - center_z
                # 90° clockwise around Y: X' = Z, Z' = -X
                plot_x_rotated = z_centered + center_x
                plot_z_rotated = -x_centered + center_z
                
                # Rotate 180 degrees clockwise around Z axis
                # Rotation matrix for 180° clockwise around Z: X' = -X, Y' = -Y, Z' = Z
                # Center of rotation (middle of mask)
                center_x = mask_z / 2
                center_y = mask_y / 2
                # Translate to origin, rotate, translate back
                x_centered = plot_x_rotated - center_x
                y_centered = plot_y_rotated - center_y
                # 180° clockwise around Z: X' = -X, Y' = -Y
                plot_x_rotated = -x_centered + center_x
                plot_y_rotated = -y_centered + center_y
                
                # Rotate 180 degrees around Y axis
                # Rotation matrix for 180° around Y: X' = -X, Y' = Y, Z' = -Z
                # Center of rotation (middle of mask)
                center_x = mask_z / 2
                center_z = mask_x / 2
                # Translate to origin, rotate, translate back
                x_centered = plot_x_rotated - center_x
                z_centered = plot_z_rotated - center_z
                # 180° around Y: X' = -X, Z' = -Z
                plot_x_rotated = -x_centered + center_x
                plot_z_rotated = -z_centered + center_z
                
                # Final Y-axis flip to fix upside-down orientation (centroids are still upside down)
                plot_y_rotated = mask_y - 1 - plot_y_rotated
                
                # Apply centroid rotations from alignment knobs
                if self.centroid_rotation_x != 0 or self.centroid_rotation_y != 0 or self.centroid_rotation_z != 0:
                    # Convert to radians
                    rx = np.radians(self.centroid_rotation_x)
                    ry = np.radians(self.centroid_rotation_y)
                    rz = np.radians(self.centroid_rotation_z)
                    
                    # Center of rotation
                    center_x = mask_z / 2
                    center_y = mask_y / 2
                    center_z = mask_x / 2
                    
                    # Translate to origin
                    x_centered = plot_x_rotated - center_x
                    y_centered = plot_y_rotated - center_y
                    z_centered = plot_z_rotated - center_z
                    
                    # Apply rotations (X, then Y, then Z)
                    # Rotation around X
                    if rx != 0:
                        y_temp = y_centered * np.cos(rx) - z_centered * np.sin(rx)
                        z_temp = y_centered * np.sin(rx) + z_centered * np.cos(rx)
                        y_centered = y_temp
                        z_centered = z_temp
                    
                    # Rotation around Y
                    if ry != 0:
                        x_temp = x_centered * np.cos(ry) + z_centered * np.sin(ry)
                        z_temp = -x_centered * np.sin(ry) + z_centered * np.cos(ry)
                        x_centered = x_temp
                        z_centered = z_temp
                    
                    # Rotation around Z
                    if rz != 0:
                        x_temp = x_centered * np.cos(rz) - y_centered * np.sin(rz)
                        y_temp = x_centered * np.sin(rz) + y_centered * np.cos(rz)
                        x_centered = x_temp
                        y_centered = y_temp
                    
                    # Translate back
                    plot_x_rotated = x_centered + center_x
                    plot_y_rotated = y_centered + center_y
                    plot_z_rotated = z_centered + center_z
                
                voxel_x_list.append(plot_x_rotated)
                voxel_y_list.append(plot_y_rotated)
                voxel_z_list.append(plot_z_rotated)  # Z unchanged for rotation around Z axis
                
                # Get ROI name
                if roi_col:
                    roi_name = str(centroid_df[roi_col].iloc[i])
                    roi_names.append(roi_name)
                else:
                    roi_name = f"ROI_{i}"
                    roi_names.append(roi_name)
                
                # Use same color/size scheme as regular visualization
                if gtv_name and roi_name == gtv_name:
                    colors.append('#FF6B6B')
                    sizes.append(30)  # Same as regular plot
                elif selected_roi and roi_name == selected_roi:
                    colors.append('#4ECDC4')
                    sizes.append(25)  # Same as regular plot
                else:
                    colors.append('#95E1D3')
                    sizes.append(20)  # Same as regular plot
            
            if len(voxel_x_list) > 0:
                print(f"  Plotting {len(voxel_x_list)} centroids...")
                print(f"  Voxel coordinate ranges: X=[{min(voxel_x_list):.1f}, {max(voxel_x_list):.1f}], "
                      f"Y=[{min(voxel_y_list):.1f}, {max(voxel_y_list):.1f}], "
                      f"Z=[{min(voxel_z_list):.1f}, {max(voxel_z_list):.1f}]")
                
                # Store plotted positions for distance line drawing
                self.plotted_centroid_positions = {
                    'x': voxel_x_list.copy(),
                    'y': voxel_y_list.copy(),
                    'z': voxel_z_list.copy(),
                    'roi_names': roi_names.copy()
                }
                
                # Plot centroids - use same visualization as regular plot (colors, sizes, styling)
                self.ax.scatter(voxel_x_list, voxel_y_list, voxel_z_list,
                             c=colors, s=sizes, alpha=0.8, edgecolors='white', 
                             linewidths=1, depthshade=True, zorder=10)
                
                # Add labels for all centroids (same as regular plot)
                for i, roi_name in enumerate(roi_names):
                    if gtv_name and roi_name == gtv_name:
                        text_obj = self.ax.text(voxel_x_list[i], voxel_y_list[i], voxel_z_list[i],
                                   f'  {roi_name} (GTV)', 
                                   color='#FF6B6B', fontsize=9, fontweight='bold', zorder=11)
                        # GTV label: no rotation (rotation=0) - ensure it's visible
                        # No rotation needed for GTV
                    elif selected_roi and roi_name == selected_roi:
                        # Use same approach as GTV ROI (ax.text with 3D coordinates)
                        # Apply rotation using transform
                        text_obj = self.ax.text(voxel_x_list[i], voxel_y_list[i], voxel_z_list[i],
                                   f'  {roi_name}', 
                                   color='#4ECDC4', fontsize=8, zorder=11)
                        # Try to apply 90 degree clockwise rotation using transform
                        try:
                            from matplotlib import transforms
                            # Get the text's transform and apply rotation
                            t = text_obj.get_transform()
                            rot_transform = transforms.Affine2D().rotate_deg(90) + t
                            text_obj.set_transform(rot_transform)
                        except:
                            # If rotation fails, just use the text without rotation
                            pass
                    else:
                        self.ax.text(voxel_x_list[i], voxel_y_list[i], voxel_z_list[i],
                                   f'  {roi_name}', 
                                   color='#95E1D3', fontsize=7, alpha=0.7, zorder=11)
                
                print(f"  Successfully overlaid {len(voxel_x_list)} centroids on 3D mask")
            else:
                print("  Warning: No centroids could be converted to voxel coordinates")
                print("  Check coordinate conversion and spatial data")
                
        except Exception as e:
            print(f"  Error overlaying centroids: {e}")
            import traceback
            traceback.print_exc()
    
    def _world_to_voxel_v4(self, world_x, world_y, world_z, spatial_data, mask):
        """Convert world coordinates (mm) to voxel coordinates - EXACT copy from experimental_sagittal_mask_v4.py"""
        if not spatial_data or len(spatial_data) == 0:
            return None, None, None
        
        # Get first slice spatial data as reference
        ref_spatial = spatial_data[0]
        
        # Check if coordinates are already in voxel space (if they're small integers)
        # Centroids from CSV might already be in voxel coordinates
        if abs(world_x) < 1000 and abs(world_y) < 1000 and abs(world_z) < 1000:
            # Might already be voxel coordinates, but check against mask size
            if hasattr(self, 'mask_3d') and self.mask_3d is not None:
                mask_z, mask_y, mask_x = self.mask_3d.shape
                if (0 <= world_x < mask_x and 0 <= world_y < mask_y and 0 <= world_z < mask_z):
                    # Already in voxel coordinates
                    return world_x, world_y, world_z
        
        # Calculate voxel coordinates from world coordinates
        # X: column index (ImagePositionPatient[0] is X position of first pixel)
        voxel_x = (world_x - ref_spatial['xImagePosition']) / ref_spatial['xImageDim']
        
        # Y: row index (ImagePositionPatient[1] is Y position of first pixel)
        voxel_y = (world_y - ref_spatial['yImagePosition']) / ref_spatial['yImageDim']
        
        # Z: slice index (find closest slice by Z position)
        z_positions = [s['zImagePosition'] for s in spatial_data]
        if len(z_positions) > 1:
            # Find closest Z slice
            z_diffs = [abs(world_z - z_pos) for z_pos in z_positions]
            voxel_z = np.argmin(z_diffs)
        else:
            voxel_z = 0
        
        return voxel_x, voxel_y, voxel_z
    
    def _on_scroll(self, event):
        """Handle mouse scroll wheel for zooming"""
        if event.inaxes != self.ax:
            return
        
        # Zoom factor
        zoom_factor = 1.1 if event.button == 'up' else 0.9
        
        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        
        # Calculate centers
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        # Calculate new ranges
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor
        z_range = (zlim[1] - zlim[0]) * zoom_factor
        
        # Set new limits
        self.ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
        self.ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
        self.ax.set_zlim([z_center - z_range/2, z_center + z_range/2])
        
        # Redraw
        self.draw()
    
    def _draw_head_mask_simple(self, mask, spatial_data, centroid_df=None):
        """FAST approach: draw outline contours from mask slices in voxel coordinates
        
        This is much faster than marching cubes and doesn't freeze the GUI.
        Uses voxel coordinates to match centroids.
        """
        try:
            if mask is None or spatial_data is None:
                print("ERROR: Cannot draw mask - mask or spatial data is None")
                return False
            
            print(f"[_draw_head_mask_simple] Drawing head mask using fast contour method...")
            print(f"[_draw_head_mask_simple]   Mask shape: {mask.shape}")
            print(f"[_draw_head_mask_simple]   Mask dtype: {mask.dtype}")
            print(f"[_draw_head_mask_simple]   Mask non-zero voxels: {np.sum(mask)}")
            
            # Centroids are in voxel coordinates, so we draw contours in voxel coordinates too
            # Draw every 50th slice for MUCH faster rendering
            slice_step = 50  # Draw every 50th slice for speed (increased from 30)
            contours_drawn = 0
            
            # Get mask dimensions for coordinate flipping
            mask_height = mask.shape[0]  # rows
            mask_width = mask.shape[1]   # cols
            
            for z_idx in range(0, mask.shape[2], slice_step):
                slice_mask = mask[:, :, z_idx]
                if np.any(slice_mask):
                    # Get contours from the slice
                    from skimage import measure
                    contours = measure.find_contours(slice_mask, 0.5)
                    
                    # Centroids use voxel coordinates (x = col, y = row, z = slice)
                    # So we plot directly in voxel space
                    # FIX: Flip Y coordinates correctly to match centroid coordinate system
                    for contour in contours:
                        if len(contour) > 10:  # Only draw substantial contours
                            # contour is in (row, col) format = (y, x)
                            # Convert to (x, y, z) for plotting
                            # IMPORTANT: Match centroid coordinate system exactly
                            # Centroids are stored as: x = col, y = row, z = slice (in voxel coordinates)
                            # Centroids are flipped: x = max_x - x, z = max_z - z (but NOT y)
                            # So mask should match: x = col (flipped), y = row (NOT flipped), z = slice (flipped)
                            
                            x_coords = contour[:, 1]  # Column index = X (same as centroid x)
                            # Flip X to match flipped centroids: x = max_x - x
                            if centroid_df is not None and len(centroid_df) > 0:
                                possible_x = ['x coordinate', 'x_coordinate', 'X', 'x']
                                x_col = next((col for col in possible_x if col in centroid_df.columns), None)
                                if x_col:
                                    max_x = centroid_df[x_col].max()
                                    x_coords = max_x - x_coords  # Flip X to match centroids
                                else:
                                    x_coords = mask_width - 1 - x_coords  # Fallback: use mask width
                            else:
                                x_coords = mask_width - 1 - x_coords  # Fallback: use mask width
                            
                            # Y is NOT flipped for centroids, so don't flip Y for mask
                            y_coords = contour[:, 0]  # Row index = Y (NOT flipped, matches centroids)
                            
                            # Flip Z to match flipped centroids: z = max_z - z
                            if centroid_df is not None and len(centroid_df) > 0:
                                possible_z = ['z coordinate', 'z_coordinate', 'Z', 'z']
                                z_col = next((col for col in possible_z if col in centroid_df.columns), None)
                                if z_col:
                                    max_z = centroid_df[z_col].max()
                                    z_coords = np.full_like(x_coords, max_z - z_idx)  # Flip Z to match centroids
                                else:
                                    z_coords = np.full_like(x_coords, mask.shape[2] - 1 - z_idx)  # Fallback
                            else:
                                z_coords = np.full_like(x_coords, mask.shape[2] - 1 - z_idx)  # Fallback
                            
                            # Plot as semi-transparent line
                            self.ax.plot(x_coords, y_coords, z_coords, 
                                       color='#808080', alpha=0.6, linewidth=1.5)
                            contours_drawn += 1
            
            print(f"  Drew {contours_drawn} contour lines from {mask.shape[2] // slice_step} slices")
            
            # Force plot refresh
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.draw()
            
            print(f"✓ Head/neck mask drawn using fast contour method")
            return True
            
        except Exception as e:
            print(f"Error in _draw_head_mask_simple: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_scroll(self, event):
        """Handle mouse scroll wheel for zooming"""
        if event.inaxes != self.ax:
            return
        
        # Zoom factor
        zoom_factor = 1.1 if event.button == 'up' else 0.9
        
        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        
        # Calculate centers
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        # Calculate new ranges
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor
        z_range = (zlim[1] - zlim[0]) * zoom_factor
        
        # Set new limits
        self.ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
        self.ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
        self.ax.set_zlim([z_center - z_range/2, z_center + z_range/2])
        
        # Redraw
        self.draw()


# Main window class - V3 with GPU/Parallel support
class EnhancedROIVisualizerV3(QMainWindow):
    """Enhanced ROI Distance Generator and Visualizer"""
    
    def __init__(self):
        super().__init__()
        self.mrn_folder = None
        self.scan_results = {}
        self.centroid_df = None
        self.distance_df = None
        self.gtv_name = None
        self.use_gpu = False
        self.use_parallel = True
        self.start_time = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Check GPU availability - use simple check to avoid hanging
        try:
            gpu_available, gpu_status = check_gpu()
        except Exception:
            # If GPU check fails, assume no GPU
            gpu_available = False
            gpu_status = "GPU check failed"
        
        self.setWindowTitle(f"Enhanced ROI Distance Generator and Visualizer | GPU: {'✓' if gpu_available else '✗'}")
        self.setGeometry(50, 50, 1200, 600)  # Increased width from 900 to 1200 for better table visibility
        
        # Apply dark theme (same as V2)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #4ECDC4;
                color: #1a1a1a;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
                min-height: 24px;
                max-height: 28px;
            }
            QPushButton:hover {
                background-color: #45b8b0;
            }
            QPushButton:pressed {
                background-color: #3a9d96;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QPushButton#primary {
                background-color: #FF6B6B;
                font-size: 11px;
                padding: 7px 20px;
                min-height: 26px;
                max-height: 30px;
            }
            QPushButton#primary:hover {
                background-color: #ee5a5a;
            }
            QPushButton#success {
                background-color: #51CF66;
            }
            QPushButton#success:hover {
                background-color: #40c057;
            }
            QGroupBox {
                border: 2px solid #4ECDC4;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 6px 16px;
                margin-right: 2px;
                font-size: 10px;
            }
            QTabBar::tab:selected {
                background-color: #4ECDC4;
                color: #1a1a1a;
            }
            QListWidget, QTreeWidget {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                color: #ffffff;
            }
            QListWidget::item:selected, QTreeWidget::item:selected {
                background-color: #4ECDC4;
                color: #1a1a1a;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                color: #ffffff;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e1a;
            }
            QProgressBar::chunk {
                background-color: #4ECDC4;
            }
            QCheckBox {
                color: #ffffff;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #4ECDC4;
                border: 2px solid #4ECDC4;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)
        
        # Title
        title = QLabel("Enhanced ROI Distance Generator and Visualizer")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #4ECDC4; padding: 8px;")
        main_layout.addWidget(title)
        
        # GPU/CPU status
        gpu_status_label = QLabel(f"GPU Status: {gpu_status}")
        gpu_status_label.setAlignment(Qt.AlignCenter)
        gpu_status_label.setStyleSheet(f"color: {'#51CF66' if gpu_available else '#FF6B6B'}; padding: 5px; font-size: 10px;")
        main_layout.addWidget(gpu_status_label)
        
        # Create tabbed interface
        tabs = QTabWidget()
        
        # Tab 1: Workflow
        workflow_tab = self.create_workflow_tab()
        tabs.addTab(workflow_tab, "📁 Workflow")
        
        # Tab 2: Analysis
        analysis_tab = self.create_analysis_tab()
        tabs.addTab(analysis_tab, "📊 Analysis")
        
        # Tab 3: Visualization
        viz_tab = self.create_visualization_tab()
        tabs.addTab(viz_tab, "🎨 Visualization")
        
        main_layout.addWidget(tabs)
        
        # Footer with copyright and export button
        footer_layout = QHBoxLayout()
        footer = QLabel("Property of Fuller Lab. Unpublished work by Cem Dede")
        footer.setStyleSheet("color: #95E1D3; padding: 5px; font-size: 9px; font-style: italic;")
        footer_layout.addWidget(footer)
        
        footer_layout.addStretch()  # Push button to the right
        
        self.export_gtv_distances_btn = QPushButton("📊 Export GTV Distances")
        self.export_gtv_distances_btn.setEnabled(False)  # Disabled by default
        self.export_gtv_distances_btn.setToolTip("Export all distances for the selected GTV ROI to Excel")
        self.export_gtv_distances_btn.clicked.connect(self.export_gtv_distances)
        self.export_gtv_distances_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 10px;
                font-size: 9px;
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QPushButton:hover:enabled {
                background-color: #353535;
            }
            QPushButton:disabled {
                color: #666;
                background-color: #1e1e1e;
            }
        """)
        footer_layout.addWidget(self.export_gtv_distances_btn)
        
        main_layout.addLayout(footer_layout)
        
        # Status bar
        status_msg = f"Ready - Select MRN folder to begin (GPU: {'Available' if gpu_available else 'Not Available'}, Parallel: {mp.cpu_count()} cores)"
        self.statusBar().showMessage(status_msg)
        self.statusBar().setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # Add timer label to status bar (bottom right)
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setStyleSheet("""
            QLabel {
                color: #95E1D3;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                font-weight: bold;
                padding: 5px 10px;
                background-color: #2b2b2b;
                border: 1px solid #4ECDC4;
                border-radius: 4px;
            }
        """)
        self.timer_label.setMinimumWidth(80)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.statusBar().addPermanentWidget(self.timer_label)
        
        # Initialize and start timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # Update every second
        self.start_time = pd.Timestamp.now()
        self.update_timer()  # Initial update
        
        # Add exit button to status bar
        exit_btn = QPushButton("Exit")
        exit_btn.setMaximumWidth(80)
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ee5a5a;
            }
        """)
        exit_btn.clicked.connect(self.close)
        self.statusBar().addPermanentWidget(exit_btn)
    
    def update_timer(self):
        """Update the timer display"""
        if self.start_time is None:
            self.timer_label.setText("00:00:00")
            return
        
        elapsed = pd.Timestamp.now() - self.start_time
        total_seconds = int(elapsed.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        self.timer_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def create_workflow_tab(self):
        """Create the workflow tab with GPU/Parallel options"""
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        main_layout.setSpacing(8)
        
        # Left side: Steps 1, 3, 4
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)
        
        # Step 1: MRN Folder Selection
        step1_group = QGroupBox("Step 1: Select MRN Folder")
        step1_layout = QVBoxLayout()
        
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color: #95E1D3; padding: 5px;")
        folder_layout.addWidget(self.folder_label)
        
        select_folder_btn = QPushButton("📂 Select MRN Folder")
        select_folder_btn.setObjectName("primary")
        select_folder_btn.clicked.connect(self.select_mrn_folder)
        folder_layout.addWidget(select_folder_btn)
        
        scan_btn = QPushButton("🔍 Scan Folder")
        scan_btn.clicked.connect(self.scan_folder)
        scan_btn.setEnabled(False)
        self.scan_btn = scan_btn
        folder_layout.addWidget(scan_btn)
        
        step1_layout.addLayout(folder_layout)
        step1_group.setLayout(step1_layout)
        left_layout.addWidget(step1_group)
        
        # Step 3: Processing Options
        step3_group = QGroupBox("Step 3: Processing Options")
        step3_layout = QVBoxLayout()
        
        # Manual classification button
        self.classify_btn = QPushButton("🔧 Manually Classify RTSTRUCT Files")
        self.classify_btn.clicked.connect(lambda: self.manual_classify_rtstructs(self.scan_results))
        self.classify_btn.setVisible(False)
        step3_layout.addWidget(self.classify_btn)
        
        # Analysis buttons
        analysis_layout = QHBoxLayout()
        analyze_rtstruct_btn = QPushButton("🔬 Analyze RTSTRUCT")
        analyze_rtstruct_btn.clicked.connect(self.analyze_rtstruct)
        analyze_rtstruct_btn.setEnabled(False)
        self.analyze_rtstruct_btn = analyze_rtstruct_btn
        
        analyze_abas_btn = QPushButton("🔬 Analyze ABAS RTSTRUCT")
        analyze_abas_btn.clicked.connect(self.analyze_abas)
        analyze_abas_btn.setEnabled(False)
        self.analyze_abas_btn = analyze_abas_btn
        
        analysis_layout.addWidget(analyze_rtstruct_btn)
        analysis_layout.addWidget(analyze_abas_btn)
        step3_layout.addLayout(analysis_layout)
        
        # Standardization and merging
        process_layout = QHBoxLayout()
        standardize_btn = QPushButton("📝 Standardize & Name")
        standardize_btn.clicked.connect(self.standardize_names)
        standardize_btn.setEnabled(False)
        self.standardize_btn = standardize_btn
        
        merge_btn = QPushButton("🔀 Merge RTSTRUCT & ABAS")
        merge_btn.setObjectName("success")
        merge_btn.clicked.connect(self.merge_rtstructs)
        merge_btn.setEnabled(False)
        self.merge_btn = merge_btn
        
        process_layout.addWidget(standardize_btn)
        process_layout.addWidget(merge_btn)
        step3_layout.addLayout(process_layout)
        
        step3_group.setLayout(step3_layout)
        left_layout.addWidget(step3_group)
        
        # Step 4: CSV Generation with GPU/Parallel options
        step4_group = QGroupBox("Step 4: Generate CSV Files")
        step4_layout = QVBoxLayout()
        
        csv_info = QLabel("Generate centroid and distance CSV files for visualization\n"
                         "Note: Step 3 generates analysis reports. Step 4 generates the CSV files.")
        csv_info.setStyleSheet("color: #95E1D3; padding: 5px;")
        step4_layout.addWidget(csv_info)
        
        # GPU checkbox (parallel CPU is always used as fallback)
        processing_options_layout = QVBoxLayout()
        self.gpu_checkbox = QCheckBox("Use GPU acceleration (if available)")
        self.gpu_checkbox.setChecked(GPU_AVAILABLE)
        # Always allow user to attempt GPU; we’ll fallback safely if unavailable
        self.gpu_checkbox.setEnabled(True)
        if not GPU_AVAILABLE:
            self.gpu_checkbox.setToolTip("GPU not detected (CuPy missing). You can still check this box; it will fall back to CPU if GPU is unavailable.")
        processing_options_layout.addWidget(self.gpu_checkbox)
        
        # CPU cores selection
        cores_layout = QHBoxLayout()
        cores_label = QLabel("CPU Cores:")
        cores_label.setStyleSheet("color: #95E1D3; font-size: 10px;")
        cores_layout.addWidget(cores_label)
        
        # Get actual CPU count for this machine
        available_cores = mp.cpu_count()
        
        self.cores_spinbox = QSpinBox()
        self.cores_spinbox.setMinimum(1)
        self.cores_spinbox.setMaximum(available_cores)  # allow full available cores
        self.cores_spinbox.setValue(min(4, available_cores))  # Default to 4 or less if fewer cores
        self.cores_spinbox.setToolTip(f"Number of CPU cores to use (1-{available_cores}). Lower values use less memory but are slower. Higher values are faster but need more RAM.")
        self.cores_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                color: #ffffff;
                padding: 3px;
                border-radius: 3px;
                min-width: 60px;
            }
        """)
        cores_layout.addWidget(self.cores_spinbox)
        
        # Add label showing available cores
        cores_info_label = QLabel(f"(Available: {available_cores})")
        cores_info_label.setStyleSheet("color: #95E1D3; font-size: 9px; font-style: italic; padding-left: 5px;")
        cores_info_label.setToolTip(f"This machine has {available_cores} CPU cores available")
        cores_layout.addWidget(cores_info_label)
        
        cores_layout.addStretch()
        processing_options_layout.addLayout(cores_layout)
        
        # Info label about parallel processing
        max_cores = mp.cpu_count()
        parallel_info = QLabel(f"Note: Select cores based on available RAM. More cores = faster but use more memory.")
        parallel_info.setStyleSheet("color: #95E1D3; padding: 3px; font-size: 9px; font-style: italic;")
        processing_options_layout.addWidget(parallel_info)
        step4_layout.addLayout(processing_options_layout)
        
        generate_csv_btn = QPushButton("📊 Generate CSV Files")
        generate_csv_btn.setObjectName("primary")
        generate_csv_btn.clicked.connect(self.generate_csv_files)
        generate_csv_btn.setEnabled(False)
        self.generate_csv_btn = generate_csv_btn
        step4_layout.addWidget(generate_csv_btn)
        
        # Help text
        self.csv_help_text = QLabel("Prerequisites: CT folder detected AND RTSTRUCT file available")
        self.csv_help_text.setStyleSheet("color: #95E1D3; padding: 5px; font-size: 10px; font-style: italic;")
        step4_layout.addWidget(self.csv_help_text)
        
        self.csv_status = QLabel("CSV files not generated yet")
        self.csv_status.setStyleSheet("color: #FF6B6B; padding: 5px;")
        step4_layout.addWidget(self.csv_status)
        
        step4_group.setLayout(step4_layout)
        left_layout.addWidget(step4_group)
        
        left_layout.addStretch()
        
        # Right side: Step 2 and Progress Box
        right_layout = QVBoxLayout()
        right_layout.setSpacing(6)
        
        # Step 2: File Detection Results
        step2_group = QGroupBox("Step 2: Detected Files")
        step2_layout = QVBoxLayout()
        
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["File Type", "Path", "Status"])
        self.file_tree.setColumnWidth(0, 120)
        self.file_tree.setColumnWidth(1, 300)
        step2_layout.addWidget(self.file_tree)
        
        step2_group.setLayout(step2_layout)
        right_layout.addWidget(step2_group)
        
        # Progress Box
        progress_group = QGroupBox("Progress Log")
        progress_layout = QVBoxLayout()
        
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(200)
        self.progress_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                color: #95E1D3;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9px;
            }
        """)
        progress_layout.addWidget(self.progress_text)
        
        # Clear progress button
        clear_progress_btn = QPushButton("Clear Log")
        clear_progress_btn.setMaximumHeight(24)
        clear_progress_btn.clicked.connect(lambda: self.progress_text.clear())
        clear_progress_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: #ffffff;
                padding: 4px 12px;
                font-size: 9px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        progress_layout.addWidget(clear_progress_btn)
        
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)
        
        right_layout.addStretch()
        
        # Add left and right to main layout
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        
        return widget
    
    def create_analysis_tab(self):
        """Create analysis and reporting tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Analysis Results Section
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        # Results display with tabs for RTSTRUCT and ABAS
        results_tabs = QTabWidget()
        
        # RTSTRUCT Results Tab
        rtstruct_results_widget = QWidget()
        rtstruct_results_layout = QVBoxLayout(rtstruct_results_widget)
        self.rtstruct_results_text = QTextEdit()
        self.rtstruct_results_text.setReadOnly(True)
        self.rtstruct_results_text.setPlaceholderText("Click 'Analyze RTSTRUCT' to see results here...")
        rtstruct_results_layout.addWidget(self.rtstruct_results_text)
        results_tabs.addTab(rtstruct_results_widget, "RTSTRUCT Analysis")
        
        # ABAS Results Tab
        abas_results_widget = QWidget()
        abas_results_layout = QVBoxLayout(abas_results_widget)
        self.abas_results_text = QTextEdit()
        self.abas_results_text.setReadOnly(True)
        self.abas_results_text.setPlaceholderText("Click 'Analyze ABAS RTSTRUCT' to see results here...")
        abas_results_layout.addWidget(self.abas_results_text)
        results_tabs.addTab(abas_results_widget, "ABAS RTSTRUCT Analysis")
        
        results_layout.addWidget(results_tabs)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Report area
        report_group = QGroupBox("Combined Analysis Report")
        report_layout = QVBoxLayout()
        
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMaximumHeight(150)
        report_layout.addWidget(self.report_text)
        
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        generate_report_btn = QPushButton("📄 Generate Combined Report")
        generate_report_btn.clicked.connect(self.generate_report)
        btn_layout.addWidget(generate_report_btn)
        
        export_report_btn = QPushButton("💾 Export Report")
        export_report_btn.clicked.connect(self.export_report)
        btn_layout.addWidget(export_report_btn)
        
        clear_results_btn = QPushButton("🗑️ Clear Results")
        clear_results_btn.clicked.connect(self.clear_analysis_results)
        btn_layout.addWidget(clear_results_btn)
        
        layout.addLayout(btn_layout)
        return widget
    
    def create_visualization_tab(self):
        """Create visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - 3D plot
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot = Centroid3DPlot(plot_container)
        # Set parent window reference so plot can prompt for files if needed
        self.plot.parent_window = self
        # Enable interactive navigation - toolbar provides zoom, pan, rotate controls
        self.toolbar = NavigationToolbar(self.plot, plot_container)
        # Enable mouse interaction for 3D rotation (left-click drag to rotate)
        self.plot.ax.mouse_init()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.plot)
        
        # Alignment knobs (temporary for alignment purposes) - collapsible with clickable arrow
        alignment_header_btn = QPushButton("▶ Alignment Controls (Temporary)")
        alignment_header_btn.setCheckable(True)
        alignment_header_btn.setChecked(False)  # Start collapsed
        alignment_header_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px;
                font-weight: bold;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QPushButton:hover {
                background-color: #353535;
                color: white;
            }
            QPushButton:checked {
                background-color: #2b2b2b;
                color: white;
            }
        """)
        alignment_header_btn.toggled.connect(self.on_alignment_group_toggled)
        plot_layout.addWidget(alignment_header_btn)
        
        # Content widget that will be shown/hidden
        self.alignment_content = QWidget()
        alignment_layout = QVBoxLayout(self.alignment_content)
        alignment_layout.setContentsMargins(10, 5, 10, 5)
        self.alignment_content.setVisible(False)  # Start hidden
        
        # Mask rotation knobs
        mask_label = QLabel("<b>Mask Rotation (degrees):</b>")
        alignment_layout.addWidget(mask_label)
        mask_knobs_layout = QHBoxLayout()
        
        # Mask X rotation
        mask_x_layout = QVBoxLayout()
        mask_x_layout.addWidget(QLabel("Mask X:"))
        self.mask_x_slider = QSlider(Qt.Horizontal)
        self.mask_x_slider.setMinimum(-180)
        self.mask_x_slider.setMaximum(180)
        self.mask_x_slider.setValue(0)
        self.mask_x_slider.setTickPosition(QSlider.TicksBelow)
        self.mask_x_slider.setTickInterval(45)
        self.mask_x_slider.valueChanged.connect(self.on_mask_rotation_changed)
        mask_x_layout.addWidget(self.mask_x_slider)
        self.mask_x_label = QLabel("0°")
        self.mask_x_label.setAlignment(Qt.AlignCenter)
        mask_x_layout.addWidget(self.mask_x_label)
        mask_knobs_layout.addLayout(mask_x_layout)
        
        # Mask Y rotation
        mask_y_layout = QVBoxLayout()
        mask_y_layout.addWidget(QLabel("Mask Y:"))
        self.mask_y_slider = QSlider(Qt.Horizontal)
        self.mask_y_slider.setMinimum(-180)
        self.mask_y_slider.setMaximum(180)
        self.mask_y_slider.setValue(0)
        self.mask_y_slider.setTickPosition(QSlider.TicksBelow)
        self.mask_y_slider.setTickInterval(45)
        self.mask_y_slider.valueChanged.connect(self.on_mask_rotation_changed)
        mask_y_layout.addWidget(self.mask_y_slider)
        self.mask_y_label = QLabel("0°")
        self.mask_y_label.setAlignment(Qt.AlignCenter)
        mask_y_layout.addWidget(self.mask_y_label)
        mask_knobs_layout.addLayout(mask_y_layout)
        
        # Mask Z rotation
        mask_z_layout = QVBoxLayout()
        mask_z_layout.addWidget(QLabel("Mask Z:"))
        self.mask_z_slider = QSlider(Qt.Horizontal)
        self.mask_z_slider.setMinimum(-180)
        self.mask_z_slider.setMaximum(180)
        self.mask_z_slider.setValue(0)
        self.mask_z_slider.setTickPosition(QSlider.TicksBelow)
        self.mask_z_slider.setTickInterval(45)
        self.mask_z_slider.valueChanged.connect(self.on_mask_rotation_changed)
        mask_z_layout.addWidget(self.mask_z_slider)
        self.mask_z_label = QLabel("0°")
        self.mask_z_label.setAlignment(Qt.AlignCenter)
        mask_z_layout.addWidget(self.mask_z_label)
        mask_knobs_layout.addLayout(mask_z_layout)
        
        alignment_layout.addLayout(mask_knobs_layout)
        
        # Centroid rotation knobs
        centroid_label = QLabel("<b>Centroid Rotation (degrees):</b>")
        alignment_layout.addWidget(centroid_label)
        centroid_knobs_layout = QHBoxLayout()
        
        # Centroid X rotation
        centroid_x_layout = QVBoxLayout()
        centroid_x_layout.addWidget(QLabel("Centroid X:"))
        self.centroid_x_slider = QSlider(Qt.Horizontal)
        self.centroid_x_slider.setMinimum(-180)
        self.centroid_x_slider.setMaximum(180)
        self.centroid_x_slider.setValue(0)
        self.centroid_x_slider.setTickPosition(QSlider.TicksBelow)
        self.centroid_x_slider.setTickInterval(45)
        self.centroid_x_slider.valueChanged.connect(self.on_centroid_rotation_changed)
        centroid_x_layout.addWidget(self.centroid_x_slider)
        self.centroid_x_label = QLabel("0°")
        self.centroid_x_label.setAlignment(Qt.AlignCenter)
        centroid_x_layout.addWidget(self.centroid_x_label)
        centroid_knobs_layout.addLayout(centroid_x_layout)
        
        # Centroid Y rotation
        centroid_y_layout = QVBoxLayout()
        centroid_y_layout.addWidget(QLabel("Centroid Y:"))
        self.centroid_y_slider = QSlider(Qt.Horizontal)
        self.centroid_y_slider.setMinimum(-180)
        self.centroid_y_slider.setMaximum(180)
        self.centroid_y_slider.setValue(0)
        self.centroid_y_slider.setTickPosition(QSlider.TicksBelow)
        self.centroid_y_slider.setTickInterval(45)
        self.centroid_y_slider.valueChanged.connect(self.on_centroid_rotation_changed)
        centroid_y_layout.addWidget(self.centroid_y_slider)
        self.centroid_y_label = QLabel("0°")
        self.centroid_y_label.setAlignment(Qt.AlignCenter)
        centroid_y_layout.addWidget(self.centroid_y_label)
        centroid_knobs_layout.addLayout(centroid_y_layout)
        
        # Centroid Z rotation
        centroid_z_layout = QVBoxLayout()
        centroid_z_layout.addWidget(QLabel("Centroid Z:"))
        self.centroid_z_slider = QSlider(Qt.Horizontal)
        self.centroid_z_slider.setMinimum(-180)
        self.centroid_z_slider.setMaximum(180)
        self.centroid_z_slider.setValue(0)
        self.centroid_z_slider.setTickPosition(QSlider.TicksBelow)
        self.centroid_z_slider.setTickInterval(45)
        self.centroid_z_slider.valueChanged.connect(self.on_centroid_rotation_changed)
        centroid_z_layout.addWidget(self.centroid_z_slider)
        self.centroid_z_label = QLabel("0°")
        self.centroid_z_label.setAlignment(Qt.AlignCenter)
        centroid_z_layout.addWidget(self.centroid_z_label)
        centroid_knobs_layout.addLayout(centroid_z_layout)
        
        alignment_layout.addLayout(centroid_knobs_layout)
        
        # Reset buttons
        reset_buttons_layout = QHBoxLayout()
        
        reset_mask_btn = QPushButton("Reset Mask Rotation")
        reset_mask_btn.clicked.connect(self.reset_mask_rotation)
        reset_mask_btn.setToolTip("Reset all mask rotation knobs to 0°")
        reset_buttons_layout.addWidget(reset_mask_btn)
        
        reset_centroid_btn = QPushButton("Reset Centroid Rotation")
        reset_centroid_btn.clicked.connect(self.reset_centroid_rotation)
        reset_centroid_btn.setToolTip("Reset all centroid rotation knobs to 0°")
        reset_buttons_layout.addWidget(reset_centroid_btn)
        
        alignment_layout.addLayout(reset_buttons_layout)
        
        # Store reference to header button for toggle handler
        self.alignment_header_btn = alignment_header_btn
        
        plot_layout.addWidget(self.alignment_content)
        
        splitter.addWidget(plot_container)
        
        # Right side - Table and info
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Info panel
        info_group = QGroupBox("ROI Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        # Remove maximum height so it can expand with the GUI
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        # Distance table
        table_group = QGroupBox("Distance Measurements")
        table_layout = QVBoxLayout()
        
        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search ROI names...")
        self.filter_input.textChanged.connect(self.filter_table)
        filter_layout.addWidget(self.filter_input)
        table_layout.addLayout(filter_layout)
        
        self.distance_table = QTableWidget()
        self.distance_table.setColumnCount(7)
        # Store column headers as a list for reference
        self.distance_table_headers = [
            "Reference ROI", "Target ROI", "Distance (mm)", 
            "Phi (°)", "Theta (°)", "Overlap (%)", "5th %ile (mm)"
        ]
        self.distance_table.setHorizontalHeaderLabels(self.distance_table_headers)
        # Use Interactive mode with minimum column widths to ensure all text is visible
        self.distance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        # Set minimum column widths to ensure all text is visible
        self.distance_table.setColumnWidth(0, 150)  # Reference ROI
        self.distance_table.setColumnWidth(1, 150)  # Target ROI
        self.distance_table.setColumnWidth(2, 100)  # Distance (mm)
        self.distance_table.setColumnWidth(3, 80)   # Phi (°)
        self.distance_table.setColumnWidth(4, 80)   # Theta (°)
        self.distance_table.setColumnWidth(5, 100)  # Overlap (%)
        self.distance_table.setColumnWidth(6, 100)  # 5th %ile (mm)
        # Allow horizontal scrolling if content is wider
        self.distance_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.distance_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.distance_table.setSelectionMode(QTableWidget.SingleSelection)
        self.distance_table.itemSelectionChanged.connect(self.on_table_selection)
        self.distance_table.setAlternatingRowColors(True)
        # Set better colors for alternating rows - improved contrast for readability
        self.distance_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #2a2a2a;
                color: #ffffff;
                gridline-color: #444444;
            }
            QTableWidget::item {
                padding: 4px;
                color: #ffffff;
                background-color: #1e1e1e;
            }
            QTableWidget::item:alternate {
                background-color: #2a2a2a;
                color: #ffffff;
            }
            QTableWidget::item:selected {
                background-color: #4a90e2;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 6px;
                border: 1px solid #444444;
                font-weight: bold;
            }
        """)
        table_layout.addWidget(self.distance_table)
        table_group.setLayout(table_layout)
        right_layout.addWidget(table_group)
        
        splitter.addWidget(right_panel)
        # Set right panel to take more width (60% of GUI) for better table visibility
        splitter.setStretchFactor(0, 2)  # Plot takes 40%
        splitter.setStretchFactor(1, 3)  # Right panel (ROI Info + Distance Measurements) takes 60%
        # Set minimum width for right panel to ensure table is readable
        right_panel.setMinimumWidth(600)  # Increased from 450 to 600 for better table visibility
        
        layout.addWidget(splitter)
        return widget
    
    def create_control_panel(self):
        """Create control panel for visualization"""
        panel = QGroupBox("Visualization Controls")
        layout = QHBoxLayout()
        
        # Load buttons
        load_centroid_btn = QPushButton("Load Centroids CSV")
        load_centroid_btn.clicked.connect(self.load_centroids)
        layout.addWidget(load_centroid_btn)
        
        load_distance_btn = QPushButton("Load Distances CSV")
        load_distance_btn.clicked.connect(self.load_distances)
        layout.addWidget(load_distance_btn)
        
        layout.addWidget(QLabel("|"))
        
        # GTV selection
        layout.addWidget(QLabel("GTV ROI:"))
        self.gtv_combo = QComboBox()
        self.gtv_combo.setMinimumWidth(200)
        self.gtv_combo.currentTextChanged.connect(self.on_gtv_changed)
        layout.addWidget(self.gtv_combo)
        
        layout.addWidget(QLabel("Select ROI:"))
        self.roi_combo = QComboBox()
        self.roi_combo.setMinimumWidth(200)
        self.roi_combo.currentTextChanged.connect(self.on_roi_selected)
        layout.addWidget(self.roi_combo)
        
        layout.addWidget(QLabel("|"))
        
        # Reveal All button
        reveal_all_btn = QPushButton("Reveal All")
        reveal_all_btn.clicked.connect(self.reveal_all_rois)
        reveal_all_btn.setToolTip("Show all ROIs on the plot")
        layout.addWidget(reveal_all_btn)
        
        # Clear All button
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_rois)
        clear_all_btn.setToolTip("Hide all ROIs from the plot")
        layout.addWidget(clear_all_btn)
        
        layout.addWidget(QLabel("|"))
        
        # Show all checkbox - unchecked by default (user must select GTV or click Reveal All)
        self.show_all_check = QCheckBox("Show All ROIs")
        self.show_all_check.setChecked(False)  # Changed to False - don't show all by default
        self.show_all_check.stateChanged.connect(self.update_plot)
        layout.addWidget(self.show_all_check)
        
        # Show head mask checkbox
        self.show_head_mask_check = QCheckBox("Show Head/Neck Mask")
        self.show_head_mask_check.setChecked(False)
        self.show_head_mask_check.stateChanged.connect(self.on_head_mask_checkbox_changed)
        self.show_head_mask_check.setToolTip("Display semi-transparent head/neck mask for anatomical reference")
        layout.addWidget(self.show_head_mask_check)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def load_centroids(self):
        """Load centroid CSV file - don't show anything until user selects GTV or clicks Reveal All"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Centroids CSV", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                # Check if head mask is currently shown (before clearing)
                head_mask_was_shown = (hasattr(self, 'show_head_mask_check') and 
                                      self.show_head_mask_check.isChecked() and
                                      hasattr(self.plot, 'head_mask_data') and 
                                      self.plot.head_mask_data is not None)
                
                # Store mask data if it exists (so we can re-draw it)
                mask_data = None
                mask_spatial_data = None
                mask_name = None
                if head_mask_was_shown:
                    mask_data = self.plot.head_mask_data
                    mask_spatial_data = self.plot.head_mask_spatial_data
                    mask_name = self.plot.head_mask_name
                    print("Preserving head mask data for re-drawing after loading centroids...")
                
                self.centroid_df = pd.read_csv(file_path)
                # Reset selections - don't show anything by default
                self.gtv_name = None
                self.gtv_combo.setCurrentText("None")
                self.roi_combo.setCurrentText("None")
                self.show_all_check.setChecked(False)  # Don't show all by default
                self.update_roi_combos()
                
                # Clear the plot - don't show anything until user selects GTV or clicks Reveal All
                self.plot.ax.clear()
                self.plot.ax.set_facecolor('#1e1e1e')
                
                # Re-draw head mask if it was shown before loading centroids
                if head_mask_was_shown and mask_data is not None:
                    print("Re-drawing head mask after loading centroids...")
                    # Restore mask data
                    self.plot.head_mask_data = mask_data
                    self.plot.head_mask_spatial_data = mask_spatial_data
                    self.plot.head_mask_name = mask_name
                    
                    # Set up axes for mask visualization (same as in _on_head_mask_loaded)
                    self.plot.ax.xaxis.pane.fill = False
                    self.plot.ax.yaxis.pane.fill = False
                    self.plot.ax.zaxis.pane.fill = False
                    self.plot.ax.xaxis.pane.set_edgecolor('#404040')
                    self.plot.ax.yaxis.pane.set_edgecolor('#404040')
                    self.plot.ax.zaxis.pane.set_edgecolor('#404040')
                    # Remove grid lines
                    self.plot.ax.grid(False)
                    # Remove axis lines
                    self.plot.ax.xaxis.line.set_visible(False)
                    self.plot.ax.yaxis.line.set_visible(False)
                    self.plot.ax.zaxis.line.set_visible(False)
                    # Remove tick marks
                    self.plot.ax.set_xticks([])
                    self.plot.ax.set_yticks([])
                    self.plot.ax.set_zticks([])
                    self.plot.ax.set_xlabel('X', color='white', fontsize=10)
                    self.plot.ax.set_ylabel('Y', color='white', fontsize=10)
                    self.plot.ax.set_zlabel('Z', color='white', fontsize=10)
                    
                    # Set initial view angle to show mask clearly
                    self.plot.ax.view_init(elev=20, azim=45)
                    
                    # Determine which centroids to show based on Show All checkbox
                    if self.show_all_check.isChecked():
                        centroids_to_show = self.centroid_df
                    else:
                        # Only show selected centroids (GTV and/or selected ROI) - none selected yet
                        centroids_to_show = None
                    
                    # Re-draw the mask with centroids
                    self.plot._draw_head_mask(mask_data, mask_spatial_data, mask_name, centroids_to_show)
                
                self.plot.draw()
                self.statusBar().showMessage(f"Loaded centroids: {len(self.centroid_df)} ROIs - Select GTV ROI or click 'Reveal All' to view")
                self.update_export_button_state()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load centroids:\n{str(e)}")
    
    def load_distances(self):
        """Load distance CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Distances CSV", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                # Read CSV and handle "-" values as NaN for numeric columns
                self.distance_df = pd.read_csv(file_path, na_values=['-', 'N/A', 'nan', 'NaN', ''])
                
                # Convert numeric columns, replacing "-" or invalid values with NaN
                numeric_cols = ['Eucledian Distance (mm)', 'Phi (degrees)', 'Theta (degrees)', 
                              '% of Target Overlap', 'Eucledian Distance (mm) 5th Percentile']
                for col in numeric_cols:
                    if col in self.distance_df.columns:
                        self.distance_df[col] = pd.to_numeric(self.distance_df[col], errors='coerce')
                
                self.update_distance_table()
                self.statusBar().showMessage(f"Loaded distances: {len(self.distance_df)} measurements")
                self.update_export_button_state()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load distances:\n{str(e)}")
    
    def update_roi_combos(self):
        """Update ROI combo boxes"""
        if self.centroid_df is None:
            return
        
        possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
        roi_col = next((col for col in possible_roi if col in self.centroid_df.columns), self.centroid_df.columns[0])
        roi_names = sorted(self.centroid_df[roi_col].unique())
        
        self.gtv_combo.clear()
        self.gtv_combo.addItem("None")
        self.gtv_combo.addItems(roi_names)
        
        self.roi_combo.clear()
        self.roi_combo.addItem("None")
        self.roi_combo.addItems(roi_names)
    
    def on_gtv_changed(self, gtv_name):
        """Handle GTV selection change"""
        if gtv_name == "None":
            self.gtv_name = None
        else:
            self.gtv_name = gtv_name
        self.update_plot()
        self.update_info()
        self.update_export_button_state()
    
    def on_roi_selected(self, roi_name):
        """Handle ROI selection change"""
        selected = None if roi_name == "None" else roi_name
        self.update_plot(selected_roi=selected)
        self.update_info()
    
    def on_head_mask_checkbox_changed(self, state):
        """Handle head mask checkbox state change - directly prompt for CT folder"""
        if state == Qt.Checked:
            # Checkbox is checked - check if mask is already loaded
            if (hasattr(self.plot, 'head_mask_data') and self.plot.head_mask_data is not None and
                hasattr(self.plot, 'head_mask_spatial_data') and self.plot.head_mask_spatial_data is not None):
                # Mask already loaded - just re-draw it with current centroids
                print("Mask already loaded - re-drawing with current centroids...")
                # Determine which centroids to show based on Show All checkbox
                if self.show_all_check.isChecked():
                    centroids_to_show = self.centroid_df
                else:
                    centroids_to_show = None
                    if self.centroid_df is not None and len(self.centroid_df) > 0:
                        gtv_name = self.gtv_name
                        selected_roi = self.roi_combo.currentText()
                        if selected_roi == "None":
                            selected_roi = None
                        if gtv_name or selected_roi:
                            possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
                            roi_col = next((col for col in possible_roi if col in self.centroid_df.columns), self.centroid_df.columns[0])
                            rois_to_show = []
                            if gtv_name:
                                rois_to_show.append(gtv_name)
                            if selected_roi:
                                rois_to_show.append(selected_roi)
                            if rois_to_show:
                                centroids_to_show = self.centroid_df[self.centroid_df[roi_col].isin(rois_to_show)]
                
                # Re-draw mask with current centroids
                self.plot._draw_head_mask(self.plot.head_mask_data, self.plot.head_mask_spatial_data, 
                                         self.plot.head_mask_name, centroids_to_show)
                self.plot.draw()
            else:
                # Mask not loaded - prompt for CT folder and load it
                # Always pass full centroid_df - filtering will happen in _on_head_mask_loaded based on Show All checkbox
                # This ensures the mask stays visible and centroids are filtered correctly
                self.plot._load_head_mask_async(self.centroid_df)
        else:
            # Checkbox is unchecked - remove mask but keep centroids visible
            if hasattr(self.plot, 'head_surface') and self.plot.head_surface:
                try:
                    self.plot.head_surface.remove()
                except:
                    pass
                self.plot.head_surface = None
            self.plot.head_mask_data = None  # Clear mask data
            # Cancel any ongoing head mask loading
            if hasattr(self.plot, 'head_mask_loader') and self.plot.head_mask_loader and self.plot.head_mask_loader.isRunning():
                self.plot.head_mask_loader.terminate()
                self.plot.head_mask_loader.wait()
                self.plot.loading_head_mask = False
            # Update plot to remove mask (but keep centroids if they were shown)
            self.update_plot()
    
    def on_alignment_group_toggled(self, checked):
        """Handle alignment group toggle - show/hide content and update arrow"""
        if hasattr(self, 'alignment_content'):
            self.alignment_content.setVisible(checked)
        
        if hasattr(self, 'alignment_header_btn'):
            if checked:
                # Arrow down = expanded/visible
                self.alignment_header_btn.setText("▼ Alignment Controls (Temporary)")
            else:
                # Arrow up = collapsed/hidden
                self.alignment_header_btn.setText("▶ Alignment Controls (Temporary)")
    
    def on_mask_rotation_changed(self):
        """Handle mask rotation slider changes"""
        if hasattr(self, 'mask_x_slider'):
            self.plot.mask_rotation_x = self.mask_x_slider.value()
            self.mask_x_label.setText(f"{self.plot.mask_rotation_x}°")
        if hasattr(self, 'mask_y_slider'):
            self.plot.mask_rotation_y = self.mask_y_slider.value()
            self.mask_y_label.setText(f"{self.plot.mask_rotation_y}°")
        if hasattr(self, 'mask_z_slider'):
            self.plot.mask_rotation_z = self.mask_z_slider.value()
            self.mask_z_label.setText(f"{self.plot.mask_rotation_z}°")
        
        # Immediately redraw if mask is shown
        if self.plot.head_mask_data is not None and self.show_head_mask_check.isChecked():
            # Determine which centroids to show
            if self.show_all_check.isChecked():
                centroids_to_show = self.centroid_df
            else:
                centroids_to_show = None
                if self.centroid_df is not None and len(self.centroid_df) > 0:
                    gtv_name = self.gtv_name
                    selected_roi = self.roi_combo.currentText()
                    if selected_roi == "None":
                        selected_roi = None
                    if gtv_name or selected_roi:
                        possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
                        roi_col = next((col for col in possible_roi if col in self.centroid_df.columns), self.centroid_df.columns[0])
                        rois_to_show = []
                        if gtv_name:
                            rois_to_show.append(gtv_name)
                        if selected_roi:
                            rois_to_show.append(selected_roi)
                        if rois_to_show:
                            centroids_to_show = self.centroid_df[self.centroid_df[roi_col].isin(rois_to_show)]
            
            # Re-draw mask with new rotation
            self.plot._draw_head_mask(self.plot.head_mask_data, self.plot.head_mask_spatial_data, 
                                     self.plot.head_mask_name, centroids_to_show)
            self.plot.draw()
    
    def on_centroid_rotation_changed(self):
        """Handle centroid rotation slider changes"""
        if hasattr(self, 'centroid_x_slider'):
            self.plot.centroid_rotation_x = self.centroid_x_slider.value()
            self.centroid_x_label.setText(f"{self.plot.centroid_rotation_x}°")
        if hasattr(self, 'centroid_y_slider'):
            self.plot.centroid_rotation_y = self.centroid_y_slider.value()
            self.centroid_y_label.setText(f"{self.plot.centroid_rotation_y}°")
        if hasattr(self, 'centroid_z_slider'):
            self.plot.centroid_rotation_z = self.centroid_z_slider.value()
            self.centroid_z_label.setText(f"{self.plot.centroid_rotation_z}°")
        
        # Immediately redraw if centroids are shown
        if self.centroid_df is not None:
            self.update_plot()
    
    def reset_mask_rotation(self):
        """Reset all mask rotation knobs to 0"""
        if hasattr(self, 'mask_x_slider'):
            self.mask_x_slider.setValue(0)
        if hasattr(self, 'mask_y_slider'):
            self.mask_y_slider.setValue(0)
        if hasattr(self, 'mask_z_slider'):
            self.mask_z_slider.setValue(0)
        # The valueChanged signal will trigger on_mask_rotation_changed automatically
    
    def reset_centroid_rotation(self):
        """Reset all centroid rotation knobs to 0"""
        if hasattr(self, 'centroid_x_slider'):
            self.centroid_x_slider.setValue(0)
        if hasattr(self, 'centroid_y_slider'):
            self.centroid_y_slider.setValue(0)
        if hasattr(self, 'centroid_z_slider'):
            self.centroid_z_slider.setValue(0)
        # The valueChanged signal will trigger on_centroid_rotation_changed automatically
    
    def update_plot(self, selected_roi=None):
        """Update the 3D plot"""
        if self.centroid_df is None:
            return
        
        if selected_roi is None:
            selected_roi = self.roi_combo.currentText()
            if selected_roi == "None":
                selected_roi = None
        
        plot_df = self.centroid_df.copy()
        
        if not self.show_all_check.isChecked():
            # Only show selected ROIs (GTV and/or selected ROI)
            possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
            roi_col = next((col for col in possible_roi if col in self.centroid_df.columns), self.centroid_df.columns[0])
            
            # Build list of ROIs to show
            rois_to_show = []
            if self.gtv_name:
                rois_to_show.append(self.gtv_name)
            if selected_roi:
                rois_to_show.append(selected_roi)
            
            # Only plot if at least one ROI is selected
            if rois_to_show:
                plot_df = plot_df[plot_df[roi_col].isin(rois_to_show)]
            else:
                # No ROIs selected - clear the plot
                self.plot.ax.clear()
                self.plot.ax.set_facecolor('#1e1e1e')
                self.plot.draw()
                return
        
        # Pass head mask checkbox state to plot
        show_head_mask = hasattr(self, 'show_head_mask_check') and self.show_head_mask_check.isChecked()
        self.plot.plot_centroids(plot_df, self.distance_df, 
                                self.gtv_name, selected_roi, 
                                show_head_mask=show_head_mask,
                                scan_results=self.scan_results)
    
    def update_distance_table(self):
        """Update the distance table - FIXED: uses stored header list instead of horizontalHeaderLabels()"""
        if self.distance_df is None:
            return
        
        self.distance_table.setRowCount(len(self.distance_df))
        
        # Use stored header list instead of horizontalHeaderLabels() which doesn't exist
        for i, row in self.distance_df.iterrows():
            for j, col in enumerate(self.distance_table_headers):
                if col == "Reference ROI":
                    value = str(row['Reference ROI'])
                elif col == "Target ROI":
                    value = str(row['Target ROI'])
                elif col == "Distance (mm)":
                    value = f"{float(row['Eucledian Distance (mm)']):.2f}"
                elif col == "Phi (°)":
                    value = f"{float(row['Phi (degrees)']):.2f}"
                elif col == "Theta (°)":
                    value = f"{float(row['Theta (degrees)']):.2f}"
                elif col == "Overlap (%)":
                    value = f"{float(row['% of Target Overlap']) * 100:.2f}"
                elif col == "5th %ile (mm)":
                    value = f"{float(row['Eucledian Distance (mm) 5th Percentile']):.2f}"
                else:
                    value = str(row.iloc[j])
                
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                # Ensure white text color for all items (better contrast on both backgrounds)
                item.setForeground(QColor('#ffffff'))
                self.distance_table.setItem(i, j, item)
        
        self.filter_table()
    
    def filter_table(self):
        """Filter table based on search input"""
        filter_text = self.filter_input.text().lower()
        
        for i in range(self.distance_table.rowCount()):
            should_show = True
            if filter_text:
                ref_item = self.distance_table.item(i, 0)
                target_item = self.distance_table.item(i, 1)
                if ref_item and target_item:
                    ref_text = ref_item.text().lower()
                    target_text = target_item.text().lower()
                    should_show = filter_text in ref_text or filter_text in target_text
            
            self.distance_table.setRowHidden(i, not should_show)
    
    def reveal_all_rois(self):
        """Reveal all ROIs on the plot"""
        if self.centroid_df is None:
            QMessageBox.warning(self, "No Data", "Please load centroids CSV first.")
            return
        self.show_all_check.setChecked(True)
        self.update_plot()
        self.statusBar().showMessage("All ROIs revealed")
    
    def clear_all_rois(self):
        """Clear all ROIs from the plot"""
        self.show_all_check.setChecked(False)
        self.gtv_combo.setCurrentText("None")
        self.roi_combo.setCurrentText("None")
        self.gtv_name = None
        self.update_plot()
        self.statusBar().showMessage("All ROIs cleared")
    
    def on_table_selection(self):
        """Handle table row selection"""
        selected_rows = self.distance_table.selectedItems()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        ref_roi = self.distance_table.item(row, 0).text()
        target_roi = self.distance_table.item(row, 1).text()
        
        index = self.roi_combo.findText(target_roi)
        if index >= 0:
            self.roi_combo.setCurrentIndex(index)
        
        self.update_info()
    
    def update_info(self):
        """Update information panel"""
        info_text = ""
        
        if self.centroid_df is not None:
            info_text += f"<b>Total ROIs:</b> {len(self.centroid_df)}<br>"
        
        if self.gtv_name and self.centroid_df is not None:
            possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
            possible_x = ['x coordinate', 'x_coordinate', 'X', 'x']
            possible_y = ['y coordinate', 'y_coordinate', 'Y', 'y']
            possible_z = ['z coordinate', 'z_coordinate', 'Z', 'z']
            
            roi_col = next((col for col in possible_roi if col in self.centroid_df.columns), self.centroid_df.columns[0])
            x_col = next((col for col in possible_x if col in self.centroid_df.columns), self.centroid_df.columns[1] if len(self.centroid_df.columns) > 1 else self.centroid_df.columns[0])
            y_col = next((col for col in possible_y if col in self.centroid_df.columns), self.centroid_df.columns[2] if len(self.centroid_df.columns) > 2 else self.centroid_df.columns[0])
            z_col = next((col for col in possible_z if col in self.centroid_df.columns), self.centroid_df.columns[3] if len(self.centroid_df.columns) > 3 else self.centroid_df.columns[0])
            
            gtv_row = self.centroid_df[self.centroid_df[roi_col] == self.gtv_name]
            if len(gtv_row) > 0:
                gtv = gtv_row.iloc[0]
                info_text += f"<b>GTV:</b> {self.gtv_name}<br>"
                info_text += f"  Position: ({gtv[x_col]:.1f}, {gtv[y_col]:.1f}, {gtv[z_col]:.1f})<br>"
        
        selected_roi = self.roi_combo.currentText()
        if selected_roi != "None" and self.centroid_df is not None:
            possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
            possible_x = ['x coordinate', 'x_coordinate', 'X', 'x']
            possible_y = ['y coordinate', 'y_coordinate', 'Y', 'y']
            possible_z = ['z coordinate', 'z_coordinate', 'Z', 'z']
            
            roi_col = next((col for col in possible_roi if col in self.centroid_df.columns), self.centroid_df.columns[0])
            x_col = next((col for col in possible_x if col in self.centroid_df.columns), self.centroid_df.columns[1] if len(self.centroid_df.columns) > 1 else self.centroid_df.columns[0])
            y_col = next((col for col in possible_y if col in self.centroid_df.columns), self.centroid_df.columns[2] if len(self.centroid_df.columns) > 2 else self.centroid_df.columns[0])
            z_col = next((col for col in possible_z if col in self.centroid_df.columns), self.centroid_df.columns[3] if len(self.centroid_df.columns) > 3 else self.centroid_df.columns[0])
            
            roi_row = self.centroid_df[self.centroid_df[roi_col] == selected_roi]
            if len(roi_row) > 0:
                roi = roi_row.iloc[0]
                info_text += f"<b>Selected ROI:</b> {selected_roi}<br>"
                info_text += f"  Position: ({roi[x_col]:.1f}, {roi[y_col]:.1f}, {roi[z_col]:.1f})<br>"
                
                if self.gtv_name and self.gtv_name != selected_roi:
                    gtv_row = self.centroid_df[self.centroid_df[roi_col] == self.gtv_name]
                    if len(gtv_row) > 0:
                        gtv = gtv_row.iloc[0]
                        dx = roi[x_col] - gtv[x_col]
                        dy = roi[y_col] - gtv[y_col]
                        dz = roi[z_col] - gtv[z_col]
                        distance = np.sqrt(dx**2 + dy**2 + dz**2)
                        info_text += f"<b>Center-to-Center Distance:</b> {distance:.2f} mm<br>"
        
        if self.distance_df is not None:
            info_text += f"<b>Total Measurements:</b> {len(self.distance_df)}<br>"
        
        self.info_text.setHtml(info_text)
    
    def select_mrn_folder(self):
        """Select MRN folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select MRN Folder")
        if folder:
            self.mrn_folder = folder
            self.folder_label.setText(f"Selected: {os.path.basename(folder)}")
            self.scan_btn.setEnabled(True)
            self.statusBar().showMessage(f"Folder selected: {folder}")
    
    def scan_folder(self):
        """Scan folder for DICOM files"""
        if not self.mrn_folder:
            return
        
        self.scan_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.scanner = DICOMScanner(self.mrn_folder)
        self.scanner.progress.connect(self.on_scan_progress)
        self.scanner.finished.connect(self.on_scan_finished)
        self.scanner.error.connect(self.on_scan_error)
        self.scanner.start()
    
    def on_scan_progress(self, value, message):
        """Update scan progress"""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
        # Update progress text box
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        self.progress_text.append(f"[{timestamp}] [{value}%] {message}")
        # Auto-scroll to bottom
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum()
        )
    
    def on_scan_finished(self, results):
        """Handle scan completion"""
        self.scan_results = results
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        # Add completion message to progress log
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        self.progress_text.append(f"[{timestamp}] [100%] ✓ Scan complete!")
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum()
        )
        
        # Automatically start generating head/neck mask if CT folder is found
        # Generate only once per scan; skip if already loaded or if mask already exists
        if results.get('ct_folder') and os.path.exists(results['ct_folder']):
            # Check if mask is already loaded in plot
            mask_already_loaded = (hasattr(self, 'plot') and self.plot and 
                                 self.plot.head_mask_data is not None and 
                                 self.plot.head_mask_spatial_data is not None)
            
            # Only generate if not already done and mask not already loaded
            if not getattr(self, '_auto_mask_done', False) and not mask_already_loaded:
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                self.progress_text.append(f"[{timestamp}] Starting head/neck mask generation...")
                self.progress_text.verticalScrollBar().setValue(
                    self.progress_text.verticalScrollBar().maximum()
                )
                
                # Find RTSTRUCT file (priority: MergedSS.dcm > RTSTRUCT.dcm > ABAS.dcm)
                rtstruct_path = None
                if results.get('merged_rtstruct') and os.path.exists(results['merged_rtstruct']):
                    rtstruct_path = results['merged_rtstruct']
                elif results.get('rtstruct_files') and len(results['rtstruct_files']) > 0:
                    # Prefer RTSTRUCT.dcm over ABAS if both exist
                    rtstruct_path = results['rtstruct_files'][0]
                elif results.get('abas_rtstruct_files') and len(results['abas_rtstruct_files']) > 0:
                    rtstruct_path = results['abas_rtstruct_files'][0]
                
                # Start mask generation in background using the plot's method
                if hasattr(self, 'plot') and self.plot:
                    self.plot._generate_mask_from_scan_results(results['ct_folder'], rtstruct_path)
                    # Mark as done to avoid repeated generation on subsequent events
                    self._auto_mask_done = True
        
        # Update file tree
        self.file_tree.clear()
        
        # Add MRN
        mrn_item = QTreeWidgetItem(self.file_tree, ["MRN", results.get('mrn', 'Unknown'), "✓"])
        mrn_item.setExpanded(True)
        
        # Add CT folder
        if results.get('ct_folder'):
            ct_item = QTreeWidgetItem(mrn_item, ["CT Folder", results['ct_folder'], "✓"])
        
        # Add RTSTRUCT files - ensure no duplicates
        normal_rtstructs = []
        abas_rtstructs = []
        
        # Get unique files - prioritize ABAS classification
        abas_set = set(results.get('abas_rtstruct_files', []))
        normal_set = set(results.get('rtstruct_files', []))
        
        # Remove any files from normal that are also in ABAS
        normal_set = normal_set - abas_set
        
        normal_rtstructs = list(normal_set)
        abas_rtstructs = list(abas_set)
        
        if normal_rtstructs:
            rtstruct_item = QTreeWidgetItem(mrn_item, ["RTSTRUCT Files", "", f"{len(normal_rtstructs)} found"])
            for rtstruct in normal_rtstructs:
                QTreeWidgetItem(rtstruct_item, ["RTSTRUCT", os.path.basename(rtstruct), "✓"])
        
        # Add ABAS RTSTRUCT files
        if abas_rtstructs:
            abas_item = QTreeWidgetItem(mrn_item, ["ABAS RTSTRUCT Files", "", f"{len(abas_rtstructs)} found"])
            for abas in abas_rtstructs:
                QTreeWidgetItem(abas_item, ["ABAS RTSTRUCT", os.path.basename(abas), "✓"])
        
        # Always show manual classification button (user may want to reclassify)
        self.classify_btn.setVisible(len(normal_rtstructs) > 0 or len(abas_rtstructs) > 0)
        
        # Add merged RTSTRUCT
        if results.get('merged_rtstruct'):
            QTreeWidgetItem(mrn_item, ["Merged RTSTRUCT", os.path.basename(results['merged_rtstruct']), "✓"])
        
        # Enable buttons based on actual unique files
        abas_set = set(results.get('abas_rtstruct_files', []))
        normal_set = set(results.get('rtstruct_files', [])) - abas_set
        
        normal_count = len(normal_set)
        abas_count = len(abas_set)
        
        self.analyze_rtstruct_btn.setEnabled(normal_count > 0)
        self.analyze_abas_btn.setEnabled(abas_count > 0)
        self.standardize_btn.setEnabled(True)
        self.merge_btn.setEnabled(normal_count > 0 and abas_count > 0)
        
        # Enable CSV generation if we have CT folder and at least one RTSTRUCT
        has_ct = bool(results.get('ct_folder'))
        has_rtstruct = bool(results.get('merged_rtstruct') or 
                           normal_count > 0 or 
                           abas_count > 0)
        self.generate_csv_btn.setEnabled(has_ct and has_rtstruct)
        
        # Update help text
        if has_ct and has_rtstruct:
            self.csv_help_text.setText("✓ Ready to generate CSV files!")
            self.csv_help_text.setStyleSheet("color: #51CF66; padding: 5px; font-size: 10px; font-weight: bold;")
        elif not has_ct:
            self.csv_help_text.setText("⚠ Missing: CT folder not detected")
            self.csv_help_text.setStyleSheet("color: #FF6B6B; padding: 5px; font-size: 10px;")
        elif not has_rtstruct:
            self.csv_help_text.setText("⚠ Missing: No RTSTRUCT file found")
            self.csv_help_text.setStyleSheet("color: #FF6B6B; padding: 5px; font-size: 10px;")
        else:
            self.csv_help_text.setText("Prerequisites: CT folder detected AND RTSTRUCT file available")
            self.csv_help_text.setStyleSheet("color: #95E1D3; padding: 5px; font-size: 10px; font-style: italic;")
        
        self.statusBar().showMessage("Scan complete!")
        
        # Check for existing CSV files and offer to load them
        self._check_and_load_existing_csvs(results)
    
    def _check_and_load_existing_csvs(self, results):
        """Check for existing CT_centroid.csv and CT_distances.csv files and offer to load them"""
        # Determine where to look for CSV files (MRN folder or CT folder)
        search_paths = []
        if results.get('mrn_folder'):
            search_paths.append(results['mrn_folder'])
        if results.get('ct_folder'):
            search_paths.append(results['ct_folder'])
        if results.get('merged_rtstruct'):
            search_paths.append(os.path.dirname(results['merged_rtstruct']))
        elif results.get('rtstruct_files'):
            search_paths.append(os.path.dirname(results['rtstruct_files'][0]))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in search_paths:
            if path and path not in seen:
                seen.add(path)
                unique_paths.append(path)
        
        # Look for CSV files in these paths
        centroid_path = None
        distance_path = None
        
        # Debug: log search paths
        print(f"[_check_and_load_existing_csvs] Searching for CSV files in {len(unique_paths)} paths:")
        for path in unique_paths:
            print(f"  - {path}")
        
        for search_path in unique_paths:
            if not os.path.exists(search_path):
                print(f"  Path does not exist: {search_path}")
                continue
            
            potential_centroid = os.path.join(search_path, 'CT_centroid.csv')
            potential_distance = os.path.join(search_path, 'CT_distances.csv')
            
            if os.path.exists(potential_centroid) and centroid_path is None:
                centroid_path = potential_centroid
                print(f"  Found centroid CSV: {centroid_path}")
            if os.path.exists(potential_distance) and distance_path is None:
                distance_path = potential_distance
                print(f"  Found distance CSV: {distance_path}")
        
        print(f"[_check_and_load_existing_csvs] Final results: centroid={centroid_path is not None}, distance={distance_path is not None}")
        
        # Always ask about both files, showing which ones are found
        if centroid_path or distance_path:
            # Build message showing status of both files
            message_lines = ["Found existing CSV files:\n"]
            
            if centroid_path:
                message_lines.append(f"✓ CT_centroid.csv: {os.path.basename(centroid_path)}")
            else:
                message_lines.append(f"✗ CT_centroid.csv: Not found")
            
            if distance_path:
                message_lines.append(f"✓ CT_distances.csv: {os.path.basename(distance_path)}")
            else:
                message_lines.append(f"✗ CT_distances.csv: Not found")
            
            message_lines.append("\nWould you like to load the available files?")
            
            reply = QMessageBox.question(
                self, 
                "CSV Files Found", 
                "\n".join(message_lines),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                try:
                    loaded_centroid = False
                    loaded_distance = False
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    
                    # Load centroid CSV if found
                    if centroid_path:
                        self.centroid_df = pd.read_csv(centroid_path)
                        self.progress_text.append(f"[{timestamp}] ✓ Loaded existing centroid CSV: {os.path.basename(centroid_path)}")
                        self.update_roi_combos()
                        loaded_centroid = True
                    
                    # Load distance CSV if found
                    if distance_path:
                        try:
                            self.distance_df = pd.read_csv(distance_path, na_values=['-', 'N/A', 'nan', 'NaN', ''])
                            # Convert numeric columns
                            numeric_cols = ['Eucledian Distance (mm)', 'Phi (degrees)', 'Theta (degrees)', 
                                          '% of Target Overlap', 'Eucledian Distance (mm) 5th Percentile']
                            for col in numeric_cols:
                                if col in self.distance_df.columns:
                                    self.distance_df[col] = pd.to_numeric(self.distance_df[col], errors='coerce')
                            self.progress_text.append(f"[{timestamp}] ✓ Loaded existing distance CSV: {os.path.basename(distance_path)}")
                            self.update_distance_table()
                            loaded_distance = True
                        except Exception as e:
                            self.progress_text.append(f"[{timestamp}] ✗ Failed to load distance CSV: {str(e)}")
                            print(f"Error loading distance CSV: {e}")
                    
                    self.progress_text.verticalScrollBar().setValue(
                        self.progress_text.verticalScrollBar().maximum()
                    )
                    self.update_export_button_state()
                    
                    # Update status
                    status_parts = []
                    if loaded_centroid:
                        status_parts.append(f"{len(self.centroid_df)} ROIs")
                    if loaded_distance:
                        status_parts.append(f"{len(self.distance_df)} distances")
                    
                    if status_parts:
                        self.statusBar().showMessage(
                            f"Loaded existing CSV files: {', '.join(status_parts)}"
                        )
                    
                    # Show success message
                    success_parts = []
                    if loaded_centroid:
                        success_parts.append(f"Centroid CSV: {len(self.centroid_df)} ROIs")
                    if loaded_distance:
                        success_parts.append(f"Distance CSV: {len(self.distance_df)} measurements")
                    
                    if not loaded_centroid:
                        success_parts.append("Centroid CSV: Not found (you may need to generate it)")
                    if not loaded_distance:
                        success_parts.append("Distance CSV: Not found (you may need to generate it)")
                    
                    QMessageBox.information(
                        self, 
                        "CSV Files Loaded", 
                        f"Successfully loaded available CSV files!\n\n" + 
                        "\n".join(success_parts) + 
                        "\n\nYou can now use the Visualization tab to view the data."
                    )
                    
                    # If mask is already loaded, automatically check the checkbox and show it
                    if (hasattr(self, 'plot') and self.plot and 
                        self.plot.head_mask_data is not None and 
                        hasattr(self, 'show_head_mask_check')):
                        self.show_head_mask_check.setChecked(True)
                        # Re-draw mask with loaded centroids
                        if loaded_centroid and self.centroid_df is not None:
                            # Determine which centroids to show
                            if self.show_all_check.isChecked():
                                centroids_to_show = self.centroid_df
                            else:
                                centroids_to_show = None
                            self.plot._draw_head_mask(self.plot.head_mask_data, 
                                                     self.plot.head_mask_spatial_data, 
                                                     self.plot.head_mask_name, 
                                                     centroids_to_show)
                            self.plot.draw()
                except Exception as e:
                    QMessageBox.warning(
                        self, 
                        "Load Error", 
                        f"Failed to load existing CSV files:\n{str(e)}\n\n"
                        f"You can manually load them from the Visualization tab."
                    )
    
    def on_scan_error(self, error_msg):
        """Handle scan error"""
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        QMessageBox.critical(self, "Scan Error", f"Error scanning folder:\n{error_msg}")
    
    def manual_classify_rtstructs(self, results):
        """Manually classify RTSTRUCT files with two-box interface"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Manually Classify RTSTRUCT Files")
        dialog.setMinimumSize(800, 500)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-weight: bold;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: 2px solid #4ECDC4;
                border-radius: 5px;
                color: #ffffff;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #404040;
            }
            QListWidget::item:selected {
                background-color: #4ECDC4;
                color: #1a1a1a;
            }
            QPushButton {
                background-color: #4ECDC4;
                color: #1a1a1a;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #45b8b0;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        
        main_layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel("Move files between boxes to classify them as RTSTRUCT or ABAS RTSTRUCT")
        instructions.setStyleSheet("color: #95E1D3; padding: 10px;")
        main_layout.addWidget(instructions)
        
        # Two-box layout
        boxes_layout = QHBoxLayout()
        
        # Left box - Normal RTSTRUCT
        left_layout = QVBoxLayout()
        left_label = QLabel("Normal RTSTRUCT")
        left_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_label)
        
        self.normal_list = QListWidget()
        self.normal_list.setSelectionMode(QListWidget.SingleSelection)
        left_layout.addWidget(self.normal_list)
        boxes_layout.addLayout(left_layout)
        
        # Middle - Move buttons
        button_layout = QVBoxLayout()
        button_layout.addStretch()
        
        move_to_abas_btn = QPushButton("→ ABAS")
        move_to_abas_btn.clicked.connect(lambda: self.move_item(self.normal_list, self.abas_list))
        button_layout.addWidget(move_to_abas_btn)
        
        move_to_normal_btn = QPushButton("← Normal")
        move_to_normal_btn.clicked.connect(lambda: self.move_item(self.abas_list, self.normal_list))
        button_layout.addWidget(move_to_normal_btn)
        
        button_layout.addStretch()
        boxes_layout.addLayout(button_layout)
        
        # Right box - ABAS RTSTRUCT
        right_layout = QVBoxLayout()
        right_label = QLabel("ABAS RTSTRUCT")
        right_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_label)
        
        self.abas_list = QListWidget()
        self.abas_list.setSelectionMode(QListWidget.SingleSelection)
        right_layout.addWidget(self.abas_list)
        boxes_layout.addLayout(right_layout)
        
        main_layout.addLayout(boxes_layout)
        
        # Populate lists with current classification
        all_rtstructs = set(results.get('rtstruct_files', []) + results.get('abas_rtstruct_files', []))
        current_abas = set(results.get('abas_rtstruct_files', []))
        current_normal = set(results.get('rtstruct_files', [])) - current_abas
        
        for rtstruct_path in sorted(current_normal):
            item = QListWidgetItem(f"{os.path.basename(rtstruct_path)}\n{rtstruct_path}")
            item.setData(Qt.UserRole, rtstruct_path)  # Store full path
            self.normal_list.addItem(item)
        
        for rtstruct_path in sorted(current_abas):
            item = QListWidgetItem(f"{os.path.basename(rtstruct_path)}\n{rtstruct_path}")
            item.setData(Qt.UserRole, rtstruct_path)  # Store full path
            self.abas_list.addItem(item)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        main_layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            # Get final classification
            new_abas = []
            new_normal = []
            
            for i in range(self.abas_list.count()):
                item = self.abas_list.item(i)
                new_abas.append(item.data(Qt.UserRole))
            
            for i in range(self.normal_list.count()):
                item = self.normal_list.item(i)
                new_normal.append(item.data(Qt.UserRole))
            
            # Update results
            self.scan_results['abas_rtstruct_files'] = new_abas
            self.scan_results['rtstruct_files'] = new_normal
            
            # Refresh display
            self.on_scan_finished(self.scan_results)
            QMessageBox.information(self, "Classification Updated", 
                                   f"Reclassified {len(new_abas)} as ABAS, "
                                   f"{len(new_normal)} as normal RTSTRUCT")
    
    def move_item(self, source_list, dest_list):
        """Move selected item from source list to destination list"""
        current_item = source_list.currentItem()
        if current_item:
            # Create new item with same data
            new_item = QListWidgetItem(current_item.text())
            new_item.setData(Qt.UserRole, current_item.data(Qt.UserRole))
            dest_list.addItem(new_item)
            
            # Remove from source
            row = source_list.row(current_item)
            source_list.takeItem(row)
            
            # Select the moved item in destination
            dest_list.setCurrentItem(new_item)
    
    def analyze_rtstruct(self):
        """Analyze RTSTRUCT files"""
        abas_set = set(self.scan_results.get('abas_rtstruct_files', []))
        all_rtstructs = set(self.scan_results.get('rtstruct_files', []))
        normal_rtstructs = list(all_rtstructs - abas_set)
        
        if not normal_rtstructs:
            QMessageBox.warning(self, "No Files", "No normal RTSTRUCT files found")
            return
        
        results_html = "<h2 style='color: #4ECDC4;'>RTSTRUCT Analysis Results</h2>"
        results_html += f"<p><b>Analysis Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        results_html += f"<p><b>Number of Files Analyzed:</b> {len(normal_rtstructs)}</p>"
        results_html += "<hr><h3>Files Analyzed:</h3><ul>"
        for f in normal_rtstructs:
            results_html += f"<li><b>{os.path.basename(f)}</b><br><small style='color: #95E1D3;'>{f}</small></li>"
        results_html += "</ul>"
        
        results_html += "<hr><h3>ROI Names in RTSTRUCT Files:</h3>"
        all_roi_names = []
        for rtstruct_file in normal_rtstructs:
            try:
                ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
                roi_names = []
                if hasattr(ds, 'StructureSetROISequence'):
                    for roi in ds.StructureSetROISequence:
                        if hasattr(roi, 'ROIName'):
                            roi_name = str(roi.ROIName)
                            roi_names.append(roi_name)
                            if roi_name not in all_roi_names:
                                all_roi_names.append(roi_name)
                    
                    results_html += f"<h4>{os.path.basename(rtstruct_file)}:</h4>"
                    if roi_names:
                        results_html += "<ul>"
                        for roi_name in sorted(roi_names):
                            results_html += f"<li>{roi_name}</li>"
                        results_html += f"</ul><p><b>Total ROIs:</b> {len(roi_names)}</p>"
                    else:
                        results_html += "<p style='color: #FF6B6B;'>No ROI names found</p>"
                else:
                    results_html += f"<h4>{os.path.basename(rtstruct_file)}:</h4>"
                    results_html += "<p style='color: #FF6B6B;'>No StructureSetROISequence found</p>"
            except Exception as e:
                results_html += f"<h4>{os.path.basename(rtstruct_file)}:</h4>"
                results_html += f"<p style='color: #FF6B6B;'>Error reading file: {str(e)}</p>"
        
        if all_roi_names:
            results_html += f"<hr><h3>Summary - All Unique ROI Names ({len(all_roi_names)} total):</h3>"
            results_html += "<ul>"
            for roi_name in sorted(all_roi_names):
                results_html += f"<li>{roi_name}</li>"
            results_html += "</ul>"
        
        metadata_info = ""
        try:
            import scipy.io
            if os.path.exists('direct_metas.mat'):
                metas = scipy.io.loadmat('direct_metas.mat')
                metadata_info = "<h3 style='color: #51CF66;'>✓ Metadata Loaded Successfully</h3>"
                metadata_info += "<p><b>Source:</b> direct_metas.mat</p>"
                
                if metas:
                    metadata_info += "<h4>Metadata Variables:</h4><ul>"
                    for key in metas.keys():
                        if not key.startswith('__'):
                            var = metas[key]
                            var_type = type(var).__name__
                            if isinstance(var, np.ndarray):
                                var_shape = var.shape
                                metadata_info += f"<li><b>{key}</b>: {var_type} {var_shape}</li>"
                            elif isinstance(var, (list, dict)):
                                metadata_info += f"<li><b>{key}</b>: {var_type} (length: {len(var)})</li>"
                            else:
                                metadata_info += f"<li><b>{key}</b>: {var_type}</li>"
                    metadata_info += "</ul>"
                
                results_html += metadata_info
            else:
                metadata_info = "<h3 style='color: #FF6B6B;'>⚠ Metadata Not Found</h3>"
                metadata_info += "<p><b>Source:</b> direct_metas.mat</p>"
                metadata_info += "<p>File not found in current directory</p>"
                results_html += metadata_info
        except Exception as e:
            metadata_info = f"<h3 style='color: #FF6B6B;'>✗ Error Loading Metadata</h3>"
            metadata_info += f"<p><b>Error:</b> {str(e)}</p>"
            results_html += metadata_info
        
        self.rtstruct_results_text.setHtml(results_html)
        
        # Show pop-up message and stay on current tab
        QMessageBox.information(self, "RTSTRUCT Analyzed", 
                               f"RTSTRUCT analyzed successfully!\n\n"
                               f"Click Analysis tab for more information.")
    
    def analyze_abas(self):
        """Analyze ABAS RTSTRUCT files"""
        abas_files = list(set(self.scan_results.get('abas_rtstruct_files', [])))
        
        if not abas_files:
            QMessageBox.warning(self, "No Files", "No ABAS RTSTRUCT files found")
            return
        
        results_html = "<h2 style='color: #4ECDC4;'>ABAS RTSTRUCT Analysis Results</h2>"
        results_html += f"<p><b>Analysis Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        results_html += f"<p><b>Number of Files Analyzed:</b> {len(abas_files)}</p>"
        results_html += "<hr><h3>Files Analyzed:</h3><ul>"
        for f in abas_files:
            results_html += f"<li><b>{os.path.basename(f)}</b><br><small style='color: #95E1D3;'>{f}</small></li>"
        results_html += "</ul>"
        
        results_html += "<hr><h3>ROI Names in ABAS RTSTRUCT Files:</h3>"
        all_roi_names = []
        for rtstruct_file in abas_files:
            try:
                ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
                roi_names = []
                if hasattr(ds, 'StructureSetROISequence'):
                    for roi in ds.StructureSetROISequence:
                        if hasattr(roi, 'ROIName'):
                            roi_name = str(roi.ROIName)
                            roi_names.append(roi_name)
                            if roi_name not in all_roi_names:
                                all_roi_names.append(roi_name)
                    
                    results_html += f"<h4>{os.path.basename(rtstruct_file)}:</h4>"
                    if roi_names:
                        results_html += "<ul>"
                        for roi_name in sorted(roi_names):
                            results_html += f"<li>{roi_name}</li>"
                        results_html += f"</ul><p><b>Total ROIs:</b> {len(roi_names)}</p>"
                    else:
                        results_html += "<p style='color: #FF6B6B;'>No ROI names found</p>"
                else:
                    results_html += f"<h4>{os.path.basename(rtstruct_file)}:</h4>"
                    results_html += "<p style='color: #FF6B6B;'>No StructureSetROISequence found</p>"
            except Exception as e:
                results_html += f"<h4>{os.path.basename(rtstruct_file)}:</h4>"
                results_html += f"<p style='color: #FF6B6B;'>Error reading file: {str(e)}</p>"
        
        if all_roi_names:
            results_html += f"<hr><h3>Summary - All Unique ROI Names ({len(all_roi_names)} total):</h3>"
            results_html += "<ul>"
            for roi_name in sorted(all_roi_names):
                results_html += f"<li>{roi_name}</li>"
            results_html += "</ul>"
        
        metadata_info = ""
        try:
            import scipy.io
            if os.path.exists('direct_metas_ABAS.mat'):
                metas = scipy.io.loadmat('direct_metas_ABAS.mat')
                metadata_info = "<h3 style='color: #51CF66;'>✓ Metadata Loaded Successfully</h3>"
                metadata_info += "<p><b>Source:</b> direct_metas_ABAS.mat</p>"
                
                if metas:
                    metadata_info += "<h4>Metadata Variables:</h4><ul>"
                    for key in metas.keys():
                        if not key.startswith('__'):
                            var = metas[key]
                            var_type = type(var).__name__
                            if isinstance(var, np.ndarray):
                                var_shape = var.shape
                                metadata_info += f"<li><b>{key}</b>: {var_type} {var_shape}</li>"
                            elif isinstance(var, (list, dict)):
                                metadata_info += f"<li><b>{key}</b>: {var_type} (length: {len(var)})</li>"
                            else:
                                metadata_info += f"<li><b>{key}</b>: {var_type}</li>"
                    metadata_info += "</ul>"
                
                results_html += metadata_info
            else:
                metadata_info = "<h3 style='color: #FF6B6B;'>⚠ Metadata Not Found</h3>"
                metadata_info += "<p><b>Source:</b> direct_metas_ABAS.mat</p>"
                metadata_info += "<p>File not found in current directory</p>"
                results_html += metadata_info
        except Exception as e:
            metadata_info = f"<h3 style='color: #FF6B6B;'>✗ Error Loading Metadata</h3>"
            metadata_info += f"<p><b>Error:</b> {str(e)}</p>"
            results_html += metadata_info
        
        self.abas_results_text.setHtml(results_html)
        
        # Show pop-up message and stay on current tab
        QMessageBox.information(self, "ABAS RTSTRUCT Analyzed", 
                               f"ABAS RTSTRUCT analyzed successfully!\n\n"
                               f"Click Analysis tab for more information.")
    
    def clear_analysis_results(self):
        """Clear all analysis results"""
        reply = QMessageBox.question(self, 'Clear Results', 
                                    'Are you sure you want to clear all analysis results?',
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.rtstruct_results_text.clear()
            self.abas_results_text.clear()
            self.report_text.clear()
            QMessageBox.information(self, "Cleared", "All analysis results have been cleared")
    
    def standardize_names(self):
        """Standardize ROI names using metadata"""
        QMessageBox.information(self, "Standardization", 
                               "ROI name standardization would use metadata from .mat files\n"
                               "This feature can be extended based on your specific requirements")
    
    def _do_merge_rtstruct_dicom(self, normal_paths, abas_paths, output_path):
        """Chain-merge DICOM RTSTRUCT files: base=Normal[0], then Normal[1..n], then ABAS[0..m].
        Skips ROI names already present (first-occurrence wins, Normal priority).
        Returns dict with output_path, normal_rois, abas_rois."""
        from copy import deepcopy

        if not normal_paths:
            raise ValueError("At least one Normal RTSTRUCT is required")
        if not abas_paths:
            raise ValueError("At least one ABAS RTSTRUCT is required")

        # Step 1: base = Normal[0]
        ds_base = pydicom.dcmread(normal_paths[0])
        if not hasattr(ds_base, 'StructureSetROISequence'):
            raise ValueError("Base RTSTRUCT has no StructureSetROISequence")

        existing_names = set()
        for roi in ds_base.StructureSetROISequence:
            existing_names.add(str(roi.ROIName).strip().lower())

        max_roi_num = 0
        for roi in ds_base.StructureSetROISequence:
            n = int(roi.ROINumber)
            if n > max_roi_num:
                max_roi_num = n

        if not hasattr(ds_base, 'ROIContourSequence') or ds_base.ROIContourSequence is None:
            ds_base.ROIContourSequence = pydicom.sequence.Sequence()

        normal_count = len(ds_base.StructureSetROISequence)
        abas_count = 0

        def _add_from_file(ds_src, is_abas):
            """Add ROIs from ds_src to ds_base, skipping duplicate names."""
            nonlocal max_roi_num, normal_count, abas_count
            roi_contour_src = ds_src.ROIContourSequence if hasattr(ds_src, 'ROIContourSequence') else []
            added = 0
            for roi in ds_src.StructureSetROISequence:
                roi_name_key = str(roi.ROIName).strip().lower()
                if roi_name_key in existing_names:
                    continue
                existing_names.add(roi_name_key)
                orig_number = roi.ROINumber
                max_roi_num += 1
                new_number = max_roi_num
                new_roi = deepcopy(roi)
                new_roi.ROINumber = new_number
                ds_base.StructureSetROISequence.append(new_roi)
                for roi_contour in roi_contour_src:
                    if (hasattr(roi_contour, 'ReferencedROINumber') and
                            int(roi_contour.ReferencedROINumber) == int(orig_number)):
                        new_contour = deepcopy(roi_contour)
                        new_contour.ReferencedROINumber = new_number
                        ds_base.ROIContourSequence.append(new_contour)
                        break
                added += 1
                if is_abas:
                    abas_count += 1
                else:
                    normal_count += 1
            return added

        # Step 2: Merge Normal[1], Normal[2], ... (skip duplicates, first wins)
        for path in normal_paths[1:]:
            ds_src = pydicom.dcmread(path)
            if hasattr(ds_src, 'StructureSetROISequence'):
                _add_from_file(ds_src, is_abas=False)

        # Step 3: Merge ABAS[0], ABAS[1], ... (skip duplicates, Normal priority)
        for path in abas_paths:
            ds_src = pydicom.dcmread(path)
            if hasattr(ds_src, 'StructureSetROISequence'):
                _add_from_file(ds_src, is_abas=True)

        # Step 4: Output MergedSS.dcm
        ds_base.save_as(output_path)
        return {'output_path': output_path, 'normal_rois': normal_count, 'abas_rois': abas_count}

    def merge_rtstructs(self):
        """Merge RTSTRUCT and ABAS RTSTRUCT (chain merge: Normal[0..n] then ABAS[0..m], skip duplicates)"""
        if not (self.scan_results.get('rtstruct_files') and 
                self.scan_results.get('abas_rtstruct_files')):
            QMessageBox.warning(self, "Missing Files", 
                              "Need both RTSTRUCT and ABAS RTSTRUCT files to merge")
            return

        abas_set = set(self.scan_results.get('abas_rtstruct_files', []))
        normal_paths = [f for f in self.scan_results.get('rtstruct_files', []) if f not in abas_set]
        abas_paths = list(abas_set)

        if not normal_paths or not abas_paths:
            QMessageBox.warning(self, "Missing Files",
                              "Need both Normal and ABAS RTSTRUCT files to merge")
            return

        output_folder = os.path.dirname(normal_paths[0])
        output_path = os.path.join(output_folder, 'MergedSS.dcm')

        try:
            # Add merge start message to progress log
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            self.progress_text.append(f"[{timestamp}] Starting RTSTRUCT merge "
                                      f"({len(normal_paths)} Normal + {len(abas_paths)} ABAS)...")
            self.progress_text.verticalScrollBar().setValue(
                self.progress_text.verticalScrollBar().maximum()
            )

            # Chain merge: base=Normal[0], then Normal[1..n], then ABAS[0..m] (skip duplicate ROI names)
            merge_result = self._do_merge_rtstruct_dicom(normal_paths, abas_paths, output_path)
            
            # Extract output path and counts from result
            if isinstance(merge_result, dict):
                output_path = merge_result.get('output_path', output_path)
                normal_count = merge_result.get('normal_rois', 0)
                abas_count = merge_result.get('abas_rois', 0)
            else:
                # Fallback if old version returns just path
                normal_count = 0
                abas_count = 0
            
            self.scan_results['merged_rtstruct'] = output_path
            # Don't call on_scan_finished again - it will trigger duplicate mask generation
            # Just update the file tree to show merged RTSTRUCT
            if self.scan_results.get('merged_rtstruct'):
                # Find MRN item in tree and add merged RTSTRUCT if not already there
                root = self.file_tree.topLevelItem(0)
                if root:
                    # Check if merged RTSTRUCT item already exists
                    merged_exists = False
                    for i in range(root.childCount()):
                        child = root.child(i)
                        if child.text(0) == "Merged RTSTRUCT":
                            merged_exists = True
                            break
                    if not merged_exists:
                        QTreeWidgetItem(root, ["Merged RTSTRUCT", os.path.basename(self.scan_results['merged_rtstruct']), "✓"])
            
            # Add merge success message to progress log
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            self.progress_text.append(f"[{timestamp}] ✓ Successfully merged {normal_count} normal ROIs and {abas_count} ABAS ROIs")
            self.progress_text.append(f"[{timestamp}]   → Merged file saved to: {os.path.basename(output_path)}")
            self.progress_text.verticalScrollBar().setValue(
                self.progress_text.verticalScrollBar().maximum()
            )
            
            QMessageBox.information(self, "Merge Complete", 
                                   f"Successfully merged RTSTRUCT files!\n\n"
                                   f"Normal ROIs: {normal_count}\n"
                                   f"ABAS ROIs: {abas_count}\n\n"
                                   f"Merged file saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Merge Error", 
                               f"Failed to merge RTSTRUCT files:\n{str(e)}")
    
    def generate_csv_files(self):
        """Generate CSV files using Python with GPU/Parallel support"""
        if not self.scan_results.get('ct_folder'):
            QMessageBox.warning(self, "Missing CT", "CT folder not found. Please scan the folder first.")
            return
        
        rtstruct_to_use = None
        rtstruct_type = ""
        
        if self.scan_results.get('merged_rtstruct'):
            rtstruct_to_use = self.scan_results['merged_rtstruct']
            rtstruct_type = "Merged"
        else:
            abas_set = set(self.scan_results.get('abas_rtstruct_files', []))
            normal_rtstructs = [f for f in self.scan_results.get('rtstruct_files', []) 
                               if f not in abas_set]
            
            if normal_rtstructs:
                rtstruct_to_use = normal_rtstructs[0]
                rtstruct_type = "Normal RTSTRUCT"
            elif self.scan_results.get('abas_rtstruct_files'):
                rtstruct_to_use = self.scan_results['abas_rtstruct_files'][0]
                rtstruct_type = "ABAS RTSTRUCT"
            else:
                QMessageBox.warning(self, "No RTSTRUCT", 
                                   "No RTSTRUCT file found.\n\n"
                                   "Please ensure you have:\n"
                                   "- A merged RTSTRUCT (MergedSS.dcm), OR\n"
                                   "- At least one RTSTRUCT file detected\n\n"
                                   "You may need to:\n"
                                   "1. Merge RTSTRUCT files first (Step 3), OR\n"
                                   "2. Ensure RTSTRUCT files are in the scanned folder")
                return
        
        if not os.path.exists(rtstruct_to_use):
            QMessageBox.critical(self, "File Not Found", 
                               f"RTSTRUCT file not found:\n{rtstruct_to_use}")
            return
        
        if not os.path.exists(self.scan_results['ct_folder']):
            QMessageBox.critical(self, "Folder Not Found", 
                               f"CT folder not found:\n{self.scan_results['ct_folder']}")
            return
        
        # Get processing mode - try GPU if requested; fallback to CPU if CuPy/GPU unavailable
        use_gpu = False
        gpu_status = ""
        if self.gpu_checkbox.isChecked():
            try:
                # Ensure CUDA DLL path is set up before importing CuPy
                _setup_cuda_dll_path()
                import cupy as cp  # noqa: F401
                global GPU_AVAILABLE
                GPU_AVAILABLE = True  # update runtime flag
                use_gpu = True
                
                # Get GPU device info for logging
                try:
                    device_id = cp.cuda.Device().id
                    device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
                    gpu_status = f"GPU: {device_name} (Device {device_id})"
                except:
                    gpu_status = "GPU: Available (device info unavailable)"
            except Exception as e:
                # Inform the user and fall back to CPU
                error_msg = f"GPU not available or CuPy not installed ({e}); falling back to CPU."
                self.progress_text.append(error_msg)
                self.progress_text.verticalScrollBar().setValue(
                    self.progress_text.verticalScrollBar().maximum()
                )
                use_gpu = False
                gpu_status = f"GPU: Failed ({str(e)[:50]})"
        else:
            gpu_status = "GPU: Checkbox not checked"
        
        num_cores = self.cores_spinbox.value()  # Get selected number of cores
        
        mode_text = ""
        if use_gpu:
            mode_text = f"GPU acceleration with parallel processing ({gpu_status})"
        else:
            mode_text = f"Parallel CPU processing ({num_cores} cores) - {gpu_status}"
        
        reply = QMessageBox.question(self, "Generate CSV Files", 
                                    f"Generate centroid and distance CSV files?\n\n"
                                    f"CT Folder: {os.path.basename(self.scan_results['ct_folder'])}\n"
                                    f"RTSTRUCT: {os.path.basename(rtstruct_to_use)} ({rtstruct_type})\n"
                                    f"Output: {os.path.dirname(rtstruct_to_use)}\n"
                                    f"Processing Mode: {mode_text}\n\n"
                                    f"This may take several minutes...",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.Yes)
        
        if reply != QMessageBox.Yes:
            return
        
        output_folder = os.path.dirname(rtstruct_to_use)
        
        self.generate_csv_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.csv_status.setText(f"Generating CSV files using {mode_text}... Please wait...")
        self.csv_status.setStyleSheet("color: #FFD93D; padding: 5px;")
        
        # Always use parallel processing (GPU if available and requested, otherwise parallel CPU)
        self.python_processor = PythonCSVGeneratorV3(
            self.scan_results['ct_folder'],
            rtstruct_to_use,
            output_folder,
            use_gpu=use_gpu,
            use_parallel=True,  # Always use parallel processing
            num_cores=num_cores  # Pass selected number of cores
        )
        self.python_processor.progress.connect(self.on_python_progress)
        self.python_processor.finished.connect(self.on_csv_generated)
        self.python_processor.error.connect(self.on_python_error)
        self.python_processor.start()
    
    def on_python_progress(self, value, message):
        """Update Python processing progress"""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        self.progress_text.append(f"[{timestamp}] [{value}%] {message}")
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum()
        )
    
    def on_csv_generated(self, centroid_path, distance_path):
        """Handle CSV generation completion - FIXED: Auto-populates distance table"""
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(100)
        self.generate_csv_btn.setEnabled(True)
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        self.progress_text.append(f"[{timestamp}] [100%] ✓ CSV files generated successfully!")
        self.progress_text.append(f"[{timestamp}]   → {os.path.basename(centroid_path)}")
        self.progress_text.append(f"[{timestamp}]   → {os.path.basename(distance_path)}")
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum()
        )
        
        centroid_size = os.path.getsize(centroid_path) / 1024
        distance_size = os.path.getsize(distance_path) / 1024
        
        try:
            centroid_df = pd.read_csv(centroid_path)
            distance_df = pd.read_csv(distance_path)
            centroid_count = len(centroid_df)
            distance_count = len(distance_df)
        except:
            centroid_count = "Unknown"
            distance_count = "Unknown"
        
        self.csv_status.setText(
            f"✓ Generated: {os.path.basename(centroid_path)} ({centroid_count} ROIs, {centroid_size:.1f} KB) "
            f"and {os.path.basename(distance_path)} ({distance_count} measurements, {distance_size:.1f} KB)"
        )
        self.csv_status.setStyleSheet("color: #51CF66; padding: 5px; font-weight: bold;")
        
        # Auto-load the generated CSV files
        try:
            self.centroid_df = pd.read_csv(centroid_path)
            self.distance_df = pd.read_csv(distance_path, na_values=['-', 'N/A', 'nan', 'NaN', ''])
            
            # Convert numeric columns
            numeric_cols = ['Eucledian Distance (mm)', 'Phi (degrees)', 'Theta (degrees)', 
                          '% of Target Overlap', 'Eucledian Distance (mm) 5th Percentile']
            for col in numeric_cols:
                if col in self.distance_df.columns:
                    self.distance_df[col] = pd.to_numeric(self.distance_df[col], errors='coerce')
            
            self.update_roi_combos()
            self.update_distance_table()
            self.update_export_button_state()
            
            # Auto-generate mask if CT folder is available
            if self.scan_results and self.scan_results.get('ct_folder'):
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                self.progress_text.append(f"[{timestamp}] Auto-generating head/neck mask...")
                self.progress_text.verticalScrollBar().setValue(
                    self.progress_text.verticalScrollBar().maximum()
                )
                # Trigger mask generation automatically (use the plot's method)
                if hasattr(self, 'plot') and hasattr(self.plot, '_load_head_mask_async'):
                    # Store CT folder path for mask generation
                    self.plot._head_mask_ct_folder = self.scan_results['ct_folder']
                    # Generate mask in background
                    self.plot._load_head_mask_async(self.centroid_df)
        except Exception as e:
            print(f"Error auto-loading CSV files: {e}")
            import traceback
            traceback.print_exc()
        
        # Auto-load the files and populate table
        try:
            self.centroid_df = pd.read_csv(centroid_path)
            self.distance_df = pd.read_csv(distance_path)
            self.update_roi_combos()
            self.update_distance_table()  # FIXED: This will now populate the table
            self.update_plot()
            
            QMessageBox.information(self, "Success", 
                                   f"CSV files generated successfully!\n\n"
                                   f"Centroid CSV: {centroid_path}\n"
                                   f"  - {centroid_count} ROIs\n"
                                   f"  - {centroid_size:.1f} KB\n\n"
                                   f"Distance CSV: {distance_path}\n"
                                   f"  - {distance_count} measurements\n"
                                   f"  - {distance_size:.1f} KB\n\n"
                                   f"Files are ready for visualization!")
        except Exception as e:
            QMessageBox.warning(self, "Files Generated But Load Failed", 
                               f"CSV files were generated but could not be loaded:\n{str(e)}\n\n"
                               f"You can manually load them from the Visualization tab.")
    
    def on_python_error(self, error_msg):
        """Handle Python processing error"""
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.generate_csv_btn.setEnabled(True)
        self.csv_status.setText(f"✗ Error: {error_msg[:50]}...")
        self.csv_status.setStyleSheet("color: #FF6B6B; padding: 5px;")
        
        detailed_msg = f"Error generating CSV files:\n\n{error_msg}\n\n"
        detailed_msg += "Troubleshooting:\n"
        detailed_msg += "1. Verify CT folder contains valid DICOM files\n"
        detailed_msg += "2. Verify RTSTRUCT file is valid and readable\n"
        detailed_msg += "3. Check that all required Python packages are installed\n"
        detailed_msg += "4. Ensure RTSTRUCT contains valid contour data\n"
        detailed_msg += "5. Check that scikit-image and scipy are properly installed\n"
        if GPU_AVAILABLE:
            detailed_msg += "6. If using GPU, check CUDA installation and CuPy\n"
        
        QMessageBox.critical(self, "Processing Error", detailed_msg)
    
    def generate_report(self):
        """Generate analysis report"""
        report = "=== ROI Distance Analysis Report ===\n\n"
        
        if self.scan_results:
            report += f"MRN: {self.scan_results.get('mrn', 'Unknown')}\n"
            report += f"CT Folder: {self.scan_results.get('ct_folder', 'Not found')}\n"
            report += f"RTSTRUCT Files: {len(self.scan_results.get('rtstruct_files', []))}\n"
            report += f"ABAS RTSTRUCT Files: {len(self.scan_results.get('abas_rtstruct_files', []))}\n\n"
        
        if self.centroid_df is not None:
            report += f"Total ROIs: {len(self.centroid_df)}\n"
            possible_roi = ['ROI', 'roi', 'ROI Name', 'Structure Name']
            roi_col = next((col for col in possible_roi if col in self.centroid_df.columns), self.centroid_df.columns[0])
            report += f"ROI Names: {', '.join(self.centroid_df[roi_col].head(10).tolist())}\n\n"
        
        if self.distance_df is not None:
            report += f"Total Distance Measurements: {len(self.distance_df)}\n"
            if len(self.distance_df) > 0:
                avg_dist = self.distance_df['Eucledian Distance (mm)'].mean()
                report += f"Average Distance: {avg_dist:.2f} mm\n"
        
        self.report_text.setPlainText(report)
    
    def export_report(self):
        """Export report to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.report_text.toPlainText())
                QMessageBox.information(self, "Success", f"Report exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report:\n{str(e)}")
    
    def update_export_button_state(self):
        """Update the export button enabled state based on data availability"""
        if hasattr(self, 'export_gtv_distances_btn'):
            # Enable button only if centroids, distances are loaded and GTV is selected
            enabled = (self.centroid_df is not None and 
                      len(self.centroid_df) > 0 and
                      self.distance_df is not None and 
                      len(self.distance_df) > 0 and
                      self.gtv_name is not None)
            self.export_gtv_distances_btn.setEnabled(enabled)
    
    def export_gtv_distances(self):
        """Export all distances for the selected GTV ROI to Excel"""
        if self.centroid_df is None or self.distance_df is None or self.gtv_name is None:
            QMessageBox.warning(self, "Export Error", 
                              "Please load centroid and distance CSVs and select a GTV ROI first.")
            return
        
        # Filter distances for the selected GTV ROI
        # Check both Reference ROI and Target ROI columns
        possible_ref = ['Reference ROI', 'Reference', 'ref_roi', 'ROI_Reference']
        possible_target = ['Target ROI', 'Target', 'target_roi', 'ROI_Target']
        
        ref_col = next((col for col in possible_ref if col in self.distance_df.columns), None)
        target_col = next((col for col in possible_target if col in self.distance_df.columns), None)
        
        if ref_col is None or target_col is None:
            QMessageBox.warning(self, "Export Error", 
                              "Could not find Reference ROI or Target ROI columns in distance CSV.")
            return
        
        # Filter for rows where GTV is either reference or target
        gtv_distances = self.distance_df[
            (self.distance_df[ref_col] == self.gtv_name) | 
            (self.distance_df[target_col] == self.gtv_name)
        ].copy()
        
        if len(gtv_distances) == 0:
            QMessageBox.information(self, "No Data", 
                                  f"No distances found for GTV ROI: {self.gtv_name}")
            return
        
        # Check for distance column and ensure numeric values
        distance_col = 'Eucledian Distance (mm)'
        if distance_col in gtv_distances.columns:
            # Convert distance column to numeric, replacing any non-numeric values with NaN
            gtv_distances[distance_col] = pd.to_numeric(gtv_distances[distance_col], errors='coerce')
            # Check if all values are NaN or invalid
            if gtv_distances[distance_col].isna().all():
                QMessageBox.warning(self, "Export Error", 
                                  f"All distance values are invalid (showing as '-').\n\n"
                                  f"This may indicate an issue with the distance CSV file.\n"
                                  f"Please check that the CSV was generated correctly.")
                return
        
        # Check other numeric columns
        numeric_cols = ['Phi (degrees)', 'Theta (degrees)', '% of Target Overlap', 
                       'Eucledian Distance (mm) 5th Percentile']
        for col in numeric_cols:
            if col in gtv_distances.columns:
                gtv_distances[col] = pd.to_numeric(gtv_distances[col], errors='coerce')
        
        # Generate default filename
        default_filename = f"GTV_Distances_{self.gtv_name.replace(' ', '_')}.xlsx"
        
        # Open file dialog with default filename
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export GTV Distances to Excel", default_filename, 
            "Excel Files (*.xlsx);;All Files (*)")
        
        if file_path:
            try:
                # Ensure .xlsx extension
                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'
                
                # Export to Excel - format numeric columns properly
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    gtv_distances.to_excel(writer, index=False, sheet_name='GTV Distances')
                    
                    # Get the worksheet to format numeric columns
                    worksheet = writer.sheets['GTV Distances']
                    
                    # Format distance column as number with 2 decimal places
                    if distance_col in gtv_distances.columns:
                        from openpyxl.styles import Font
                        from openpyxl.utils import get_column_letter
                        col_idx = list(gtv_distances.columns).index(distance_col) + 1
                        col_letter = get_column_letter(col_idx)
                        for row in range(2, len(gtv_distances) + 2):  # Start from row 2 (skip header)
                            cell = worksheet[f'{col_letter}{row}']
                            if pd.notna(gtv_distances.iloc[row-2][distance_col]):
                                cell.number_format = '0.00'
                
                QMessageBox.information(self, "Export Successful", 
                                      f"Exported {len(gtv_distances)} distance measurements for GTV ROI '{self.gtv_name}' to:\n{file_path}")
            except Exception as e:
                import traceback
                error_msg = f"Failed to export distances:\n{str(e)}\n\n"
                if "openpyxl" in str(e).lower():
                    error_msg += "Make sure 'openpyxl' is installed: pip install openpyxl"
                else:
                    error_msg += f"Traceback:\n{traceback.format_exc()}"
                QMessageBox.critical(self, "Export Error", error_msg)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop timer
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        # Clean up any running threads before showing exit dialog
        # Stop head mask loader if running
        if hasattr(self, 'plot') and hasattr(self.plot, 'head_mask_loader'):
            if self.plot.head_mask_loader and self.plot.head_mask_loader.isRunning():
                self.plot.head_mask_loader.terminate()
                self.plot.head_mask_loader.wait(1000)  # Wait up to 1 second
        
        # Stop CSV generator thread if running
        if hasattr(self, 'csv_generator') and self.csv_generator and self.csv_generator.isRunning():
            self.csv_generator.terminate()
            self.csv_generator.wait(1000)
        
        # Stop scanner thread if running
        if hasattr(self, 'scanner') and self.scanner and self.scanner.isRunning():
            self.scanner.terminate()
            self.scanner.wait(1000)
        
        reply = QMessageBox.question(self, 'Exit Application', 
                                    'Are you sure you want to exit?',
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    app.setApplicationName("Enhanced ROI Distance Generator and Visualizer")
    app.setOrganizationName("Medical Imaging Analysis")
    
    window = EnhancedROIVisualizerV3()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

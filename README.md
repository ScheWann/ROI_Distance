# Enhanced ROI Distance Generator and Visualizer

A comprehensive Python-based tool for analyzing RTSTRUCT (DICOM RT Structure Set) files, calculating ROI (Region of Interest) centroids and distances, and visualizing them in an interactive 3D environment. This tool replaces MATLAB-based workflows with a fully Python implementation, supporting GPU acceleration and parallel CPU processing.

## Features

- **Complete RTSTRUCT Workflow**: Analyze, merge, and process RTSTRUCT and ABAS RTSTRUCT files
- **CSV Generation**: Generate centroid and distance CSV files for visualization
- **3D Visualization**: Interactive 3D plot with ROI centroids, distance measurements, and anatomical mask overlay
- **GPU Acceleration**: Automatic GPU detection with CUDA support (CuPy) for faster processing
- **Parallel Processing**: Multi-core CPU processing for efficient computation
- **No MATLAB Required**: All processing done entirely in Python
- **Interactive GUI**: User-friendly PyQt5 interface with dark theme

## Requirements

### Python Version
- Python 3.7 or higher

### Required Python Packages

#### Core Dependencies
```
pandas>=1.3.0
numpy>=1.19.0
PyQt5>=5.15.0
matplotlib>=3.3.0
pydicom>=2.2.0
scipy>=1.6.0
scikit-image>=0.18.0
```

#### Optional (for GPU acceleration)
```
cupy-cuda11x>=11.0.0  # For CUDA 11.x
# OR
cupy-cuda12x>=12.0.0  # For CUDA 12.x
```

#### Optional (for Excel export)
```
openpyxl>=3.0.0
```

### System Requirements

- **Operating System**: Windows 10/11 (tested), Linux, macOS (should work but not extensively tested)
- **RAM**: Minimum 8 GB, recommended 16 GB or more for large datasets
- **GPU** (optional): NVIDIA GPU with CUDA support for GPU acceleration
- **Storage**: Sufficient space for DICOM files and generated CSV files

## Installation

### 1. Install Python

Download and install Python 3.7 or higher from [python.org](https://www.python.org/downloads/).

### 2. Install Required Packages

#### Using pip (recommended)

```bash
pip install pandas numpy PyQt5 matplotlib pydicom scipy scikit-image openpyxl
```

#### For GPU Support (Optional)

If you have an NVIDIA GPU with CUDA installed:

**For CUDA 11.x:**
```bash
pip install cupy-cuda11x
```

**For CUDA 12.x:**
```bash
pip install cupy-cuda12x
```

**Note**: The tool will automatically detect and use GPU if CuPy is installed. If GPU is not available or CuPy is not installed, it will fall back to CPU processing.

### 3. Verify Installation

Run the tool using the provided batch file or directly:

```bash
python roi_distance_visualizer_enhanced_V_Lab.py
```

## Usage

### Quick Start

1. **Launch the Application**
   - Double-click `START_V3A.bat` (Windows)
   - Or run: `python roi_distance_visualizer_enhanced_V_Lab.py`

2. **Select MRN Folder**
   - Click "📂 Select MRN Folder" in the Workflow tab
   - Navigate to your patient folder containing CT and RTSTRUCT files

3. **Scan Folder**
   - Click "🔍 Scan Folder" to detect CT and RTSTRUCT files
   - Review detected files in the "Detected Files" panel

4. **Analyze RTSTRUCT Files** (Optional)
   - Click "🔬 Analyze RTSTRUCT" to analyze normal RTSTRUCT files
   - Click "🔬 Analyze ABAS RTSTRUCT" to analyze ABAS RTSTRUCT files
   - View results in the Analysis tab

5. **Merge RTSTRUCT Files** (Optional)
   - Click "🔀 Merge RTSTRUCT & ABAS" to combine RTSTRUCT and ABAS files
   - Merged file will be saved as `MergedSS.dcm`

6. **Generate CSV Files**
   - Configure processing options:
     - Check "Use GPU acceleration" if GPU is available
     - Select number of CPU cores (based on available RAM)
   - Click "📊 Generate CSV Files"
   - Wait for processing to complete
   - CSV files will be saved in the MRN folder:
     - `CT_centroid.csv` - ROI centroid coordinates
     - `CT_distances.csv` - Distance measurements between ROIs

7. **Visualize Results**
   - Go to the Visualization tab
   - Load CSV files:
     - Click "Load Centroids CSV" and select `CT_centroid.csv`
     - Click "Load Distances CSV" and select `CT_distances.csv`
   - Select GTV ROI from the dropdown
   - Select a target ROI to view distance measurements
   - Use "Show Head/Neck Mask" to overlay anatomical reference (optional)

### Batch Processing (CLI, No GUI)

For processing multiple patients on a remote server or in the background, use the command-line batch script. It does **not** require PyQt5 or a display.

```bash
Usage:

python batch_process.py /path/to/parent_folder [options]

Options:

--no-merge: Do not merge; only use the first available RTSTRUCT.

--skip-existing: Skip if a CSV already exists.

--cores: N (Number of CPU cores; default 4).

--verbose, -v: Verbose output.


# Run in background (e.g. on remote server)
nohup python batch_process.py /path/to/parent_folder --skip-existing --cores 10 > batch.log 2>&1 &
```

The batch script will, for each patient subfolder:
1. Scan for CT and RTSTRUCT files
2. Merge Normal + ABAS RTSTRUCT if both exist (chain merge, skip duplicate ROI names)
3. Generate `CT_centroid.csv` and `CT_distances.csv` in the patient folder

### Workflow Overview

The tool is organized into three main tabs:

#### 1. Workflow Tab
- **Step 1**: Select and scan MRN folder
- **Step 2**: View detected files (CT, RTSTRUCT, ABAS)
- **Step 3**: Analyze and merge RTSTRUCT files
- **Step 4**: Generate CSV files with GPU/CPU options

#### 2. Analysis Tab
- View RTSTRUCT analysis results
- View ABAS RTSTRUCT analysis results
- Generate and export combined reports

#### 3. Visualization Tab
- Load centroid and distance CSV files
- Interactive 3D plot with:
  - ROI centroids (color-coded: GTV=red, Selected=cyan, Others=light green)
  - Distance lines between selected ROIs
  - Distance annotations
  - Optional head/neck mask overlay
- ROI Information panel
- Distance Measurements table with filtering

### GPU Acceleration

The tool automatically detects GPU availability:

- **GPU Available**: Uses CUDA (CuPy) for faster array operations
- **GPU Not Available**: Falls back to parallel CPU processing
- **Note**: Distance transform calculations use CPU even with GPU (CuPy limitation)

To enable GPU:
1. Install CuPy matching your CUDA version
2. Ensure NVIDIA GPU drivers are installed
3. The tool will automatically detect and use GPU if available

### Parallel Processing

- **CPU Cores**: Selectable number of CPU cores (1 to available cores)
- **Memory Considerations**: More cores = faster processing but higher RAM usage
- **Recommendation**: Start with 4-8 cores, increase based on available RAM

## File Structure

### Input Files
- **CT Folder**: Contains DICOM CT image series
- **RTSTRUCT Files**: DICOM RT Structure Set files
  - Normal RTSTRUCT files (e.g., `RTSTRUCT.dcm`)
  - ABAS RTSTRUCT files (e.g., `ABAS_*.dcm`)

### Output Files
- **CT_centroid.csv**: ROI centroid coordinates (voxel coordinates)
- **CT_distances.csv**: Distance measurements between all ROI pairs
- **MergedSS.dcm**: Merged RTSTRUCT file (if merge is performed)

## Troubleshooting

### GPU Not Detected
- **Issue**: GPU shows as "Not Available" even with NVIDIA GPU
- **Solution**: 
  - Install CuPy matching your CUDA version: `pip install cupy-cuda11x` or `cupy-cuda12x`
  - Ensure NVIDIA GPU drivers are up to date
  - Check that CUDA toolkit is installed

### Memory Errors
- **Issue**: "MemoryError" or "Unable to allocate array"
- **Solution**:
  - Reduce number of CPU cores in processing options
  - Close other applications to free RAM
  - Process smaller datasets or use GPU acceleration

### GUI Hangs on Startup
- **Issue**: Application hangs at "Checking dependencies..."
- **Solution**:
  - Check if seaborn is installed (can cause hangs)
  - Try running with minimal dependencies first
  - Check system resources (CPU, RAM usage)

### CSV Files Not Generated
- **Issue**: CSV generation fails or produces empty files
- **Solution**:
  - Ensure CT folder and RTSTRUCT file are detected
  - Check that RTSTRUCT file contains valid ROI contours
  - Review Progress Log for error messages

### Distance Measurements Show as "-" or 0
- **Issue**: Distance CSV has missing or zero values
- **Solution**:
  - Verify ROI contours are valid in RTSTRUCT file
  - Check that ROIs have sufficient volume
  - Ensure RTSTRUCT and CT are from the same scan

## Performance Tips

1. **Use GPU**: Install CuPy for significant speedup on large datasets
2. **Optimize CPU Cores**: Balance between speed and memory usage
3. **Close Other Applications**: Free up RAM for processing
4. **Process in Batches**: For multiple patients, process one at a time

## Known Limitations

- Distance transform uses CPU even with GPU (CuPy limitation)
- Large datasets (>500 ROIs) may require significant RAM
- 3D text rotation in matplotlib has limitations (labels may not rotate perfectly in 3D space)

## Support

For issues or questions:
- Check the Progress Log in the Workflow tab for detailed error messages
- Review terminal/console output for additional debugging information

## License

Property of Fuller Lab. Unpublished work by Cem Dede.

## Version

Current version: Enhanced ROI Distance Generator and Visualizer
- Python-only implementation
- GPU acceleration support
- Parallel processing support
- No MATLAB dependency

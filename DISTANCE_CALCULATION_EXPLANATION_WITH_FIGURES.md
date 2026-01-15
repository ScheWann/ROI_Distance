# Distance Calculation Process: From Centroid Identification to Final Measurements

## Overview

This document explains how ROI distances are calculated in the Enhanced ROI Distance Visualizer, starting from centroid identification through to the final distance measurements.

---

## Step 1: RTSTRUCT to Binary Mask Conversion

### Process Flow

```
RTSTRUCT File (DICOM)
    │
    ├─ Extract Contour Points (3D coordinates)
    │   └─ Each ROI has multiple contour sequences
    │       └─ Each contour = list of (x, y, z) points in world coordinates
    │
    ├─ Convert to Voxel Coordinates
    │   └─ Transform using CT image spatial data:
    │       • Image Position (Patient)
    │       • Pixel Spacing
    │       • Slice Thickness
    │
    └─ Create Binary 3D Mask
        └─ Fill voxels inside contours = 1, outside = 0
```

### Visual Representation

```
RTSTRUCT Contour (2D slice view):
┌─────────────────────────────────┐
│                                 │
│     ●───●                       │
│    ╱     ╲                      │
│   ●       ●  ← Contour points   │
│    ╲     ╱                      │
│     ●───●                       │
│                                 │
└─────────────────────────────────┘

Converted to Binary Mask:
┌─────────────────────────────────┐
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │
│ 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 │
│ 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 │
│ 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 │
│ 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 │
│ 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 │
│ 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 │
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │
└─────────────────────────────────┘
    1 = Inside ROI, 0 = Outside ROI
```

**Code Reference:**
```323:346:roi_distance_visualizer_enhanced_v3a.py
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
```

---

## Step 2: Centroid Identification

### Method 1: Using `regionprops` (Primary Method)

**Process:**
1. Label connected components in the binary mask
2. Use `skimage.measure.regionprops` to calculate properties
3. Extract centroid from the largest connected region

**Mathematical Definition:**
```
Centroid = (x̄, ȳ, z̄)

Where:
  x̄ = (1/N) × Σ(x_i)  for all voxels i in ROI
  ȳ = (1/N) × Σ(y_i)  for all voxels i in ROI
  z̄ = (1/N) × Σ(z_i)  for all voxels i in ROI

N = Total number of voxels in ROI
```

### Visual Representation

```
3D Binary Mask (voxel view):
┌─────────────────────────────┐
│ Slice Z=0                   │
│ 0 0 0 0 0 0 0 0            │
│ 0 1 1 1 1 1 0 0            │
│ 0 1 1 1 1 1 0 0            │
│ 0 1 1 1 1 1 0 0            │
│ 0 0 0 0 0 0 0 0            │
└─────────────────────────────┘
         │
         │ Calculate mean position
         ▼
    Centroid (C)
    ┌─────┐
    │  C  │ ← Center of mass
    └─────┘
```

**Coordinate System:**
- **regionprops** returns: `(row, col, slice)` = `(y, x, z)`
- **CSV output**: `(x, y, z)` coordinates
- **Conversion**: `x = col, y = row, z = slice`

### Method 2: Manual Calculation (Fallback)

If `regionprops` fails, calculate manually:
```python
coords = np.where(mask)  # Get all (y, x, z) coordinates where mask = 1
centroid_x = np.mean(coords[1])  # Mean of x coordinates
centroid_y = np.mean(coords[0])  # Mean of y coordinates  
centroid_z = np.mean(coords[2])  # Mean of z coordinates
```

---

## Step 3: Distance Calculation Between ROIs

### Overview

The distance calculation uses **Euclidean Distance Transform** to find the **minimum surface-to-surface distance** between two ROIs.

### Process Flow

```
Reference ROI (GTV)          Target ROI (OAR)
     │                            │
     ├─ Binary Mask 1             ├─ Binary Mask 2
     │                            │
     └─ Calculate Distance        └─ Extract distances
         Transform                    at target voxels
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
         Distance Transform Map
         (Each voxel = distance to nearest reference boundary)
                    │
                    ▼
         Extract distances at all target voxels
                    │
                    ▼
         Calculate Statistics:
         • Minimum distance (R)
         • 5th percentile (R5)
         • Angles (Phi, Theta)
         • Overlap percentage
```

### Visual Representation: Distance Transform

#### Step 3.1: Reference ROI Boundary

```
Reference ROI (GTV) - 2D Slice View:
┌─────────────────────────────┐
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │
│ 0 0 1 1 1 1 1 1 1 1 0 0 0 0 │
│ 0 1 1 1 1 1 1 1 1 1 1 0 0 0 │
│ 0 1 1 1 1 1 1 1 1 1 1 0 0 0 │
│ 0 0 1 1 1 1 1 1 1 1 0 0 0 0 │
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │
└─────────────────────────────┘
    1 = Inside GTV
    0 = Outside GTV
```

#### Step 3.2: Distance Transform Calculation

**Distance Transform** calculates the distance from each voxel to the nearest boundary of the reference ROI.

```
Distance Transform Result (values in mm):
┌─────────────────────────────────────────┐
│ 2.1 1.4 0.7 0.0 0.0 0.0 0.7 1.4 2.1 ... │
│ 1.4 0.7 0.0 0.0 0.0 0.0 0.0 0.7 1.4 ... │
│ 0.7 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.7 ... │
│ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ... │ ← Inside GTV = 0
│ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ... │
│ 0.7 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.7 ... │
│ 1.4 0.7 0.0 0.0 0.0 0.0 0.0 0.7 1.4 ... │
│ 2.1 1.4 0.7 0.0 0.0 0.0 0.7 1.4 2.1 ... │
└─────────────────────────────────────────┘
    ↑
    Distance increases as you move away from GTV boundary
```

**Code Implementation:**
```python
# Calculate distance transform from reference ROI boundary
dist_ref = distance_transform_edt(~ref_cropped, sampling=aspect)
# ~ref_cropped = outside of reference ROI
# Result: distance from each voxel to nearest boundary
```

#### Step 3.3: Extract Distances at Target ROI Voxels

```
Target ROI (OAR) Overlay:
┌─────────────────────────────────────────┐
│         [Distance Transform Map]       │
│                                         │
│     ┌─────────┐                        │
│     │  OAR    │ ← Extract distances    │
│     │  Mask   │    at these voxels     │
│     └─────────┘                        │
│                                         │
└─────────────────────────────────────────┘

Extracted Distances:
[2.3, 2.8, 3.1, 2.9, 2.5, 3.2, 2.7, ...]
  ↑
  Minimum distance = 2.3 mm
  5th percentile = 2.4 mm
```

**Code Reference:**
```609:626:roi_distance_visualizer_enhanced_v3a.py
        # Calculate signed distance transform
        aspect = [ydim, xdim, zdim]
        
        dist_ref = distance_transform_edt(~ref_cropped, sampling=aspect).astype(np.float32)
        dist_not_ref = distance_transform_edt(ref_cropped, sampling=aspect).astype(np.float32)
        
        # Signed distance
        signed_dist = dist_not_ref - dist_ref
        
        # Get distances at target voxels
        target_voxels = target_cropped == 1
        distances = signed_dist[target_voxels]
        
        if len(distances) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        min_dist = np.min(distances)
        r5 = np.percentile(distances, 5)
```

---

## Step 4: Angle Calculation (Phi and Theta)

### Coordinate System

```
3D Space:
        Z (Superior)
        │
        │
        │
        │
        └─────── X (Anterior)
       ╱
      ╱
     Y (Left)
```

### Angle Definitions

**Phi (φ)**: Elevation angle (vertical angle)
- Range: -90° to +90°
- 0° = horizontal plane
- Positive = above, Negative = below

**Theta (θ)**: Azimuth angle (horizontal angle)
- Range: -180° to +180°
- 0° = along +Y axis
- Positive = counter-clockwise

### Calculation Process

```
1. Calculate Centroids:
   Reference Centroid: C_ref = (x_ref, y_ref, z_ref)
   Target Centroid:    C_tgt = (x_tgt, y_tgt, z_tgt)

2. Calculate Vector:
   Vector = C_tgt - C_ref
   = (Δx, Δy, Δz)

3. Scale by Voxel Spacing:
   Vector_scaled = (Δx × dx, Δy × dy, Δz × dz)
   Where dx, dy, dz are voxel dimensions in mm

4. Calculate Angles:
   r = ||Vector_scaled||  (magnitude)
   
   Phi (φ) = arcsin(Δz_scaled / r) × 180/π
   Theta (θ) = arctan2(Δx_scaled, -Δy_scaled) × 180/π
```

### Visual Representation

```
3D View:
                    Target ROI
                       ●
                      ╱│
                     ╱ │
                    ╱  │
                   ╱   │ Δz
                  ╱    │
                 ╱     │
                ╱      │
               ╱  φ    │
              ╱  ╱     │
             ╱  ╱      │
            ╱  ╱       │
           ╱  ╱        │
          ╱  ╱         │
         ╱  ╱          │
        ╱  ╱           │
       ╱  ╱            │
      ╱  ╱             │
     ╱  ╱              │
    ╱  ╱               │
   ╱  ╱                │
  ╱  ╱                 │
 ╱  ╱                  │
●───┴──────────────────┴───
Reference ROI          │
                       │
                    Projection
                    on XY plane
                       │
                       │
                       ●
                    (x, y, 0)
                    
Theta (θ) = angle in XY plane
```

**Code Reference:**
```628:648:roi_distance_visualizer_enhanced_v3a.py
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
```

---

## Step 5: Overlap Calculation

### Definition

**Overlap Percentage** = (Volume of Overlap / Volume of Target ROI) × 100%

### Visual Representation

```
Reference ROI (GTV)        Target ROI (OAR)
┌─────────────┐           ┌─────────────┐
│             │           │             │
│  ┌──────┐   │           │  ┌──────┐   │
│  │      │   │           │  │      │   │
│  │ OVER │   │  ← Overlap│  │ LAP  │   │
│  │      │   │     Region│  │      │   │
│  └──────┘   │           │  └──────┘   │
│             │           │             │
└─────────────┘           └─────────────┘

Overlap Mask = GTV ∩ OAR (intersection)
Overlap % = (Overlap Voxels / OAR Voxels) × 100%
```

**Code Reference:**
```650:656:roi_distance_visualizer_enhanced_v3a.py
        # Calculate overlap
        overlap_mask = (reference == 1) & (target == 1)
        overlap_volume = np.sum(target == 1)
        if overlap_volume > 0:
            overlap_percent = np.sum(overlap_mask) / overlap_volume
        else:
            overlap_percent = 0.0
```

---

## Step 6: Final Output Metrics

### CSV Output Columns

| Column | Description | Calculation |
|--------|-------------|-------------|
| **Reference ROI** | Name of reference ROI (typically GTV) | From RTSTRUCT |
| **Target ROI** | Name of target ROI | From RTSTRUCT |
| **Eucledian Distance (mm)** | Minimum surface-to-surface distance | `min(distances)` |
| **Phi (degrees)** | Elevation angle | `arcsin(Δz/r) × 180/π` |
| **Theta (degrees)** | Azimuth angle | `arctan2(Δx, -Δy) × 180/π` |
| **% of Target Overlap** | Overlap percentage | `(overlap_voxels / target_voxels) × 100` |
| **Eucledian Distance (mm) 5th Percentile** | 5th percentile distance | `percentile(distances, 5)` |

### Example Output

```
Reference ROI,Target ROI,Eucledian Distance (mm),Phi (degrees),Theta (degrees),% of Target Overlap,Eucledian Distance (mm) 5th Percentile
Eye_L,Eye_R,47.05,2.3,45.2,0.0,47.10
Eye_L,Lens_L,12.34,-5.1,30.5,0.0,12.45
```

---

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: RTSTRUCT File                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Convert RTSTRUCT to Binary Masks                       │
│  • Extract contour points                                       │
│  • Convert to voxel coordinates                                 │
│  • Create 3D binary masks (1 = inside, 0 = outside)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Calculate Centroids                                    │
│  • For each ROI mask:                                           │
│    - Label connected components                                 │
│    - Calculate centroid using regionprops                      │
│    - Output: (x, y, z) coordinates                               │
│  • Save to CT_centroid.csv                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Calculate Distances (for each ROI pair)                │
│                                                                 │
│  For Reference ROI (e.g., GTV):                                 │
│    • Calculate distance transform                               │
│      (distance from each voxel to ROI boundary)                 │
│                                                                 │
│  For Target ROI (e.g., OAR):                                   │
│    • Extract distances at all target voxels                     │
│    • Calculate:                                                 │
│      - Minimum distance (R)                                    │
│      - 5th percentile (R5)                                      │
│                                                                 │
│  Step 4: Calculate Angles                                       │
│    • Compute centroid-to-centroid vector                       │
│    • Calculate Phi (elevation) and Theta (azimuth)             │
│                                                                 │
│  Step 5: Calculate Overlap                                      │
│    • Find intersection of Reference and Target ROIs            │
│    • Calculate overlap percentage                              │
│                                                                 │
│  • Save to CT_distances.csv                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: CSV Files                            │
│  • CT_centroid.csv: ROI centroids                               │
│  • CT_distances.csv: Distance measurements                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Surface-to-Surface Distance vs. Center-to-Center Distance

**Surface-to-Surface Distance** (what we calculate):
- Distance from the **boundary** of Reference ROI to **boundary** of Target ROI
- More clinically relevant for radiation therapy planning
- Accounts for ROI shape and size

**Center-to-Center Distance** (not used):
- Distance between centroids
- Less accurate for irregular shapes
- Doesn't account for ROI boundaries

### 2. Signed Distance Transform

The code uses a **signed distance transform**:
- **Positive values**: Outside the reference ROI (distance from boundary)
- **Zero**: On the boundary
- **Negative values**: Inside the reference ROI

```
Signed Distance = dist_not_ref - dist_ref

Where:
  dist_not_ref = distance from inside to boundary
  dist_ref = distance from outside to boundary
```

### 3. Memory Optimization

For large ROIs, the code crops to a **bounding box**:
- Only processes voxels near both ROIs
- Reduces memory usage significantly
- Maintains accuracy

---

## Summary

1. **Centroid Identification**: Calculated as the mean position of all voxels in the ROI using `regionprops` or manual calculation.

2. **Distance Calculation**: Uses Euclidean Distance Transform to find minimum surface-to-surface distance between ROI boundaries.

3. **Angle Calculation**: Computes Phi (elevation) and Theta (azimuth) angles from centroid-to-centroid vector.

4. **Overlap Calculation**: Measures percentage of target ROI that overlaps with reference ROI.

5. **Statistics**: Reports minimum distance and 5th percentile distance for robustness.

All calculations account for **voxel spacing** (dx, dy, dz) to ensure measurements are in **millimeters** rather than voxel units.


LAST NOTE: Previous method with MATLAB vs PYTHON

MATLAB:
Uses: bwdistsc_CEC — a custom Euclidean distance transform
Documentation states: "BWDISTSC computes Euclidean distance transform"
Method: Custom optimized algorithm (forward-backward scan)
Accounts for voxel spacing: [ydim, xdim, zdim]

Python:
Uses: scipy.ndimage.distance_transform_edt — SciPy's Euclidean Distance Transform
Method: SciPy library implementation
Accounts for voxel spacing: sampling=[ydim, xdim, zdim]

Both use the same approach:
Compute Euclidean distance transform from the reference ROI
Account for anisotropic voxel spacing
Compute signed distance (positive = outside reference, negative = inside)
Find the minimum distance from reference to target


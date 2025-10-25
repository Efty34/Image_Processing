# Image Processing Laboratory Collection

This repository contains a comprehensive collection of image processing laboratory exercises, implementations, and projects covering fundamental to advanced image processing techniques. The codebase is organized into five main laboratory modules, each focusing on specific aspects of image processing, along with additional practice materials and a complete project implementation.

## Table of Contents

- [Image Processing Laboratory Collection](#image-processing-laboratory-collection)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Laboratory Modules](#laboratory-modules)
    - [Lab 1: Image Convolution and Filtering](#lab-1-image-convolution-and-filtering)
    - [Lab 1: Image Convolution and Filtering](#lab-1-image-convolution-and-filtering-1)
    - [Lab 2: Edge Detection](#lab-2-edge-detection)
    - [Lab 3: Histogram Processing](#lab-3-histogram-processing)
    - [Lab 4: Texture Analysis](#lab-4-texture-analysis)
    - [Lab 5: Frequency Domain Processing](#lab-5-frequency-domain-processing)
  - [Additional Components](#additional-components)
    - [my_lib/](#my_lib)
    - [practices/](#practices)
  - [Project: LumenTrace](#project-lumentrace)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dependencies](#dependencies)
  - [Related Repositories](#related-repositories)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Repository Structure

```
Image_Processing/
├── Lab_1/                  # Image Convolution and Filtering
├── Lab_2/                  # Edge Detection
├── Lab_3/                  # Histogram Processing
├── Lab_4/                  # Texture Analysis
├── Lab_5/                  # Frequency Domain Processing
├── my_lib/                 # Custom Image Processing Library
├── practices/              # Additional Practice Materials
├── project/                # LumenTrace Project
└── 19_Slides/              # Presentation Slides
```

## Laboratory Modules

### Lab 1: Image Convolution and Filtering

| Week   | Topics Covered                                        |
| ------ | ----------------------------------------------------- |
| Week 1 | Convolution operations                                |
| Week 2 | Segmentation applying edge detection and thresholding |
| Week 3 | Histogram equalization and matching                   |
| Week 4 | Frequency Domain Filtering                            |
| Week 5 | Region Descriptors                                    |

### Lab 1: Image Convolution and Filtering

**Topics Covered:**

- 2D Convolution operations
- Gaussian smoothing filters
- Laplacian of Gaussian (LoG) sharpening filters
- Manual kernel implementation
- Image padding and border handling
- Color space processing (RGB and HSV)

**Key Files:**

- [`Lab_1/lab_task/manual_convolution.py`](Lab_1/lab_task/manual_convolution.py) - Manual implementation of 2D convolution
- [`Lab_1/assignment/1.ipynb`](Lab_1/assignment/1.ipynb) - Applying smoothing and sharpening filters to grayscale images
- [`Lab_1/assignment/2.ipynb`](Lab_1/assignment/2.ipynb) - Convolution on RGB and HSV color channels
- [`my_lib/my_lib.py`](my_lib/my_lib.py) - Core convolution functions and kernel generators

**Key Functions:**

- `convolution2D()` - 2D convolution with padding
- `gaussian_kernel()` - Gaussian smoothing kernel generator
- `log_kernel()` - Laplacian of Gaussian sharpening kernel generator

### Lab 2: Edge Detection

**Topics Covered:**

- Gradient computation using Gaussian derivatives
- Edge magnitude calculation
- Double thresholding
- Hysteresis edge linking
- Canny edge detection implementation
- Edge visualization and post-processing

**Key Files:**

- [`Lab_2/lab_task/double_thresholding.ipynb`](Lab_2/lab_task/double_thresholding.ipynb) - Edge detection with thresholding
- [`Lab_2/assignment/1.ipynb`](Lab_2/assignment/1.ipynb) - Complete edge detection pipeline
- [`Lab_2/assignment/2.ipynb`](Lab_2/assignment/2.ipynb) - OpenCV Canny edge detection

**Key Functions:**

- `double_threshold()` - Double thresholding for edge detection
- `hysteresis_thresholding()` - Hysteresis edge linking
- `colorize_edges()` - Edge visualization

### Lab 3: Histogram Processing

**Topics Covered:**

- Histogram calculation and visualization
- Probability density functions (PDF)
- Cumulative distribution functions (CDF)
- Histogram equalization
- Histogram specification with custom distributions
- Color image histogram processing in BGR and HSV spaces

**Key Files:**

- [`Lab_3/lab_task/lab_task.ipynb`](Lab_3/lab_task/lab_task.ipynb) - Histogram equalization implementation
- [`Lab_3/assignment/1.ipynb`](Lab_3/assignment/1.ipynb) - Color image histogram processing
- [`Lab_3/assignment/2.ipynb`](Lab_3/assignment/2.ipynb) - Histogram specification with Erlang distribution

**Key Functions:**

- `histogram_equalization()` - Complete histogram equalization pipeline
- `histogram_specification()` - Custom histogram matching
- `erlang_pdf()` - Erlang distribution for specific histograms

### Lab 4: Texture Analysis

**Topics Covered:**

- Gray-Level Co-occurrence Matrix (GLCM) computation
- Texture feature extraction (energy, entropy, contrast, homogeneity)
- Region descriptors for shape analysis
- Object classification using texture features
- Image similarity metrics

**Key Files:**

- [`Lab_4/assignment/glcm_lib.py`](Lab_4/assignment/glcm_lib.py) - GLCM implementation library
- [`Lab_4/assignment/assignment_1.ipynb`](Lab_4/assignment/assignment_1.ipynb) - GLCM analysis of different textures
- [`Lab_4/assignment/assignment_2.ipynb`](Lab_4/assignment/assignment_2.ipynb) - Patch-based texture analysis
- [`Lab_4/lab_task/lab4.py`](Lab_4/lab_task/lab4.py) - Shape descriptors and similarity metrics

**Key Functions:**

- `manual_horizontal_glcm_fn()` - Horizontal GLCM computation
- `manual_vertical_glcm_fn()` - Vertical GLCM computation
- `manual_diagonal_glcm_fn()` - Diagonal GLCM computation
- `energy()`, `entropy()`, `contrast()`, `homogenity()` - Texture feature extractors

### Lab 5: Frequency Domain Processing

**Topics Covered:**

- Fourier Transform and spectrum analysis
- Frequency domain filtering
- Notch filtering for periodic noise removal
- Phase and magnitude analysis
- Image reconstruction from filtered frequency components

**Key Files:**

- [`Lab_5/labtask/labtask.py`](Lab_5/labtask/labtask.py) - Notch filter implementation
- [`Lab_5/assets/fourier_base.py`](Lab_5/assets/fourier_base.py) - Basic Fourier transform operations

**Key Functions:**

- `notch_filter()` - Creates notch filters for periodic noise removal

## Additional Components

### my_lib/

Custom image processing library containing core functions used across all laboratories:

- [`my_lib/my_lib.py`](my_lib/my_lib.py) - Core image processing functions

### practices/

Additional practice materials and implementations:

- [`practices/image_lib.py`](practices/image_lib.py) - Additional image processing utilities
- Various Jupyter notebooks for edge detection, histogram equalization, and frequency domain processing

## Project: LumenTrace

**LumenTrace** is a comprehensive project for determining the time a historical photograph was taken by analyzing shadow geometry. The project implements a complete image processing pipeline to restore photographs, isolate shadows, and perform geometric analysis to calculate sun position.

**Project Features:**

- Shadow detection using color space analysis (HSV and LAB)
- Object segmentation using K-means clustering
- Shadow-object pairing based on proximity
- Sun elevation angle calculation from object-shadow ratios
- Time estimation using solar position algorithms
- 3D visualization of the sun-object-shadow relationship

**Key Files:**

- [`project/LumenTrace.ipynb`](https://github.com/Efty34/LumenTrace) - Complete project implementation
- [`project/ProjectProposal/2007052_LumenTrace.pdf`](project/ProjectProposal/2007052_LumenTrace.pdf) - Project proposal and methodology

**Key Functions:**

- `detect_shadows()` - Multi-method shadow detection
- `kmeans_object_segmentation()` - Object-background separation
- `calculate_sun_elevation()` - Sun angle calculation
- `estimate_time_from_elevation()` - Time estimation from solar position

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Image_Processing
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Each laboratory module can be run independently:

1. **For Lab 1 (Convolution and Filtering):**

   ```bash
   cd Lab_1/assignment
   jupyter notebook 1.ipynb
   ```

2. **For Lab 2 (Edge Detection):**

   ```bash
   cd Lab_2/assignment
   jupyter notebook 1.ipynb
   ```

3. **For Lab 3 (Histogram Processing):**

   ```bash
   cd Lab_3/assignment
   jupyter notebook 1.ipynb
   ```

4. **For Lab 4 (Texture Analysis):**

   ```bash
   cd Lab_4/assignment
   jupyter notebook assignment_1.ipynb
   ```

5. **For Lab 5 (Frequency Domain):**

   ```bash
   cd Lab_5/labtask
   python labtask.py
   ```

6. **For LumenTrace Project:**
   ```bash
   cd project
   jupyter notebook LumenTrace.ipynb
   ```

## Dependencies

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Jupyter Notebook

## Related Repositories

- [Image-Processing-Lab by Turzo](https://github.com/vallagenakisu/Image-Processing-Lab) - Turzo is from IG1
- [IPCV_Labs by Faysal](https://github.com/Faysal-star/IPCV_Labs) - Faysal is from IG2
- [ImageCodes](https://github.com/abusaeed2433/ImageCodes) - Abu Saeed bhai is from 19
- [LumenTrace](https://github.com/Efty34/LumenTrace) - My Project of this lab.

## License

This repository is for educational purposes as part of the Image Processing course.

## Acknowledgments

This work is based on the curriculum and assignments from the Image Processing course, implementing fundamental image processing techniques as described in "Digital Image Processing" by Gonzalez and Woods.

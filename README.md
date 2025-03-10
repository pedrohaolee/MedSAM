# Ultrasound-to-Underwater: Cross-Domain Segmentation with MedSAM

This repository explores how **MedSAM (Medical Segment Anything Model)** can be adapted for **underwater sonar imaging**, specifically targeting fish segmentation using the **Caltech Fish Counting (CFC) dataset**. By leveraging the similarities between **ultrasound** and **sonar** images—particularly their noise characteristics and diffused boundaries—this project aims to demonstrate MedSAM’s potential for robust cross-domain generalization.

---

## Repository Contents

- **`project_ultrasound_to_underwater.ipynb`**  
  An interactive Jupyter Notebook that walks through the main pipeline, including data loading, preprocessing, segmentation via MedSAM, and preliminary results analysis.

- **`histogram.py`**  
  Contains functions to compute and compare **grayscale histograms**. Metrics such as correlation, Chi-Square distance, intersection, and Bhattacharyya distance quantify the similarity between ultrasound and sonar images.

- **`nps.py`**  
  Provides utilities to compute the **Noise Power Spectrum (NPS)**, a measure of how noise energy is distributed across spatial frequencies. Helpful for examining the parallel noise profiles in ultrasound and sonar data.

- **`speckle.py`**  
  Implements **speckle noise analysis**, including speckle contrast (standard deviation over mean) and Shannon entropy calculations. These metrics illustrate how sonar images exhibit noise traits akin to ultrasound, guiding potential noise-reduction strategies.

- **`dsa.py`**  
  Demonstrates a **Digital Subtraction Angiography (DSA)**-inspired approach for sonar imaging. It computes the absolute difference between frames (or images) to highlight dynamic regions—such as moving fish—while reducing background clutter.

---

## Quick Start

1. **Clone** this repository:
   ```bash
   git clone https://github.com/<username>/UltrasoundToUnderwater.git
   cd UltrasoundToUnderwater
   ```

2. **Install** dependencies (preferably in a virtual environment or conda environment):
   ```bash
   pip install -r requirements.txt
   ```
   - Make sure you have Python 3.7+ and a suitable version of PyTorch if you plan to integrate with MedSAM.

3. **Run the notebook** for a step-by-step demo:
   ```bash
   jupyter notebook project_ultrasound_to_underwater.ipynb
   ```
   This notebook showcases MedSAM segmentation results on example Caltech fish sonar images.

4. **Scripts**:
   - **`histogram.py`**:  
     ```bash
     python histogram.py
     ```
     Analyze histograms and their similarities between images.  
   - **`nps.py`**:  
     ```bash
     python nps.py
     ```
     Visualizes and compares noise power spectra between images.  
   - **`speckle.py`**:  
     ```bash
     python speckle.py
     ```
     Prints out speckle contrast and image entropy.  
   - **`dsa.py`**:  
     ```bash
     python dsa.py
     ```
     Displays side-by-side subtraction of sonar frames to isolate fish objects.

---

## Project Goals

1. **Cross-Domain Segmentation**: Demonstrate how MedSAM, despite being trained on medical data, can be repurposed for sonar imagery by exploiting parallels with ultrasound.
2. **Noise Characterization**: Show the resemblance in noise power spectrum and speckle properties between ultrasound and sonar images.
3. **Enhancement Techniques**: Evaluate DSA-inspired subtraction as a preprocessing step to improve fish segmentation accuracy.

---

## Potential Extensions

- **Integration with MedSAM**: Further refine prompts and bounding boxes for sonar images to address over-segmentation issues.
- **Extended Datasets**: Explore additional sonar datasets, or incorporate more diverse ultrasound samples to bolster cross-domain evidence.
- **Noise Reduction**: Investigate advanced filters or deep-learning-based denoising methods to complement NPS and speckle analyses.

---

## References

- **MedSAM**  
  Ma, J., He, Y., Li, F., Han, L., You, C., & Wang, B. (2024). _Segment Anything in Medical Images_. *Nature Communications, 15*(1), 654.

- **Caltech Fish Counting Dataset**  
  Kay, J., Kulits, P., Stathatos, S., Deng, S., Young, E., Beery, S., ... & Perona, P. (2022). *The Caltech Fish Counting dataset: a benchmark for multiple-object tracking and counting*. In _European Conference on Computer Vision_ (pp. 290-311). Cham: Springer.

# MedSAM for Underwater Sonar Image Segmentation

This repository adapts the **MedSAM (Medical Segment Anything Model)** to the domain of underwater sonar imaging, specifically targeting fish segmentation using the Caltech Fish Counting (CFC) dataset.

## Overview
MedSAM, originally developed for medical image segmentation, is applied here to underwater sonar imagery, leveraging its capabilities in handling challenging imaging conditions such as high noise, diffuse boundaries, and complex textures typical of both ultrasound and sonar data.

## Installation

1. Create and activate a conda environment:
   ```bash
   conda create -n medsam python=3.10 -y
   conda activate medsam
   ```

2. Install PyTorch:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. Clone this repository:
   ```bash
   git clone https://github.com/bowang-lab/MedSAM
   cd MedSAM
   pip install -e .
   ```

## Data Preparation

This project uses the Caltech Fish Counting Dataset (CFC), which contains sonar video clips annotated for fish detection and tracking. Specifically, data from the Kenai River (KL subset) was used for training and validation, with testing on different subsets (KR, KC, NU, EL) to evaluate model generalization across diverse sonar environments.

- Dataset details: [Caltech Fish Counting Dataset](https://github.com/visipedia/caltech-fish-counting)

## Usage

### Model Inference

Use the provided script to run segmentation on your sonar images:

```bash
python MedSAM_Inference.py -i input_image.png -o output_path --box x1 y1 x2 y2
```

### Interactive Notebooks

A Colab notebook tutorial demonstrating the full pipeline for sonar image segmentation using MedSAM is provided:

- [Run on Google Colab](https://colab.research.google.com)

### GUI Application

Run the GUI tool for interactive segmentation:

```bash
pip install PyQt5
python gui.py
```

Load sonar images and draw bounding boxes to segment targets.

## Results

MedSAM shows promising segmentation results in sonar imagery, surpassing baseline methods in both qualitative and quantitative assessments.

## Future Work

- Enhance model robustness and generalization across varied underwater environments.
- Investigate real-time performance optimization.

## Acknowledgments

- Thanks to the Caltech Fish Counting Dataset creators for making their dataset publicly available.
- Thanks to Meta AI for the Segment Anything model and codebase.

## Citation

If you find this adaptation useful, please cite the original MedSAM paper:

```bibtex
@article{MedSAM2024,
  author = {Jun Ma et al.},
  title = {Segment Anything in Medical Images},
  journal = {Nature Communications},
  volume = {15},
  pages = {654},
  year = {2024}
}
```


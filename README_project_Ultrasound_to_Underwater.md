# MedSAM for Underwater Sonar Image Segmentation

## Project Overview

This project adapts the **Medical Segment Anything Model (MedSAM)**—originally designed for medical image segmentation—to the domain of underwater sonar imaging. By leveraging the structural similarities between ultrasound and sonar imaging, such as low signal-to-noise ratios, diffuse boundaries, and complex textures, this project aims to effectively segment sonar images, specifically for fish tracking and ecological monitoring tasks.

### Objectives
- Adapt and fine-tune MedSAM for sonar image segmentation tasks.
- Evaluate segmentation performance using the Caltech Fish Counting Dataset (CFC).
- Provide quantitative and qualitative analyses comparing MedSAM to baseline models.

## MedSAM Background

MedSAM is a deep learning segmentation model, trained on a large-scale dataset of 1.6 million medical image-mask pairs across various medical imaging modalities. The model has demonstrated exceptional performance in medical contexts, notably with challenging ultrasound imaging data.

Key components include:
- **Image Encoder:** Extracts features from the input images.
- **Prompt Encoder:** Integrates interactive or automated prompts to guide segmentation.
- **Mask Decoder:** Produces precise segmentation masks from encoded features.

## Dataset

The Caltech Fish Counting Dataset (CFC) was utilized for this project. This dataset includes over 1,500 sonar video clips sourced from five diverse environmental locations:
- Kenai River, Alaska (KL, KR, KC)
- Nushagak River, Alaska (NU)
- Elwha River, Washington (EL)

The data presents multiple challenges such as:
- Low signal-to-noise ratios.
- Visually indistinct targets.
- Complex and variable backgrounds.

Dataset annotations include over half a million bounding boxes, covering thousands of fish across numerous video frames, enabling detailed performance assessments.

## Implementation

### Setup and Installation

To replicate the environment:
```bash
conda create -n medsam python=3.10 -y
conda activate medsam

# Install PyTorch
pip install torch torchvision torchaudio

# Clone MedSAM repository
git clone https://github.com/bowang-lab/MedSAM
cd MedSAM
pip install -e .
```

Download and place the pretrained MedSAM model checkpoint in `work_dir/MedSAM/medsam_vit_b`.

### Adapting MedSAM
- Sonar images were preprocessed using normalization and augmentation techniques suitable for underwater imagery.
- The MedSAM model was fine-tuned specifically for sonar-based segmentation tasks using data from the KL subset for training and validation.

## Evaluation Metrics
Performance was evaluated using:
- **Dice Similarity Coefficient (DSC)**: Measures segmentation accuracy.
- **Mean Absolute Error (MAE)**: Assesses precision in segmentation.
- Tracking-specific metrics such as MOTA (Multiple Object Tracking Accuracy) and HOTA (Higher-Order Tracking Accuracy).

## Results

### Quantitative Analysis
MedSAM demonstrated strong segmentation performance, outperforming traditional baseline methods in handling noisy and diffuse sonar imagery.

### Qualitative Analysis
Visual comparisons highlighted MedSAM’s robustness in segmenting fish accurately, despite challenging environmental conditions.

## Key Findings
- MedSAM effectively leverages structural similarities between medical ultrasound and sonar images, demonstrating significant adaptability and robustness.
- Superior handling of low signal-to-noise ratios and diffuse boundaries compared to baseline methods.

## Challenges
- Performance varied with image quality and complexity, particularly with small or stationary fish against highly noisy backgrounds.
- High noise levels in some dataset subsets impacted segmentation accuracy.

## Future Directions
- Further optimization for real-time segmentation and tracking.
- Extending the model to other marine species and varying environmental contexts.

## Acknowledgements
Special thanks to dataset curators and Meta AI for their foundational contributions to the Segment Anything framework.


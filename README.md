# Pokémon Image Processing Project

![Sample Processing Pipeline](processed_pokemon/collages/collage_pikachu.png)

A complete digital image processing pipeline for Pokémon images, implementing key computer vision techniques including color adjustment, edge detection, and feature extraction.

## 📌 Project Overview

This project demonstrates:
- **Image preprocessing** (brightness/contrast adjustment, denoising)
- **Feature extraction** (color histograms, edge detection)
- **Classification preparation** (feature engineering for machine learning)
- **Batch processing** of Pokémon images from the [Pokémon Images Dataset](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)

## 🛠️ Technical Implementation

### Core Techniques
| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| Color Adjustment | `cv2.convertScaleAbs()` | Enhance image visibility |
| Gaussian Blur | `cv2.GaussianBlur()` | Noise reduction |
| Edge Detection | `cv2.Canny()` | Contour extraction |
| Denoising | `cv2.fastNlMeansDenoisingColored()` | Image cleanup |
| Feature Extraction | Color histograms + Edge density | ML preparation |

### Extracted Features
1. **3D Color Histogram** (32 bins per RGB channel)
2. **Edge Density** (% of edge pixels)
3. **Aspect Ratio** (width/height)

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Kaggle environment (for dataset access)

### Installation
```bash
pip install -r requirements.txt

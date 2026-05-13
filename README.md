# LSAN: Lightweight Encrypted JPEG Image Retrieval Model Based on Self-Attention Networks

A comprehensive Python implementation of a lightweight encrypted JPEG image retrieval system that leverages self-attention networks for efficient image matching and retrieval from encrypted datasets.

## 📋 Overview

LSAN provides a complete pipeline for encrypted image handling and retrieval, combining three core functionalities:

1. **Image Encryption**: Secure JPEG image encryption with support for RGB color space
2. **Feature Extraction**: Extract discriminative features using DCT and self-attention mechanisms
3. **Image Retrieval**: Retrieve and rank images based on extracted features

## 📁 Project Structure

```
LSAN/
├── README.md                          # Project documentation
├── Encryption/                        # Image encryption module
│   ├── cipherimageRgbGenerate.py     # RGB cipher image generation
│   ├── encryption_utils.py           # Encryption utility functions
│   ├── owner_encryption.py           # Owner-side encryption logic
│   ├── JPEG/                         # JPEG-related encryption components
│   ├── transform_stage4change.mat     # Transformation matrix for encryption
│   └── __pycache__/                  # Python cache
│
├── Feature_extraction/                # Feature extraction module
│   ├── All_feature.py                # Unified feature extraction interface
│   ├── dct_histogram.py              # DCT-based histogram feature extraction
│   └── __pycache__/                  # Python cache
│
├── Retrieval_Model/                   # Image retrieval and ranking module
│   ├── model.py                      # Self-attention network model architecture
│   ├── train.py                      # Model training script
│   ├── test.py                       # Model testing and evaluation script
│   ├── LossFunction.py               # Custom loss functions
│   ├── model_evaluate.py             # Model evaluation utilities
│   ├── re_ranking.py                 # Re-ranking algorithm for retrieval results
│   ├── utils.py                      # General utility functions
│   └── __pycache__/                  # Python cache
│
└── data/                              # Data storage directory
    ├── plainimages/                  # Original plaintext images folder
    ├── cipherimages/                 # Encrypted images folder
    ├── JPEGBitStream/                # JPEG bitstream data
    ├── Label/                        # Image labels for classification
    ├── plainimages.npy               # Preprocessed plaintext images (numpy array)
    ├── img_size.npy                  # Image size information (numpy array)
    └── transform_stage4change.mat     # Reference transformation matrix
```

## 🔧 Module Descriptions

### Encryption Module (`Encryption/`)
Handles secure encryption of JPEG images:
- **cipherimageRgbGenerate.py**: Generates encrypted RGB images from plaintext
- **encryption_utils.py**: Core encryption utilities and helper functions
- **owner_encryption.py**: Owner-side encryption orchestration and management
- **JPEG/**: JPEG format-specific encryption operations

### Feature Extraction Module (`Feature_extraction/`)
Extracts discriminative features from encrypted images:
- **All_feature.py**: Unified interface for multi-method feature extraction
- **dct_histogram.py**: DCT (Discrete Cosine Transform) based histogram features

### Retrieval Model Module (`Retrieval_Model/`)
Self-attention network for image matching and ranking:
- **model.py**: Self-attention network architecture
- **train.py**: Training loop and optimization
- **test.py**: Testing, inference, and performance evaluation
- **LossFunction.py**: Customized loss functions for optimization
- **model_evaluate.py**: Evaluation metrics and performance assessment
- **re_ranking.py**: Re-ranking algorithm to improve retrieval results
- **utils.py**: Data loading, preprocessing, and utility functions

### Data Directory (`data/`)
Data storage and preprocessing:
- **plainimages/**: Original images before encryption (input directory)
- **cipherimages/**: Encrypted image outputs
- **JPEGBitStream/**: JPEG bitstream representations
- **Label/**: Ground truth labels for evaluation
- **plainimages.npy**: Preprocessed image array
- **img_size.npy**: Image dimensions and metadata

## 🚀 Usage Guide

### Prerequisites
- Python 3.7+
- Required dependencies:
  - NumPy
  - PyTorch (for neural network operations)
  - PIL/Pillow (for image processing)
  - SciPy (for DCT and other algorithms)

### Quick Start

1. **Image Encryption**
   ```python
   from Encryption.owner_encryption import EncryptionManager
   
   # Initialize encryption manager
   encryptor = EncryptionManager()
   # Process images - place plaintext images in data/plainimages/
   encryptor.encrypt_images()
   # Encrypted images saved to data/cipherimages/
   ```

2. **Feature Extraction**
   ```python
   from Feature_extraction.All_feature import FeatureExtractor
   
   # Extract features from encrypted images
   extractor = FeatureExtractor()
   features = extractor.extract_all_features()
   ```

3. **Model Training**
   ```python
   from Retrieval_Model.train import train_model
   
   # Train the self-attention based retrieval model
   train_model(epochs=100, batch_size=32)
   ```

4. **Image Retrieval**
   ```python
   from Retrieval_Model.test import retrieve_images
   
   # Retrieve similar images
   results = retrieve_images(query_image, top_k=10)
   ```

## 💾 Data Flow

```
plaintext images
       ↓
   Encryption
       ↓
encrypted images → Feature Extraction → feature vectors
       ↓
  Retrieval Model (Self-Attention)
       ↓
  ranked results → Re-ranking → final results
```

## 📊 Key Features

- ✅ **Lightweight Design**: Optimized for efficiency with minimal computational overhead
- ✅ **Encryption Support**: Secure JPEG image encryption with integrity preservation
- ✅ **Self-Attention Networks**: Modern attention mechanisms for improved retrieval accuracy
- ✅ **Multi-feature Extraction**: DCT-based and histogram-based feature extraction
- ✅ **Re-ranking Algorithm**: Iterative re-ranking to refine retrieval results

## 🔐 Security Considerations

- Images are encrypted before feature extraction
- Features are computed on encrypted data to maintain privacy
- Owner-side encryption ensures only authorized parties can decrypt original images

## 📝 Important Notes

⚠️ **Data Organization**:
- All plaintext images should be placed in `/LSAN/data/plainimages/` before processing
- Ensure consistent image formats (JPEG recommended)
- Image sizes should be uniform or resized accordingly

## 🛠️ Configuration

Key parameters can be adjusted in respective modules:
- Encryption parameters in `Encryption/encryption_utils.py`
- Feature extraction parameters in `Feature_extraction/dct_histogram.py`
- Model hyperparameters in `Retrieval_Model/model.py`

## 📚 References

This implementation is based on research combining:
- Lightweight encrypted image processing
- Self-attention mechanisms for visual understanding
- JPEG domain operations for efficiency

## 📄 License

See repository for license information.

## 👤 Author

**FrankZanyar**

---

**Last Updated**: 2026-05-13

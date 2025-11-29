# Esercitazione 11: U-Net per Segmentazione di Materia Cerebrale

## Indice

- [Introduzione](#introduzione)
- [Teoria](#teoria)
  - [U-Net Architecture](#unet-architecture)
  - [Segmentazione Semantica](#segmentazione-semantica)
  - [DICE Coefficient](#dice-coefficient)
  - [Transfer Learning](#transfer-learning)
- [Dataset](#dataset)
- [Implementazione](#implementazione)
- [Utilizzo](#utilizzo)
- [Risultati Attesi](#risultati-attesi)
- [Bibliografia](#bibliografia)

---

## Introduzione

Questa esercitazione implementa **U-Net** (Ronneberger et al. 2015), un'architettura CNN encoder-decoder per la segmentazione semantica di immagini mediche. Obiettivo: segmentare automaticamente la materia cerebrale (gray + white matter) da immagini MRI T1.

**Strategia a due fasi con Transfer Learning**:
1. **Task 1 (Facile)**: Segmentazione skull (intero cranio) → DICE >0.99
2. **Task 2 (Difficile)**: Segmentazione brain matter usando pretrained weights → DICE >0.85

**Dataset**: BrainWeb simulator (1810 slice MRI T1 da 20 cervelli normali, 256×256 px)

---

## Teoria

### U-Net Architecture

**U-Net** è un'architettura fully convolutional per segmentazione, caratterizzata da:

**Struttura**:
```
INPUT (256×256×1)
    ↓
ENCODER (Contracting Path)
    Block 1: [Conv3-32, Conv3-32] → Pool  (256 → 128)
    Block 2: [Conv3-64, Conv3-64] → Pool  (128 → 64)
    Block 3: [Conv3-128, Conv3-128] → Pool  (64 → 32)
    Block 4: [Conv3-256, Conv3-256]      (32 → 32, bottleneck)
    ↓
DECODER (Expanding Path)
    Block 3': UpConv + Skip + [Conv3-128, Conv3-128]  (32 → 64)
    Block 2': UpConv + Skip + [Conv3-64, Conv3-64]    (64 → 128)
    Block 1': UpConv + Skip + [Conv3-32, Conv3-32]    (128 → 256)
    ↓
OUTPUT: Conv1-1 → Sigmoid  (256×256×1, binary mask)
```

**Componenti chiave**:

1. **Encoder (Contracting Path)**:
   - Serie di conv blocks con MaxPooling
   - Cattura contesto (receptive field crescente)
   - Riduce risoluzione spaziale: 256 → 128 → 64 → 32

2. **Bottleneck**:
   - Layer più profondo (minima risoluzione, massimo depth)
   - Dropout per regolarizzazione

3. **Decoder (Expanding Path)**:
   - UpConvolution (Conv2DTranspose) per upsampling
   - **Skip Connections**: Concatena feature da encoder corrispondente
   - Ripristina risoluzione spaziale: 32 → 64 → 128 → 256

4. **Skip Connections**:
   - Essenziali per preservare dettagli spaziali
   - Combinano feature high-level (semantica) + low-level (dettagli)
   - Mitigano vanishing gradient

**Vantaggi U-Net**:
- Output stessa risoluzione input (segmentazione pixel-perfect)
- Skip connections → dettagli fini preservati
- Efficace con pochi dati (data augmentation)
- Fully convolutional → input size flessibile

### Segmentazione Semantica

**Segmentazione** assegna una label ad ogni pixel:
- **Semantic**: Ogni pixel ha classe (background, brain, skull, etc.)
- **Instance**: Distingue istanze multiple stessa classe
- **Panoptic**: Combina semantic + instance

**Loss Functions**:

1. **Binary Cross-Entropy** (pixel-wise):
```
BCE = -Σ [y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]
```

2. **DICE Loss**:
```
DICE_loss = 1 - DICE_coefficient
```

3. **Combined Loss** (usato in questo progetto):
```
L = α * BCE + (1-α) * DICE_loss
```
- Combina pixel-wise accuracy (BCE) e overlap globale (DICE)
- α=0.5 tipicamente

### DICE Coefficient

**DICE** (Sørensen-Dice coefficient) misura overlap tra predizione e ground truth:

```
DICE = 2 * |A ∩ B| / (|A| + |B|)
     = 2 * TP / (2*TP + FP + FN)
```

- Range: [0, 1], dove 1 = overlap perfetto
- Equivalente a F1-score
- Più robusto di accuracy per classi sbilanciate

**Relazione con IoU** (Intersection over Union / Jaccard):
```
DICE = 2*IoU / (1 + IoU)
IoU = DICE / (2 - DICE)
```

**Perché DICE > Accuracy per segmentazione?**
- Background domina (es. 95% pixel) → accuracy bias
- DICE pesa di più la regione di interesse
- DICE sensibile a FP e FN (non solo background corretto)

### Transfer Learning

**Strategia**:
1. Pretrain su **task facile** (skull segmentation):
   - Background vs skull (alto contrasto)
   - DICE >0.99 facilmente raggiungibile

2. Fine-tune su **task difficile** (brain matter):
   - Gray matter vs white matter (basso contrasto)
   - Riusa weights encoder (feature generiche)
   - Faster convergence, better performance

**Benefici**:
- Convergenza più veloce (meno epochs)
- Performance migliore (soprattutto con dati limitati)
- Regolarizzazione implicita (evita overfitting)

**Learning rate**: Ridotto per fine-tuning (0.0005 vs 0.001)

---

## Dataset

### BrainWeb Simulator

Fonte: https://brainweb.bic.mni.mcgill.ca/anatomic_normal_20.html

**Caratteristiche**:
- 20 cervelli normali
- Simulatore MRI: genera T1/T2/PD da atlante anatomico
- 12 classi tissutali (CSF, gray matter, white matter, fat, skull, etc.)
- Risoluzione originale: 362×434×362
- Interpolato a: 256×256×181

**Dataset Processato**:
```
data/
├── MR/              # 1810 immagini T1 (256×256, PNG)
├── GRAY_MASK_B/     # 1810 maschere skull (task 1, binarie)
└── GRAY_MASK_C/     # 1810 maschere brain matter (task 2, binarie)
```

- **GRAY_MASK_B**: Background (0) vs Skull (1) - tutto tranne background
- **GRAY_MASK_C**: Background (0) vs Brain Matter (1) - solo gray+white matter

**Split**: 70% train (1267), 15% val (272), 15% test (271)

---

## Implementazione

### Struttura Progetto

```
es_11__25_05_2022_unet_segmentazione/
├── README.md
├── requirements.txt
├── data/
│   ├── MR/                    # 1810 T1 images
│   ├── GRAY_MASK_B/           # Skull masks
│   └── GRAY_MASK_C/           # Brain matter masks
├── results/
│   ├── skull/                 # Task 1 results
│   │   ├── training_history.png
│   │   ├── segmentation_results.png
│   │   ├── best_model.h5
│   │   └── metrics.npz
│   └── brain/                 # Task 2 results
│       ├── training_history.png
│       ├── segmentation_results.png
│       ├── best_model.h5
│       └── metrics.npz
├── docs/
│   └── Esercitazione_10_18_05_2022.pdf
└── src/
    ├── __init__.py
    ├── utils.py                      # U-Net + utilities (~600 righe)
    ├── train_skull_segmentation.py  # Task 1 (~350 righe)
    └── train_brain_segmentation.py  # Task 2 con transfer learning (~400 righe)
```

### Pipeline

**Task 1: Skull Segmentation** (easy task for pretraining)

```bash
cd es_11__25_05_2022_unet_segmentazione/src

python train_skull_segmentation.py \
    --data_dir ../data \
    --output_dir ../results/skull \
    --epochs 30 \
    --batch_size 8 \
    --image_size 256
```

**Task 2: Brain Matter Segmentation** (hard task with transfer learning)

```bash
python train_brain_segmentation.py \
    --data_dir ../data \
    --pretrained_model ../results/skull/best_model.h5 \
    --output_dir ../results/brain \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 0.0005
```

---

## Utilizzo

### Training Skull Segmentation

```bash
# Standard training
python train_skull_segmentation.py --data_dir ../data --epochs 30

# Quick test (first 100 samples, smaller images)
python train_skull_segmentation.py \
    --data_dir ../data \
    --max_samples 100 \
    --image_size 128 \
    --epochs 10

# Custom architecture
python train_skull_segmentation.py \
    --encoder_depth 3 \
    --num_first_filters 64 \
    --batch_size 16
```

### Training Brain Matter Segmentation

```bash
# With transfer learning (recommended)
python train_brain_segmentation.py \
    --data_dir ../data \
    --pretrained_model ../results/skull/best_model.h5 \
    --epochs 50

# From scratch (no transfer learning)
python train_brain_segmentation.py \
    --data_dir ../data \
    --epochs 100 \
    --learning_rate 0.001

# Custom loss function
python train_brain_segmentation.py \
    --pretrained_model ../results/skull/best_model.h5 \
    --loss dice \
    --epochs 60
```

### Python Script Personalizzato

```python
import numpy as np
from src.utils import (
    load_dataset,
    build_unet,
    compile_unet,
    calculate_segmentation_metrics
)
from sklearn.model_selection import train_test_split

# Load data
X, y = load_dataset(
    image_dir='data/MR',
    mask_dir='data/GRAY_MASK_C',
    target_size=(256, 256),
    verbose=True
)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build U-Net
model = build_unet(input_shape=(256, 256, 1), num_classes=1)
model = compile_unet(model, learning_rate=0.001, loss='combined')

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=8
)

# Evaluate
y_pred = model.predict(X_test)
metrics = calculate_segmentation_metrics(y_test, y_pred)
print(f"Test DICE: {metrics['dice']:.4f}")
```

---

## Risultati Attesi

### Task 1: Skull Segmentation

**Training** (~30 epochs, ~10 min su GPU):
- DICE: >0.99
- IoU: >0.98
- Pixel Accuracy: >0.99

**Test Set**:
- DICE: >0.99
- IoU: >0.98
- Pixel Accuracy: >0.99

**Interpretazione**: Task molto facile (alto contrasto skull vs background). Rete converge rapidamente.

### Task 2: Brain Matter Segmentation

**Senza Transfer Learning** (~100 epochs):
- Test DICE: 0.82-0.84
- Test IoU: 0.70-0.73
- Test Accuracy: >0.95

**Con Transfer Learning** (~50 epochs):
- Test DICE: 0.85-0.88
- Test IoU: 0.74-0.78
- Test Accuracy: >0.97

**Beneficio Transfer Learning**: +3-4% DICE, convergenza 2x più veloce

**Note**:
- Accuracy alta (>95%) ma DICE moderato (0.85) → background domina accuracy
- DICE metrica più significativa per valutare segmentazione
- Qualità visiva può sembrare limitata anche con DICE 0.85 (errori concentrati su bordi)

### Esempi Visuali

**Output generati**:
1. `training_history.png`: Loss + DICE curves (train/val)
2. `segmentation_results.png`: Griglia 6 esempi (image, GT, prediction, overlay)
3. `best_model.h5`: Model checkpoint (best validation DICE)
4. `metrics.npz`: Metriche salvate (train/val/test)

---

## Bibliografia

### U-Net e Segmentazione

1. **Ronneberger, O., Fischer, P., & Brox, T. (2015)**
   *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
   **MICCAI 2015**. DOI: 10.1007/978-3-319-24574-4_28
   (Architettura originale U-Net)

2. **Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016)**
   *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.*
   **MICCAI 2016**. DOI: 10.1007/978-3-319-46723-8_49
   (Estensione 3D)

3. **Milletari, F., Navab, N., & Ahmadi, S. A. (2016)**
   *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.*
   **3DV 2016**. DOI: 10.1109/3DV.2016.79
   (DICE loss per segmentazione)

4. **Isensee, F., Jaeger, P. F., Kohl, S. A., et al. (2021)**
   *nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation.*
   **Nature Methods**, 18, 203-211. DOI: 10.1038/s41592-020-01008-z
   (U-Net state-of-the-art con auto-configuration)

### Transfer Learning

5. **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014)**
   *How transferable are features in deep neural networks?*
   **NIPS 2014**, pp. 3320-3328.
   (Analisi transferability CNN features)

6. **Tajbakhsh, N., Shin, J. Y., Gurudu, S. R., et al. (2016)**
   *Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?*
   **IEEE TMI**, 35(5), 1299-1312. DOI: 10.1109/TMI.2016.2535302
   (Transfer learning medical imaging)

### Brain MRI Segmentation

7. **Kamnitsas, K., Ledig, C., Newcombe, V. F., et al. (2017)**
   *Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation.*
   **Medical Image Analysis**, 36, 61-78. DOI: 10.1016/j.media.2016.10.004
   (DeepMedic per brain segmentation)

8. **Zhang, W., Li, R., Deng, H., et al. (2015)**
   *Deep Convolutional Neural Networks for Multi-Modality Isointense Infant Brain Image Segmentation.*
   **NeuroImage**, 108, 214-224. DOI: 10.1016/j.neuroimage.2014.12.061
   (Infant brain segmentation con CNN)

### BrainWeb Dataset

9. **Collins, D. L., Zijdenbos, A. P., Kollokian, V., et al. (1998)**
   *Design and construction of a realistic digital brain phantom.*
   **IEEE TMI**, 17(3), 463-468. DOI: 10.1109/42.712135
   (BrainWeb simulator)

10. **Cocosco, C. A., Kollokian, V., Kwan, R. K. S., et al. (1997)**
    *BrainWeb: Online Interface to a 3D MRI Simulated Brain Database.*
    **NeuroImage**, 5(4), S425.
    (BrainWeb database description)

---

**Autore**: Corso di Imaging Biomedico
**Data**: Maggio 2022 (Conversione Python: 2025)
**Versione**: 1.0

# Report: Esercitazione 11 - U-Net Segmentazione Cerebrale

## Obiettivo
Segmentazione automatica della materia cerebrale (sostanza grigia + bianca) da immagini MRI T1 usando architettura U-Net con Transfer Learning.

## Metodologia

### Architettura U-Net
- **Encoder**: Cattura contesto, riduce risoluzione
- **Decoder**: Ripristina risoluzione, localizza feature
- **Skip Connections**: Preserva dettagli spaziali

### Strategia Transfer Learning (2 Fasi)
1. **Task 1 (Facile)**: Segmentazione cranio (skull) → DICE > 0.99
2. **Task 2 (Difficile)**: Segmentazione materia cerebrale → DICE > 0.85

### Loss Functions
- Binary Cross-Entropy (BCE)
- DICE Loss: 1 - DICE coefficient
- Combined Loss: Combinazione pesata BCE + DICE

### Coefficiente DICE
```
DICE = 2 * |A ∩ B| / (|A| + |B|)
```
Range [0, 1], robusto per classi sbilanciate.

## Dataset
- **Fonte**: BrainWeb Simulator
- **Immagini**: 1810 slice MRI T1 (256x256)
- **Soggetti**: 20 cervelli normali
- **Maschere**: Skull + Brain Matter (Gray + White)

## Pipeline
1. Caricamento immagini T1 e maschere
2. Training Task 1 (Skull Segmentation)
3. Salvataggio pesi pre-addestrati
4. Fine-tuning Task 2 (Brain Segmentation)
5. Valutazione DICE su test set

## Risultati Attesi

| Task | Target DICE |
|------|-------------|
| Skull Segmentation | > 0.99 |
| Brain Segmentation | > 0.85 |

## Output
- `results/skull/`: Training Task 1
- `results/brain/`: Training Task 2
- `training_history.png`: Curve Loss e DICE
- `segmentation_results.png`: Esempi visuali
- `best_model.h5`: Modello addestrato

## Dipendenze
- tensorflow/keras
- numpy, scikit-learn
- matplotlib

## Esecuzione

### Demo Rapida
```bash
cd src
python train_skull_segmentation.py --epochs 2 --max_samples 50
python train_brain_segmentation.py --pretrained_model ../results/skull/best_model.h5 --epochs 2
```

### Training Completo (GPU)
```bash
python train_skull_segmentation.py --epochs 30
python train_brain_segmentation.py --pretrained_model ../results/skull/best_model.h5 --epochs 50
```

## Riferimenti
- Ronneberger et al. (2015): U-Net for Biomedical Image Segmentation
- BrainWeb: Simulated Brain Database

---
**Data conversione**: 2025-12-29
**Fonte**: Esercitazione MATLAB 25/05/2022

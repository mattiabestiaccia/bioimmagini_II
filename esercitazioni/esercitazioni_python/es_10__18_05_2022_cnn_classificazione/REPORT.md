# Report: Esercitazione 10 - Classificazione CNN Slice Cardiache

## Obiettivo
Classificazione automatica di slice cardiache MRI in asse corto (Short-Axis) secondo il modello AHA usando Convolutional Neural Networks.

## Classi di Classificazione (Modello AHA)
- **Basal**: Regione basale (vicino valvole, diametro massimo)
- **Middle**: Regione medio-ventricolare (muscoli papillari visibili)
- **Apical**: Regione apicale (punta del cuore, diametro ridotto)

## Metodologia

### Architettura CNN
- **Modello**: VGG-small (ispirato a VGG)
- **Input**: Immagini 128x128 pixel
- **Filtri**: 3x3 con raddoppio dopo ogni pooling
- **Regolarizzazione**: Dropout

### Pipeline
1. Caricamento immagini DICOM
2. Center Crop + Resize a 128x128
3. Normalizzazione [0, 1]
4. Split: Train 70%, Val 15%, Test 15%
5. Training con Data Augmentation
6. Valutazione con matrici di confusione

### Data Augmentation
- Flip orizzontale/verticale
- Rotazioni casuali
- Zoom

## Dataset
- **753 immagini** DICOM
- Sequenze miste: Perfusion, Cine, T2*, LGE
- Organizzate in cartelle per classe

## Risultati

### Performance Ottenute
- **Accuracy**: ~33-35%
- La classificazione e risultata difficile a causa di:
  - Dataset limitato (753 immagini)
  - Variabilita delle sequenze MRI
  - Possibili ambiguita nel labeling

### Miglioramenti Proposti
1. Transfer Learning da ImageNet
2. Aumento dataset
3. Architetture piu semplici

## Output
- `results_aug/training_history.png`: Curve di training
- `results_aug/confusion_matrix_*.png`: Matrici di confusione
- `results_aug/misclassified_samples.png`: Esempi errori

## Dipendenze
- tensorflow/keras
- numpy, scikit-learn
- pydicom, matplotlib

## Esecuzione
```bash
cd src
python cardiac_slice_classifier.py \
    --data_dir ../data \
    --epochs 50 \
    --use_data_augmentation
```

## Riferimenti
- VGG: Visual Geometry Group architecture
- Modello AHA per segmentazione cardiaca

---
**Data conversione**: 2025-12-29
**Fonte**: Esercitazione MATLAB 18/05/2022

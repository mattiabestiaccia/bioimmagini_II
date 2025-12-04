# Esercitazione 11: U-Net per Segmentazione di Materia Cerebrale

## Indice

- [Introduzione](#introduzione)
- [Teoria](#teoria)
  - [Architettura U-Net](#architettura-u-net)
  - [Segmentazione Semantica](#segmentazione-semantica)
  - [Coefficiente DICE](#coefficiente-dice)
  - [Transfer Learning](#transfer-learning)
- [Dataset](#dataset)
- [Implementazione](#implementazione)
- [Utilizzo](#utilizzo)
  - [Demo Rapida](#demo-rapida)
  - [Training Completo](#training-completo)
- [Risultati Attesi](#risultati-attesi)
- [Bibliografia](#bibliografia)

---

## Introduzione

Questa esercitazione implementa **U-Net** (Ronneberger et al. 2015), un'architettura CNN encoder-decoder per la segmentazione semantica di immagini mediche. L'obiettivo è segmentare automaticamente la materia cerebrale (sostanza grigia + bianca) da immagini MRI T1.

**Strategia a due fasi con Transfer Learning**:
1. **Task 1 (Facile)**: Segmentazione del cranio (skull segmentation) → DICE >0.99
2. **Task 2 (Difficile)**: Segmentazione della materia cerebrale usando pesi pre-addestrati → DICE >0.85

**Dataset**: Simulatore BrainWeb (1810 slice MRI T1 da 20 cervelli normali, 256×256 px).

---

## Teoria

### Architettura U-Net

**U-Net** è un'architettura fully convolutional per la segmentazione, caratterizzata da:

**Struttura**:
- **Encoder (Contracting Path)**: Cattura il contesto e riduce la risoluzione spaziale.
- **Decoder (Expanding Path)**: Ripristina la risoluzione spaziale e localizza le feature.
- **Skip Connections**: Concatenano le feature dell'encoder con quelle del decoder per preservare i dettagli spaziali.

### Segmentazione Semantica

La segmentazione assegna una classe ad ogni pixel dell'immagine.
Le funzioni di perdita (Loss Functions) utilizzate sono:
- **Binary Cross-Entropy (BCE)**: Accuratezza pixel-wise.
- **DICE Loss**: 1 - DICE coefficient, ottimizza l'overlap globale.
- **Combined Loss**: Combinazione pesata di BCE e DICE.

### Coefficiente DICE

Il **DICE** misura l'overlap tra la predizione e il ground truth:
`DICE = 2 * |A ∩ B| / (|A| + |B|)`
Range: [0, 1], dove 1 è overlap perfetto. È più robusto dell'accuratezza per classi sbilanciate (es. molto background).

### Transfer Learning

Strategia utilizzata:
1. Pre-addestramento su un task facile (segmentazione cranio).
2. Fine-tuning sul task difficile (segmentazione materia cerebrale).
Questo approccio migliora la convergenza e le performance finali.

---

## Dataset

Il dataset proviene dal simulatore **BrainWeb**:
- **Immagini**: MRI T1 (256x256).
- **Maschere Task 1**: Background vs Skull.
- **Maschere Task 2**: Background vs Brain Matter (Gray + White Matter).

Struttura dati:
```
data/
├── MR/              # Immagini T1
├── GRAY_MASK_B/     # Maschere Skull
└── GRAY_MASK_C/     # Maschere Brain Matter
```

---

## Implementazione

Il codice è organizzato in:
- `src/utils.py`: Definizione modello U-Net, caricamento dati, metriche.
- `src/train_skull_segmentation.py`: Script training Task 1.
- `src/train_brain_segmentation.py`: Script training Task 2.

---

## Utilizzo

### Demo Rapida

Per testare rapidamente il codice e generare output dimostrativi (pochi campioni, poche epoche):

1. **Attivare l'ambiente virtuale** (se presente):
   ```bash
   source ../venv/bin/activate
   ```

2. **Installare le dipendenze**:
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Eseguire Training Skull (Demo)**:
   ```bash
   cd src
   python train_skull_segmentation.py \
       --data_dir ../data \
       --output_dir ../results/skull \
       --epochs 2 \
       --max_samples 50 \
       --batch_size 4
   ```

4. **Eseguire Training Brain (Demo)**:
   ```bash
   python train_brain_segmentation.py \
       --data_dir ../data \
       --pretrained_model ../results/skull/best_model.h5 \
       --output_dir ../results/brain \
       --epochs 2 \
       --max_samples 50 \
       --batch_size 4
   ```

### Training Completo

Per ottenere le massime performance (richiede GPU):

```bash
# Task 1
python train_skull_segmentation.py --data_dir ../data --epochs 30

# Task 2 (con transfer learning)
python train_brain_segmentation.py \
    --data_dir ../data \
    --pretrained_model ../results/skull/best_model.h5 \
    --epochs 50
```

---

## Risultati Attesi

Gli script generano nella cartella `results/`:
- **training_history.png**: Grafici di Loss e DICE durante il training.
- **segmentation_results.png**: Esempi visivi di segmentazione (Input, Ground Truth, Predizione, Overlay).
- **metrics.npz**: Metriche quantitative salvate.
- **best_model.h5**: Il modello addestrato.

### Esempi Visuali

#### Training History (Skull Segmentation)
![Training History Skull](results/skull/training_history.png)

#### Risultati Segmentazione (Brain Matter)
![Segmentation Results Brain](results/brain/segmentation_results.png)

---

## Bibliografia

1. Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
2. BrainWeb: Simulated Brain Database.

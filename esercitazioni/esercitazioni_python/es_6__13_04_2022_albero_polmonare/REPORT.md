# Report: Esercitazione 6 - Analisi Albero Bronchiale

## Obiettivo
Analisi automatica dell'albero bronchiale da immagini CT toraciche per segmentazione del lume, estrazione centerline e misurazione del diametro.

## Metodologia

### Algoritmi Implementati
1. **Conversione Hounsfield Units**: Calibrazione fisica CT
2. **Region Growing 3D**: Segmentazione lume bronchiale da seed in trachea
3. **Interpolazione Isotropa**: Uniformazione voxel per algoritmi 3D
4. **Skeletonization 3D**: Estrazione centerline (medial axis)
5. **Sphere Method**: Misurazione diametro lungo la centerline

### Pipeline
1. Caricamento DICOM CT (148 slice)
2. Conversione a Hounsfield Units (aria = -1000 HU)
3. Region Growing da seed nella trachea
4. Riempimento buchi e pulizia maschera
5. Skeletonization per centerline
6. Sphere method per diametro vascolare
7. Generazione grafico diametro vs distanza

## Valori di Riferimento Anatomici

| Struttura | Diametro | Lunghezza |
|-----------|----------|-----------|
| Trachea | 15-18 mm | ~12 cm |
| Bronchi primari | 10-12 mm | ~4.8 cm |
| Bronchi lobari | 8-10 mm | ~1.9 cm |

## Dataset
- **Fonte**: Cancer Imaging Archive (LIDC-IDRI)
- **Modalita**: CT toracica ad alta risoluzione
- **Copertura**: Trachea e prima biforcazione

## Output
- `results/diameter_measurements.txt`: Misurazioni lungo centerline
- `results/diameter_plot.png`: Grafico diametro vs distanza
- `results/*.npy`: Maschere salvate (opzionale)

## Dipendenze
- numpy, scipy (interpolazione, morfologia)
- scikit-image (skeletonization)
- pydicom, matplotlib

## Esecuzione
```bash
cd src
python bronchial_tree_analysis.py --save_mask
```

## Applicazioni Cliniche
- Diagnosi BPCO
- Valutazione asma severa
- Analisi bronchiectasie
- Follow-up post-trapianto

## Riferimenti
- Tschirren et al. (2005): Intrathoracic airway trees segmentation
- Kiraly et al. (2002): Sphere method per diametro vascolare

---
**Data conversione**: 2025-12-29
**Fonte**: Esercitazione MATLAB 13/04/2022

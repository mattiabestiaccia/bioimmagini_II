# Report: Esercitazione 5 - Segmentazione Grasso Addominale

## Obiettivo
Segmentazione automatica del grasso addominale (SAT/VAT) da acquisizioni MRI T1-pesate per quantificare il rischio cardiovascolare e metabolico.

## Metodologia

### Algoritmi Implementati
1. **K-means Clustering (K=3)**: Separazione aria/tessuto/grasso
2. **Connected Component Labeling 3D**: Rimozione braccia e componenti spurie
3. **Active Contours (Chan-Vese)**: Doppi snake per delimitare SAT
4. **EM-GMM**: Classificazione VAT nella regione intra-addominale

### Pipeline
1. Caricamento volume DICOM T1-weighted
2. K-means per identificazione iniziale grasso
3. Active contour esterno (cute) e interno (fascia muscolare)
4. SAT = regione tra i due contorni
5. EM-GMM su regione intra-addominale per VAT
6. Calcolo volumi e rapporto VAT/SAT

## Risultati Attesi

| Compartimento | Volume Riferimento | Significato Clinico |
|---------------|-------------------|---------------------|
| SAT | ~2091 cm³ | Grasso sottocutaneo |
| VAT | ~970 cm³ | Grasso viscerale |
| VAT/SAT | ~46% | Indice rischio metabolico |

### Interpretazione VAT/SAT
- < 30%: Basso rischio metabolico
- 30-50%: Rischio moderato
- > 50%: Alto rischio (obesita viscerale)

## Output
- `results/fat_volumes.txt`: Volumi SAT/VAT calcolati
- `results/fat_segmentation_results.png`: Visualizzazione overlay
- `results/slice_*_detailed.png`: Analisi dettagliata per slice

## Dipendenze
- numpy, scipy, scikit-learn (K-means, GMM)
- scikit-image (Active Contours)
- pydicom, matplotlib

## Esecuzione
```bash
cd src
python fat_segmentation.py
```

## Riferimenti
- Positano et al. (2004): Accurate segmentation of subcutaneous adipose tissue
- Chan & Vese (2001): Active contours without edges

---
**Data conversione**: 2025-12-29
**Fonte**: Esercitazione MATLAB 06/04/2022

# Report Esercitazione 2: Filtraggio 3D su Immagini CT

**Data completamento**: 2025-12-29

**Directory**: `es_2__16_03_2022_filtraggio/`

**Branch**: `feature/es_2`

---

## Sommario Lavoro Svolto

Implementazione e validazione di una pipeline completa per il confronto di algoritmi di filtraggio 3D su immagini CT. L'esercitazione valuta le prestazioni di tre filtri (media mobile, Gaussiano, Wiener adattivo) in termini di SNR (Signal-to-Noise Ratio) e conservazione delle transizioni (acutezza dei contorni).

---

## Obiettivi Raggiunti

1. **Caricamento DICOM** con conversione corretta in Hounsfield Units (HU)
2. **Interpolazione trilineare** per ottenere volume isotropo
3. **Implementazione filtri 3D**: media mobile, Gaussiano, Wiener adattivo
4. **Ottimizzazione sigma** per filtro Gaussiano
5. **Calcolo metriche**: SNR e acutezza transizioni
6. **Visualizzazione risultati**: plot confronto, profili, differenze

---

## File Implementati

### `src/main_filtering.py` (~467 righe)

Script principale che orchestra l'intera pipeline:
- Caricamento volume DICOM CT
- Verifica e correzione isotropia
- Applicazione filtri 3D
- Calcolo e visualizzazione metriche

### `src/dicom_utils.py` (~200 righe)

Gestione file DICOM:
- `load_dicom_volume()`: Caricamento con conversione HU corretta
- `make_isotropic()`: Interpolazione trilineare per voxel cubici
- `check_isotropy()`: Verifica isotropia volume

### `src/filters_3d.py` (~200 righe)

Implementazione filtri 3D:
- `moving_average_filter_3d()`: Filtro a media mobile
- `gaussian_filter_3d()`: Filtro Gaussiano con sigma configurabile
- `wiener_filter_3d()`: Filtro Wiener adattivo 3D (non disponibile in MATLAB)
- `variance_map_3d()`: Mappa varianza locale 3D
- `estimate_noise_variance()`: Stima varianza rumore da ROI

### `src/metrics.py` (~170 righe)

Calcolo metriche di qualita:
- `calculate_snr()`: Signal-to-Noise Ratio
- `calculate_edge_sharpness()`: Acutezza transizioni (gradiente massimo)
- `create_circular_roi()`: Creazione ROI circolare
- `extract_profile()`: Estrazione profilo 1D

### `src/interactive_roi_selection.py` (~160 righe)

Tool interattivo per selezione ROI e profilo:
- Visualizzazione slice centrale
- Selezione interattiva centro/raggio ROI
- Definizione profilo per misura transizioni

---

## Risultati Validazione

### Dataset

- **Fonte**: RIDER Phantom PET-CT
- **Serie**: CT 2.5mm (63 slices, 512x512)
- **Spacing originale**: 0.68 x 0.68 x 2.5 mm
- **Volume isotropo**: 512 x 512 x 230 (dopo interpolazione)

### Parametri Configurazione

```
Kernel size: 7x7x7
Sigma Gaussiano ottimale: 1.55
ROI centro: (256, 256), raggio: 80 pixel
Rumore stimato (σ): 13.71 HU
```

### Tabella Risultati

| Filtro | SNR | Acutezza | Note |
|--------|-----|----------|------|
| Originale | 79.05 | 33.62 | Baseline |
| Media Mobile | 83.64 | 33.10 | +5.8% SNR, -1.5% acutezza |
| Gaussiano (σ=1.55) | 87.21 | 34.56 | +10.3% SNR, +2.8% acutezza |
| Wiener Adattivo | 93.48 | 33.54 | **+18.2% SNR**, -0.2% acutezza |

### Interpretazione

1. **Filtro Wiener Adattivo**: Miglior incremento SNR (+18.2%) con minima perdita di acutezza. Il filtro si adatta localmente, filtrando nelle zone omogenee e preservando i contorni.

2. **Filtro Gaussiano**: Ottimo compromesso con +10.3% SNR e addirittura leggero miglioramento acutezza (+2.8%). L'ottimizzazione del sigma (1.55) ha trovato il punto di equilibrio.

3. **Media Mobile**: Minimo incremento SNR (+5.8%) con maggior perdita di acutezza. Essendo un filtro uniforme, sfoca indiscriminatamente.

---

## Output Generati

```
results/
├── confronto_filtri.png      # Confronto visivo slice centrali
├── confronto_profili.png     # Profili sovrapposti per confronto transizioni
├── differenze_filtri.png     # Mappe differenza rispetto all'originale
├── ottimizzazione_sigma.png  # SNR e acutezza vs sigma
└── risultati.txt             # Tabella risultati in formato testo
```

---

## Pipeline Esecuzione

```bash
# 1. Posizionarsi nel worktree
cd /home/brusc/Projects/bioimmagini_positano_worktrees/es_2

# 2. Attivare ambiente virtuale
source /home/brusc/Projects/bioimmagini_positano/esercitazioni/esercitazioni_python/venv/bin/activate

# 3. Eseguire esercitazione
cd esercitazioni/esercitazioni_python
python -m es_2__16_03_2022_filtraggio.src.main_filtering

# 4. (Opzionale) Selezione interattiva ROI
python -m es_2__16_03_2022_filtraggio.src.interactive_roi_selection
```

---

## Dipendenze

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pydicom>=2.2.0
```

---

## Note Tecniche

### Conversione Hounsfield Units

MATLAB `dicomread` (fino a versione 2021b) ignora erroneamente RescaleIntercept/Slope. La nostra implementazione Python applica correttamente:

```python
HU = RescaleIntercept + RescaleSlope * PixelValue
```

### Filtro Wiener 3D

Non disponibile in MATLAB (solo `wiener2` per 2D). Implementazione custom:

```python
def wiener_filter_3d(volume, kernel_size, noise_variance):
    # Media mobile
    I_MM = moving_average_filter_3d(volume, kernel_size)

    # Mappa varianza locale
    I_VAR = variance_map_3d(volume, kernel_size)

    # Coefficiente adattivo
    alpha = np.maximum(0, (I_VAR - noise_variance) / I_VAR)

    # Applicazione Wiener
    return I_MM + alpha * (volume - I_MM)
```

### Calcolo Efficiente Varianza Locale

Implementazione O(1) per pixel usando identita `Var(X) = E[X²] - E[X]²`:

```python
def variance_map_3d(volume, kernel_size):
    mean_local = uniform_filter(volume, kernel_size)
    mean_sq = uniform_filter(volume**2, kernel_size)
    return mean_sq - mean_local**2
```

---

## Confronto MATLAB vs Python

| Operazione | MATLAB | Python |
|------------|--------|--------|
| DICOM loading | `dicomread` (bug HU) | `pydicom` (corretto) |
| Media 3D | `fspecial3` + `imfilter` | `scipy.ndimage.uniform_filter` |
| Gaussiano 3D | `imgaussfilt3` | `scipy.ndimage.gaussian_filter` |
| Wiener 3D | **Non disponibile** | Implementazione custom |
| Varianza 3D | **Non disponibile** | Implementazione O(1) |
| Interpolazione | `interp3` | `scipy.ndimage.zoom` |

---

## Conclusioni

L'esercitazione 2 e stata completata con successo:

- **Pipeline completa** per valutazione filtri 3D su immagini CT
- **Tre algoritmi** implementati e confrontati (media, Gaussiano, Wiener)
- **Ottimizzazione automatica** del parametro sigma per Gaussiano
- **Metriche quantitative** (SNR, acutezza) per valutazione oggettiva
- **Visualizzazione** completa con plot e tabelle

Il **filtro Wiener adattivo** si conferma superiore per imaging CT, con il miglior incremento SNR (+18.2%) e conservazione quasi perfetta dei contorni. Questa implementazione 3D rappresenta un valore aggiunto rispetto alla versione MATLAB originale.

# Report Branch feature/es_4

**Data**: 2024-12-28
**Branch**: feature/es_4
**Autore**: Claude Code

---

## Riepilogo Sessione

Questa sessione ha lavorato su due esercitazioni:
1. **Esercitazione 4**: Funzione Cardiaca (Segmentazione a Contorni)
2. **Esercitazione 5**: Segmentazione Grasso Addominale (SAT/VAT)

---

## Esercitazione 4: Funzione Cardiaca

**Status**: ⚠️ IN CORSO - Richiede ottimizzazione

**Directory**: `esercitazioni_python/es_4__30_03_2022_funzione_cardiaca/`

**Source MATLAB**: `esercitazioni_matlab/LEZIONE_09_30_03_2022 (Esercitazione Contorni)/`

**Topic**: Analisi funzione cardiaca del ventricolo sinistro da MRI cine con Active Contours (Chan-Vese)

### Overview

Implementazione della pipeline per quantificazione della funzione cardiaca:
- **EDLV/ESLV**: Volumi ventricolari end-diastolico e end-sistolico
- **Stroke Volume**: Volume di eiezione (EDLV - ESLV)
- **Ejection Fraction**: Frazione di eiezione (SV/EDLV)
- **Cardiac Output**: Gittata cardiaca

### Dataset

- **Formato**: 450 file DICOM (15 slices x 30 frame temporali)
- **Risoluzione**: 256x256 pixel
- **Pixel spacing**: 1.41 mm
- **Slice thickness**: 8.0 mm
- **Sequenza**: Cine MRI SSFP short-axis
- **Fasi**: Diastole (frame 27, 693ms), Sistole (frame 11, 288ms)

### Pipeline Implementata

1. **Caricamento 4D**: Parsing DICOM con raggruppamento per TriggerTime/SlicePosition
2. **Identificazione fasi**: Automatica da TriggerTime o manuale
3. **Segmentazione Chan-Vese**: `morphological_chan_vese` di scikit-image
4. **Find LV Center**: Rilevamento automatico centro ventricolo sinistro
5. **Refine Segmentation**: Filtro componenti per seed_center e max_area
6. **Calcolo volumi**: Metodo Simpson (somma aree x spessore slice)
7. **Calcolo parametri**: SV, EF, CO, valori indicizzati per BSA

### Risultati

| Parametro | Calcolato | Riferimento | Errore | Status |
|-----------|-----------|-------------|--------|--------|
| EDLV | 143 mL | 114 mL | +25% | Accettabile |
| ESLV | 113 mL | 41 mL | **+176%** | CRITICO |
| Stroke Volume | 31 mL | 73 mL | -58% | CRITICO |
| Ejection Fraction | 21% | 63% | -67% | CRITICO |

**Comando test**:
```bash
python src/cardiac_function_analysis.py --seed_radius 15 --smoothing 4 --n_iterations 150
```

### Criticita' Identificate

1. **Sovrastima aree sistole**: Chan-Vese non cattura contrazione ventricolare
2. **Find LV center**: Talvolta identifica RV invece di LV
3. **Slice selection**: Parametri empirici (4-11 diastole, 5-10 sistole)
4. **Propagazione errori**: Seed errato si propaga a slice successive

### File Creati/Modificati

- `src/utils.py`: Aggiunta `find_lv_center()`, modificata `refine_segmentation()`
- `src/cardiac_function_analysis.py`: Propagazione centri diastole->sistole
- `WORK_IN_PROGRESS.md`: Documentazione dettagliata criticita'
- `requirements.txt`: Dipendenze Python
- `.venv/`: Virtual environment

### Prossimi Passi

- [ ] Provare approccio threshold-based invece di Chan-Vese
- [ ] Implementare edge-based active contours
- [ ] Aggiungere seed interattivo
- [ ] Migliorare pre-processing immagini

---

## Esercitazione 5: Segmentazione Grasso Addominale

**Status**: ✅ COMPLETATA

**Directory**: `esercitazioni_python/es_5__06_04_2022_segmentazione_grasso/`

**Source MATLAB**: `esercitazioni_matlab/LEZIONE_11_06_04_2022 (Esercitazione segmentazione grasso)/`

**Topic**: Quantificazione grasso addominale subcutaneo (SAT) e viscerale (VAT) da MRI T1-weighted

### Overview

Pipeline completa per segmentazione automatica grasso addominale:
- **SAT (Subcutaneous Adipose Tissue)**: Grasso sottocutaneo
- **VAT (Visceral Adipose Tissue)**: Grasso viscerale
- **VAT/SAT ratio**: Indice rischio cardiovascolare

### Dataset

- **Formato**: 18 slice DICOM assiali T1-weighted
- **Risoluzione**: 256x256 pixel
- **Pixel spacing**: 1.875 mm
- **Slice thickness**: 5.0 mm
- **Sequenza**: T1-weighted (grasso = segnale alto)

### Pipeline Implementata

1. **K-means Clustering (K=3)**: Separazione aria/muscolo/grasso
2. **Connected Component Labeling**: Rimozione braccia (keep largest)
3. **Active Contours (Chan-Vese)**:
   - Contorno esterno (cute): 150 iterazioni
   - Contorno interno (fascia): 100 iterazioni
4. **EM-GMM (2 componenti)**: Classificazione VAT nella regione intra-addominale
5. **Calcolo volumi**: Somma voxel x volume voxel

### Risultati

| Parametro | Calcolato | Riferimento | Errore | Status |
|-----------|-----------|-------------|--------|--------|
| SAT | 1840.5 cm³ | 2091 cm³ | **-12%** | Buono |
| VAT | 1123.8 cm³ | 970 cm³ | **+16%** | Buono |
| VAT/SAT | 61.1% | 46% | +15 punti | Accettabile |

**Comando esecuzione**:
```bash
python src/fat_segmentation.py
```

### Output Generati

```
results/
├── fat_segmentation_results.png   # Visualizzazione 6 pannelli
└── fat_volumes.txt                # Report numerico
```

### File Struttura

```
es_5__06_04_2022_segmentazione_grasso/
├── .venv/                    # Virtual environment
├── data/dicom/               # 18 file DICOM (copiati)
├── docs/                     # PDF esercitazione + paper
├── results/                  # Output generati
├── src/
│   ├── utils.py              # Funzioni core (600+ righe)
│   ├── fat_segmentation.py   # Pipeline principale
│   └── visualize_results.py  # Visualizzazione avanzata
├── requirements.txt
└── README.md                 # Documentazione completa
```

### Valutazione

- ✅ Pipeline funzionante end-to-end
- ✅ Risultati entro margine accettabile (errore <20%)
- ✅ Visualizzazioni chiare e informative
- ✅ Documentazione completa

---

## Riepilogo Finale

| Esercitazione | Status | Accuratezza | Note |
|---------------|--------|-------------|------|
| Es. 4 - Funzione Cardiaca | ⚠️ WIP | Bassa | ESLV sovrastimato, EF errata |
| Es. 5 - Grasso SAT/VAT | ✅ OK | Buona | Errori <20%, visivamente corretto |

### Commit Suggeriti

```bash
# Es. 5 (pronto per commit)
git add esercitazioni/esercitazioni_python/es_5__06_04_2022_segmentazione_grasso/
git commit -m "feat(es_5): complete fat segmentation pipeline (SAT/VAT)"

# Es. 4 (work in progress)
git add esercitazioni/esercitazioni_python/es_4__30_03_2022_funzione_cardiaca/
git commit -m "wip(es_4): cardiac function analysis - needs optimization"
```

---

## Ambiente di Sviluppo

- **Python**: 3.13
- **Package Manager**: uv
- **Dipendenze principali**:
  - numpy, scipy
  - scikit-image, scikit-learn
  - pydicom
  - matplotlib

---

*Report generato automaticamente da Claude Code*

# Report Branch feature/es_4

**Data**: 2024-12-28
**Branch**: feature/es_4
**Autore**: Claude Code

---

## Riepilogo Sessione

Questo branch contiene il lavoro sull'esercitazione 4:
- **Esercitazione 4**: Funzione Cardiaca (Segmentazione a Contorni)

> **Nota**: L'esercitazione 5 e' stata spostata nel branch dedicato `feature/es_5`

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

### Prossimi Passi

- [ ] Provare approccio threshold-based invece di Chan-Vese
- [ ] Implementare edge-based active contours
- [ ] Aggiungere seed interattivo
- [ ] Migliorare pre-processing immagini

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

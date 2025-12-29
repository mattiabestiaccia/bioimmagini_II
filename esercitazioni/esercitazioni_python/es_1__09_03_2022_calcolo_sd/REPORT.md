# Report Esercitazione 1 - Calcolo della Deviazione Standard in Immagini MRI

**Data completamento**: 2025-12-29
**Versione Python**: 3.12
**Branch**: feature/es_1

---

## 1. Sintesi dell'Esercitazione

### Obiettivo
Analizzare e stimare il rumore nelle immagini di Risonanza Magnetica (MRI) attraverso diversi metodi statistici, comprendendo le differenze tra approcci manuali (ROI-based) e automatici (SD map).

### Algoritmi Implementati
- **Calcolo SD Map**: Mappa di deviazione standard con kernel sliding window (equivalente a `stdfilt` MATLAB)
- **Stima sigma da istogramma**: Identificazione del massimo dell'istogramma della SD map
- **Correzione Rayleigh**: Fattore correttivo (1.526) per il rumore di fondo in immagini MRI magnitude
- **Test Monte Carlo**: Simulazione per studiare la convergenza delle stime statistiche

### Dataset Utilizzato
- **Tipo**: Immagini DICOM (fantoccio MRI) e immagini sintetiche
- **Dimensioni**:
  - Sintetica: 512x512 pixel
  - Fantoccio: 256x256 pixel
- **Caratteristiche**: Rumore gaussiano con sigma noto (5.0 per sintetica, sconosciuto per fantoccio)

---

## 2. Analisi dei Risultati

### Risultati Script 1: Immagine Sintetica (calcolo_sd.py)

| Metodo | Valore Stimato | Valore Vero | Errore (%) |
|--------|----------------|-------------|------------|
| Media SD map | 6.8141 | 5.0 | 36.28% |
| Mediana SD map | 4.8957 | 5.0 | 2.09% |
| Max Istogramma | 4.9919 | 5.0 | **0.16%** |

**Osservazioni**:
- Il **massimo dell'istogramma** è il metodo più accurato (errore < 1%)
- La **media** presenta bias elevato a causa delle variazioni di intensità tra pattern
- La **mediana** offre buona robustezza

### Risultati Script 2: Fantoccio MRI (esempio_calcolo_sd.py)

#### Analisi Manuale (ROI-based)
| ROI | Media | Std Dev | N pixel |
|-----|-------|---------|---------|
| Olio | 390.22 | 98.87 | 1260 |
| Acqua | 361.51 | 136.99 | 1260 |
| Background | 63.29 | 117.95 | 700 |
| Background (corretto) | - | **180.04** | - |

#### Analisi Automatica (SD Map)
| Configurazione | Mean | Hist Max |
|----------------|------|----------|
| Kernel 3x3 | 21.03 | 7.92 |
| Kernel 9x9, th>100 | 69.10 | 12.89 |

### Risultati Script 3: Test Monte Carlo (test_m_sd.py)

| Dim ROI | Est. Mean | Err Mean (%) | Est. SD | Err SD (%) |
|---------|-----------|--------------|---------|------------|
| 2x2 | 29.98 | 0.08 | 20.40 | 7.27 |
| 4x4 | 30.69 | 2.31 | 21.59 | 1.87 |
| 8x8 | 29.64 | 1.21 | 22.06 | 0.28 |
| 16x16 | 30.01 | 0.03 | 22.01 | 0.06 |
| 32x32 | 30.04 | 0.13 | 22.07 | 0.31 |
| 64x64 | 29.99 | 0.02 | 22.03 | 0.12 |
| 128x128 | 30.02 | **0.07** | 22.00 | **0.02** |

**Valori imposti**: Mean = 30.0, SD = 22.0

---

## 3. Performance

### Tempi di Esecuzione
| Script | Tempo | Note |
|--------|-------|------|
| calcolo_sd.py | ~2s | Immagine sintetica 512x512 |
| esempio_calcolo_sd.py | ~1s | Fantoccio 256x256 |
| test_m_sd.py | ~3s | 100 simulazioni Monte Carlo |
| **Totale** | ~6s | |

### Risorse Utilizzate
- **RAM**: ~200MB (picco durante caricamento TensorFlow)
- **CPU**: Utilizzo moderato, operazioni vettorizzate

---

## 4. Problemi Riscontrati

### Difficoltà Tecniche

1. **Conflitto nome modulo types.py**
   - **Descrizione**: Il file `types.py` causava ImportError per conflitto con il modulo standard Python
   - **Causa**: Python importava il file locale invece del modulo standard `types`
   - **Soluzione**: Rinominato in `type_defs.py`

2. **Path DICOM errato**
   - **Descrizione**: Lo script `esempio_calcolo_sd.py` non trovava `phantom.dcm`
   - **Causa**: Costruzione errata del path relativo (`script_dir.parent` invece di `script_dir`)
   - **Soluzione**: Corretto il path a `script_dir / args.dicom`

### Differenze MATLAB/Python
- **`stdfilt` vs `generic_filter`**: Implementazione equivalente usando `scipy.ndimage.generic_filter` con `np.std`
- **`ddof=1`**: Necessario per replicare il calcolo sample SD di MATLAB
- **ROI interattive**: In MATLAB `drawcircle()`, in Python implementato con classe custom

### Limitazioni Note
- Le ROI nell'analisi del fantoccio sono fisse (non interattive in modalità batch)
- I grafici vengono salvati ma non visualizzati in ambiente non-interattivo

---

## 5. Miglioramenti Didattici Suggeriti

### Cosa Ha Funzionato Bene
- Struttura modulare con `utils.py` riutilizzabile
- Documentazione completa nel README
- Output formattato con tabelle chiare
- Salvataggio automatico dei grafici

### Cosa Potrebbe Essere Migliorato

#### Nella Documentazione
- Aggiungere spiegazione visiva della correzione Rayleigh
- Includere diagramma del workflow di analisi

#### Nel Codice
- Implementare modalità interattiva ROI cross-platform
- Aggiungere unit test automatizzati

---

## 6. Conclusioni

### Competenze Acquisite
1. **Analisi statistica del rumore** in immagini mediche
2. **Correzione Rayleigh** per distribuzioni non-gaussiane
3. **Metodi di stima robusta** (mediana vs media vs max istogramma)
4. **Convergenza statistica** attraverso simulazioni Monte Carlo
5. **Conversione MATLAB-Python** mantenendo equivalenza numerica

### Punti Chiave per Discussione con Professore
1. Perché il max dell'istogramma è il metodo più accurato per la stima del rumore?
2. Quando è necessaria la correzione Rayleigh e perché il fattore è ~1.526?
3. Trade-off tra dimensione ROI e accuratezza/precisione delle misure
4. Applicazioni cliniche: come queste tecniche vengono usate in diagnostica?

### Validazione
- Tutti e 3 gli script eseguono correttamente
- Risultati numerici coerenti con i valori attesi documentati nel README
- Grafici generati e salvati in `results/`

---

## File Modificati

| File | Modifica | Motivo |
|------|----------|--------|
| `src/types.py` → `src/type_defs.py` | Rinominato | Conflitto con modulo standard Python |
| `src/esempio_calcolo_sd.py` | Fix path linea 441 | DICOM non trovato |

---

*Report generato automaticamente durante la revisione dell'esercitazione*

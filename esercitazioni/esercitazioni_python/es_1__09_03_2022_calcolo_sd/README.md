# Esercitazione 1 - Calcolo della Deviazione Standard in Immagini MRI

Implementazione Python delle esercitazioni MATLAB per l'analisi del rumore in immagini di Risonanza Magnetica.

## Descrizione

Questa esercitazione copre tre aspetti fondamentali dell'analisi del rumore nelle immagini MRI:

1. **Analisi su immagine sintetica** - Validazione dei metodi di stima del rumore
2. **Analisi su fantoccio MRI** - Applicazione su dati reali con tecniche manuali e automatiche
3. **Test Monte Carlo** - Valutazione della convergenza statistica al variare della dimensione ROI

## Struttura del Progetto

```
esercitazione_1/
├── src/                    # Codice sorgente Python
│   ├── __init__.py        # Inizializzazione modulo
│   ├── utils.py           # Funzioni utility comuni
│   ├── calcolo_sd.py      # Script 1: Analisi immagine sintetica
│   ├── esempio_calcolo_sd.py  # Script 2: Analisi fantoccio MRI
│   └── test_m_sd.py       # Script 3: Test Monte Carlo
├── data/                   # Dati DICOM
│   ├── phantom.dcm        # Fantoccio MRI
│   ├── IMG-0001-00001.dcm
│   ├── IMG-0002-00001.dcm
│   └── esempio_LGE/       # Serie cardiache LGE (18 DICOM, ~2.4MB)
│       └── 20050330 092439 [8397 - CUORE DE]/
│           ├── Series 112-117  # 6 serie MRI cardiache
│           └── README.md       # Documentazione dati LGE
├── results/               # Output (grafici, tabelle)
├── notebooks/             # Jupyter notebooks (opzionale)
├── tests/                 # Unit tests
├── requirements.txt       # Dipendenze Python
└── README.md             # Questa documentazione
```

## Installazione

### Prerequisiti

- Python 3.8 o superiore
- pip (package manager)

### Setup Ambiente

```bash
# 1. Creare un virtual environment (raccomandato)
python -m venv venv

# 2. Attivare l'ambiente virtuale
# Su Linux/Mac:
source venv/bin/activate
# Su Windows:
venv\Scripts\activate

# 3. Installare le dipendenze
pip install -r requirements.txt
```

## Utilizzo

### Script 1: Analisi Immagine Sintetica

Crea un'immagine sintetica con rumore gaussiano noto e confronta diversi metodi di stima del rumore.

```bash
cd src
python calcolo_sd.py
```

**Output:**
- Immagine con rumore gaussiano e relativo istogramma
- Mappa di deviazione standard (SD map)
- Confronto tra 4 metodi di stima sigma: valore vero, media, mediana, massimo istogramma

**Concetti illustrati:**
- Generazione di immagini sintetiche con pattern multipli
- Calcolo della SD map con kernel sliding window
- Validazione quantitativa dei metodi di stima

---

### Script 2: Analisi Fantoccio MRI

Analizza il rumore su un'immagine reale di fantoccio MRI utilizzando approcci manuali e automatici.

```bash
cd src
python esempio_calcolo_sd.py [opzioni]
```

**Opzioni:**
- `--interactive`: Abilita selezione interattiva delle ROI
- `--no-display`: Non mostra i grafici (solo salvataggio)
- `--dicom PATH`: Specifica percorso file DICOM (default: `../data/phantom.dcm`)

**Esempio con ROI interattive:**
```bash
python esempio_calcolo_sd.py --interactive
```

**Output:**

**Sezione 1 - Analisi Manuale con ROI:**
- Selezione di 3 ROI circolari (olio, acqua, background)
- Calcolo della deviazione standard per ciascuna ROI
- Applicazione della correzione Rayleigh per il background (fattore 1.526)

**Sezione 2 - Analisi Automatica (kernel 3×3):**
- Calcolo SD map automatica
- Esclusione zero-padding
- Confronto: media, mediana, massimo istogramma

**Sezione 3 - Analisi Automatica (kernel 9×9, soglia >100):**
- Kernel più grande per smoothing
- Applicazione soglia di intensità
- Effetti sulla stima del rumore

**Concetti illustrati:**
- Differenza tra metodi manuali (ROI-based) e automatici (SD map)
- Correzione Rayleigh per distribuzioni non-gaussiane nel background
- Effetto della dimensione del kernel e delle soglie di intensità

---

### Script 3: Test Monte Carlo ROI

Valuta la convergenza delle stime di media e deviazione standard al crescere della dimensione ROI attraverso simulazioni Monte Carlo.

```bash
cd src
python test_m_sd.py
```

**Output:**
- 100 simulazioni per 7 dimensioni ROI (da 2×2 a 128×128 pixel)
- Grafici di convergenza per media e deviazione standard
- Analisi della precisione (varianza delle stime)
- Tabella riassuntiva con errori percentuali

**Concetti illustrati:**
- Legge dei grandi numeri applicata all'imaging
- Convergenza più rapida della media rispetto alla deviazione standard
- Trade-off tra dimensione ROI e accuratezza/precisione
- Importanza della dimensione campionaria nelle misure statistiche

---

## Modulo Utils

Il modulo `utils.py` fornisce funzioni riutilizzabili:

### Funzioni Principali

```python
from src.utils import (
    compute_sd_map,              # Calcola mappa SD (equivalente a stdfilt MATLAB)
    estimate_sigma_from_histogram,  # Stima sigma dal massimo istogramma
    apply_rayleigh_correction,   # Applica correzione Rayleigh
    create_synthetic_image,      # Genera immagine sintetica
    compute_statistics,          # Statistiche complete
    exclude_zero_padding         # Maschera zero-padding
)
```

### Esempio d'Uso

```python
import numpy as np
from src.utils import compute_sd_map, apply_rayleigh_correction

# Caricare un'immagine
image = ...  # array numpy

# Calcolare mappa SD
sd_map = compute_sd_map(image, kernel_size=5)

# Misurare rumore nel background
background_sd = np.std(background_roi)
corrected_sd = apply_rayleigh_correction(background_sd)
print(f"SD corretta: {corrected_sd:.4f}")
```

## Teoria: Correzione Rayleigh

Nelle immagini MRI magnitude, il rumore nel **background** (dove non c'è segnale) segue una **distribuzione di Rayleigh**, non gaussiana.

La relazione tra SD misurata nel background (σ_bkg) e il vero rumore gaussiano (σ_true) è:

```
σ_true = σ_bkg × √(2 / (4 - π)) ≈ σ_bkg × 1.526
```

Questo fattore correttivo è implementato in `apply_rayleigh_correction()`.

**Riferimento:**
> Constantinides, C. D., Atalar, E., & McVeigh, E. R. (1997).
> Signal-to-noise measurements in magnitude images from NMR phased arrays.

## Best Practices Implementate

### 1. **Struttura Modulare**
- Separazione tra utility, script ed esecuzione
- Codice riutilizzabile in `utils.py`
- Import relativi per portabilità

### 2. **Documentazione**
- Docstring completi in stile NumPy/SciPy
- Type hints per parametri e return values
- Esempi d'uso inline

### 3. **Gestione I/O**
- Path relativi con `pathlib.Path`
- Organizzazione output in cartelle dedicate
- Supporto salvataggio risultati

### 4. **Visualizzazione**
- Grafici chiari e annotati
- Comparazioni quantitative
- Salvata automatico in alta risoluzione (150 DPI)

### 5. **Parametrizzazione**
- Argparse per configurazione da CLI
- Parametri di default sensati
- Modalità interattiva/batch

### 6. **Compatibilità MATLAB**
- Equivalenza funzionale confermata
- Commenti che referenziano codice MATLAB originale
- Stessi parametri e algoritmi

## Confronto con MATLAB

| Feature | MATLAB | Python |
|---------|--------|--------|
| SD Map | `stdfilt()` | `scipy.ndimage.generic_filter()` |
| Histogram | `hist()` | `numpy.histogram()` / `matplotlib.hist()` |
| DICOM I/O | `dicomread()` | `pydicom.dcmread()` |
| ROI Selection | `drawcircle()` | Custom `ROISelector` class |
| Random Numbers | `randn()` | `numpy.random.normal()` |
| Standard Deviation | `std()` | `numpy.std(ddof=1)` |

**Nota:** In Python usiamo `ddof=1` per replicare il comportamento di MATLAB che calcola la sample standard deviation per default.

## Risultati Attesi

### Script 1 (Immagine Sintetica)
- Errore medio < 2% per tutti i metodi
- Massimo istogramma: metodo più accurato
- Media e mediana: leggero bias positivo

### Script 2 (Fantoccio MRI)
- SD background (raw): ~5-10 unità
- SD background (corretto): ~7-15 unità
- SD acqua: ~3-8 unità
- SD olio: ~2-6 unità

### Script 3 (Monte Carlo)
- Convergenza media: rapida (ROI ≥ 16×16)
- Convergenza SD: più lenta (ROI ≥ 64×64)
- Errore < 1% per ROI 128×128

## Estensioni Possibili

### Jupyter Notebooks
Creare notebook interattivi nella cartella `notebooks/`:
```bash
jupyter notebook notebooks/
```

### Unit Testing
Implementare test in `tests/`:
```python
# tests/test_utils.py
import pytest
from src.utils import compute_sd_map

def test_sd_map_shape():
    image = np.random.randn(100, 100)
    sd_map = compute_sd_map(image, kernel_size=5)
    assert sd_map.shape == image.shape
```

### Analisi Serie DICOM
Estendere `esempio_calcolo_sd.py` per processare serie complete:
```python
for dcm_file in series_path.glob("*.dcm"):
    analyze_noise(dcm_file)
```

## Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'pydicom'"
**Soluzione:** Installare le dipendenze
```bash
pip install -r requirements.txt
```

### Problema: "FileNotFoundError: phantom.dcm not found"
**Soluzione:** Verificare che i file DICOM siano in `data/`
```bash
ls -l data/*.dcm
```

### Problema: Grafici non vengono visualizzati
**Soluzione:** Backend matplotlib
```python
# Aggiungere in testa allo script:
import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg'
```

### Problema: ROI interattive non funzionano
**Soluzione:** Usare modalità non interattiva
```bash
python esempio_calcolo_sd.py  # senza --interactive
```

## Autori e Riferimenti

**Corso:** Bioimmagini - Positano
**Esercitazione Originale:** MATLAB (09/03/2022)
**Conversione Python:** 2025

### Riferimenti Bibliografici

1. Constantinides et al. (1997) - Rayleigh correction in MRI
2. Consip_MRI.pdf - Specifiche tecniche risonanza magnetica
3. Documentazione originale: `Esercitazione_01_09_03_2022.pdf`

## Licenza

Materiale didattico - Solo uso educativo

---

## Quick Start

```bash
# Setup completo
git clone <repository>
cd esercitazione_1
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Eseguire tutte le analisi
cd src
python calcolo_sd.py
python esempio_calcolo_sd.py
python test_m_sd.py

# Risultati disponibili in: ../results/
```

## Contatti

Per domande o segnalazioni relative alle esercitazioni, contattare i docenti del corso.

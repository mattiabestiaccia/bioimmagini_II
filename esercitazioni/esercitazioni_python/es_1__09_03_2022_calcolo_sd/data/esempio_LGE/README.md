# Esempio LGE - Late Gadolinium Enhancement Cardiac MRI

## Descrizione

Questo dataset contiene immagini DICOM di un caso clinico reale di **Late Gadolinium Enhancement (LGE)** cardiaco, acquisito con risonanza magnetica.

## Informazioni Studio

- **Data acquisizione**: 30 Marzo 2005
- **Studio**: Cardiaco con mezzo di contrasto (Gadolinio)
- **Paziente**: CUORE DE [ID: 8397]
- **Tecnica**: Late Gadolinium Enhancement (LGE)

## Struttura Dati

```
esempio_LGE/
└── 20050330 092439 [8397 - CUORE DE]/
    ├── Series 112 [MR - SC 2DDel Enh 250 2 C]  → 1 slice (2 chambers)
    ├── Series 113 [MR - SC 2DDel Enh 250 4 C]  → 1 slice (4 chambers)
    ├── Series 114 [MR - SC 2DDel Enh 250 2 C]  → 1 slice (2 chambers)
    ├── Series 115 [MR - SC 2DDel Enh 250 FX]   → 13 slices (short axis) ⭐
    ├── Series 116 [MR - SC 2DDel Enh 250 2 C]  → 1 slice (2 chambers)
    └── Series 117 [MR - SC 2DDel Enh 250 4 C]  → 1 slice (4 chambers)
```

**Totale**: 18 file DICOM (~2.4 MB)

## Serie Principali

### Series 115 - Short Axis (FX)
- **13 slice** attraverso l'asse corto del cuore
- Copertura completa del ventricolo sinistro
- Ideale per analisi volumetriche e segmentazione

### Series 112, 114, 116 - 2 Chamber Views (2C)
- Viste long-axis a 2 camere
- Visualizzazione atrio sinistro + ventricolo sinistro

### Series 113, 117 - 4 Chamber Views (4C)
- Viste long-axis a 4 camere
- Visualizzazione completa delle camere cardiache

## Tecnica LGE

Il **Late Gadolinium Enhancement** è una tecnica MRI usata per:

1. **Identificare tessuto necrotico/fibrosi** nel miocardio
2. **Valutare infarto miocardico** (acute e cronico)
3. **Caratterizzare cardiomiopatie** (dilatativa, ipertrofica)
4. **Studiare viabilità miocardica**

### Principio
- Gadolinio (contrasto paramagnetico) si accumula in tessuto danneggiato
- Acquisizione 10-20 minuti dopo iniezione
- Aree iperintense = necrosi/fibrosi

## Possibili Analisi

### Analisi Base
- Visualizzazione slice singole
- Calcolo intensità medie per camera cardiaca
- Misurazione spessore parietale

### Analisi Avanzate
- **Segmentazione automatica** del miocardio
- **Quantificazione enhancement** (% di miocardio affetto)
- **Ricostruzione 3D** dello stack short-axis
- **Analisi regionale** (modello AHA 17-segmenti)
- **Calcolo masse ventricolari**

## Formato File

I file DICOM potrebbero richiedere `force=True` in pydicom:

```python
import pydicom
dcm = pydicom.dcmread(file_path, force=True)
```

**Nota**: Alcuni file potrebbero non avere il preambolo DICOM standard.

## Utilizzo negli Script

Questi dati **non sono utilizzati** negli script base dell'Esercitazione 1:
- `calcolo_sd.py` - usa immagine sintetica
- `esempio_calcolo_sd.py` - usa `phantom.dcm`
- `test_m_sd.py` - usa simulazioni Monte Carlo

Sono disponibili per:
- Esercitazioni avanzate future
- Progetti personalizzati
- Analisi di casi clinici reali

## Notebook di Esempio

Per esplorare questi dati, puoi creare un notebook Jupyter:

```python
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to Series 115 (short axis)
series_path = Path("data/esempio_LGE/20050330 092439 [8397 - CUORE DE]/Series 115 [MR - SC 2DDel Enh 250 FX]")

# Load all slices
slices = []
for dcm_file in sorted(series_path.glob("*.dcm")):
    dcm = pydicom.dcmread(dcm_file, force=True)
    slices.append(dcm.pixel_array)

# Create 3D volume
volume = np.stack(slices, axis=0)
print(f"Volume shape: {volume.shape}")

# Display middle slice
plt.figure(figsize=(8, 8))
plt.imshow(volume[len(volume)//2], cmap='gray')
plt.title('LGE Cardiac MRI - Short Axis')
plt.axis('off')
plt.show()
```

## Privacy e Uso

⚠️ **IMPORTANTE**: Questi sono dati clinici reali (anche se anonimizzati).

- **Solo uso didattico/ricerca**
- **Non redistribuire** senza autorizzazione
- **Rispettare privacy** del paziente

## Riferimenti

### Tecniche LGE
- Kim, R.J., et al. (2000). "The use of contrast-enhanced magnetic resonance imaging to identify reversible myocardial dysfunction." NEJM
- Mahrholdt, H., et al. (2002). "Cardiovascular magnetic resonance assessment of human myocarditis." Circulation

### Segmentazione Cardiaca
- Cerqueira, M.D., et al. (2002). "Standardized myocardial segmentation and nomenclature for tomographic imaging of the heart." AHA Scientific Statement

## Contatti

Per domande sull'uso di questi dati contattare i docenti del corso.

---

**Data aggiunta**: 2025-11-10
**Fonte originale**: Esercitazione MATLAB 1 (09/03/2022)
**Formato**: DICOM (può richiedere force=True)
# Dati Esercitazione 3 - Perfusione Cardiaca MRI

## Contenuto

### 1. DICOM Perfusione (`perfusione/`)

**Descrizione**: Serie temporale di immagini MRI di perfusione cardiaca first-pass

**File**: I01, I02, ..., I79 (79 frame temporali)

**Specifiche tecniche**:
- **Dimensioni**: 256 × 256 pixel
- **Formato**: DICOM senza estensione
- **Tipo**: Immagini T1-pesate con saturazione (PREP + GRE)
- **Intervallo temporale**: ~0.8s per frame (durata ciclo cardiaco)
- **Durata totale**: ~63 secondi
- **Trigger Time**: memorizzato nel campo DICOM `TriggerTime` (in millisecondi)
- **Instance Number**: sequenza 2-80

**Fase cardiaca**: Diastolica (acquisizione sincronizzata ECG)

**Contrasto**: Gadolinio iniettato come bolo

**Dinamica visibile**:
1. **Frame 1-10**: Pre-contrasto, baseline
2. **Frame 10-15**: Passaggio ventricolo destro (RV)
3. **Frame 15-25**: Passaggio ventricolo sinistro (LV)
4. **Frame 25-40**: Perfusione miocardio
5. **Frame 40-79**: Wash-out e plateau

---

### 2. Gold Standard (`GoldStandard.mat`)

**Descrizione**: Maschere di segmentazione manuale di riferimento

**Formato**: File MATLAB (.mat)

**Contenuto**:
```python
{
    'DXmask': np.ndarray(256, 256, dtype=bool),  # Ventricolo destro
    'SXmask': np.ndarray(256, 256, dtype=bool),  # Ventricolo sinistro
    'MyoMask': np.ndarray(256, 256, dtype=bool)  # Miocardio
}
```

**Mapping nomi**:
- `DXmask` → Right Ventricle (RV, destro in italiano)
- `SXmask` → Left Ventricle (LV, sinistro in italiano)
- `MyoMask` → Myocardium (Miocardio)

**Utilizzo**:
```python
import scipy.io
from pathlib import Path

# Carica gold standard
data = scipy.io.loadmat(Path("data/GoldStandard.mat"))

rv_mask = data['DXmask'].astype(bool)
lv_mask = data['SXmask'].astype(bool)
myo_mask = data['MyoMask'].astype(bool)

# Statistiche
print(f"RV pixels: {rv_mask.sum()}")
print(f"LV pixels: {lv_mask.sum()}")
print(f"Myo pixels: {myo_mask.sum()}")
```

**Frame di riferimento**: Le maschere gold standard sono basate su un frame rappresentativo della serie (tipicamente al picco del contrasto, frame ~12-15).

---

## Caricamento Dati

### Python (usando utilities del progetto)

```python
from pathlib import Path
from src.utils import load_perfusion_series, load_gold_standard

# Carica serie DICOM
data_dir = Path("data/perfusione")
image_stack, trigger_times = load_perfusion_series(data_dir)

print(f"Shape: {image_stack.shape}")  # (256, 256, 79)
print(f"Trigger times: {trigger_times[0]:.2f} - {trigger_times[-1]:.2f} s")

# Carica gold standard
masks = load_gold_standard(Path("data/GoldStandard.mat"))
print(f"Tissues: {list(masks.keys())}")  # ['rv', 'lv', 'myo']
```

### Python (raw)

```python
import numpy as np
import pydicom
from pathlib import Path
import scipy.io

# Carica DICOM manualmente
dicom_dir = Path("data/perfusione")
dicom_files = sorted(dicom_dir.glob("I*"))

images = []
for dcm_path in dicom_files:
    dcm = pydicom.dcmread(dcm_path, force=True)
    images.append(dcm.pixel_array)

image_stack = np.stack(images, axis=2)

# Carica MAT manualmente
mat_data = scipy.io.loadmat("data/GoldStandard.mat")
rv_mask = mat_data['DXmask']
lv_mask = mat_data['SXmask']
myo_mask = mat_data['MyoMask']
```

---

## Statistiche Dataset

### Dimensioni

```
Immagini DICOM:
  - Numero file: 79
  - Dimensione singolo file: ~131 KB
  - Dimensione totale: ~10 MB
  - Dimensioni spaziali: 256×256 pixel
  - Range intensità: [0, ~600]

Gold Standard:
  - Dimensione file: 1.3 KB (compresso)
  - Dimensioni: 256×256 pixel
  - Tipo: maschere binarie
```

### Tessuti (pixel count approssimativo)

```
Background:    ~50,000 pixel
Right Ventricle:  ~1,000 pixel
Left Ventricle:   ~1,500 pixel
Myocardium:       ~3,000 pixel
Altri tessuti: ~10,000 pixel
```

---

## Note Tecniche

### Sincronizzazione ECG

Le immagini sono acquisite in fase diastolica, una per ogni battito cardiaco. Il campo `TriggerTime` nel DICOM indica il ritardo dall'onda R dell'ECG.

### Respiro Trattenuto

L'acquisizione dovrebbe essere effettuata a respiro trattenuto per evitare disallineamenti. Se presente movimento respiratorio, è necessaria registrazione post-acquisizione (non implementata in questa esercitazione).

### Sequenza MRI

- **PREP**: impulso di saturazione per reset magnetizzazione
- **GRE**: Gradient Echo veloce per acquisizione
- **T1-pesata**: maximizza contrasto Gadolinio vs tessuto

### Limitazioni Gold Standard

- Segmentazione manuale su singolo frame rappresentativo
- Soggettività dell'operatore (variabilità inter/intra-operator)
- Bordi miocardio sfumati (parziale volum effect)

---

## Riferimenti

- Gerber et al., "Myocardial First-Pass Perfusion CMR", JCMR 2008
- Documentazione PyDICOM: https://pydicom.github.io/
- Documentazione SciPy.io: https://docs.scipy.org/doc/scipy/reference/io.html

---

**Ultima revisione**: 2025-11-20
**Autore**: Bioimmagini Positano

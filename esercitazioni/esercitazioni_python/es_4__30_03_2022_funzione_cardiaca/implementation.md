# Esercitazione 4: Analisi Funzione Cardiaca - Implementazione

**Data**: 30/03/2022
**Obiettivo**: Dettagli implementativi e guida all'utilizzo per la segmentazione del ventricolo sinistro.

## Indice

1. [Pipeline di Analisi](#pipeline-di-analisi)
2. [Struttura del Codice](#struttura-del-codice)
3. [Utilizzo](#utilizzo)
4. [Dettagli Implementativi](#dettagli-implementativi)

---

## Pipeline di Analisi

### Step 1: Load 4D Dataset

Il dataset DICOM viene caricato come un volume 4D (frames, slices, height, width).

```python
volume_4d, datasets, metadata = load_cardiac_4d('data/FUNZIONE')
# Shape: (30 frames, 15 slices, 256, 256)
```

**Logica**:

1. Leggi tutti i 450 file DICOM
2. Raggruppa per `ImagePositionPatient` (Z-coordinate) → n_slices = 15
3. Ordina ogni slice per `TriggerTime` → frame sequence
4. Ricostruisci volume 4D

### Step 2: Identify Cardiac Phases

Identificazione automatica o manuale delle fasi diastolica e sistolica.

**Opzione A: Da TriggerTime** (preferita):

```python
diastolic_frame = argmin(|TriggerTime - 693 ms|)  # Frame 28
systolic_frame = argmin(|TriggerTime - 288 ms|)   # Frame 12
```

**Opzione B: Da Stima Volume**:

```python
# Stima volume da intensita' regione centrale (LV cavity)
center_intensities = [mean(volume_4d[f, 5:11, center_region]) for f in range(30)]
diastolic_frame = argmax(center_intensities)  # Max intensity = larger cavity
systolic_frame = argmin(center_intensities)   # Min intensity = smaller cavity
```

### Step 3: Segment LV Endocardium

Per ogni fase (diastole, sistole), segmentiamo le slice contenenti il ventricolo.

1. **Seleziona slices**:

   - Diastole: slices 3-14
   - Sistole: slices 4-13

2. **Inizializzazione (Seed)**:

   - Prima slice: Seed circolare al centro
   - Slices successive: Maschera della slice precedente (propagazione)

3. **Active Contours (Chan-Vese)**:
   Evoluzione del contorno per minimizzare l'energia.

4. **Refinement**:
   - Rimozione piccoli oggetti
   - Riempimento buchi (muscoli papillari)
   - Smoothing morfologico

### Step 4: Compute Volumes

Calcolo del volume fisico usando il metodo di Simpson.

```python
# Somma aree su tutte le slices
EDLV = sum(diastolic_masks) * pixel_spacing[0] * pixel_spacing[1] * slice_thickness / 1000
ESLV = sum(systolic_masks) * pixel_spacing[0] * pixel_spacing[1] * slice_thickness / 1000
```

### Step 5: Calculate Parameters

Calcolo dei parametri clinici (SV, EF, CO) e normalizzazione per BSA.

---

## Struttura del Codice

```
es_4__30_03_2022_funzione_cardiaca/
├── src/
│   ├── utils.py                        # Funzioni core (segmentazione, calcoli)
│   └── cardiac_function_analysis.py    # Script principale (pipeline)
├── data/
│   └── FUNZIONE/                       # 450 DICOM files
├── docs/                               # Documentazione
├── results/                            # Output plots + report
└── README.md                           # Landing page
```

### File: `utils.py`

Contiene le funzioni riutilizzabili:

- `load_cardiac_4d`: Caricamento e parsing DICOM
- `segment_lv_active_contour`: Wrapper per Chan-Vese
- `compute_volume_from_masks`: Calcolo volumi fisici
- `calculate_cardiac_parameters`: Calcolo SV, EF, CO, BSA

### File: `cardiac_function_analysis.py`

Script eseguibile che orchestra la pipeline:

1. Caricamento dati
2. Identificazione fasi
3. Loop di segmentazione su diastole e sistole
4. Generazione grafici e report

---

## Utilizzo

### Requisiti

```bash
pip install numpy scipy scikit-image matplotlib pydicom
```

### Esecuzione Base

```bash
cd es_4__30_03_2022_funzione_cardiaca/src
python cardiac_function_analysis.py
```

**Parametri Default**:

- `--data_dir`: `../data/FUNZIONE`
- `--output_dir`: `../results`
- Auto-detect fasi cardiache
- `--seed_radius`: 30 pixels
- `--n_iterations`: 100

### Parametri Configurabili

**Specifica fasi cardiache manualmente**:

```bash
python cardiac_function_analysis.py --diastolic_frame 28 --systolic_frame 12
```

**Modifica parametri paziente**:

```bash
python cardiac_function_analysis.py --weight 70 --height 175 --heart_rate 75
```

**Tuning Segmentazione**:

```bash
python cardiac_function_analysis.py --n_iterations 150 --smoothing 3.0
```

### Output Files

Il programma genera in `results/`:

1. **cardiac_4d_overview.png**: Montage 15 slices x 10 frames
2. **cardiac_phases_comparison.png**: Diastole vs Sistole
3. **segmentation_diastolic.png**: Segmentazione fase diastolica
4. **segmentation_systolic.png**: Segmentazione fase sistolica
5. **cardiac_volumes.png**: Grafici volumi ed EF
6. **cardiac_function_report.txt**: Report completo

---

## Dettagli Implementativi

### Segmentazione con Chan-Vese

Implementazione in `src/utils.py`:

```python
def segment_lv_active_contour(image: np.ndarray,
                              seed_mask: np.ndarray,
                              n_iterations: int = 100,
                              smoothing: float = 2.0,
                              lambda1: float = 1.0,
                              lambda2: float = 1.0) -> np.ndarray:
    """
    Segment left ventricle using Chan-Vese active contours.
    """
    # Normalize image to [0, 1]
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)

    # Apply morphological Chan-Vese
    segmentation_mask = morphological_chan_vese(
        image_norm,
        num_iter=n_iterations,
        init_level_set=seed_mask,
        smoothing=int(smoothing),
        lambda1=lambda1,
        lambda2=lambda2
    )

    return segmentation_mask.astype(np.uint8)
```

### Calcolo Volumi

Implementazione in `src/utils.py`:

```python
def compute_volume_from_masks(masks: np.ndarray,
                              pixel_spacing: Tuple[float, float],
                              slice_thickness: float) -> float:
    """
    Compute ventricular volume from segmentation masks.
    V = sum(A_i) * dx * dy * dz
    """
    # Sum areas across slices
    total_area_pixels = np.sum(masks)

    # Convert to mm^2
    dy, dx = pixel_spacing
    pixel_area_mm2 = dx * dy
    total_area_mm2 = total_area_pixels * pixel_area_mm2

    # Multiply by slice thickness to get volume in mm^3
    volume_mm3 = total_area_mm2 * slice_thickness

    # Convert to mL
    volume_ml = volume_mm3 / 1000.0

    return volume_ml
```

### Calcolo Parametri Cardiaci

Implementazione in `src/utils.py`:

```python
def calculate_cardiac_parameters(edlv: float,
                                 eslv: float,
                                 heart_rate: float,
                                 bsa: float) -> Dict[str, float]:
    # Stroke volume
    stroke_volume = edlv - eslv

    # Ejection fraction (percentage)
    ejection_fraction = (stroke_volume / edlv) * 100.0 if edlv > 0 else 0.0

    # Cardiac output (L/min)
    cardiac_output = (stroke_volume * heart_rate) / 1000.0

    # Indexed values (normalized by BSA)
    edlv_indexed = edlv / bsa if bsa > 0 else 0.0
    eslv_indexed = eslv / bsa if bsa > 0 else 0.0
    sv_indexed = stroke_volume / bsa if bsa > 0 else 0.0

    return {
        'stroke_volume': stroke_volume,
        'ejection_fraction': ejection_fraction,
        'cardiac_output': cardiac_output,
        'edlv_indexed': edlv_indexed,
        'eslv_indexed': eslv_indexed,
        'sv_indexed': sv_indexed
    }
```

### Seed Propagation Strategy

Per garantire coerenza spaziale 3D e ridurre l'intervento manuale, usiamo la maschera della slice precedente come inizializzazione per la successiva:

```python
# Snippet da cardiac_function_analysis.py
previous_mask = None

for slice_idx in slice_range:
    # ...
    if slice_idx in seed_centers:
        # User-specified seed center
        seed_mask = create_circular_seed(...)
    elif previous_mask is not None:
        # Use previous slice mask as seed (PROPAGATION)
        seed_mask = previous_mask
    else:
        # Default: center seed
        seed_mask = create_circular_seed(...)

    # Segment...
    mask = segment_lv_active_contour(image, seed_mask, ...)
    previous_mask = mask
```

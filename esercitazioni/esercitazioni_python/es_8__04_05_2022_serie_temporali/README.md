# Esercitazione 8: Registrazione Serie Temporali con Demons Algorithm

**Data**: 04/05/2022
**Obiettivo**: Registrazione non-rigida di serie temporali MRI con motion artifacts usando Demons algorithm e hierarchical clustering

## Indice

1. [Panoramica](#panoramica)
2. [Dataset](#dataset)
3. [Background Teorico](#background-teorico)
   - [Demons Algorithm](#demons-algorithm)
   - [Hierarchical Clustering](#hierarchical-clustering)
   - [Multi-Scale Registration](#multi-scale-registration)
4. [Pipeline di Registrazione](#pipeline-di-registrazione)
5. [Implementazione](#implementazione)
6. [Utilizzo](#utilizzo)
7. [Risultati Attesi](#risultati-attesi)
8. [Riferimenti](#riferimenti)

---

## Panoramica

Questa esercitazione implementa un sistema completo per la registrazione di serie temporali MRI affette da motion artifacts (principalmente respiratori). Il problema nasce dalla necessità di estrarre curve di perfusione accurate dalla corticale renale, dove il movimento respiratorio introduce artefatti che degradano la qualità delle misure dinamiche.

### Problema Clinico

Durante l'acquisizione di sequenze dinamiche di perfusione renale:
- Il paziente respira naturalmente durante i ~5-10 minuti di acquisizione
- Il movimento respiratorio sposta il rene di diversi millimetri tra i frame
- Le curve di perfusione estratte da una ROI fissa mostrano oscillazioni spurie
- Queste oscillazioni mascherano i veri pattern di enhancement del contrasto

### Soluzione Proposta

**Pipeline in due fasi**:

1. **Hierarchical Clustering**: Raggruppa immagini simili (pre/post contrasto) usando distanze MSE
2. **Demons Registration**: Allinea le immagini usando registrazione non-rigida (deformabile)

Il risultato è una serie temporale allineata che permette l'estrazione di curve di perfusione smooth e clinicamente interpretabili.

---

## Dataset

### RENAL_PERF

- **Modalità**: MRI (T1-weighted)
- **Tipo**: Serie temporale dinamica (2D+T)
- **Numero di frame**: 70 immagini
- **Dimensioni immagine**: 256x256 pixel (tipicamente)
- **Protocollo**: Perfusione renale con mezzo di contrasto gadolinio
- **Timing**:
  - Frame 1-10: Baseline (pre-contrasto)
  - Frame 11-30: Fase arteriosa (rapido enhancement)
  - Frame 31-70: Fase venosa e wash-out

### Motion Artifacts

Il movimento respiratorio causa:
- **Traslazioni**: ~5-15 mm in direzione cranio-caudale
- **Deformazioni**: Cambiamenti di forma del parenchima renale
- **Cicli respiratori**: ~12-20 respiri/minuto durante acquisizione

---

## Background Teorico

### Demons Algorithm

Il **Demons Algorithm** e' un metodo di registrazione non-rigida ispirato alla fisica termodinamica (demoni di Maxwell). Ogni pixel agisce come un "demone" che spinge l'immagine mobile verso l'immagine fissa lungo il gradiente di intensità.

#### Formulazione Matematica

Dati:
- **F(x)**: Moving image (immagine da registrare)
- **R(x)**: Reference image (immagine fissa)
- **U(x)**: Displacement field (campo di spostamento)

L'obiettivo e' trovare **U** tale che:

```
F(x + U(x)) ≈ R(x)
```

#### Regola di Aggiornamento

L'aggiornamento iterativo del displacement field e':

```
U^(n+1)(x) = U^(n)(x) + ΔU(x)
```

Dove l'incremento **ΔU** e' calcolato come:

```
ΔU(x) = (F(n)(x) - R(x)) · ∇R(x) / (||∇R(x)||² + α² · (F(n)(x) - R(x))²)
```

**Componenti**:
- **F(n)(x)**: Moving image warped con U^(n)
- **∇R(x)**: Gradiente della reference image
- **α**: Parametro di regolarizzazione (previene divisione per zero)
- **Numeratore**: Forza proporzionale alla differenza di intensità lungo il gradiente
- **Denominatore**: Normalizzazione che evita spostamenti eccessivi

#### Intuizione Fisica

L'analogia con i demoni di Maxwell:
1. Ogni pixel e' un "demone" che osserva la differenza di intensità locale
2. Il demone "spinge" il pixel lungo la direzione del gradiente
3. La forza della spinta e' proporzionale alla differenza di intensità
4. I demoni coordinano i loro spostamenti per allineare le immagini

#### Regolarizzazione: Diffusion

Dopo ogni aggiornamento, il displacement field viene smoothato con filtro Gaussiano:

```
U^(n+1)_smooth = Gaussian_σ(U^(n+1))
```

Questo garantisce:
- **Smoothness**: Campi di spostamento regolari (vincolo di fisica)
- **Topology preservation**: Nessun folding o self-intersection
- **Robustezza**: Riduzione del rumore negli spostamenti

#### Convergenza

L'algoritmo converge quando la variazione di MSE tra iterazioni e' inferiore a una tolleranza:

```
|MSE^(n+1) - MSE^(n)| < ε
```

### Hierarchical Clustering

Il **clustering gerarchico** raggruppa le immagini in base alla loro similarità, formando una struttura ad albero (dendrogram).

#### Distance Matrix

La matrice di distanze D tra immagini i e j e' calcolata con MSE:

```
D(i,j) = mean((I_i - I_j)²)
```

Per N immagini, D e' una matrice simmetrica NxN.

#### Linkage Methods

Il metodo di linkage determina come misurare la distanza tra cluster:

1. **Single linkage**: `d(C1, C2) = min{d(i,j) : i∈C1, j∈C2}`
   - Pro: Identifica catene di similarità
   - Contro: Sensibile a outliers

2. **Complete linkage**: `d(C1, C2) = max{d(i,j) : i∈C1, j∈C2}`
   - Pro: Cluster compatti
   - Contro: Sensibile a outliers

3. **Average linkage**: `d(C1, C2) = mean{d(i,j) : i∈C1, j∈C2}`
   - Pro: Bilanciato, robusto
   - **Usato in questa esercitazione**

4. **Ward**: Minimizza varianza intra-cluster
   - Pro: Cluster equilibrati
   - Contro: Assume cluster convessi

#### Dendrogram

Il dendrogram visualizza la struttura gerarchica:
- **Asse X**: Indici delle immagini
- **Asse Y**: Distanza di fusione
- **Cut height**: Determina il numero di cluster

#### Applicazione alla Perfusione

Per serie di perfusione renale:
- **Cluster 0**: Immagini pre-contrasto (intensità baseline)
- **Cluster 1**: Immagini post-contrasto (enhanced)

Questa separazione e' cruciale perche' immagini di cluster diversi hanno distribuzioni di intensità molto differenti, rendendo la registrazione diretta instabile.

### Multi-Scale Registration

La **registrazione multi-scala** (piramidale) migliora convergenza, robustezza e velocità.

#### Image Pyramid

Si costruisce una piramide di immagini con risoluzione decrescente:

```
Level 0 (fine):     256x256  (original)
Level 1:            128x128  (downsampled by 2)
Level 2 (coarse):   64x64    (downsampled by 4)
```

#### Coarse-to-Fine Strategy

La registrazione procede dal livello coarse al fine:

1. **Coarse level** (es. 64x64):
   - Cattura spostamenti grandi (macroscopici)
   - Convergenza veloce (meno pixel)
   - Meno soggetto a minimi locali

2. **Intermediate level** (es. 128x128):
   - Raffina la registrazione
   - Bilancia velocità e accuratezza

3. **Fine level** (256x256):
   - Dettagli fini e bordi
   - Registrazione precisa

#### Propagazione del Displacement Field

Il displacement field viene propagato tra livelli:

```python
# Upsample displacement field
U_fine = upsample(U_coarse, scale_factor)

# Scale displacement magnitudes
U_fine *= scale_factor

# Refine at fine level
U_fine = demons_registration(moving, fixed, initial_displacement=U_fine)
```

#### Vantaggi

1. **Capture range**: Gestisce spostamenti più grandi (fino a ~image_size/2)
2. **Convergenza**: Evita minimi locali nella funzione di costo
3. **Velocità**: Livelli coarse convergono rapidamente
4. **Robustezza**: Meno sensibile a rumore e artefatti locali

---

## Pipeline di Registrazione

La pipeline completa implementa un approccio sofisticato in 6 passi:

### Step 1: Load Temporal Series

```python
images, datasets = load_dicom_series('data/RENAL_PERF')
# Shape: (70, 256, 256)
```

### Step 2: Subset Selection (Opzionale)

Per l'esercitazione, si usa un subset di 20 immagini (come nel PDF):

```python
subset_indices = select_subset(images, n_subset=20, strategy='uniform')
images_subset = images[subset_indices]
```

Questo riduce il computational cost mantenendo la rappresentatività temporale.

### Step 3: Compute Distance Matrix

```python
distance_matrix = compute_distance_matrix(images_subset, metric='mse')
# Shape: (20, 20), symmetric
```

La matrice MSE quantifica la similarità tra tutte le coppie di immagini.

### Step 4: Hierarchical Clustering

```python
clustering_results = hierarchical_clustering(
    distance_matrix,
    n_clusters=2,
    method='average'
)
labels = clustering_results['labels']  # [0, 0, 0, ..., 1, 1, 1]
```

Output:
- **Cluster 0**: Pre-contrast images (low intensity)
- **Cluster 1**: Post-contrast images (high intensity)

### Step 5: Within-Cluster Registration

Per ogni cluster, si seleziona un'immagine di riferimento (mediana) e si registrano tutte le immagini del cluster ad essa:

```python
for cluster_id in [0, 1]:
    cluster_images = images_subset[labels == cluster_id]
    reference = cluster_images[len(cluster_images) // 2]  # Median

    for img in cluster_images:
        displacement, registered = multi_scale_demons(
            moving=img,
            fixed=reference,
            scales=[4, 2, 1],
            n_iterations=50
        )
```

**Razionale**: Immagini dello stesso cluster hanno intensità simili, rendendo la registrazione stabile.

### Step 6: Between-Cluster Registration

Si registrano i reference di ogni cluster al reference globale (Cluster 0):

```python
global_reference = images_subset[cluster_references[0]]

for cluster_id in [1, ...]:
    cluster_ref = images_subset[cluster_references[cluster_id]]

    displacement_between, _ = multi_scale_demons(
        moving=cluster_ref,
        fixed=global_reference,
        scales=[4, 2, 1],
        n_iterations=50
    )
```

**Razionale**: Allinea i cluster tra loro, permettendo un sistema di coordinate comune.

### Step 7: Compose Displacements

Per ogni immagine, il displacement field totale e':

```
U_total(i) = U_within_cluster(i) + U_between_cluster(cluster_of_i)
```

Questa e' un'approssimazione del composition. Una composizione esatta richiederebbe:

```
x' = x + U_within(x)
x'' = x' + U_between(x')
```

Ma l'approssimazione additiva e' sufficiente per piccoli displacement.

### Step 8: Apply Displacements

```python
for i in range(n_frames):
    registered_series[i] = warp_image(images[i], U_total[i])
```

### Step 9: Extract Perfusion Curves

```python
curve_before = extract_perfusion_curve(images_subset, roi_coords=(y1, y2, x1, x2))
curve_after = extract_perfusion_curve(registered_series, roi_coords=(y1, y2, x1, x2))
```

---

## Implementazione

### Struttura del Codice

```
es_8__04_05_2022_serie_temporali/
├── src/
│   ├── utils.py                    # Core functions (Demons, clustering)
│   └── temporal_registration.py    # Main pipeline script
├── data/
│   └── RENAL_PERF/                # DICOM temporal series (70 frames)
├── docs/
│   ├── Esercitazione_08_04_05_2022.pdf
│   └── testDaemons.m              # MATLAB reference
├── results/                       # Output visualizations
├── notebooks/                     # Jupyter analysis (optional)
├── tests/                         # Unit tests (optional)
└── README.md                      # This file
```

### File: `utils.py`

Contiene le funzioni principali:

1. **I/O Functions**:
   - `load_dicom_series()`: Carica serie DICOM temporali
   - `normalize_image()`: Normalizzazione [0,1] con percentile clipping

2. **Clustering Functions**:
   - `compute_distance_matrix()`: Matrice MSE tra immagini
   - `hierarchical_clustering()`: Clustering gerarchico con linkage

3. **Demons Algorithm**:
   - `compute_image_gradient()`: Gradiente con Gaussian smoothing
   - `demons_step()`: Singola iterazione Demons
   - `warp_image()`: Warp con bilinear interpolation
   - `demons_registration()`: Registrazione Demons completa
   - `multi_scale_demons()`: Approccio piramidale multi-scala

4. **Analysis Functions**:
   - `apply_displacement_to_series()`: Applica displacement a serie
   - `extract_perfusion_curve()`: Estrae curva di perfusione da ROI

### File: `temporal_registration.py`

Script principale con:
- Argparse CLI per parametri configurabili
- Pipeline completa (step 1-9)
- Visualizzazioni:
  - Dendrogram (clustering gerarchico)
  - Cluster assignments (sample images per cluster)
  - Registration comparisons (before/after, difference, checkerboard)
  - Perfusion curves (before/after registration)
- Statistiche di valutazione (MSE, smoothness)

---

## Utilizzo

### Requisiti

```bash
pip install numpy scipy scikit-image matplotlib pydicom
```

### Esecuzione Base

```bash
cd es_8__04_05_2022_serie_temporali/src
python temporal_registration.py
```

**Default parameters**:
- `--data_dir`: `../data/RENAL_PERF`
- `--output_dir`: `../results`
- `--n_subset`: 20 (immagini da usare)
- `--n_clusters`: 2
- `--n_iterations`: 50 (per scala)
- `--alpha`: 2.5
- `--sigma_diffusion`: 1.0
- Multi-scale: Enabled (scales=[4, 2, 1])

### Parametri Avanzati

**Usare serie completa (70 frames)**:
```bash
python temporal_registration.py --n_subset 0
```

**Specificare ROI per perfusion curve**:
```bash
python temporal_registration.py --roi 100 150 120 170
# ROI: y=[100,150], x=[120,170]
```

**Disabilitare multi-scale** (single-scale registration):
```bash
python temporal_registration.py --no_multiscale
```

**Aumentare iterazioni per convergenza migliore**:
```bash
python temporal_registration.py --n_iterations 100
```

**Regolarizzazione più forte** (displacement più smooth):
```bash
python temporal_registration.py --alpha 5.0 --sigma_diffusion 2.0
```

**3 cluster** (es. pre-contrast, arterial, venous):
```bash
python temporal_registration.py --n_clusters 3
```

### Output Files

Il programma genera in `results/`:

1. **dendrogram.png**: Dendrogram del clustering gerarchico
2. **cluster_assignment.png**: Visualizzazione dei cluster con sample images
3. **Within_Cluster_X_Registration.png**: Esempi di registrazione intra-cluster
4. **Between_Clusters_X_to_0_Registration.png**: Registrazione inter-cluster
5. **perfusion_curves.png**: Curve di perfusione prima/dopo registrazione

### Esempio Output Console

```
======================================================================
TEMPORAL SERIES REGISTRATION WITH DEMONS ALGORITHM
======================================================================

Loading DICOM series from ../data/RENAL_PERF...
Loaded 70 frames of size 256x256

Selecting subset of 20 images (uniformly spaced)...
Selected indices: [ 0  3  7 10 14 17 21 24 28 31 35 38 42 45 49 52 56 59 63 66]

Extracting perfusion curve before registration...
  Using whole image (no ROI specified)

=== TEMPORAL SERIES REGISTRATION ===
Number of frames: 20
Image shape: 256 x 256

Step 1: Computing distance matrix (MSE metric)...

Step 2: Hierarchical clustering (n_clusters=2)...
  Cluster 0: 10 images (indices: [0 3 7 10 14 ...]...)
  Cluster 1: 10 images (indices: [21 24 28 31 35 ...]...)

Step 3: Selecting reference images for each cluster...
  Cluster 0 reference: Image 10
  Cluster 1 reference: Image 31

Step 4: Registering images within each cluster...

  Cluster 0:

Scale 1/3: downsampling factor = 4
  Iteration 10/50: MSE = 0.002453, Change = 0.000123
  ...
  Converged at iteration 35

Scale 2/3: downsampling factor = 2
  ...

Scale 3/3: downsampling factor = 1
  ...

...

Step 5: Registering between clusters...

  Registering Cluster 1 reference to Cluster 0 reference...
  ...

Step 6: Applying combined displacement fields...

Registration complete!

Extracting perfusion curve after registration...

=== PERFUSION CURVE STATISTICS ===
Before registration:
  Mean: 145.32, Std: 52.18
  Range: [89.45, 254.67]

After registration:
  Mean: 145.28, Std: 52.21
  Range: [89.52, 254.63]

Curve smoothness (variance of derivative):
  Before: 124.5673
  After: 45.2341
  Improvement: 63.7%

=== RESULTS SAVED TO ../results ===
...

Registration pipeline completed successfully!
```

### Interpretazione Risultati

#### Perfusion Curves

**Before Registration**:
- Oscillazioni spurie dovute a respiratory motion
- Alta variance of derivative (non-smooth)
- Difficile identificare fasi di enhancement

**After Registration**:
- Curva smooth e interpretabile
- Riduzione variance of derivative (>50% tipicamente)
- Fasi di perfusione chiaramente visibili:
  - Baseline flat
  - Rapid arterial enhancement
  - Gradual venous wash-out

#### Checkerboard Overlay

Il checkerboard pattern permette di valutare l'allineamento:
- **Before**: Edges misaligned, "double contours"
- **After**: Smooth transitions, good alignment

#### MSE Reduction

La riduzione di MSE tra moving e fixed indica:
- MSE reduction >50%: Excellent registration
- MSE reduction 30-50%: Good registration
- MSE reduction <30%: Check parameters or image quality

---

## Risultati Attesi

### Quantitativi

1. **MSE Reduction**: 40-60% per within-cluster registration
2. **Perfusion Curve Smoothness**: Improvement 50-70%
3. **Cluster Separation**: Clear separation of pre/post contrast (MSE ratio ~5-10x)

### Qualitativi

1. **Dendrogram**: Two main branches (pre/post contrast)
2. **Cluster Assignment**: Temporal continuity within clusters
3. **Registration**: Aligned kidney contours in checkerboard
4. **Perfusion Curves**: Smooth, physiologically plausible curves

### Parametri Ottimali

Based on empirical tuning:

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `alpha` | 2.5 | 1.0-5.0 | Higher = smoother but slower convergence |
| `sigma_diffusion` | 1.0 | 0.5-2.0 | Higher = smoother displacement field |
| `n_iterations` | 50 | 30-100 | Per scale; 50 usually sufficient |
| `scales` | [4,2,1] | [8,4,2,1] | More scales = better for large motion |
| `n_clusters` | 2 | 2-4 | 2 for simple pre/post, 3+ for arterial/venous |

### Common Issues

**Issue 1: Poor registration (high MSE after)**
- **Causa**: Alpha troppo basso, iterazioni insufficienti
- **Soluzione**: Aumentare `alpha` a 3-5, `n_iterations` a 100

**Issue 2: Over-smoothing (loss of detail)**
- **Causa**: `sigma_diffusion` troppo alto
- **Soluzione**: Ridurre `sigma_diffusion` a 0.5-0.8

**Issue 3: Cluster non separano pre/post**
- **Causa**: Subset non rappresentativo
- **Soluzione**: Aumentare `n_subset` o usare serie completa

**Issue 4: Slow convergence**
- **Causa**: Multi-scale disabilitato o scale insufficienti
- **Soluzione**: Abilitare multi-scale con scales=[8,4,2,1]

---

## Riferimenti

### Papers

1. **Thirion, J.P. (1998)**
   *"Image Matching as a Diffusion Process: An Analogy with Maxwell's Demons"*
   Medical Image Analysis, 2(3):243-260
   DOI: 10.1016/S1361-8415(98)80022-4
   **Nota**: Paper originale del Demons algorithm

2. **Vercauteren, T. et al. (2009)**
   *"Diffeomorphic demons: Efficient non-parametric image registration"*
   NeuroImage, 45(1):S61-S72
   DOI: 10.1016/j.neuroimage.2008.10.040
   **Nota**: Estensione diffeomorfica (preserva topologia)

3. **Wang, H. et al. (2005)**
   *"Validation of an accelerated 'demons' algorithm for deformable image registration in radiation therapy"*
   Physics in Medicine and Biology, 50(12):2887
   **Nota**: Validazione clinica in radioterapia

4. **Kroon, D.J. et al. (2009)**
   *"MRIModalitiy Transformation in Demon Registration"*
   IEEE International Symposium on Biomedical Imaging
   **Nota**: Demons per multi-modalità (T1/T2)

### Textbooks

5. **Modersitzki, J. (2009)**
   *"FAIR: Flexible Algorithms for Image Registration"*
   SIAM, Philadelphia
   **Nota**: Trattazione completa di registrazione, include Demons

6. **Goshtasby, A.A. (2012)**
   *"Image Registration: Principles, Tools and Methods"*
   Springer
   **Nota**: Overview di metodi rigid e non-rigid

### Software & Tools

7. **SimpleITK**
   https://simpleitk.org/
   Implementa Demons registration (C++/Python bindings)

8. **ANTs (Advanced Normalization Tools)**
   http://stnava.github.io/ANTs/
   Suite completa per registrazione (include SyN, evoluzione di Demons)

9. **scikit-image**
   https://scikit-image.org/
   Libreria Python per image processing (usata in questa implementazione)

### Clinical Applications

10. **Renal Perfusion MRI**
    - **Annet, L. et al. (2004)**: *"Glomerular filtration rate: assessment with dynamic contrast-enhanced MRI"*
    - **Sourbron, S. et al. (2008)**: *"Quantification of renal perfusion and function"*

11. **Motion Correction in MRI**
    - **Melbourne, A. et al. (2007)**: *"Registration of dynamic contrast-enhanced MRI"*
    - **Buonaccorsi, G.A. et al. (2007)**: *"Tracer kinetic model-driven registration"*

---

## Appendice: Dettagli Implementativi

### Numerical Stability

**Issue**: Division by zero nel denominatore di Demons update

**Solution**: Aggiungere epsilon nel denominatore

```python
denominator = grad_magnitude_sq + alpha**2 * diff**2
denominator = np.maximum(denominator, 1e-10)  # Prevent division by zero
```

### Bilinear Interpolation

Il warping usa `scipy.ndimage.map_coordinates` con `order=1` (bilinear):

```python
warped = ndimage.map_coordinates(image, coords, order=1, mode='nearest')
```

**Alternatives**:
- `order=0`: Nearest neighbor (veloce, blocky)
- `order=3`: Cubic (smooth, più lento)

### Displacement Field Storage

Displacement field shape: `(2, height, width)`
- `displacement[0]`: y-displacement (vertical)
- `displacement[1]`: x-displacement (horizontal)

Convention: `new_position = old_position + displacement`

### Memory Optimization

Per serie grandi (>100 frames):
- Processare a batch per evitare OOM
- Usare `dtype=np.float32` invece di `float64`
- Deallocare array intermedi con `del`

```python
# Process in batches
batch_size = 10
for i in range(0, n_frames, batch_size):
    batch = images[i:i+batch_size]
    registered_batch = process_batch(batch)
    # Save to disk immediately
    save_batch(registered_batch, i)
    del batch, registered_batch  # Free memory
```

---

## Note Finali

Questa implementazione fornisce:
- ✅ Algoritmo Demons completo con multi-scale
- ✅ Hierarchical clustering per serie eterogenee
- ✅ Pipeline end-to-end automatica
- ✅ Visualizzazioni comprehensive
- ✅ CLI flessibile con parametri configurabili

Per domande o issues, consultare il codice sorgente commentato in `src/utils.py` e `src/temporal_registration.py`.

**Autori**: Corso di Biomedical Imaging
**Data**: 04/05/2022 (Conversion to Python: 2025)

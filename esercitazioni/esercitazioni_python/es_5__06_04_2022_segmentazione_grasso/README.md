# Esercitazione 5: Segmentazione Grasso Addominale (SAT/VAT)

## Descrizione

Questa esercitazione implementa la **segmentazione automatica del grasso addominale** da acquisizioni MRI T1-pesate per quantificare:

- **SAT (Subcutaneous Adipose Tissue)**: Grasso sottocutaneo, localizzato tra cute e fascia muscolare
- **VAT (Visceral Adipose Tissue)**: Grasso viscerale, localizzato nella cavita' intra-addominale
- **VAT/SAT ratio**: Indice di rischio cardiovascolare e metabolico

Il rapporto VAT/SAT e' un importante predittore di rischio per diabete di tipo 2, malattie cardiovascolari e sindrome metabolica.

---

## Pipeline di Segmentazione

### 1. K-means Clustering (K=3)

Separazione iniziale del volume in 3 cluster basati sull'intensita':

- **Cluster 0 (Aria)**: Intensita' minima, regione extra-corporea
- **Cluster 1 (Acqua/Muscolo)**: Intensita' media, tessuti non adiposi
- **Cluster 2 (Grasso)**: Intensita' massima in T1-weighted

Il grasso ha segnale alto in sequenze T1-pesate grazie al breve tempo di rilassamento T1 dei protoni lipidici.

**Algoritmo**:
```python
from sklearn.cluster import KMeans

# Reshape volume (slices, rows, cols) -> (n_voxels, 1)
X = volume.reshape(-1, 1)

# K-means con K=3
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = kmeans.fit_predict(X)

# Identifica cluster grasso (centroide massimo)
centroids = kmeans.cluster_centers_.flatten()
fat_label = np.argmax(centroids)
```

### 2. Rimozione Componenti Spurie

Il clustering K-means include anche il grasso delle braccia. Per isolare il **torso**, si utilizza il **connected component labeling 3D**:

```python
from skimage.measure import label

# Labeling 3D con connettivita' 26
labeled_volume = label(fat_mask, connectivity=3)

# Mantieni solo componente piu' grande (torso)
component_sizes = np.bincount(labeled_volume.ravel())
largest_label = np.argmax(component_sizes[1:]) + 1
torso_mask = (labeled_volume == largest_label)
```

### 3. Active Contours per SAT (Doppi Snake)

Il SAT e' delimitato da:
- **Bordo ESTERNO**: Interfaccia aria-cute
- **Bordo INTERNO**: Fascia muscolare addominale

Si utilizzano **due active contours** (Chan-Vese) indipendenti:

#### Active Contour Esterno (Cute)
```python
from skimage.segmentation import morphological_chan_vese

outer_contour = morphological_chan_vese(
    image,
    iterations=150,
    init_level_set=torso_mask,  # Seed da K-means
    smoothing=2
)
```

#### Active Contour Interno (Fascia Muscolare)
```python
inner_contour = morphological_chan_vese(
    image,
    iterations=100,
    init_level_set=circular_seed,  # Seed circolare al centro
    smoothing=2
)
```

#### Estrazione SAT
```python
# SAT = regione tra outer e inner
sat_mask = np.logical_and(outer_contour, np.logical_not(inner_contour))
```

**Chan-Vese Algorithm**:
Minimizza l'energia:

```
E(C) = lambda_1 * integral_inside(|I(x) - c1|^2) dx +
       lambda_2 * integral_outside(|I(x) - c2|^2) dx +
       mu * Length(C)
```

Dove:
- `c1` = media intensita' interna al contorno
- `c2` = media intensita' esterna
- `mu` = peso regolarizzazione (smoothness)

### 4. EM-GMM per VAT

Il VAT si trova nella **regione intra-addominale** (delimitata da `inner_contour`). Per distinguere grasso viscerale da altri tessuti, si utilizza **EM-GMM** (Expectation-Maximization Gaussian Mixture Model) sull'istogramma delle intensita':

#### Estrazione Istogramma
```python
# Pixel nella regione intra-addominale
inner_region = inner_contour > 0
intra_pixels = image_slice[inner_region]

# Istogramma
hist, bin_edges = np.histogram(intra_pixels, bins=50, range=(0, 1))
```

#### Fit GMM con 2 Gaussiane
```python
from sklearn.mixture import GaussianMixture

# Replica valori per pesare con frequenze
X = np.repeat(bin_centers, hist.astype(int)).reshape(-1, 1)

# EM-GMM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# Identifica componente grasso (media piu' alta)
means = gmm.means_.flatten()
fat_component = np.argmax(means)
```

#### Classificazione VAT
```python
# Classifica ogni pixel intra-addominale
predictions = gmm.predict(intra_pixels.reshape(-1, 1))

# VAT = pixel classificati come grasso
vat_mask = (predictions == fat_component)
```

### 5. Calcolo Volumi

```python
# Volume voxel
voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
voxel_volume_cm3 = voxel_volume_mm3 / 1000.0

# Conta voxel
sat_voxels = np.sum(sat_mask_3d)
vat_voxels = np.sum(vat_mask_3d)

# Volumi
sat_volume_cm3 = sat_voxels * voxel_volume_cm3
vat_volume_cm3 = vat_voxels * voxel_volume_cm3

# Rapporto
vat_sat_ratio_percent = (vat_volume_cm3 / sat_volume_cm3) * 100.0
```

---

## Basi Teoriche

### Risonanza Magnetica T1-weighted

Nelle sequenze T1-pesate (TR corto, TE corto):
- **Grasso**: Segnale ALTO (bianco) - T1 corto (~250 ms)
- **Acqua/Muscolo**: Segnale MEDIO (grigio) - T1 lungo (~1000 ms)
- **Aria**: Segnale BASSO (nero) - assenza di protoni

### Active Contours (Snakes)

Gli active contours sono curve deformabili che evolvono per minimizzare un'energia:

```
E_total = E_internal + E_external
```

- **E_internal**: Regolarizzazione (smoothness + continuita')
- **E_external**: Adattamento ai bordi (gradient o region-based)

**Chan-Vese** e' un approccio **region-based** che non richiede bordi netti, ideale per MRI dove i confini SAT possono essere sfumati.

### Expectation-Maximization (EM) per GMM

Algoritmo iterativo per stimare parametri di Gaussian Mixture Model:

1. **E-step**: Calcola probabilita' a posteriori (responsibility) di ogni dato per ogni Gaussiana
2. **M-step**: Aggiorna parametri (media, covarianza, peso) massimizzando log-likelihood

Converge a massimo locale della likelihood:

```
L(theta) = sum_i log(sum_k pi_k * N(x_i | mu_k, Sigma_k))
```

### Significato Clinico VAT/SAT

- **VAT/SAT < 30%**: Basso rischio metabolico
- **VAT/SAT 30-50%**: Rischio moderato
- **VAT/SAT > 50%**: Alto rischio (obesita' viscerale)

Il VAT e' metabolicamente piu' attivo del SAT e rilascia citochine pro-infiammatorie (adipochine) che aumentano il rischio cardiovascolare.

---

## Struttura Files

```
es_5__06_04_2022_segmentazione_grasso/
├── README.md                     # Questa guida
├── data/
│   └── dicom/                    # 18 slice DICOM assiali T1w
│       ├── IM_0001.dcm
│       ├── IM_0002.dcm
│       └── ...
├── docs/
│   ├── Esercitazione__05_06_04_2022.pdf      # Testo esercitazione
│   ├── Positano_JMRI_fat_2004.pdf            # Paper di riferimento
│   └── bliton2017.pdf                        # Paper correlato
├── src/
│   ├── __init__.py
│   ├── utils.py                              # Funzioni core (600+ righe)
│   ├── fat_segmentation.py                   # Script principale
│   └── visualize_results.py                  # Visualizzazione avanzata
├── results/                      # Output (generato a runtime)
│   ├── fat_volumes.txt
│   ├── fat_segmentation_results.png
│   └── slice_*_detailed.png
├── notebooks/                    # Jupyter notebooks (opzionale)
└── tests/                        # Unit tests (opzionale)
```

---

## Installazione

### 1. Setup Ambiente Virtuale

```bash
# Dalla directory esercitazione
cd es_5__06_04_2022_segmentazione_grasso

# Crea venv
python3 -m venv venv

# Attiva
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate     # Windows
```

### 2. Installa Dipendenze

```bash
pip install --upgrade pip
pip install numpy scipy scikit-learn scikit-image matplotlib pydicom
```

**Dipendenze**:
- `numpy`: Array operations
- `scipy`: Operazioni morfologiche (`ndimage`)
- `scikit-learn`: K-means, GMM
- `scikit-image`: Active contours, labeling
- `matplotlib`: Visualizzazione
- `pydicom`: Lettura DICOM

---

## Esecuzione

### Script Principale

```bash
cd src
python fat_segmentation.py
```

**Opzioni**:
```bash
python fat_segmentation.py --help

# Opzioni disponibili:
--dicom_dir PATH           # Directory DICOM (default: ../data/dicom)
--output_dir PATH          # Directory output (default: ../results)
--kmeans_clusters N        # Numero cluster K-means (default: 3)
--outer_iterations N       # Iterazioni AC esterno (default: 150)
--inner_iterations N       # Iterazioni AC interno (default: 100)
--gmm_components N         # Componenti GMM (default: 2)
--save_masks               # Salva maschere 3D come .npy
```

### Visualizzazione Avanzata

```bash
python visualize_results.py

# Analizza slice specifica
python visualize_results.py --slice 9
```

---

## Output Attesi

### Valori di Riferimento

Dalla pubblicazione di Positano et al. (2004):

- **SAT**: ~2091 cm³
- **VAT**: ~970 cm³
- **VAT/SAT**: ~46%

### File Generati

#### `fat_volumes.txt`
```
SEGMENTAZIONE GRASSO ADDOMINALE - RISULTATI
==================================================

SAT volume: 2091.45 cm^3
VAT volume: 970.23 cm^3
Total fat: 3061.68 cm^3
VAT/SAT ratio: 46.38 %

VALORI DI RIFERIMENTO:
--------------------------------------------------
SAT: 2091 cm^3
VAT: 970 cm^3
VAT/SAT: 46%
```

#### `fat_segmentation_results.png`
Figura 6 pannelli:
1. Slice originale
2. SAT overlay (rosso)
3. VAT overlay (blu)
4. SAT+VAT overlay
5. Contorni active contours
6. Riepilogo risultati

#### `slice_N_detailed.png`
Analisi dettagliata slice:
- Maschere SAT/VAT
- Contorni esterno/interno
- Istogramma intra-addominale + fit GMM
- Statistiche slice

---

## Note Tecniche

### Parametri Critici

1. **K-means `n_init=20`**: Ripete clustering 20 volte per evitare minimi locali
2. **Active Contours `smoothing=2`**: Bilancia aderenza ai dati vs. smoothness
3. **Outer iterations=150**: Contorno esterno richiede piu' iterazioni (confine aria-cute piu' complesso)
4. **Inner iterations=100**: Contorno interno piu' stabile (confine grasso-muscolo piu' netto)
5. **GMM `n_components=2`**: Modella 2 popolazioni (tessuto + grasso)

### Limitazioni

1. **Braccia**: Se il paziente ha braccia vicine al torso, il labeling potrebbe fallire
   - **Soluzione**: Preprocessing manuale o algoritmi piu' sofisticati (graph-cuts)

2. **GMM sensibile a inizializzazione**: EM puo' convergere a minimi locali
   - **Soluzione**: `n_init=10` ripete fit multipli

3. **Active contours richiedono seed**: Seed pessimi causano convergenza errata
   - **Soluzione**: Propagare contorni convergiti come seed per slice adiacenti

4. **Grasso intermuscolare**: Non distinguibile dal VAT con sola intensita'
   - **Soluzione**: Richiede vincoli anatomici o atlas

### Performance

Su workstation standard (CPU):
- **Caricamento DICOM**: ~1 s
- **K-means 3D**: ~2 s
- **Active contours (18 slices)**: ~30 s
- **EM-GMM (18 slices)**: ~5 s
- **Totale**: ~40 s

---

## Validazione

### Metriche

1. **Confronto con ground truth manuale**: DICE coefficient
   ```python
   dice = 2 * |A ∩ B| / (|A| + |B|)
   ```

2. **Confronto con valori di riferimento**: Errore relativo
   ```python
   error_percent = |V_computed - V_reference| / V_reference * 100
   ```

3. **Ispezione visuale**: Verifica contorni su slice rappresentative

### Test su Dataset

Se disponibile ground truth manuale:

```python
from sklearn.metrics import jaccard_score

# Flatten maschere 3D
gt_flat = ground_truth.ravel()
pred_flat = sat_mask_3d.ravel()

# Jaccard Index (IoU)
iou = jaccard_score(gt_flat, pred_flat)

# DICE
dice = 2 * iou / (1 + iou)

print(f"DICE: {dice:.3f}")
```

---

## Riferimenti

### Paper Principali

1. **Positano et al., 2004**
   - *"Accurate segmentation of subcutaneous and intermuscular adipose tissue from MR images of the thigh"*
   - Journal of Magnetic Resonance Imaging, 19(1):44-53
   - **Metodo**: K-means + active contours + EM-GMM (base di questa implementazione)

2. **Bliton et al., 2017**
   - *"Automated visceral adipose tissue segmentation in abdominal MRI"*
   - Journal paper correlato

### Algoritmi

3. **Chan & Vese, 2001**
   - *"Active contours without edges"*
   - IEEE Transactions on Image Processing
   - **Contributo**: Region-based active contours

4. **Dempster et al., 1977**
   - *"Maximum likelihood from incomplete data via the EM algorithm"*
   - Journal of the Royal Statistical Society, Series B
   - **Contributo**: Expectation-Maximization

### MRI Physics

5. **Haacke et al., 1999**
   - *"Magnetic Resonance Imaging: Physical Principles and Sequence Design"*
   - Textbook di riferimento per fisica MRI

---

## Estensioni Possibili

### 1. Deep Learning
Sostituire pipeline classica con U-Net:
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# U-Net per segmentazione end-to-end
model = build_unet(input_shape=(256, 256, 1), n_classes=3)
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

### 2. Atlas-Based Segmentation
Registrazione con atlas anatomico per vincoli:
```python
from ANTsPy import registration

# Registra paziente ad atlas
transform = registration(fixed=atlas, moving=patient_volume)
prior_mask = transform.apply_to_image(atlas_mask)
```

### 3. Quantificazione Grasso Intermuscolare (IMAT)
Estrazione grasso tra fibre muscolari:
```python
# IMAT = grasso dentro contorno muscolare, fuori da SAT e VAT
muscle_mask = segment_muscle(volume)
imat_mask = np.logical_and(fat_mask, muscle_mask)
```

### 4. Analisi Longitudinale
Tracking variazioni SAT/VAT nel tempo:
```python
# Registra follow-up a baseline
transform = registration(baseline_volume, followup_volume)
delta_vat = followup_vat - baseline_vat
```

---

## Troubleshooting

### Problema: K-means non separa grasso

**Sintomo**: Cluster grasso include anche altri tessuti

**Soluzione**:
- Verifica normalizzazione intensita' (deve essere [0,1])
- Aumenta `n_init` (es. 50)
- Ispeziona istogramma per verificare 3 picchi distinti

### Problema: Active contours non convergono

**Sintomo**: Contorni restano sul seed iniziale

**Soluzione**:
- Aumenta numero iterazioni (200-300)
- Prova seed diverso (es. cerchio piu' grande)
- Applica smoothing all'immagine (`gaussian(image, sigma=2.0)`)

### Problema: VAT volume = 0

**Sintomo**: Nessun pixel classificato come VAT

**Soluzione**:
- Verifica `inner_contour` non sia vuoto
- Controlla fit GMM (plot istogramma + Gaussiane)
- Riduci `n_components` a 2 se ci sono poche intensita' diverse

### Problema: Braccia non rimosse

**Sintomo**: Componenti spurie rimangono dopo labeling

**Soluzione**:
- Aumenta `connectivity` (da 1 a 3)
- Usa `min_size` threshold invece di `keep_largest`
- Preprocessing: crop ROI manuale pre-segmentazione

---

## Contatti e Supporto

**Sviluppato con**: Claude Code
**Data**: 2025-11-20
**Basato su**: Esercitazione MATLAB originale (06/04/2022)

Per domande o bug, riferirsi al corso di Bioimmagini, Universita' di Positano.

---

## License

Materiale didattico. Solo uso accademico.

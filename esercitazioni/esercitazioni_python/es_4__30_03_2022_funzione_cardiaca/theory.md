# Esercitazione 4: Analisi Funzione Cardiaca - Teoria

**Data**: 30/03/2022
**Obiettivo**: Segmentazione del ventricolo sinistro e calcolo parametri di funzione cardiaca usando Active Contours (Chan-Vese) su immagini MRI cardiache cine

## Indice

1. [Panoramica](#panoramica)
2. [Dataset](#dataset)
3. [Background Teorico](#background-teorico)
   - [Funzione Cardiaca MRI](#funzione-cardiaca-mri)
   - [Active Contours (Chan-Vese)](#active-contours-chan-vese)
   - [Parametri Cardiaci](#parametri-cardiaci)
4. [Risultati Attesi](#risultati-attesi)
5. [Riferimenti](#riferimenti)

---

## Panoramica

Questa esercitazione implementa una pipeline completa per l'analisi della funzione ventricolare sinistra da immagini MRI cardiache cine. L'obiettivo è calcolare i parametri funzionali standard:

- **EDLV** (End-Diastolic Left Ventricular Volume): Volume telediastolico
- **ESLV** (End-Systolic Left Ventricular Volume): Volume telesistolico
- **SV** (Stroke Volume): Volume di eiezione
- **EF** (Ejection Fraction): Frazione di eiezione
- **CO** (Cardiac Output): Gittata cardiaca

### Problema Clinico

L'analisi della funzione ventricolare e' fondamentale per:

- **Cardiomiopatie**: Valutazione insufficienza cardiaca
- **Cardiopatie congenite**: Follow-up post-chirurgico
- **Cardiotossicita'**: Monitoraggio chemioterapia
- **Valvulopatie**: Assessment pre/post-intervento

---

## Dataset

### FUNZIONE (Cardiac Cine MRI)

- **Modalita'**: MRI T1-weighted (SSFP - Steady-State Free Precession)
- **Tipo**: 4D cine (3D+T)
- **Numero immagini**: 450 DICOM (15 slices x 30 temporal frames)
- **Dimensioni slice**: Tipicamente 256x256 pixel
- **Pixel spacing**: ~1.4 x 1.4 mm
- **Slice thickness**: 6-8 mm (linee guida: ≤10 mm)
- **Inter-slice distance**: 10 mm (o meno, secondo linee guida)
- **Temporal resolution**: ~45 ms (30 frames sul ciclo cardiaco)
- **View**: Short-axis (asse corto)

### Organizzazione Dati

Le 450 immagini DICOM sono organizzate come:

```
15 slices  x  30 frames  =  450 images
```

**Identificazione fase temporale**:

- **TriggerTime**: Timestamp dall'ECG trigger (ms)
- **CardiacNumberOfImages**: 30 (numero di frames per ciclo)
- **ImagePositionPatient**: Posizione 3D (identifica la slice)

**Fasi cardiache** (dal referto):

- **Diastole**: Frame 28 (693 ms) - Massimo volume (rilassamento)
- **Sistole**: Frame 12 (288 ms) - Minimo volume (contrazione)

### Slices Ventricolari

Non tutte le 15 slices contengono il ventricolo sinistro:

- **Slices 1-2**: Atrio sinistro (sopra il ventricolo)
- **Slices 3-14**: Ventricolo sinistro (DIASTOLE)
- **Slices 4-13**: Ventricolo sinistro (SISTOLE, cuore accorciato)
- **Slices 15+**: Al di sotto del ventricolo

---

## Background Teorico

### Funzione Cardiaca MRI

#### Ciclo Cardiaco

Il ciclo cardiaco comprende due fasi principali:

**1. Diastole (Rilassamento)**:

- Ventricolo si riempie di sangue
- Valvola mitrale aperta, valvola aortica chiusa
- Volume massimo (EDLV)
- Pressione minima

**2. Sistole (Contrazione)**:

- Ventricolo espelle sangue nell'aorta
- Valvola mitrale chiusa, valvola aortica aperta
- Volume minimo (ESLV)
- Pressione massima

#### Acquisizione MRI Cine

**SSFP (Steady-State Free Precession)**:

- Alta SNR (Signal-to-Noise Ratio)
- Buon contrasto sangue/miocardio
- Acquisizione rapida (breath-hold)

**ECG-gating**:

- Sincronizzazione con ECG per trigger
- 25-30 fasi per ciclo cardiaco
- Risoluzione temporale ~30-50 ms

**Short-axis view**:

- Perpendicolare all'asse lungo cardiaco
- Slice stack copre tutto il ventricolo
- Ottimale per calcolo volumi (metodo Simpson)

#### Linee Guida Cliniche

Secondo raccomandazioni SCMR (Society for Cardiovascular Magnetic Resonance):

- **Slice thickness**: 6-8 mm (max 10 mm)
- **Slice gap**: ≤4 mm (idealmente 0 mm)
- **Temporal resolution**: ≤45 ms (≥25 fps)
- **In-plane resolution**: ~1.5-2.0 mm

### Active Contours (Chan-Vese)

Gli **Active Contours** (contorni attivi o "snakes") sono curve deformabili che evolvono verso i bordi degli oggetti, guidate da forze interne (smoothness) ed esterne (image features).

#### Formulazione Classica (Snakes)

**Kass, Witkin, Terzopoulos (1988)**:

Minimizza energia:

```
E = E_internal + E_external

E_internal = ∫ (α|v'(s)|² + β|v''(s)|²) ds   (smoothness)
E_external = ∫ -|∇I(v(s))|² ds              (edge attraction)
```

Dove v(s) e' la curva parametrizzata, α controlla tensione, β controlla rigidita'.

**Limitazioni**:

- Sensibile all'inizializzazione
- Difficolta' con topologia variabile (splitting/merging)
- Richiede edge forti

#### Chan-Vese Model (2001)

Il **Chan-Vese model** e' un active contour **region-based** (non edge-based), piu' robusto e flessibile.

**Formulazione**:

Minimizza energia di Mumford-Shah semplificata:

```
E(C, c1, c2) = λ1 ∫_inside(C) |I(x) - c1|² dx
             + λ2 ∫_outside(C) |I(x) - c2|² dx
             + μ · Length(C)
             + ν · Area(inside(C))
```

**Dove**:

- **C**: Contorno (curva chiusa)
- **c1**: Intensita' media dentro il contorno
- **c2**: Intensita' media fuori il contorno
- **λ1, λ2**: Pesi per fitting interno/esterno (tipicamente 1.0)
- **μ**: Peso per lunghezza contorno (smoothness)
- **ν**: Bias per contrazione/espansione (non usato in questa implementazione)

**Intuizione**:

- Il contorno separa l'immagine in due regioni omogenee
- Dentro: pixel simili a c1 (es. cavita' ventricolare, bright)
- Fuori: pixel simili a c2 (es. miocardio + background, dark)
- Penalizzazione sulla lunghezza mantiene smoothness

**Vantaggi**:

1. **Topologia flessibile**: Puo' gestire multiple componenti, splitting, merging
2. **No edges forti richiesti**: Funziona con intensita' omogenee
3. **Robusto a rumore**: Approccio region-based
4. **Convergenza affidabile**: Ottimizzazione convessa (level set formulation)

#### Level Set Formulation

Chan-Vese usa **level set methods** per rappresentare il contorno:

```
C = {x : φ(x) = 0}
inside(C) = {x : φ(x) > 0}
outside(C) = {x : φ(x) < 0}
```

Evoluzione del level set φ:

```
∂φ/∂t = δ(φ) · [μ·div(∇φ/|∇φ|) - ν - λ1(I - c1)² + λ2(I - c2)²]
```

**Dove**:

- δ(φ): Delta di Dirac (concentrata sul contorno)
- div(∇φ/|∇φ|): Curvatura (smoothness term)
- (I - c1)², (I - c2)²: Fitting term

#### Morphological Chan-Vese (scikit-image)

`morphological_chan_vese` implementa Chan-Vese con **operatori morfologici** invece di level sets:

**Vantaggi**:

- Piu' veloce (no PDE solving)
- Meno parametri
- Convergenza piu' rapida

**Parametri**:

- `num_iter`: Numero iterazioni (50-200 tipicamente)
- `init_level_set`: Maschera iniziale (seed)
- `smoothing`: Smoothing factor (1-3, higher = smoother)
- `lambda1, lambda2`: Pesi fitting inside/outside (default: 1, 1)

### Parametri Cardiaci

#### Volume Ventricolare

**Metodo Simpson**:

Il volume e' calcolato sommando le aree endocardiche su tutte le slices:

```
V = Σ A_i · dx · dy · dz
```

**Dove**:

- A_i: Area endocardica slice i (in pixel)
- dx, dy: Pixel spacing in-plane (mm)
- dz: Slice thickness (mm)

**Conversione**:

```
1 mL = 1 cm³ = 1000 mm³
```

#### Stroke Volume (SV)

Volume di sangue espulso per battito:

```
SV = EDLV - ESLV    (mL)
```

#### Ejection Fraction (EF)

Frazione di volume espulso rispetto al volume diastolico:

```
EF = (EDLV - ESLV) / EDLV × 100    (%)
```

**Range normali**:

- Normale: 55-70%
- Disfunzione lieve: 45-54%
- Disfunzione moderata: 30-44%
- Disfunzione severa: <30%

#### Cardiac Output (CO)

Gittata cardiaca, volume pompato al minuto:

```
CO = SV × HR / 1000    (L/min)
```

Dove HR = Heart Rate (bpm)

**Range normali**: 4-8 L/min (a riposo)

#### Body Surface Area (BSA)

Per normalizzare i parametri alla taglia del paziente:

**Mosteller formula** (usata nel referto):

```
BSA = √[(Height_cm × Weight_kg) / 3600]    (m²)
```

**DuBois formula**:

```
BSA = 0.007184 × Height_cm^0.725 × Weight_kg^0.425
```

#### Indexed Values

Parametri normalizzati per BSA:

```
EDLV_indexed = EDLV / BSA    (mL/m²)
ESLV_indexed = ESLV / BSA    (mL/m²)
SV_indexed   = SV / BSA      (mL/m²)
```

**Range normali (indexed)**:

- EDLV/BSA: 65-110 mL/m² (male), 55-95 mL/m² (female)
- ESLV/BSA: 20-40 mL/m² (male), 15-35 mL/m² (female)
- SV/BSA: 40-75 mL/m²

---

## Risultati Attesi

### Valori dal Referto (FUNZIONE20140224_FNRES.pdf)

| Parametro            | Valore Referto | Unita'     |
| -------------------- | -------------- | ---------- |
| Fase Diastolica      | 29 (833 ms)    | frame (ms) |
| Fase Sistolica       | 12 (333 ms)    | frame (ms) |
| ED Volume (LV)       | 114            | mL         |
| ES Volume (LV)       | 41             | mL         |
| Stroke Volume (LV)   | 73             | mL         |
| Peso                 | 47             | kg         |
| Altezza              | 180            | cm         |
| BMI                  | 14.5062        | -          |
| BSA                  | 1.52692        | m²         |
| ED Volume / BSA      | 75             | mL/m²      |
| ES Volume / BSA      | 27             | mL/m²      |
| Stroke Volume / BSA  | 47             | mL/m²      |
| Gittata Cardiaca     | 4.97366        | L/min      |
| Frazione di Eiezione | 63             | %          |
| Massa LV ED          | 43             | g          |
| Massa LV ES          | 47             | g          |

**Note**:

- Nell'esercitazione calcoliamo solo volumi endocardici (contorno verde)
- Non calcoliamo massa miocardica (richiede contorno epicardico, arancione)
- Valori attesi: EDLV ~114 mL, ESLV ~41 mL, EF ~63%

### Variabilita' Attesa

**Fattori di variabilita'**:

1. **Fase diastolica**: Referto usa frame 29, TriggerTime 693 ms puo' corrispondere a frame 28
2. **Seed initialization**: Posizione/dimensione seed influenza convergenza
3. **Parametri Chan-Vese**: `num_iter`, `smoothing` influenzano risultato finale
4. **Slice selection**: Operatore puo' includere/escludere slices borderline

**Variabilita' accettabile** (rispetto al referto):

- EDLV, ESLV: ±5-10 mL (±5-10%)
- EF: ±3-5% (assoluto)
- Se differenze > 15%: controllare fase cardiaca, slices selezionate, seed

---

## Riferimenti

### Papers Fondamentali

1. **Chan, T.F., & Vese, L.A. (2001)**
   _"Active contours without edges"_
   IEEE Transactions on Image Processing, 10(2):266-277
   DOI: 10.1109/83.902291
   **Nota**: Paper originale del Chan-Vese model

2. **Kass, M., Witkin, A., & Terzopoulos, D. (1988)**
   _"Snakes: Active contour models"_
   International Journal of Computer Vision, 1(4):321-331
   **Nota**: Formulazione classica active contours

3. **Petitjean, C., & Dacher, J.N. (2011)**
   _"A review of segmentation methods in short axis cardiac MR images"_
   Medical Image Analysis, 15(2):169-184
   DOI: 10.1016/j.media.2010.12.004
   **Nota**: Review completa metodi segmentazione cardiaca

4. **Zhuang, X. (2013)**
   _"Challenges and methodologies of fully automatic whole heart segmentation"_
   Journal of Healthcare Engineering, 4(3):371-407
   **Nota**: State-of-the-art segmentazione cardiaca

### Linee Guida Cliniche

5. **Kramer, C.M. et al. (2013)**
   _"Standardized cardiovascular magnetic resonance (CMR) protocols 2013 update"_
   Journal of Cardiovascular Magnetic Resonance, 15:91
   **Nota**: SCMR consensus statement su protocolli CMR

6. **Schulz-Menger, J. et al. (2020)**
   _"Standardized image interpretation and post-processing in cardiovascular magnetic resonance - 2020 update"_
   Journal of Cardiovascular Magnetic Resonance, 22:19
   DOI: 10.1186/s12968-020-00610-6
   **Nota**: Linee guida SCMR 2020 per post-processing

7. **Hundley, W.G. et al. (2009)**
   _"ACCF/ACR/AHA/NASCI/SCMR 2010 expert consensus document on cardiovascular magnetic resonance"_
   Circulation, 121:2462-2508
   **Nota**: Consensus multi-society su CMR

### Textbooks

8. **Bogaert, J., Dymarkowski, S., & Taylor, A.M. (2012)**
   _"Clinical Cardiac MRI"_
   Springer, 2nd edition
   **Nota**: Testo completo su cardiac MRI clinica

9. **Manning, W.J., & Pennell, D.J. (2010)**
   _"Cardiovascular Magnetic Resonance"_
   Elsevier Health Sciences
   **Nota**: Reference standard per CMR

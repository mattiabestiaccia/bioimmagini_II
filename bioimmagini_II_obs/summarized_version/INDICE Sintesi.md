# Bioimmagini II - Versione Sintetizzata per lo Studio

**Formato**: Concetti chiave • Frasi concise • Formule essenziali • Tabelle riepilogative

---

## 1. Modello Immagine

| File | Contenuto Chiave |
|------|------------------|
| [[1. Modello Immagine/1. Dal Post-processing - Sintesi\|Dal Post-processing]] | Analisi qualitativa/semi-quant/quantitativa, normativa DM |
| [[1. Modello Immagine/2. Definizione Immagine - Sintesi\|Definizione Immagine]] | DICOM, parametri 2D/3D, FOV, voxel |
| [[1. Modello Immagine/3. Computer Vision - Sintesi\|Computer Vision]] | Pipeline: acquisition→preprocessing→segmentation→decision |
| [[1. Modello Immagine/4. Modello Immagine - Sintesi\|Modello Immagine]] | $I = [I_0 + n_B]*h \cdot g + n$, PVE, entropia |
| [[1. Modello Immagine/5. Qualità Immagine - Sintesi\|Qualità Immagine]] | SNR, CNR, PSF, JND, rumore Riciano |
| [[1. Modello Immagine/6. Case Study CONSIP - Sintesi\|Case Study CONSIP]] | Bando MRI, phantom, uniformità, artefatti |

---

## 2. Interpolazione e Filtraggio

| File | Contenuto Chiave |
|------|------------------|
| [[2. Interpolazione Filtraggio/1. Intro - Sintesi\|Intro]] | Preprocessing = secondo livello pipeline |
| [[2. Interpolazione Filtraggio/2. Interpolazione - Sintesi\|Interpolazione]] | NN, bilineare, bicubica, spline, reslicing 3D |
| [[2. Interpolazione Filtraggio/3. Filtraggio - Sintesi\|Filtraggio]] | Puntuali (LUT, windowing), locali (convoluzione), globali (eq. istogramma) |
| [[2. Interpolazione Filtraggio/4. Compressione - Sintesi\|Compressione]] | Lossless (RLE, LZW, Huffman), Lossy (DCT, wavelet), PSNR |
| [[2. Interpolazione Filtraggio/5. Super-Resolution - Sintesi\|Super-Resolution]] | PVE, back-projection iterativa, regolarizzazione |

---

## 3. Segmentazione

| File | Contenuto Chiave |
|------|------------------|
| [[3. Segmentazione/1. Intro - Sintesi\|Intro]] | Mask vs contorni, even-odd rule |
| [[3. Segmentazione/2. Fondamenti ML - Sintesi\|Fondamenti ML]] | Supervised/unsupervised, loss functions, learning curve |
| [[3. Segmentazione/3. Clustering - Sintesi\|Clustering]] | Otsu, K-means, FCM, EM, metriche distanza |
| [[3. Segmentazione/4. Regioni Contorni - Sintesi\|Regioni e Contorni]] | Labeling, region growing, Canny, snakes, level set, watershed |

---

## 4. Validazione

| File | Contenuto Chiave |
|------|------------------|
| [[4. Validazione/1. Intro - Sintesi\|Intro]] | Gold standard, variabilità inter/intra-osservatore |
| [[4. Validazione/2. Valutazione Segmentazione - Sintesi\|Valutazione Segmentazione]] | Jaccard, Dice, F1-score |
| [[4. Validazione/3. Misure Statistiche - Sintesi\|Misure Statistiche]] | CoV, regressione, Bland-Altman |
| [[4. Validazione/4. Test Statistici Diagnosi - Sintesi\|Test Statistici]] | p-value, t-test, matrice confusione, ROC, AUC |
| [[4. Validazione/5. Valutazione Software - Sintesi\|Valutazione Software]] | Criteri (accessibilità, usabilità, riproducibilità) |
| [[4. Validazione/5b. Certificazione - Sintesi\|Certificazione DM]] | Classi rischio A/B/C, ISO, MDR |

---

## 5. Registrazione e Fusione

| File | Contenuto Chiave |
|------|------------------|
| [[5. Registrazione Fusione/1. Intro - Sintesi\|Intro]] | Unimodale/multimodale, modello AHA |
| [[5. Registrazione Fusione/2. Concetti Fondamentali - Sintesi\|Concetti]] | Search space, metriche, trasformazioni rigida/affine/non-rigida |
| [[5. Registrazione Fusione/3. Mutua Informazione - Sintesi\|Mutua Informazione]] | Entropia Shannon, $MI = H(X) + H(Y) - H(X,Y)$ |
| [[5. Registrazione Fusione/4. Serie Temporali - Sintesi\|Serie Temporali]] | Strategie: riferimento fisso, progressiva, clustering gerarchico |
| [[5. Registrazione Fusione/5. Esempi Registrazione - Sintesi\|Esempi]] | Intra-operatoria, perfusione renale |
| [[5. Registrazione Fusione/6. Fusione Immagini - Sintesi\|Fusione]] | Alpha blending, wavelet, checkerboard, applicazioni cliniche |
| [[5. Registrazione Fusione/7. Ottimizzatori Globali - Sintesi\|Ottimizzatori Globali]] | No Free Lunch, MultiStart, Amdahl |
| [[5. Registrazione Fusione/8. Cellular Automata - Sintesi\|Cellular Automata]] | Computazione evolutiva, segmentazione |
| [[5. Registrazione Fusione/9. Algoritmi Genetici - Sintesi\|Algoritmi Genetici]] | Selezione, crossover, mutazione, elitismo |

---

## 6. Classificazione e Deep Learning

| File | Contenuto Chiave |
|------|------------------|
| [[6. Classificazione DL/1. Intro Classificazione - Sintesi\|Intro]] | Data-driven, train/val/test, cross-validation |
| [[6. Classificazione DL/2. Classificatori Lineari - Sintesi\|Classificatori Lineari]] | k-NN, SVM (hinge loss), Softmax (cross-entropy), gradient descent |
| [[6. Classificazione DL/3. Reti Neurali - Sintesi\|Reti Neurali]] | Neuroni, ReLU, backpropagation, vanishing gradient |
| [[6. Classificazione DL/4. CNN - Sintesi\|CNN]] | CONV, POOL, FC, architetture (LeNet→ResNet) |
| [[6. Classificazione DL/5. Training Ottimizzazione - Sintesi\|Training]] | Regolarizzazione, data augmentation, transfer learning |
| [[6. Classificazione DL/6. Applicazioni Biomediche - Sintesi\|Applicazioni]] | U-Net, semantic segmentation, IoU, Dice |

---

## 7. PACS/RIS

| File | Contenuto Chiave |
|------|------------------|
| [[7. PACS RIS/1. Intro PACS - Sintesi\|Intro PACS]] | Vantaggi digitale, DICOM |
| [[7. PACS RIS/2. Sistemi PACS - Sintesi\|Sistemi PACS]] | Componenti, RAID, dimensionamento, workstation |

---

## Formule Essenziali

### Qualità Immagine
$$SNR = \frac{M_i}{\sigma_i}, \quad CNR = \frac{|M_1 - M_2|}{\sqrt{(\sigma_1^2 + \sigma_2^2)/2}}$$

### Modello Immagine
$$I(x,y) = [I_0(x,y) + n_B(x,y)] \otimes h(x,y) \cdot g(x,y) + n(x,y)$$

### Entropia
$$H(I) = -\sum P(I = g_i) \cdot \log_2 P(I = g_i)$$

### Validazione Segmentazione
$$Dice = \frac{2|S \cap G|}{|S| + |G|}, \quad Jaccard = \frac{|S \cap G|}{|S \cup G|}$$

### Metriche Diagnostiche
$$Sens = \frac{VP}{VP+FN}, \quad Spec = \frac{VN}{VN+FP}, \quad Acc = \frac{VP+VN}{K}$$

### Mutua Informazione
$$MI(X;Y) = H(X) + H(Y) - H(X,Y)$$

### Dimensione Output CNN
$$W_{out} = \frac{W_{in} - F + 2P}{S} + 1$$

---

**38 file** sintetizzati • **7 cartelle** • Pronto per lo studio

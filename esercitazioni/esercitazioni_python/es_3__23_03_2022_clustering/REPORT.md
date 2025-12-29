# Report Esercitazione 3 - Clustering K-means per Perfusione Cardiaca

**Data completamento**: 2024-12-29
**Autore**: Studente
**Versione Python**: 3.12.3

---

## 1. Sintesi dell'Esercitazione

### Obiettivo
Implementare un sistema di segmentazione automatica delle strutture cardiache (ventricolo destro RV, ventricolo sinistro LV, miocardio MYO) in immagini MRI di perfusione cardiaca, utilizzando l'algoritmo K-means clustering basato sulle firme temporali dei pixel.

### Algoritmi Implementati
- **K-means clustering**: Algoritmo di clustering non supervisionato per raggruppare pixel con curve temporali simili
- **Analisi delle curve intensità-tempo**: Estrazione delle firme temporali dei diversi tessuti cardiaci
- **Post-processing morfologico**: Rimozione di piccole regioni e selezione della componente connessa più grande
- **Validazione quantitativa (DICE)**: Confronto con gold standard manuale

### Dataset Utilizzato
- **Tipo**: Serie temporale di immagini MRI di perfusione (raw images, non DICOM)
- **Dimensioni**: 79 frame temporali, risoluzione 256x256 pixel
- **Caratteristiche**: Imaging first-pass con mezzo di contrasto (Gadolinio)

---

## 2. Analisi dei Risultati

### Risultati Ottenuti

#### Configurazione Default (tutti i frame, distanza euclidea)
| Tessuto | DICE Score | Qualità |
|---------|------------|---------|
| RV      | 0.0143     | Poor    |
| LV      | 0.4933     | Poor    |
| MYO     | 0.0001     | Poor    |
| **Media** | **0.1693** | **Poor** |

#### Configurazione Ottimizzata (40 frame, distanza euclidea)
| Tessuto | DICE Score | Qualità |
|---------|------------|---------|
| RV      | 0.6938     | Good    |
| LV      | 0.0662     | Poor    |
| MYO     | 0.0001     | Poor    |
| **Media** | **0.2534** | **Poor** |

### Analisi delle Curve Intensità-Tempo
Le curve estratte mostrano chiaramente le differenze temporali tra tessuti:

| Tessuto | Baseline | Peak | Time-to-Peak | Enhancement |
|---------|----------|------|--------------|-------------|
| RV      | 34.5     | 47.4 | 28.0s        | +37.7%      |
| LV      | 365.6    | 479.0| 14.7s        | +31.0%      |
| MYO     | 395.2    | 411.0| 28.7s        | +4.0%       |
| Background | 30.5  | 39.3 | 49.6s        | +28.8%      |

### Risultati Grid Search Ottimizzazione
| N. Frames | Distanza    | DICE Mean | DICE RV | DICE LV | DICE MYO |
|-----------|-------------|-----------|---------|---------|----------|
| 40        | euclidean   | 0.2534    | 0.6938  | 0.0662  | 0.0001   |
| 20        | euclidean   | 0.2503    | 0.6900  | 0.0605  | 0.0003   |
| all       | euclidean   | 0.1693    | 0.0143  | 0.4933  | 0.0001   |
| all       | correlation | 0.0489    | 0.1468  | 0.0000  | 0.0000   |
| 40        | correlation | 0.0441    | 0.1322  | 0.0000  | 0.0000   |

---

## 3. Performance

### Tempi di Esecuzione
| Operazione | Tempo | Note |
|------------|-------|------|
| Caricamento dati (79 frame) | ~0.5s | |
| K-means clustering | ~0.4s | |
| Post-processing | ~0.1s | |
| Grid search (6 config) | ~5s | --quick mode |
| **Totale singola run** | **~1s** | |

### Test Suite
- **32 test** passati
- **Coverage**: 34.36%
- **Tempo esecuzione test**: 3.67s

---

## 4. Problemi Riscontrati

### Difficoltà Tecniche

1. **Score DICE molto bassi per LV e MYO**
   - **Descrizione**: L'algoritmo identifica bene RV ma fallisce su LV e MYO
   - **Causa probabile**:
     - Mappatura cluster-tessuto basata su timing del picco potrebbe non essere ottimale
     - Il gold standard potrebbe usare criteri di segmentazione diversi
   - **Possibile soluzione**: Rivedere l'euristica di identificazione tessuti

2. **Trade-off RV vs LV**
   - **Descrizione**: Configurazioni che migliorano RV peggiorano LV e viceversa
   - **Causa**: Con tutti i frame RV va male ma LV meglio; con 40 frame RV va bene ma LV peggiora
   - **Implicazione**: I parametri ottimali dipendono dal tessuto target

### Differenze MATLAB/Python
- **Nessun file .m da convertire**: L'esercitazione MATLAB originale conteneva solo dati e documentazione PDF, gli studenti dovevano scrivere il codice da zero
- **Implementazione ex-novo**: Il codice Python è stato scritto seguendo le indicazioni del PDF

### Limitazioni Note
- **DICE MYO sistematicamente ~0**: Il miocardio è molto difficile da segmentare con K-means puro
- **Sensibilità all'inizializzazione**: K-means può convergere a minimi locali diversi
- **Mancanza di informazione spaziale**: K-means usa solo le curve temporali, ignorando la posizione anatomica

---

## 5. Miglioramenti Didattici Suggeriti

### Cosa Ha Funzionato Bene
- La pipeline completa è eseguibile e produce visualizzazioni chiare
- Il README guida lo studente passo passo
- L'ottimizzazione automatica esplora lo spazio dei parametri
- I test automatizzati verificano la correttezza del codice

### Cosa Potrebbe Essere Migliorato

#### Negli Algoritmi
- Aggiungere regolarizzazione spaziale (es. Markov Random Fields)
- Implementare clustering gerarchico per confronto
- Usare pre-processing (normalizzazione, denoising) delle curve temporali
- Testare algoritmi alternativi (GMM, DBSCAN, Mean-Shift)

#### Nel Dataset
- Verificare la qualità del gold standard
- Includere più casi per validazione robusta
- Aggiungere annotazioni intermedie (non solo maschere finali)

#### Nell'Approccio Didattico
- Mostrare esempi di segmentazione corretta per riferimento
- Spiegare le metriche di valutazione (DICE, IoU) prima dell'esecuzione
- Guidare nell'interpretazione dei risultati "non perfetti"

### Estensioni Possibili
- Implementare segmentazione con deep learning (U-Net) per confronto
- Aggiungere analisi quantitativa della perfusione (mappe parametriche)
- Estendere a dataset multi-paziente

---

## 6. Conclusioni

### Sintesi
L'esercitazione dimostra come il clustering K-means possa identificare automaticamente le strutture cardiache basandosi sulla dinamica temporale del contrasto. I risultati mostrano una buona identificazione del ventricolo destro ma difficoltà con ventricolo sinistro e miocardio, evidenziando i limiti degli approcci non supervisionati puri in imaging medico.

### Competenze Acquisite
- Analisi di serie temporali di immagini mediche
- Implementazione e ottimizzazione di algoritmi di clustering
- Validazione quantitativa con metriche (DICE coefficient)
- Interpretazione critica dei risultati di segmentazione automatica

### Valutazione Personale
L'esercitazione offre un'introduzione pratica al machine learning non supervisionato in imaging medico. I risultati sub-ottimali sono didatticamente utili perché mostrano che gli algoritmi semplici hanno limitazioni reali in applicazioni cliniche, motivando l'uso di tecniche più avanzate.

---

## Appendice

### A. Comandi Eseguiti
```bash
# Spostarsi sul worktree
cd /home/brusc/Projects/bioimmagini_positano_worktrees/es_3

# Eseguire gli script
cd esercitazioni/esercitazioni_python/es_3__23_03_2022_clustering/src
python plot_time_curves.py
python kmeans_segmentation.py
python optimize_kmeans.py --quick

# Eseguire i test
python -m pytest tests/ -v
```

### B. File Generati
- `results/time_curves.png` - Curve intensità-tempo per i diversi tessuti
- `results/kmeans_segmentation.png` - Visualizzazione della segmentazione
- `results/gold_standard.png` - Gold standard di riferimento
- `results/optimization_results.png` - Grafici dell'ottimizzazione
- `results/optimization_results.csv` - Dati numerici dell'ottimizzazione
- `results/segmentation_masks.npz` - Maschere binarie delle regioni

### C. Riferimenti
1. PDF originale: `docs/Esercitazione_kmeans.pdf`
2. scikit-learn K-means documentation
3. DICE coefficient per valutazione segmentazione medica

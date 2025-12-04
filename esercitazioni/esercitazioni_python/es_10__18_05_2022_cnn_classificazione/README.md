# Esercitazione 10: Classificazione CNN di Slice Cardiache MRI

## Indice

- [Introduzione](#introduzione)
- [Teoria](#teoria)
  - [Anatomia Cardiaca e Modello AHA](#anatomia-cardiaca-e-modello-aha)
  - [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
  - [Architettura VGG](#architettura-vgg)
- [Dataset](#dataset)
- [Implementazione](#implementazione)
  - [Struttura del Progetto](#struttura-del-progetto)
  - [Pipeline di Elaborazione](#pipeline-di-elaborazione)
- [Esecuzione](#esecuzione)
- [Risultati Ottenuti](#risultati-ottenuti)
  - [Training History](#training-history)
  - [Matrici di Confusione](#matrici-di-confusione)
  - [Analisi Errori](#analisi-errori)
  - [Discussione](#discussione)

---

## Introduzione

Questa esercitazione implementa un sistema di **classificazione automatica** di slice cardiache MRI in asse corto (Short-Axis) utilizzando una **Convolutional Neural Network (CNN)**.

L'obiettivo è assegnare automaticamente ad ogni fetta una label anatomica secondo il modello AHA (American Heart Association):

- **Basal**: Regione basale (vicino alle valvole)
- **Middle**: Regione medio-ventricolare
- **Apical**: Regione apicale (apice del cuore)

Questo task è fondamentale per automatizzare l'analisi cardiaca, permettendo di identificare correttamente le sezioni per il calcolo di parametri volumetrici e funzionali.

---

## Teoria

### Anatomia Cardiaca e Modello AHA

Il ventricolo sinistro viene suddiviso in tre regioni principali lungo l'asse lungo:
1.  **Basal (Base)**: La parte superiore, caratterizzata dalla presenza delle valvole mitrale e aortica e dal diametro massimo.
2.  **Middle (Mid-Cavity)**: La parte centrale, identificabile dalla presenza dei muscoli papillari ben definiti.
3.  **Apical (Apex)**: La punta del cuore, con diametro ridotto e assenza di muscoli papillari.

### Convolutional Neural Networks (CNN)

Le CNN sono reti neurali specializzate per l'elaborazione di dati a griglia come le immagini. I componenti chiave utilizzati in questo progetto sono:

-   **Convolutional Layer**: Estrae feature locali (bordi, texture) tramite filtri apprendibili.
-   **ReLU Activation**: Introduce non-linearità (`max(0, x)`).
-   **Pooling Layer**: Riduce la dimensionalità spaziale (down-sampling) per invarianza e riduzione parametri.
-   **Fully Connected Layer**: Classifica le feature estratte.
-   **Dropout**: Regolarizzazione per prevenire l'overfitting.

### Architettura VGG

Il modello implementato si ispira all'architettura **VGG** (Visual Geometry Group), caratterizzata da:
-   Uso esclusivo di filtri convoluzionali piccoli (3x3).
-   Stack di layer convoluzionali seguiti da Max Pooling.
-   Raddoppio del numero di filtri dopo ogni pooling.

In questa esercitazione utilizziamo una variante `vgg_small` adattata per le dimensioni delle nostre immagini (128x128).

---

## Dataset

Il dataset è composto da **753 immagini DICOM** acquisite con diverse sequenze MRI (Perfusion, Cine, T2*, LGE).

Struttura delle directory:
```
data/
├── Apical/   # Immagini regione apicale
├── Basal/    # Immagini regione basale
└── Middle/   # Immagini regione mediale
```

Le immagini vengono pre-processate con:
1.  Caricamento DICOM.
2.  Center Crop (per focalizzarsi sul cuore).
3.  Resize a 128x128 pixel.
4.  Normalizzazione [0, 1].

---

## Implementazione

### Struttura del Progetto

```
es_10__18_05_2022_cnn_classificazione/
├── data/                  # Dataset immagini
├── results_aug/           # Output generati (con data augmentation)
├── src/
│   ├── cardiac_slice_classifier.py  # Script principale
│   └── utils.py                     # Funzioni di supporto
└── requirements.txt       # Dipendenze
```

### Pipeline di Elaborazione

Lo script `cardiac_slice_classifier.py` esegue i seguenti passi:
1.  **Load Data**: Carica le immagini e le label dalle cartelle.
2.  **Split**: Divide i dati in Training (70%), Validation (15%) e Test (15%).
3.  **Build Model**: Costruisce la CNN (VGG-style).
4.  **Train**: Addestra il modello monitorando la validation loss.
5.  **Evaluate**: Valuta le performance su tutti i set.
6.  **Visualize**: Genera grafici e salva i risultati.

---

## Esecuzione

Per riprodurre i risultati, eseguire il seguente comando dalla cartella `src`:

```bash
# Assicurarsi di essere nell'ambiente virtuale con le dipendenze installate
python cardiac_slice_classifier.py \
    --data_dir ../data \
    --output_dir ../results_aug \
    --epochs 50 \
    --architecture vgg_small \
    --use_data_augmentation
```

L'opzione `--use_data_augmentation` attiva trasformazioni casuali (flip, rotazioni, zoom) durante il training per migliorare la robustezza del modello.

---

## Risultati Ottenuti

I seguenti risultati sono stati ottenuti addestrando il modello `vgg_small` per 50 epoche con data augmentation.

### Training History

![Training History](results_aug/training_history.png)

*Grafico dell'andamento di Loss e Accuracy durante il training.*
Si nota come il modello cerchi di apprendere (la training loss scende), ma la validation loss rimane alta o instabile, indicando difficoltà nella generalizzazione su questo specifico dataset o split.

### Matrici di Confusione

Le matrici di confusione mostrano come il modello classifica le immagini rispetto alle label reali.

#### Test Set
![Confusion Matrix Test](results_aug/confusion_matrix_test.png)

### Analisi Errori

Esempi di immagini misclassificate dal modello:

![Misclassified Samples](results_aug/misclassified_samples.png)

### Discussione

Dai risultati ottenuti (Accuracy ~33-35%), si evince che il task è complesso per la configurazione attuale. Possibili cause:
1.  **Dataset Limitato**: 753 immagini potrebbero non essere sufficienti per addestrare una CNN da zero senza un forte overfitting.
2.  **Variabilità delle Sequenze**: Il dataset mischia sequenze diverse (Cine, LGE, Perfusion) che hanno contrasti molto diversi, rendendo difficile l'apprendimento di feature comuni.
3.  **Labeling**: La classificazione basata su directory potrebbe contenere errori o ambiguità (slice di transizione).

Per migliorare le performance, si potrebbero considerare:
-   **Transfer Learning**: Utilizzare una rete pre-addestrata (es. su ImageNet) e fare fine-tuning.
-   **Più Dati**: Aumentare la dimensione del dataset.
-   **Architetture più Semplici**: Ridurre ulteriormente la complessità del modello per evitare overfitting.

---

## Conclusioni

Questa esercitazione ha mostrato come impostare una pipeline completa di Deep Learning per l'analisi di immagini mediche, dal caricamento dei dati alla valutazione del modello. Nonostante le sfide nel raggiungere un'alta accuratezza, i concetti di CNN, data augmentation e valutazione tramite matrici di confusione sono stati applicati e visualizzati.

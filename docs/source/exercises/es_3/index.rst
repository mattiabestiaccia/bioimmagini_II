Esercizio 3: K-means Clustering per Segmentazione Cardiaca
============================================================

*Segmentazione Automatica di Strutture Cardiache su MRI di Perfusione*

.. toctree::
   :maxdepth: 2

   theory
   usage
   api
   examples
   matlab-conversion

Panoramica
----------

Questa esercitazione copre la segmentazione automatica di strutture cardiache
(ventricolo destro, ventricolo sinistro, miocardio) su immagini MRI di perfusione
first-pass usando l'algoritmo K-means clustering.

L'algoritmo classifica ogni pixel basandosi sulla sua curva intensità/tempo,
identificando automaticamente quattro classi principali:

1. **Background** - nessun contrasto
2. **Ventricolo destro (RV)** - picco precoce
3. **Ventricolo sinistro (LV)** - picco intermedio
4. **Miocardio** - picco tardivo

Obiettivi Didattici
-------------------

* Comprendere il clustering spaziale vs temporale nelle serie dinamiche
* Implementare K-means su curve intensità/tempo
* Applicare post-processing morfologico per raffinare segmentazioni
* Valutare accuratezza usando DICE coefficient
* Ottimizzare parametri (numero frame, metrica distanza)

Moduli Implementati
-------------------

* **kmeans_segmentation.py** - Segmentazione principale
* **plot_time_curves.py** - Visualizzazione curve temporali
* **optimize_kmeans.py** - Ottimizzazione parametri
* **utils.py** - Funzioni utility (DICE, post-processing)

Dataset
-------

* **Serie perfusione cardiaca**: ``data/perfusione/`` (40+ frame DICOM)
* **Maschere gold standard**: ``data/gold_standard/`` (RV, LV, miocardio)

Struttura Dati
--------------

Le immagini sono organizzate come:

* **Stack 3D**: (height, width, n_temporal_frames)
* **Curve temporali**: Un vettore di intensità per ogni pixel
* **Maschere**: Immagini binarie per ogni struttura anatomica

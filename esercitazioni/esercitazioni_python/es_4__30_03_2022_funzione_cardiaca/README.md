# Esercitazione 4: Analisi Funzione Cardiaca con Active Contours

**Data**: 30/03/2022
**Obiettivo**: Segmentazione del ventricolo sinistro e calcolo parametri di funzione cardiaca usando Active Contours (Chan-Vese) su immagini MRI cardiache cine

---

## Documentazione

La documentazione di questa esercitazione è stata divisa in due parti per maggiore chiarezza:

### 1. [Teoria e Background Clinico](theory.md)

Contiene:

- Panoramica del problema clinico
- Descrizione del dataset
- Background teorico su funzione cardiaca e Active Contours
- Risultati attesi e valori di riferimento

### 2. [Implementazione e Utilizzo](implementation.md)

Contiene:

- Pipeline di analisi dettagliata
- Struttura del codice
- Guida all'utilizzo e parametri
- Dettagli implementativi con snippet di codice

---

## Quick Start

### Requisiti

```bash
pip install numpy scipy scikit-image matplotlib pydicom
```

### Esecuzione

```bash
cd src
python cardiac_function_analysis.py
```

Per maggiori dettagli sull'esecuzione, consultare il documento [Implementazione](implementation.md).

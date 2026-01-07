# Report: Esercitazione 4 - Analisi Funzione Cardiaca

## Obiettivo
Segmentazione del ventricolo sinistro e calcolo dei parametri di funzione cardiaca usando Active Contours (Chan-Vese) su immagini MRI cardiache cine.

## Metodologia

### Algoritmi Implementati
1. **Chan-Vese Active Contours**: Segmentazione region-based del ventricolo sinistro
2. **Analisi temporale**: Tracking del ventricolo attraverso le fasi cardiache
3. **Calcolo parametri funzionali**: EDV, ESV, Stroke Volume, Ejection Fraction

### Pipeline
1. Caricamento immagini DICOM cine MRI
2. Inizializzazione contorno (seed manuale o automatico)
3. Evoluzione Active Contour per ogni frame
4. Calcolo aree/volumi per fase cardiaca
5. Estrazione parametri funzionali

## Parametri Cardiaci Calcolati

| Parametro | Descrizione | Range Normale |
|-----------|-------------|---------------|
| EDV | Volume Telediastolico | 95-145 mL (uomini) |
| ESV | Volume Telesistolico | 35-75 mL (uomini) |
| SV | Stroke Volume (EDV-ESV) | 55-100 mL |
| EF | Frazione di Eiezione (SV/EDV) | 55-70% |

## Risultati

I risultati dell'analisi sono salvati in:
- `results/`: Output grafici e metriche
- Curve volume-tempo
- Parametri funzionali calcolati

## Dipendenze
- numpy, scipy, scikit-image
- pydicom, matplotlib

## Esecuzione
```bash
cd src
python cardiac_function_analysis.py
```

## Riferimenti
- Chan & Vese (2001): Active contours without edges
- Documentazione dettagliata: [theory.md](theory.md), [implementation.md](implementation.md)

---
**Data conversione**: 2025-12-29
**Fonte**: Esercitazione MATLAB 30/03/2022

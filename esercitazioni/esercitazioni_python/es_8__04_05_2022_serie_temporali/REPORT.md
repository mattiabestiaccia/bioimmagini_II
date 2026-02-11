# Report Esercitazione 8: Registrazione Serie Temporali con Demons Algorithm

**Data**: 2025-12-28
**Stato**: Funzionante
**Branch**: feature/es_7

---

## Obiettivo

Implementare la registrazione non-rigida di serie temporali MRI con motion artifacts usando:
- Hierarchical Clustering per raggruppare immagini simili (pre/post contrasto)
- Demons Algorithm per registrazione deformabile
- Multi-scale approach per robustezza

---

## Lavoro Svolto

### 1. Setup Iniziale

- Creato `requirements.txt` con dipendenze (numpy, scipy, scikit-image, pydicom, matplotlib)
- Verificato dataset RENAL_PERF (20 frame 512x512)

### 2. Bug Fix Critico: Segno Update Demons

**Problema**: L'MSE aumentava ad ogni iterazione invece di diminuire.

**Causa**: La formula di update del Demons aveva il segno sbagliato.

```python
# PRIMA (errato) - spingeva nella direzione sbagliata
update_y = diff * grad_y / denominator
update_x = diff * grad_x / denominator

# DOPO (corretto) - minimizza (F - R)
update_y = -diff * grad_y / denominator
update_x = -diff * grad_x / denominator
```

**Verifica**:
- Prima del fix: MSE 0.0061 → 0.0092 (+50% PEGGIORA)
- Dopo il fix: MSE 0.0061 → 0.0052 (-15% MIGLIORA)

---

## Risultati

### Pipeline Completa (20 frame)

```
=== PERFUSION CURVE STATISTICS ===
Before registration:
  Mean: 319.83, Std: 27.48
  Range: [282.62, 362.67]

After registration:
  Mean: 318.55, Std: 27.31
  Range: [281.74, 360.90]

Curve smoothness (variance of derivative):
  Before: 91.36
  After: 85.79
  Improvement: 6.1%
```

### Clustering

Il clustering gerarchico separa correttamente:
- **Cluster 0** (8 immagini): Frame 12-19 (post-contrasto, alta intensita')
- **Cluster 1** (12 immagini): Frame 0-11 (pre-contrasto, bassa intensita')

### Registrazione Within-Cluster

| Cluster | MSE Before | MSE After | Riduzione |
|---------|------------|-----------|-----------|
| Cluster 0 | 0.0088 | 0.0078 | 11.4% |
| Cluster 1 | 0.0010 | 0.0009 | 10.0% |

### Registrazione Between-Cluster

| Registrazione | MSE Before | MSE After | Riduzione |
|---------------|------------|-----------|-----------|
| Cluster 1 → 0 | 0.0273 | 0.0245 | 10.3% |

---

## Analisi Performance

### Punti di Forza

1. **Clustering funziona**: Separa correttamente pre/post contrasto
2. **Demons converge**: MSE diminuisce ad ogni iterazione
3. **Multi-scale**: Approccio piramidale migliora robustezza
4. **Smoothness migliora**: 6.1% riduzione variance derivata

### Limitazioni

1. **Miglioramento modesto (6.1%)**: Il dataset ha poco motion respiratorio
2. **Composizione displacement semplificata**: Usa somma invece di composizione esatta
3. **Tempo di esecuzione**: ~2-3 minuti per 20 frame 512x512

### Note sul Dataset

Il dataset RENAL_PERF fornito ha:
- 20 frame (non 70 come nel README teorico)
- Motion respiratorio limitato
- Principalmente variazione di intensita' (enhancement contrasto)

---

## Visualizzazioni Generate

1. `results/dendrogram.png` - Dendrogramma clustering gerarchico
2. `results/cluster_assignment.png` - Assegnazione cluster con sample images
3. `results/Within_Cluster_*_Registration.png` - Registrazione intra-cluster
4. `results/Between_Clusters_*_Registration.png` - Registrazione inter-cluster
5. `results/perfusion_curves.png` - Curve di perfusione before/after

---

## Come Eseguire

```bash
cd esercitazioni/esercitazioni_python/es_8__04_05_2022_serie_temporali

# Attiva environment
source venv/bin/activate

# Esecuzione base (20 frame, 50 iterazioni)
python src/temporal_registration.py --n_subset 0 --n_iterations 50

# Con ROI specifica per perfusion curve
python src/temporal_registration.py --roi 200 300 200 300

# Con piu' iterazioni per convergenza migliore
python src/temporal_registration.py --n_iterations 100
```

---

## Conclusioni

L'esercitazione e' **funzionante** e dimostra:

1. **Hierarchical clustering** per separare fasi di perfusione
2. **Demons algorithm** per registrazione non-rigida
3. **Multi-scale approach** per robustezza
4. **Valutazione quantitativa** con curve di perfusione

Il miglioramento della smoothness (6.1%) e' modesto ma significativo, considerando che il dataset ha poco motion. Con dati reali con piu' movimento respiratorio, il miglioramento sarebbe piu' marcato.

---

## File Modificati

- `src/utils.py` - Fix segno update Demons algorithm (riga 307-310)
- `requirements.txt` - Creato (nuovo file)

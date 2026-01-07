# Esercitazione 4 - Status Report

**Data ultimo aggiornamento**: 2024-12-28
**Stato**: IN CORSO - Richiede ottimizzazione

---

## Riepilogo

L'esercitazione implementa l'analisi della funzione cardiaca del ventricolo sinistro usando Active Contours (Chan-Vese) su immagini MRI cine. Il codice funziona end-to-end ma l'accuratezza della segmentazione non e' soddisfacente.

---

## Cosa funziona

- [x] Caricamento dataset DICOM 4D (450 immagini: 15 slices x 30 frames)
- [x] Identificazione automatica fasi cardiache (diastole/sistole) da TriggerTime
- [x] Segmentazione con Chan-Vese (morphological_chan_vese di scikit-image)
- [x] Calcolo volumi ventricolari (metodo Simpson)
- [x] Calcolo parametri cardiaci (SV, EF, CO, valori indicizzati per BSA)
- [x] Generazione grafici e report
- [x] Pipeline completa eseguibile da CLI

---

## Performance attuali vs Referto

| Parametro | Calcolato | Referto | Errore | Status |
|-----------|-----------|---------|--------|--------|
| EDLV | 143 mL | 114 mL | +25% | Accettabile |
| ESLV | 113 mL | 41 mL | +176% | **CRITICO** |
| Stroke Volume | 31 mL | 73 mL | -58% | **CRITICO** |
| Ejection Fraction | 21% | 63% | -67% | **CRITICO** |
| Cardiac Output | 2.1 L/min | 4.97 L/min | -58% | **CRITICO** |

**Comando usato per il test**:
```bash
python src/cardiac_function_analysis.py --seed_radius 15 --smoothing 4 --n_iterations 150
```

---

## Criticita' identificate

### 1. Sovrastima aree in sistole (CRITICO)
La segmentazione Chan-Vese non cattura correttamente la contrazione ventricolare. Le aree segmentate in sistole sono quasi uguali a quelle in diastole, mentre dovrebbero essere significativamente minori.

**Causa probabile**: Chan-Vese e' region-based e segmenta in base all'intensita' media. In sistole il miocardio si ispessisce ma la cavita' rimane "bright", causando sovrasegmentazione.

### 2. Selezione automatica centro LV
La funzione `find_lv_center()` a volte identifica il ventricolo destro (RV) invece del sinistro (LV), specialmente nelle slice basali e apicali.

### 3. Slice selection
Le slice da analizzare (attualmente 4-11 diastole, 5-10 sistole) sono state determinate empiricamente. Potrebbero non essere ottimali.

### 4. Propagazione errori tra slice
Se una slice viene segmentata male, l'errore si propaga alle slice successive (seed propagation).

---

## Tentativi di ottimizzazione effettuati

1. **Modifica seed radius**: 30 -> 20 -> 15 pixel
   - Risultato: Miglioramento parziale

2. **Aumento smoothing Chan-Vese**: 2 -> 4
   - Risultato: Contorni piu' regolari ma ancora sovrastimati

3. **Filtro max_area su componenti**: Limite 1800-3500 pixel
   - Risultato: Troppo restrittivo o inefficace

4. **Uso centri diastolici per sistole**: Passaggio seed_centers tra fasi
   - Risultato: Migliora consistenza ma non accuratezza

5. **Scoring basato su circolarita'** in find_lv_center()
   - Risultato: Miglioramento nella selezione LV vs RV

---

## Possibili miglioramenti futuri

### Opzione A: Threshold-based approach
Usare soglia di intensita' + operazioni morfologiche invece di Chan-Vese. Piu' semplice e potenzialmente piu' controllabile.

### Opzione B: Edge-based active contours
Usare `activecontour(..., 'edge')` che si basa sui gradienti invece che sulle regioni. Potrebbe essere piu' sensibile ai bordi del miocardio.

### Opzione C: Seed interattivo
Permettere all'utente di cliccare sul centro del LV per la prima slice, poi propagare.

### Opzione D: Pre-processing avanzato
- Filtro di contrasto locale
- Normalizzazione intensita' tra slice
- ROI detection automatica del cuore

### Opzione E: Deep Learning
Usare una rete pre-trained per segmentazione cardiaca (es. nnU-Net). Fuori scope per questa esercitazione didattica.

---

## File modificati rispetto all'originale

1. `src/utils.py`:
   - Aggiunta funzione `find_lv_center()` con scoring basato su posizione/circolarita'
   - Modificata `refine_segmentation()` per supportare seed_center e max_area

2. `src/cardiac_function_analysis.py`:
   - Modificata `segment_phase()` per restituire centri usati
   - Passaggio centri da diastole a sistole
   - Gestione EF negativo in plot

---

## Come riprendere il lavoro

1. Attivare ambiente:
   ```bash
   cd esercitazioni/esercitazioni_python/es_4__30_03_2022_funzione_cardiaca
   source .venv/bin/activate
   ```

2. Testare configurazione attuale:
   ```bash
   python src/cardiac_function_analysis.py --seed_radius 15 --smoothing 4 --n_iterations 150
   ```

3. Visualizzare risultati in `results/`

4. Focus principale: **Ridurre ESLV** (attualmente 113 mL, target 41 mL)

---

## Note

- Il referto usa segmentazione **manuale** da operatore esperto
- Variabilita' inter-operatore tipica: 5-10% sui volumi
- Per uso didattico l'implementazione dimostra i concetti chiave
- Per uso clinico servirebbero validazione e tuning approfonditi

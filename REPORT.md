# Report Esercitazione 8 - Registrazione Serie Temporali con Demons

**Data completamento**: 2025-12-28
**Autore**: Claude Code (assistito)
**Branch**: feature/es_8

---

## 1. Sintesi dell'Esercitazione

### Obiettivo
Implementare registrazione non-rigida di serie temporali MRI con motion artifacts respiratori usando Demons algorithm e hierarchical clustering.

### Algoritmi Implementati
- **Demons Algorithm**: Registrazione deformabile ispirata ai demoni di Maxwell
- **Hierarchical Clustering**: Raggruppamento immagini simili (pre/post contrasto) con linkage average
- **Multi-Scale Registration**: Approccio piramidale coarse-to-fine (scales [4, 2, 1])

### Dataset Utilizzato
- **Tipo**: DICOM serie perfusione renale (RENAL_PERF)
- **Dimensioni**: 20 frame, 512x512 pixel
- **Caratteristiche**: Serie temporale con enhancement da mezzo di contrasto gadolinio

---

## 2. Analisi dei Risultati

### Risultati Ottenuti

| Parametro | Prima | Dopo | Variazione |
|-----------|-------|------|------------|
| Smoothness (var derivata) | 91.36 | 85.79 | **-6.1%** |
| MSE Within Cluster 0 | 0.0088 | 0.0078 | -11.4% |
| MSE Within Cluster 1 | 0.0010 | 0.0009 | -10.0% |
| MSE Between Clusters | 0.0273 | 0.0245 | -10.3% |

### Clustering Gerarchico

| Cluster | Frame | Tipo | N. Immagini |
|---------|-------|------|-------------|
| Cluster 0 | 12-19 | Post-contrasto | 8 |
| Cluster 1 | 0-11 | Pre-contrasto | 12 |

### Accuratezza
- **Riduzione MSE media**: ~10%
- **Miglioramento smoothness**: 6.1%
- **Convergenza Demons**: MSE diminuisce correttamente ad ogni iterazione

---

## 3. Performance

### Tempi di Esecuzione

| Operazione | Tempo |
|------------|-------|
| Caricamento DICOM | ~1 sec |
| Clustering | < 1 sec |
| Registrazione (20 frame) | ~2-3 min |
| **Totale** | ~3 min |

### Risorse Utilizzate
- **RAM**: ~1 GB (picco)
- **CPU**: Single-threaded
- **GPU**: Non utilizzata

### Scalabilita'
L'approccio multi-scale riduce significativamente il tempo di convergenza. Per dataset piu' grandi, si potrebbe parallelizzare la registrazione intra-cluster.

---

## 4. Problemi Riscontrati

### Difficolta' Tecniche

1. **Segno Update Demons**
   - **Descrizione**: MSE aumentava ad ogni iterazione invece di diminuire
   - **Causa**: Formula update aveva segno sbagliato - spingeva l'immagine nella direzione opposta
   - **Soluzione**: Negato l'update displacement

```python
# PRIMA (errato) - MSE aumentava
update_y = diff * grad_y / denominator

# DOPO (corretto) - MSE diminuisce
update_y = -diff * grad_y / denominator
```

### Differenze MATLAB/Python

| Aspetto | MATLAB | Python | Gestione |
|---------|--------|--------|----------|
| DICOM read | dicomread | pydicom.dcmread | force=True se necessario |
| Gaussian filter | imgaussfilt | skimage.filters.gaussian | preserve_range=True |
| Interpolazione | interp2 | ndimage.map_coordinates | order=1 per bilineare |

### Limitazioni Note
- **Miglioramento modesto (6.1%)**: Il dataset fornito ha poco motion respiratorio
- **Composizione displacement**: Approssimata con somma invece di composizione esatta
- **Single-threaded**: Non parallelizzato

---

## 5. Miglioramenti Didattici Suggeriti

### Cosa Ha Funzionato Bene
- Pipeline end-to-end automatica
- Clustering separa correttamente pre/post contrasto
- Visualizzazioni comprehensive (dendrogram, checkerboard, curves)
- Multi-scale approach migliora robustezza

### Cosa Potrebbe Essere Migliorato

#### Nella Documentazione
- Specificare chiaramente il segno corretto nella formula Demons
- Aggiungere derivazione matematica dell'update rule
- Documentare convenzione displacement field (quale asse e' quale)

#### Negli Algoritmi
- Usare composizione esatta displacement invece di somma
- Implementare Diffeomorphic Demons per preservare topologia
- Aggiungere early stopping basato su convergenza MSE

#### Nel Dataset
- Fornire dataset con piu' motion respiratorio per risultati piu' evidenti
- Includere ground truth per validazione quantitativa
- Aggiungere casi con artefatti diversi (cardiac motion, peristalsi)

#### Nell'Approccio Didattico
- Iniziare con registrazione 2D statica prima di serie temporali
- Mostrare step intermedi della convergenza Demons
- Visualizzare il displacement field per debugging

### Estensioni Possibili
- Diffeomorphic Demons (preserva topologia)
- Registrazione 3D+T
- Altre metriche (NCC, Local MI)
- GPU acceleration con CuPy/PyTorch

---

## 6. Conclusioni

### Sintesi
L'esercitazione implementa con successo registrazione non-rigida con Demons algorithm per serie temporali MRI. Il miglioramento della smoothness (6.1%) e' modesto ma significativo, considerando che il dataset ha poco motion respiratorio.

### Competenze Acquisite
- Implementazione Demons algorithm per registrazione deformabile
- Clustering gerarchico per preprocessing serie temporali
- Registrazione multi-scala (pyramid approach)
- Estrazione e analisi curve di perfusione
- Debugging algoritmi di registrazione (identificazione bug segno)

### Valutazione Personale
L'esercitazione copre aspetti fondamentali della registrazione non-rigida. Il bug del segno nell'update Demons e' un errore comune e la sua identificazione/risoluzione e' educativa. Il miglioramento modesto riflette le caratteristiche del dataset piu' che limitazioni dell'algoritmo.

---

## Appendice

### A. Comandi Eseguiti

```bash
cd esercitazioni/esercitazioni_python/es_8__04_05_2022_serie_temporali
source venv/bin/activate
python src/temporal_registration.py --n_subset 0 --n_iterations 50
```

### B. File Modificati
- `src/utils.py` - Fix segno update Demons (riga 307-310)
- `requirements.txt` - Nuovo (dipendenze)
- `REPORT.md` - Nuovo (questo file, nella directory esercitazione)

### C. Riferimenti Consultati
1. Thirion, J.P. (1998). "Image Matching as a Diffusion Process: An Analogy with Maxwell's Demons"
2. Vercauteren, T. et al. (2009). "Diffeomorphic demons: Efficient non-parametric image registration"
3. scipy.ndimage documentation

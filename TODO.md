# TODO - Bioimmagini Positano Rebasing

## ✅ Completato

- [x] Setup virtual environment in `esercitazioni_python/venv/`
- [x] Installazione dipendenze base (numpy, scipy, matplotlib, pydicom, etc.)
- [x] Conversione completa Esercitazione 1 (09/03/2022 - Calcolo SD MRI)
  - [x] `Calcolo_SD.m` → `calcolo_sd.py`
  - [x] `EsempioCalcoloSD.m` → `esempio_calcolo_sd.py`
  - [x] `Test_M_SD.m` → `test_m_sd.py`
  - [x] Modulo `utils.py` con funzioni comuni
  - [x] Copia dati DICOM (phantom, LGE series)
  - [x] README completo con teoria
  - [x] Documentazione dati LGE
  - [x] Testing e validazione
- [x] Conversione completa Esercitazione 2 (16/03/2022 - Filtraggio 3D)
  - [x] Implementazione caricamento DICOM con Rescale Intercept/Slope
  - [x] Interpolazione trilineare per volume isotropo
  - [x] Filtro media mobile 3D
  - [x] Filtro Gaussiano 3D con ottimizzazione sigma
  - [x] Filtro Wiener adattivo 3D (implementazione custom)
  - [x] Calcolo SNR e acutezza transizioni
  - [x] Script interattivo per selezione ROI
  - [x] README completo con teoria filtri
  - [x] Testing e validazione con dataset RIDER Phantom
- [x] Creazione `REBASING_GUIDE.md` con direttive complete
- [x] Creazione README principale progetto
- [x] Configurazione `.claude/` per AI assistant
- [x] Catalogazione completa di tutte le 11 esercitazioni

## 🔜 Prossimi Passi Immediati

### 1. Catalogazione Esercitazioni Rimanenti

**Azione**: Esplorare `esercitazioni/esercitazioni_matlab/` e catalogare tutte le esercitazioni

```bash
# Eseguire per identificare tutte le esercitazioni
ls -la esercitazioni/esercitazioni_matlab/
```

**Output atteso**: Rinominare tutte le esercitazioni_matlab come  `es_{n°esercitazione}__{date}_{argument}`

**TODO**: Per ogni esercitazione trovata:
- [ ] Creare entry in `TODO.md`
- [ ] Documentare in `README.md` principale
- [ ] Stimare complessità (# file, # script, dimensione dati)

### 2. Prioritizzazione Esercitazioni

**Criteri**:
1. Ordine cronologico (seguire sequenza didattica)
2. Complessità crescente
3. Dipendenze tra esercitazioni

**TODO**:
- [ ] Ordinare esercitazioni per priorità (usare **Criteri**)
- [ ] Identificare dipendenze cross-esercitazione

### 3. Setup Template per Nuove Esercitazioni

**TODO**:
- [ ] Creare script bash `create_esercitazione.sh` che:
  - Crea struttura cartelle standard
  - Inizializza file base (`__init__.py`, `.gitignore`, etc.)
  - Copia template README
  - Setup git ignore

**Esempio script**:
```bash
#!/bin/bash
# create_esercitazione.sh <numero>

NUM=$1
DIR="esercitazioni/esercitazioni_python/esercitazione_$NUM"

mkdir -p $DIR/{src,data,results,notebooks,tests,docs}
touch $DIR/src/{__init__.py,utils.py}
touch $DIR/results/.gitkeep
# ... etc
```

## 📋 Esercitazioni da Convertire

### Template Entry
```markdown
### Esercitazione X - [Titolo]

**Status**: 🔜 TODO / 🔄 In Corso / ✅ Completata

**File MATLAB**:
- [ ] `Script1.m` → `script_1.py`
- [ ] `Script2.m` → `script_2.py`
- [ ] ...

**Dati**:
- [ ] File DICOM (X file, Y MB)
- [ ] Immagini

**Docs**:
- [ ] PDF documentazione
- [ ] MD

**Concetti trattati**:
- [Elenco argomenti]

**Dipendenze**:
- Da Esercitazione: [numero]
- Richiede funzioni da: [modulo]

**Note**:
- [Eventuali note speciali]
```

---

## 📚 Catalogo Esercitazioni (Ordine Cronologico)

### Esercitazione 1 - Calcolo SD MRI ✅ COMPLETATA
**Data**: 09/03/2022
**Status**: ✅ Completata
**Path**: `esercitazioni_python/es_1__09_03_2022_calcolo_sd/`

**Concetti**: Signal-to-Noise Ratio, Contrast-to-Noise Ratio, DICOM MRI, regioni omogenee

### Esercitazione 2 - Filtraggio 3D ✅ COMPLETATA
**Data**: 16/03/2022
**Status**: ✅ Completata
**Path**: `esercitazioni_python/es_2__16_03_2022_filtraggio/`

**Concetti**: Filtri 3D (media mobile, Gaussiano, Wiener adattivo), SNR, acutezza transizioni, volume isotropo, interpolazione trilineare, RIDER Phantom CT

### Esercitazione 3 - Clustering 🔜 TODO
**Data**: 23/03/2022
**Path MATLAB**: `LEZIONE_08_23_03_2022 (Esercitazione Clustering)/`

**Concetti**: Segmentazione clustering, K-means

### Esercitazione 4 - Contorni 🔜 TODO
**Data**: 30/03/2022
**Path MATLAB**: `LEZIONE_09_30_03_2022 (Esercitazione Contorni)/`

**Concetti**: Rilevamento contorni, edge detection

### Esercitazione 5 - Segmentazione Grasso 🔜 TODO
**Data**: 06/04/2022
**Path MATLAB**: `LEZIONE_11_06_04_2022 (Esercitazione segmentazione grasso)/`

**Concetti**: Segmentazione tessuto adiposo

### Esercitazione 6 🔜 TODO
**Data**: 13/04/2022
**Path MATLAB**: `Esercitazione_13_04_2022/`

**Concetti**: TBD (da esplorare)

### Esercitazione 7 🔜 TODO
**Data**: 27/04/2022
**Path MATLAB**: `Esercitazione_7_27_04_2022/`

**Concetti**: TBD (da esplorare)

### Esercitazione 8 - Serie Temporali 🔜 TODO
**Data**: 04/05/2022
**Path MATLAB**: `LEZIONE_17_04_05_2022 (Esercitazzione serie temporali)/`

**Concetti**: Analisi serie temporali

### Esercitazione 9 - Mappe Parametriche 🔜 TODO
**Data**: 11/05/2022
**Path MATLAB**: `ESERCITAZIONE_11_05_2022 (Esercitazione Mappe Parametriche)/`

**Concetti**: Mappe parametriche

### Esercitazione 10 - Classificazione CNN 🔜 TODO
**Data**: 18/05/2022
**Path MATLAB**: `ESERCITAZIONE_18_05_2022 (Classificazione CNN)/`

**Concetti**: Deep Learning, CNN, classificazione immagini

### Esercitazione 11 - Segmentazione CNN 🔜 TODO
**Data**: 23-25/05/2022
**Path MATLAB**: `ESERCITAZIONE_23_25_05_2022 (Segmentazione CNN)/`

**Concetti**: Deep Learning, CNN, segmentazione semantica


## 📝 Note e Idee

### Miglioramenti Possibili


### Tracking

| # | Titolo | Status | Data Completamento | Righe Python | Note |
|---|--------|--------|-------------------|--------------|------|
| 1 | Calcolo SD MRI | ✅ | 2025-11-10 | 1221 | SNR/CNR su MRI cardiaca |
| 2 | Filtraggio 3D | ✅ | 2025-11-10 | 1519 | Wiener 3D custom, ottimizzazione σ |
| 3 | Clustering | 🔜 | - | - | - |
| 4 | Contorni | 🔜 | - | - | - |
| 5 | Segmentazione Grasso | 🔜 | - | - | - |
| 6 | TBD | 🔜 | - | - | - |
| 7 | TBD | 🔜 | - | - | - |
| 8 | Serie Temporali | 🔜 | - | - | - |
| 9 | Mappe Parametriche | 🔜 | - | - | - |
| 10 | Classificazione CNN | 🔜 | - | - | - |
| 11 | Segmentazione CNN | 🔜 | - | - | - |

---

**Ultima modifica**: 2025-11-10
**Prossimo step**: Esercitazione 3 (Clustering)

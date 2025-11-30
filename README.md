# Bioimmagini Positano - MATLAB to Python Rebasing Project

## 🎯 Obiettivo del Progetto

Conversione sistematica delle esercitazioni MATLAB del corso "Bioimmagini - Positano" in Python, mantenendo equivalenza funzionale e seguendo le best practices moderne.

## 📁 Struttura del Progetto

```
bioimmagini_positano/
├── esercitazioni/
│   ├── esercitazioni_matlab/          # Esercitazioni originali MATLAB (riferimento)
│   │   ├── Esercitazione_1_09_03_2022/
│   │   ├── Esercitazione_2_*/
│   │   └── ...
│   └── esercitazioni_python/          # Conversioni Python
│       ├── venv/                      # Virtual environment condiviso
│       ├── activate.sh                # Script attivazione rapida
│       ├── esercitazione_1/           # ✅ COMPLETATA
│       ├── esercitazione_2/           # 🔜 TODO
│       └── ...
├── REBASING_GUIDE.md                  # 📘 Guida completa conversione
├── .claude/
│   └── project_context.md             # Contesto per AI assistant
└── README.md                          # Questo file
```

## 🚀 Quick Start

### Setup Cross-Platform (WSL + Windows)

Questo repository è configurato per essere utilizzato sia da **WSL** (per sviluppo con Cursor/VS Code) che da **Windows** (per Obsidian vault):

**Percorso WSL**:
```bash
/home/brusc/Projects/bioimmagini_positano
```

**Percorso Windows** (per aprire in Obsidian):
```
\\wsl.localhost\Ubuntu\home\brusc\Projects\bioimmagini_positano
```

**Configurazione Git**:
- `.gitattributes` gestisce automaticamente i line endings (LF)
- `.gitignore` esclude file temporanei di Obsidian
- Puoi editare documenti da entrambi gli ambienti senza conflitti

### Setup Ambiente

```bash
# Clona o naviga al progetto
cd /path/to/bioimmagini_positano

# Attiva virtual environment
cd esercitazioni/esercitazioni_python
source venv/bin/activate

# Verifica installazione
python --version
pip list | grep -E "(numpy|scipy|matplotlib|pydicom)"
```

### Eseguire Esercitazione 1

```bash
cd esercitazione_1/src

# Script 1: Analisi immagine sintetica
python calcolo_sd.py

# Script 2: Analisi fantoccio MRI
python esempio_calcolo_sd.py

# Script 3: Test Monte Carlo
python test_m_sd.py

# Risultati disponibili in:
ls ../results/
```

## 📚 Documentazione

### Per Sviluppatori/Convertitori

**Leggi prima di iniziare una nuova conversione**:
- 📘 **[REBASING_GUIDE.md](REBASING_GUIDE.md)** - Guida completa con:
  - Workflow standard
  - Regole di gestione file
  - Quality checklist
  - Equivalenze MATLAB↔Python
  - Convenzioni di naming

### Per Studenti/Utilizzatori

Ogni esercitazione ha il suo `README.md` con:
- Descrizione obiettivi
- Istruzioni installazione
- Esempi utilizzo
- Teoria e concetti
- Troubleshooting

**Esempio**: [Esercitazione 1 README](esercitazioni/esercitazioni_python/esercitazione_1/README.md)

## ✅ Stato delle Conversioni

### Completate

| # | Titolo | Data | Status | File Python | Documentazione |
|---|--------|------|--------|-------------|----------------|
| 1 | Calcolo SD in Immagini MRI | 09/03/2022 | ✅ | 1221 righe | ✅ Completa |

### In Programmazione

| # | Titolo | Data | Status |
|---|--------|------|--------|
| 2 | TBD | TBD | 🔜 Da catalogare |
| 3 | TBD | TBD | 🔜 Da catalogare |
| ... | ... | ... | ... |

## 🔧 Tecnologie

### Python Stack
- **Python**: 3.12+
- **NumPy**: 2.3+ (operazioni array)
- **SciPy**: 1.16+ (elaborazione scientifica)
- **Matplotlib**: 3.10+ (visualizzazione)
- **PyDICOM**: 3.0+ (lettura DICOM)
- **Jupyter**: 1.1+ (notebook opzionali)
- **scikit-image**: 0.25+ (elaborazione immagini)

### Ambiente di Sviluppo
- Virtual environment condiviso per tutte le esercitazioni
- VS Code con estensioni Python
- Git per version control

## 📋 Workflow di Conversione

### 1. Analisi
```bash
# Esplorare esercitazione MATLAB
ls -la esercitazioni/esercitazioni_matlab/Esercitazione_X/

# Catalogare file
find ... -name "*.m"      # Script da convertire
find ... -name "*.dcm"    # Dati da copiare
find ... -name "*.pdf"    # Documentazione da copiare
```

### 2. Setup
```bash
# Creare struttura standard
mkdir -p esercitazioni_python/esercitazione_X/{src,data,results,docs}
```

### 3. Conversione
- Copiare dati e PDF
- Convertire script MATLAB in Python
- Creare moduli utility
- Scrivere documentazione

### 4. Validazione
- Testare tutti gli script
- Verificare equivalenza numerica
- Completare quality checklist

**Vedi [REBASING_GUIDE.md](REBASING_GUIDE.md) per dettagli completi**

## 🎓 Esercitazioni - Dettagli

### Esercitazione 1: Calcolo della Deviazione Standard in Immagini MRI

**Obiettivi didattici**:
- Analisi del rumore in immagini MRI
- Metodi di stima SD (manual ROI, SD map)
- Correzione Rayleigh per background
- Convergenza statistica (Monte Carlo)

**Script Python**:
1. `calcolo_sd.py` - Analisi immagine sintetica
2. `esempio_calcolo_sd.py` - Analisi fantoccio MRI
3. `test_m_sd.py` - Test Monte Carlo ROI

**Dati**:
- Fantoccio MRI (`phantom.dcm`)
- Serie cardiache LGE (18 DICOM)
- Immagini esempio

**[→ Vai alla documentazione completa](esercitazioni/esercitazioni_python/esercitazione_1/README.md)**

## 🤝 Come Contribuire

### Convertire una Nuova Esercitazione

1. **Leggere** [REBASING_GUIDE.md](REBASING_GUIDE.md)
2. **Scegliere** esercitazione non ancora convertita
3. **Seguire** workflow standard
4. **Validare** con quality checklist
5. **Documentare** completamente
6. **Aggiornare** questo README

### Standard di Qualità

Ogni conversione deve:
- ✅ Replicare TUTTE le funzionalità MATLAB
- ✅ Copiare TUTTI i file dati
- ✅ Avere README completo
- ✅ Passare la quality checklist
- ✅ Essere equivalente numericamente

## 📖 Riferimenti

### Corso
- **Titolo**: Bioimmagini
- **Sede**: Positano
- **Docenti**: [Da specificare]

### Risorse Tecniche
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/)
- [PyDICOM Guide](https://pydicom.github.io/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Bibliografia
Vedi README delle singole esercitazioni per riferimenti specifici.

## 📄 Licenza

Materiale didattico - Solo uso educativo e di ricerca.

## 📞 Contatti

Per domande sul progetto di conversione o sulle esercitazioni:
- [Specificare contatti docenti/responsabili]

---

**Ultima modifica**: 2025-11-10
**Status progetto**: In corso (1/X esercitazioni completate)
**Prossima azione**: Catalogare esercitazioni rimanenti

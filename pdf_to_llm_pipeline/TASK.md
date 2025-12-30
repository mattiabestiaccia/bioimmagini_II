# Task: Pipeline PDF → Markdown → Chunks per LLM

> **Stato**: In attesa delle dispense PDF
> **Branch**: `claude/pipeline-prompt-setup-JMCPR`

---

## Prompt per Claude Code

Voglio costruire una pipeline in Python per trasformare dispense universitarie in PDF (50–60 pagine, piene di formule matematiche e qualche immagine) in contenuti "LLM-ready".
La pipeline deve:

### 1. Input e struttura progetto
- Leggere tutti i PDF da una cartella di input (es. `./input_pdfs`).
- Salvare gli output in una cartella di output (es. `./output_md`), mantenendo un nome file coerente per ogni PDF.
- Creare, se utile, una struttura di progetto minimale (es. `src/`, `configs/`, ecc.) con README sintetico.

### 2. Conversione PDF → Markdown
- Usare un tool moderno per conversione PDF → Markdown pensato per LLM, ad esempio:
  - `marker-pdf` (preferito, se adatto allo scenario) **oppure**
  - `microsoft/markitdown` **oppure**
  - `pdf-to-markdown` come libreria Python.
- Scegli in autonomia la libreria più adatta, spiegando in un commento perché (es. precisione sulle formule, struttura, presenza di JSON, ecc.).
- Lo script deve:
  - Leggere il PDF.
  - Produrre un file `.md` e, se disponibile, un file `.json` con la struttura (sezioni, paragrafi, formule).
  - Preservare le formule matematiche in LaTeX (inline e block).
  - Preservare titoli, sezioni e sottosezioni in formato Markdown (`#`, `##`, `###`).
- Gestire errori comuni (PDF corrotto, pagina illeggibile, ecc.) con logging chiaro.

### 3. Pulizia e normalizzazione del Markdown
- Implementare una fase di "cleaning":
  - Rimuovere numeri di pagina, header/footer ripetuti, linee vuote multiple, hyphenation a capo riga.
  - Assicurarsi che le formule LaTeX non vengano spezzate.
  - Mantenere immagini come riferimenti testuali (es. `![image](path)`), senza tentare di estrarre il contenuto se non banalmente.
- Questa fase può essere un modulo/funzione a parte (es. `clean_markdown(text: str) -> str`).

### 4. Chunking per uso con LLM / RAG
- Implementare una funzione Python che prende il Markdown pulito e lo spezza in chunk adatti a LLM, in stile RAG:
  - Target: ~500–1500 token per chunk (puoi usare come proxy il conteggio parole o caratteri, ma progetta l'interfaccia in modo che sia facile sostituire con un vero tokenizer).
  - Non spezzare:
    - In mezzo a una formula LaTeX.
    - In mezzo a un enunciato di teorema/definizione se riconoscibile.
  - Usare le intestazioni Markdown come boundary preferenziali per i chunk.
- Output:
  - Un file JSON per ogni PDF, con struttura tipo:
    ```json
    {
      "source_pdf": "nome_file.pdf",
      "chunks": [
        {
          "id": "nomefile-0001",
          "title": "Titolo sezione (se disponibile)",
          "page_range": [start_page, end_page],
          "content_markdown": "testo del chunk..."
        }
      ]
    }
    ```
  - Opzionale ma utile: salvare anche i chunk come singoli `.md` (es. `nomefile_chunk_0001.md`).

### 5. Esecuzione batch e CLI
- Fornire uno script `main.py` che:
  - Prenda come argomenti da linea di comando:
    - `--input_dir`
    - `--output_dir`
    - (eventuale) `--max_tokens_per_chunk` o equivalenti.
  - Processi tutti i PDF nella cartella input.
  - Stampi a schermo un breve riepilogo (numero di PDF, numero di chunk per PDF, eventuali warning).
- Usare `argparse` e logging standard (`logging`).

### 6. Qualità del codice
- Fornire codice Python reale, eseguibile, con:
  - Una chiara separazione in funzioni/moduli.
  - Type hints dove ha senso.
  - Docstring brevi e pratiche.
- Prevedere in un commento dove e come si potrebbero integrare:
  - Un tokenizer "vero" (es. tiktoken).
  - Un vector DB per RAG (non implementare RAG, solo indicare il punto di aggancio).

### 7. Output richiesto
- Genera:
  1. File `requirements.txt` con le dipendenze minime.
  2. File `main.py` completo.
  3. Eventuali altri moduli Python (es. `converter.py`, `cleaning.py`, `chunking.py`).
  4. Un README in Markdown, conciso, che spieghi:
     - Come installare e lanciare la pipeline.
     - Come sono organizzati gli output.
     - Come adattare facilmente la pipeline ad altri corsi/materie.

---

## Ambiente di esecuzione
- Python 3.11+ su Linux/WSL
- Accesso a Internet per installare librerie da PyPI

---

## Riferimenti utili
1. [marker-pdf GitHub](https://github.com/datalab-to/marker) - Convert PDF to markdown + JSON
2. [microsoft/markitdown](https://github.com/microsoft/markitdown) - Python tool for converting files to Markdown
3. [Deep Dive into Open Source PDF to Markdown Tools](https://jimmysong.io/blog/pdf-to-markdown-open-source-deep-dive/)
4. [Python MarkItDown: Convert Documents Into LLM-Ready Markdown](https://realpython.com/python-markitdown/)
5. [marker-pdf PyPI](https://pypi.org/project/marker-pdf/0.3.2/)

---

## Come usare questo task

1. **Caricare le dispense PDF** nella cartella `input_pdfs/`
2. **Eseguire il prompt** sopra con Claude Code per generare la pipeline
3. **Installare le dipendenze** con `pip install -r requirements.txt`
4. **Eseguire la pipeline** con `python main.py --input_dir ./input_pdfs --output_dir ./output_md`

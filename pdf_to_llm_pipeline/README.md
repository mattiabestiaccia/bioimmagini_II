# PDF to LLM Pipeline

Pipeline Python per convertire dispense universitarie PDF in contenuti pronti per LLM/RAG.

## Struttura

```
pdf_to_llm_pipeline/
├── main.py              # Script CLI principale
├── requirements.txt     # Dipendenze Python
├── input_pdfs/          # Cartella per le dispense PDF da convertire
├── output_md/           # Output: Markdown, JSON chunks
├── src/                 # Moduli Python
│   ├── __init__.py
│   ├── converter.py     # Conversione PDF → Markdown
│   ├── cleaning.py      # Pulizia e normalizzazione
│   └── chunking.py      # Chunking per LLM/RAG
├── configs/             # Configurazioni (opzionale)
├── TASK.md              # Specifica completa del task
└── README.md            # Questo file
```

## Installazione

```bash
# Requisiti: Python 3.11+

# Installa le dipendenze
pip install -r requirements.txt
```

## Uso

### Uso base

```bash
# Copia i PDF nella cartella input
cp /path/to/dispense/*.pdf ./input_pdfs/

# Esegui la pipeline
python main.py --input_dir ./input_pdfs --output_dir ./output_md
```

### Opzioni CLI

```bash
python main.py [opzioni]

Opzioni:
  -i, --input_dir PATH          Cartella con i PDF (default: ./input_pdfs)
  -o, --output_dir PATH         Cartella output (default: ./output_md)
  --min_tokens N                Minimo token per chunk (default: 500)
  --max_tokens N                Massimo token per chunk (default: 1500)
  --save_individual_chunks      Salva anche chunk come file .md separati
  --skip_chunking               Solo conversione e pulizia
  --skip_cleaning               Salta la pulizia del markdown
  -v, --verbose                 Output verboso (debug)
```

### Esempi

```bash
# Conversione completa con chunk da ~1000 token
python main.py -i ./input_pdfs -o ./output --max_tokens 1000

# Solo conversione PDF → Markdown (senza chunking)
python main.py -i ./pdfs -o ./output --skip_chunking

# Salva anche i chunk come file separati
python main.py -i ./pdfs -o ./output --save_individual_chunks
```

## Output

Per ogni PDF processato, la pipeline genera:

| File | Descrizione |
|------|-------------|
| `nome_file.md` | Markdown grezzo dalla conversione |
| `nome_file_cleaned.md` | Markdown pulito |
| `nome_file_structure.json` | Struttura del documento (sezioni) |
| `nome_file_chunks.json` | Chunks pronti per LLM/RAG |
| `nome_file_chunks/` | (opzionale) Singoli file .md per chunk |

### Formato Chunks JSON

```json
{
  "source_pdf": "dispensa.pdf",
  "total_chunks": 42,
  "total_tokens": 35000,
  "chunks": [
    {
      "id": "dispensa-0001",
      "title": "Introduzione",
      "content_markdown": "# Introduzione\n\nTesto del chunk...",
      "token_estimate": 850,
      "section_hierarchy": ["Capitolo 1", "Introduzione"]
    }
  ]
}
```

## Architettura

### Fasi della Pipeline

1. **Conversione** (`converter.py`)
   - Usa `marker-pdf` per conversione accurata
   - Preserva formule LaTeX, struttura, immagini
   - Fallback su PyMuPDF se marker non disponibile

2. **Pulizia** (`cleaning.py`)
   - Rimuove numeri di pagina, header/footer
   - Fix hyphenation a capo riga
   - Normalizza whitespace
   - Protegge formule LaTeX

3. **Chunking** (`chunking.py`)
   - Divide rispettando la struttura (sezioni)
   - Non spezza formule o teoremi
   - Target: 500-1500 token per chunk

### Personalizzazione

#### Usare un tokenizer reale (tiktoken)

In `chunking.py`, sostituire `estimate_tokens()`:

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(text: str) -> int:
    return len(enc.encode(text))
```

#### Integrare con Vector DB (RAG)

In `chunking.py` dopo `save_chunks()`:

```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("dispense")

for chunk in chunked_doc.chunks:
    collection.add(
        documents=[chunk.content_markdown],
        metadatas=[{"source": chunk.source_pdf, "title": chunk.title}],
        ids=[chunk.id]
    )
```

## Adattare ad altri corsi

1. **Configurare header/footer** specifici in `cleaning.py`:
   ```python
   config = {
       'header_pattern': r'^Nome Corso.*$',
       'footer_pattern': r'^Prof\..*$'
   }
   cleaned = clean_markdown(text, config)
   ```

2. **Aggiungere pattern** per strutture specifiche (teoremi, definizioni) in `cleaning.py:find_theorems_definitions()`

3. **Modificare token range** in base al modello LLM target:
   - GPT-4: 1000-2000 token
   - Claude: 1500-3000 token
   - Modelli piccoli: 500-1000 token

## Requisiti di sistema

- Python 3.11+
- ~2GB RAM per documenti grandi
- GPU opzionale (accelera marker-pdf)

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| `marker-pdf` non si installa | Usa fallback PyMuPDF: `pip install PyMuPDF` |
| Formule non riconosciute | Verifica che il PDF non sia scansionato (OCR) |
| Chunks troppo piccoli | Aumenta `--min_tokens` |
| Memoria insufficiente | Processa meno PDF alla volta |

## Licenza

MIT

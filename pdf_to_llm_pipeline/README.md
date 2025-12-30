# PDF to LLM Pipeline

Pipeline Python per convertire dispense universitarie PDF in contenuti pronti per LLM/RAG.

## Struttura

```
pdf_to_llm_pipeline/
├── input_pdfs/      # Cartella per le dispense PDF da convertire
├── output_md/       # Output: Markdown e chunks JSON
├── src/             # Moduli Python della pipeline
├── configs/         # Configurazioni (opzionale)
├── TASK.md          # Specifica completa del task
└── README.md        # Questo file
```

## Stato

**In attesa delle dispense PDF** - La pipeline verrà implementata quando i PDF saranno caricati in `input_pdfs/`.

## Uso previsto

```bash
# Installare dipendenze
pip install -r requirements.txt

# Eseguire la pipeline
python main.py --input_dir ./input_pdfs --output_dir ./output_md
```

## Task

Vedere [TASK.md](./TASK.md) per la specifica completa del task da eseguire.

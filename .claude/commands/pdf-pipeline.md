---
description: Avvia pipeline PDF → LLM con supporto resume
argument-hint: [run|resume|status|reset] [pdf_path]
---

# PDF to LLM Pipeline

Converte documenti PDF (dispense universitarie) in formato ottimizzato per LLM/RAG.

## Argomenti

- `run <pdf_path>` - Avvia elaborazione su un singolo PDF
- `run` - Avvia elaborazione su tutti i PDF in input_pdfs/
- `resume` - Riprende elaborazione interrotta
- `status` - Mostra stato attuale della pipeline
- `reset [pdf_name]` - Resetta stato (di un PDF o tutti)
- `force` - Forza rielaborazione completa ignorando checkpoint

## Opzioni Aggiuntive (passabili dopo l'azione)

- `--save_chunks` - Salva anche chunk individuali come file .md
- `--skip_cleaning` - Salta fase di pulizia markdown
- `--skip_chunking` - Salta fase di chunking (solo conversione)
- `--verbose` - Output dettagliato

## Istruzioni Operative

1. **Determina l'azione richiesta** da `$ARGUMENTS`:
   - Se vuoto o `run`: esegui pipeline su tutti i PDF
   - Se `run <path>`: esegui su singolo PDF
   - Se `resume`: riprendi da interruzione
   - Se `status`: mostra solo stato
   - Se `reset`: resetta stato
   - Se `force`: forza rielaborazione

2. **Directory di lavoro**: `pdf_to_llm_pipeline/`
   - Input default: `input_pdfs/`
   - Output default: `output_md/`

3. **Esegui il comando appropriato**:

   ```bash
   cd /home/brusc/Projects/bioimmagini_positano/pdf_to_llm_pipeline
   source venv/bin/activate
   ```

   Poi in base all'azione:

   - **run (tutti)**: `python main.py --save_individual_chunks`
   - **run <pdf>**: `python main.py --pdf "<pdf_path>" --save_individual_chunks`
   - **resume**: `python main.py --resume --save_individual_chunks`
   - **status**: `python main.py --status`
   - **reset**: `python main.py --reset`
   - **reset <nome>**: `python main.py --reset "<nome>"`
   - **force**: `python main.py --force --save_individual_chunks`

4. **Per elaborazioni lunghe**, usa `run_in_background: true` per eseguire in background e monitorare l'output.

5. **Al termine**, mostra:
   - Riepilogo PDF processati
   - Eventuali errori
   - Suggerimento per riprendere se interrotto

## Fasi della Pipeline

1. **Conversione** (PDF → Markdown): Usa marker-pdf per preservare formule LaTeX
2. **Pulizia**: Rimuove header/footer, normalizza whitespace
3. **Chunking**: Divide in chunk 500-1500 token per LLM/RAG

## Esempi d'Uso

```
# Elabora tutti i PDF nella cartella input
/pdf-pipeline run

# Elabora un singolo PDF
/pdf-pipeline run ./dispense/Cap_1.pdf

# Riprendi elaborazione interrotta
/pdf-pipeline resume

# Mostra stato
/pdf-pipeline status

# Forza rielaborazione
/pdf-pipeline force
```

## Note

- I checkpoint vengono salvati in `output_md/.pipeline_state.json`
- La pipeline può essere interrotta in qualsiasi momento e ripresa con `resume`
- Ogni fase (convert, clean, chunk) viene tracciata separatamente
- Per PDF grandi (>30 pagine), l'elaborazione può richiedere diversi minuti

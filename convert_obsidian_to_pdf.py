#!/usr/bin/env python3
"""
Converte i file Markdown Obsidian in PDF pronti per la stampa.
Rimuove elementi specifici di Obsidian (navigazione, tag, frontmatter).
"""

import os
import re
import subprocess
import shutil
from pathlib import Path

# Percorsi
INPUT_DIR = Path("/home/user/bioimmagini_II/bioimmagini_II_obs")
OUTPUT_DIR = Path("/home/user/bioimmagini_II/bioimmagini_II_obs_pdf")
TEMP_DIR = OUTPUT_DIR / "_temp_md"

# File da escludere
EXCLUDE_FILES = {"Template.md", "RIORGANIZZAZIONE_STATO.md"}


def clean_markdown(content: str, source_path: Path) -> str:
    """Pulisce il Markdown dagli elementi specifici di Obsidian."""

    # 1. Rimuove il frontmatter YAML (tra ---)
    content = re.sub(r'^---\n.*?\n---\n*', '', content, flags=re.DOTALL)

    # 1b. Converte i separatori --- rimanenti in * * * per evitare conflitti YAML
    content = re.sub(r'^---$', '* * *', content, flags=re.MULTILINE)

    # 2. Rimuove completamente i callout di navigazione [!nav]
    # Pattern per callout nav con contenuto multiriga
    content = re.sub(
        r'> \[!nav\][^\n]*\n(?:> [^\n]*\n)*',
        '',
        content
    )

    # 3. Converte i link wiki [[file|testo]] in solo testo
    # Prima gestisce [[file#anchor|testo]]
    content = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', content)
    # Poi gestisce [[file]] senza testo alternativo
    content = re.sub(r'\[\[([^\]|#]+)(?:#[^\]|]*)?\]\]', r'\1', content)

    # 4. Converte le immagini embed ![[image.png]] in formato standard
    def convert_image(match):
        img_name = match.group(1)
        # Cerca l'immagine nella cartella images relativa al file sorgente
        source_dir = source_path.parent
        img_path = source_dir / "images" / img_name
        if img_path.exists():
            return f'![{img_name}]({img_path})'
        # Prova anche nella stessa cartella
        img_path = source_dir / img_name
        if img_path.exists():
            return f'![{img_name}]({img_path})'
        # Ritorna comunque un riferimento
        return f'![{img_name}](images/{img_name})'

    content = re.sub(r'!\[\[([^\]]+)\]\]', convert_image, content)

    # 5. Converte altri callout in box stilizzati per print
    # [!tip], [!info], [!warning], [!example], [!note], [!important], [!quote], [!danger], [!summary]
    def convert_callout(match):
        callout_type = match.group(1).lower()
        title = match.group(2) if match.group(2) else ""

        # Mappa tipi callout a emoji/simboli per print
        type_map = {
            'tip': '**Suggerimento**',
            'info': '**Info**',
            'warning': '**Attenzione**',
            'example': '**Esempio**',
            'note': '**Nota**',
            'important': '**Importante**',
            'quote': '**Citazione**',
            'danger': '**Pericolo**',
            'summary': '**Riepilogo**'
        }

        header = type_map.get(callout_type, f'**{callout_type.title()}**')
        if title.strip():
            header = f'{header}: {title.strip()}'

        return f'> {header}\n>'

    # Pattern per callout con titolo opzionale
    content = re.sub(
        r'> \[!(\w+)\][-+]?\s*([^\n]*)\n>',
        convert_callout,
        content
    )

    # 6. Rimuove gli hashtag inline (ma mantiene quelli in titoli)
    # Rimuove solo tag isolati tipo #MRI #TAC non usati come titoli
    content = re.sub(r'(?<!#)`#[a-zA-Z0-9-]+`', '', content)  # Tag in backtick

    # 7. Rimuove righe vuote multiple
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


def convert_to_pdf(md_path: Path, pdf_path: Path, images_dir: Path = None):
    """Converte un file Markdown in PDF usando pandoc."""

    # Costruisce il comando pandoc
    cmd = [
        'pandoc',
        str(md_path),
        '-o', str(pdf_path),
        '--pdf-engine=xelatex',
        '-V', 'geometry:margin=2.5cm',
        '-V', 'fontsize=11pt',
        '-V', 'documentclass=article',
        '-V', 'lang=it',
        '--highlight-style=tango',
    ]

    # Aggiunge la directory delle risorse se specificata
    if images_dir and images_dir.exists():
        cmd.extend(['--resource-path', str(images_dir)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f"  Errore pandoc: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Timeout durante la conversione")
        return False
    except Exception as e:
        print(f"  Errore: {e}")
        return False


def process_file(md_file: Path, output_base: Path):
    """Processa un singolo file Markdown."""

    # Calcola il percorso di output mantenendo la struttura
    rel_path = md_file.relative_to(INPUT_DIR)

    # Nome file PDF
    pdf_name = rel_path.with_suffix('.pdf')
    pdf_path = output_base / pdf_name

    # Crea la directory di output
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Legge e pulisce il contenuto
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    cleaned_content = clean_markdown(content, md_file)

    # Salva il Markdown pulito temporaneamente
    temp_md = TEMP_DIR / rel_path
    temp_md.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_md, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    # Converte in PDF
    images_dir = md_file.parent / "images"
    success = convert_to_pdf(temp_md, pdf_path, images_dir if images_dir.exists() else md_file.parent)

    return success


def main():
    """Funzione principale."""

    print("=" * 60)
    print("Conversione Obsidian Markdown -> PDF")
    print("=" * 60)

    # Crea directory temporanea
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Trova tutti i file Markdown
    md_files = list(INPUT_DIR.rglob("*.md"))

    # Filtra file da escludere e cartella .obsidian
    md_files = [
        f for f in md_files
        if f.name not in EXCLUDE_FILES
        and ".obsidian" not in str(f)
    ]

    print(f"\nTrovati {len(md_files)} file Markdown da convertire\n")

    success_count = 0
    fail_count = 0

    for i, md_file in enumerate(sorted(md_files), 1):
        rel_path = md_file.relative_to(INPUT_DIR)
        print(f"[{i}/{len(md_files)}] {rel_path}")

        if process_file(md_file, OUTPUT_DIR):
            success_count += 1
            print(f"  -> OK")
        else:
            fail_count += 1
            print(f"  -> FALLITO")

    # Pulizia directory temporanea
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"Completato: {success_count} OK, {fail_count} falliti")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

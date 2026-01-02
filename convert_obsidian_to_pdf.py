#!/usr/bin/env python3
"""
Converte i file Markdown Obsidian in PDF pronti per la stampa.
Rimuove elementi specifici di Obsidian (navigazione, tag, frontmatter).
Versione migliorata con migliore formattazione liste e tipografia.
"""

import os
import re
import subprocess
import shutil
from pathlib import Path

# Percorsi
INPUT_DIR = Path("/home/brusc/Projects/bioimmagini_positano/dispense_obs")
OUTPUT_DIR = Path("/home/brusc/Projects/bioimmagini_positano/dispense_pdf")
TEMP_DIR = OUTPUT_DIR / "_temp_md"

# File da escludere
EXCLUDE_FILES = {"Template.md", "RIORGANIZZAZIONE_STATO.md"}

# Cache globale delle immagini (popolata all'avvio)
IMAGE_INDEX = {}


def build_image_index():
    """Costruisce un indice globale di tutte le immagini nel vault."""
    global IMAGE_INDEX
    for img_path in INPUT_DIR.rglob("*.png"):
        IMAGE_INDEX[img_path.name] = img_path
    for img_path in INPUT_DIR.rglob("*.jpg"):
        IMAGE_INDEX[img_path.name] = img_path
    for img_path in INPUT_DIR.rglob("*.jpeg"):
        IMAGE_INDEX[img_path.name] = img_path
    for img_path in INPUT_DIR.rglob("*.gif"):
        IMAGE_INDEX[img_path.name] = img_path
    print(f"Indicizzate {len(IMAGE_INDEX)} immagini nel vault")

# Template LaTeX per migliore formattazione
LATEX_HEADER = r"""
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{titlesec}
\usepackage{xcolor}

% Definizione colori per titoli (scala di calore)
\definecolor{title1}{RGB}{180, 0, 0}      % Rosso scuro - # titoli
\definecolor{title2}{RGB}{0, 70, 140}     % Blu - ## titoli
\definecolor{title3}{RGB}{0, 120, 60}     % Verde - ### titoli
\definecolor{title4}{RGB}{140, 80, 0}     % Arancio scuro - #### titoli

% Configurazione liste con spaziatura migliore
\setlist[itemize]{itemsep=0.3em, parsep=0.2em, topsep=0.5em}
\setlist[enumerate]{itemsep=0.3em, parsep=0.2em, topsep=0.5em}
\setlist[itemize,2]{itemsep=0.2em, parsep=0.1em}
\setlist[enumerate,2]{itemsep=0.2em, parsep=0.1em}

% Formattazione titoli con colori
\titleformat{\section}
  {\normalfont\Large\bfseries\color{title1}}{\thesection}{1em}{}
\titleformat{\subsection}
  {\normalfont\large\bfseries\color{title2}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries\color{title3}}{\thesubsubsection}{1em}{}
\titleformat{\paragraph}
  {\normalfont\normalsize\bfseries\color{title4}}{\theparagraph}{1em}{}

% Spaziatura titoli
\titlespacing*{\section}{0pt}{1.5em}{0.8em}
\titlespacing*{\subsection}{0pt}{1.2em}{0.6em}
\titlespacing*{\subsubsection}{0pt}{1em}{0.5em}

% Interlinea leggermente aumentata
\linespread{1.1}
"""


def clean_markdown(content: str, source_path: Path) -> str:
    """Pulisce il Markdown dagli elementi specifici di Obsidian."""

    # 1. Rimuove il frontmatter YAML (tra ---)
    content = re.sub(r'^---\s*\n.*?\n---\s*\n*', '', content, flags=re.DOTALL)

    # 1b. Converte i separatori --- rimanenti in linea orizzontale (più sicuro per pandoc)
    # Usa pattern più robusto che gestisce spazi e diverse situazioni
    # Usiamo asterischi per la linea orizzontale (standard markdown)
    content = re.sub(r'^---\s*$', '\n* * *\n', content, flags=re.MULTILINE)
    content = re.sub(r'\n---\s*\n', '\n\n* * *\n\n', content)

    # 2. Rimuove completamente i callout di navigazione [!nav]
    # Pattern per callout nav con contenuto multiriga
    content = re.sub(
        r'> \[!nav\][^\n]*\n(?:> [^\n]*\n)*',
        '',
        content
    )

    # 3. Converte le immagini embed ![[image.png]] in formato standard
    # IMPORTANTE: deve essere eseguito PRIMA della conversione wiki link
    # altrimenti ![[image.png]] diventa !image.png
    found_images = []

    def convert_image(match):
        img_name = match.group(1)
        # Usa l'indice globale per trovare l'immagine
        if img_name in IMAGE_INDEX:
            img_path = IMAGE_INDEX[img_name]
            found_images.append(img_path)
            # Usa path assoluto per pandoc
            return f'![{img_name}]({img_path.absolute()})'
        # Fallback: cerca nella cartella images relativa
        source_dir = source_path.parent
        img_path = source_dir / "images" / img_name
        if img_path.exists():
            found_images.append(img_path)
            return f'![{img_name}]({img_path.absolute()})'
        # Se non trovata, ritorna placeholder
        return f'[Immagine non trovata: {img_name}]'

    content = re.sub(r'!\[\[([^\]]+)\]\]', convert_image, content)

    # 4. Converte i link wiki [[file|testo]] in solo testo
    # Prima gestisce [[file#anchor|testo]]
    content = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', content)
    # Poi gestisce [[file]] senza testo alternativo
    content = re.sub(r'\[\[([^\]|#]+)(?:#[^\]|]*)?\]\]', r'\1', content)

    # 5. Converte callout in blocchi ben formattati
    def convert_callout(match):
        callout_type = match.group(1).lower()
        title = match.group(2).strip() if match.group(2) else ""
        callout_content = match.group(3) if match.group(3) else ""

        # Mappa tipi callout
        type_map = {
            'tip': '💡 **Suggerimento**',
            'info': 'ℹ️ **Info**',
            'warning': '⚠️ **Attenzione**',
            'example': '📋 **Esempio**',
            'note': '📝 **Nota**',
            'important': '❗ **Importante**',
            'quote': '💬 **Citazione**',
            'danger': '🚨 **Pericolo**',
            'summary': '📌 **Riepilogo**'
        }

        header = type_map.get(callout_type, f'**{callout_type.title()}**')
        if title:
            header = f'{header}: {title}'

        # Pulisce il contenuto del callout (rimuove i > iniziali)
        lines = callout_content.strip().split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                line = line[1:].strip()
            if line:
                clean_lines.append(line)

        # Formatta come lista se contiene elementi separati da -
        content_text = '\n'.join(clean_lines)

        # Converti elementi inline " - " in lista (ma NON se contiene formule matematiche)
        if ' - ' in content_text and not content_text.startswith('-') and '$$' not in content_text and '$' not in content_text:
            items = [item.strip() for item in content_text.split(' - ') if item.strip()]
            if len(items) > 1:
                content_text = '\n'.join(f'- {item}' for item in items)

        return f'\n\n> {header}\n>\n> {content_text}\n\n'

    # Pattern per callout completi (header + contenuto)
    content = re.sub(
        r'> \[!(\w+)\][-+]?\s*([^\n]*)\n((?:> [^\n]*\n?)*)',
        convert_callout,
        content
    )

    # 6. NUOVO: Converte pattern "**Label:** - item - item" in liste corrette
    def convert_inline_list(match):
        label = match.group(1)
        items_text = match.group(2)

        # NON convertire se contiene formule matematiche
        if '$$' in items_text or '$' in items_text:
            return match.group(0)

        # Splitta per " - "
        items = [item.strip() for item in items_text.split(' - ') if item.strip()]

        if len(items) > 1:
            # Crea lista markdown corretta
            list_items = '\n'.join(f'- {item}' for item in items)
            return f'**{label}:**\n\n{list_items}'
        else:
            return match.group(0)  # Ritorna originale se non è una lista

    # Applica conversione per pattern inline
    content = re.sub(
        r'\*\*([^*]+):\*\*\s*-\s*(.+?)(?=\n\n|\n\*\*|\n##|\n#|\Z)',
        convert_inline_list,
        content,
        flags=re.DOTALL
    )

    # 7. Assicura newline prima di ogni list item (- o numero.)
    # Questo garantisce che pandoc riconosca le liste
    content = re.sub(r'([^\n])\n(- |\d+\. )', r'\1\n\n\2', content)

    # 8. Assicura spazio dopo titoli prima di liste
    content = re.sub(r'(#+[^\n]+)\n(- |\d+\. )', r'\1\n\n\2', content)

    # 9. Rimuove gli hashtag inline (ma mantiene quelli in titoli)
    content = re.sub(r'(?<!#)`#[a-zA-Z0-9-]+`', '', content)

    # 10. Normalizza righe vuote (max 2 consecutive)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 11. Assicura che le liste nested abbiano indentazione corretta
    lines = content.split('\n')
    fixed_lines = []
    in_list = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('- ') or re.match(r'^\d+\. ', stripped):
            in_list = True
            # Controlla se è un subitem (preceduto da spazi nel sorgente)
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces >= 2:
                fixed_lines.append('  ' + stripped)  # Indenta come subitem
            else:
                fixed_lines.append(stripped)
        elif stripped == '' and in_list:
            fixed_lines.append('')
        else:
            in_list = False
            fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    return content.strip()


def convert_to_pdf(md_path: Path, pdf_path: Path, images_dir: Path = None):
    """Converte un file Markdown in PDF usando pandoc con formattazione migliorata."""

    # Crea file header LaTeX temporaneo
    header_file = md_path.parent / "_header.tex"
    with open(header_file, 'w', encoding='utf-8') as f:
        f.write(LATEX_HEADER)

    # Costruisce il comando pandoc con opzioni migliorate
    cmd = [
        'pandoc',
        str(md_path),
        '-o', str(pdf_path),
        '--pdf-engine=xelatex',
        '-V', 'geometry:margin=2.5cm',
        '-V', 'fontsize=11pt',
        '-V', 'documentclass=article',
        '-V', 'lang=it',
        '-V', 'mainfont=DejaVu Serif',
        '-V', 'sansfont=DejaVu Sans',
        '-V', 'monofont=DejaVu Sans Mono',
        '--highlight-style=tango',
        '-H', str(header_file),
        '--wrap=preserve',  # Preserva line breaks
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

        # Rimuovi header temporaneo
        if header_file.exists():
            header_file.unlink()

        if result.returncode != 0:
            print(f"  Errore pandoc: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Timeout durante la conversione")
        if header_file.exists():
            header_file.unlink()
        return False
    except Exception as e:
        print(f"  Errore: {e}")
        if header_file.exists():
            header_file.unlink()
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

    # Costruisce indice globale delle immagini
    build_image_index()

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

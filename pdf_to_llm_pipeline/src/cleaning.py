"""
cleaning.py - Modulo per pulizia e normalizzazione del Markdown

Funzionalità:
- Rimozione numeri di pagina
- Rimozione header/footer ripetuti
- Normalizzazione linee vuote
- Fix hyphenation a capo riga
- Preservazione formule LaTeX
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def clean_markdown(text: str, config: Optional[dict] = None) -> str:
    """
    Pulisce e normalizza il Markdown convertito da PDF.

    Args:
        text: Testo markdown grezzo
        config: Configurazione opzionale per la pulizia

    Returns:
        Testo markdown pulito
    """
    if not text:
        return text

    config = config or {}

    # Proteggi le formule LaTeX prima della pulizia
    text, latex_placeholders = _protect_latex(text)

    # Applica le pulizie
    text = _remove_page_numbers(text)
    text = _remove_headers_footers(text, config.get('header_pattern'), config.get('footer_pattern'))
    text = _fix_hyphenation(text)
    text = _normalize_whitespace(text)
    text = _normalize_headings(text)
    text = _clean_image_references(text)

    # Ripristina le formule LaTeX
    text = _restore_latex(text, latex_placeholders)

    return text.strip()


def _protect_latex(text: str) -> tuple[str, dict]:
    """
    Protegge le formule LaTeX sostituendole con placeholder.
    Questo evita che vengano alterate durante la pulizia.
    """
    placeholders = {}
    counter = 0

    # Pattern per formule LaTeX
    patterns = [
        # Display math: $$ ... $$ o \[ ... \]
        (r'\$\$[\s\S]*?\$\$', 'DISPLAY'),
        (r'\\\[[\s\S]*?\\\]', 'DISPLAY'),
        # Inline math: $ ... $ o \( ... \)
        (r'(?<!\$)\$(?!\$)(?:[^$]|\\\$)+\$(?!\$)', 'INLINE'),
        (r'\\\([\s\S]*?\\\)', 'INLINE'),
        # Ambienti LaTeX: \begin{...} ... \end{...}
        (r'\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}', 'ENV'),
    ]

    for pattern, latex_type in patterns:
        def replace(match, counter=counter, latex_type=latex_type, placeholders=placeholders):
            placeholder = f"__LATEX_{latex_type}_{counter}__"
            placeholders[placeholder] = match.group(0)
            return placeholder

        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):  # Reverse per mantenere gli indici corretti
            placeholder = f"__LATEX_{latex_type}_{counter}__"
            placeholders[placeholder] = match.group(0)
            text = text[:match.start()] + placeholder + text[match.end():]
            counter += 1

    return text, placeholders


def _restore_latex(text: str, placeholders: dict) -> str:
    """Ripristina le formule LaTeX dai placeholder."""
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)
    return text


def _remove_page_numbers(text: str) -> str:
    """Rimuove i numeri di pagina comuni nei PDF."""
    patterns = [
        # Numeri di pagina isolati su una riga
        r'^\s*\d{1,4}\s*$',
        # Formato "Page X" o "Pagina X"
        r'^\s*(?:Page|Pagina|Pag\.?)\s*\d+\s*$',
        # Formato "X of Y" o "X di Y"
        r'^\s*\d+\s*(?:of|di|/)\s*\d+\s*$',
        # Commenti HTML con numeri di pagina (da fallback converter)
        r'<!--\s*Page\s*\d+\s*-->',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    return text


def _remove_headers_footers(
    text: str,
    header_pattern: Optional[str] = None,
    footer_pattern: Optional[str] = None
) -> str:
    """
    Rimuove header e footer ripetuti.

    Args:
        text: Testo da pulire
        header_pattern: Pattern regex personalizzato per header
        footer_pattern: Pattern regex personalizzato per footer
    """
    lines = text.split('\n')

    # Trova linee ripetute (potenziali header/footer)
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 3:  # Ignora linee troppo corte
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # Rimuovi linee che appaiono troppe volte (probabili header/footer)
    threshold = max(3, len(lines) // 20)  # Soglia dinamica
    repeated_lines = {line for line, count in line_counts.items() if count >= threshold}

    # Applica pattern personalizzati se forniti
    if header_pattern:
        text = re.sub(header_pattern, '', text, flags=re.MULTILINE)
    if footer_pattern:
        text = re.sub(footer_pattern, '', text, flags=re.MULTILINE)

    # Filtra le linee ripetute
    cleaned_lines = []
    for line in lines:
        if line.strip() not in repeated_lines:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def _fix_hyphenation(text: str) -> str:
    """
    Ricongiunge parole spezzate a fine riga (hyphenation).
    Es: "mate-\nmatica" → "matematica"
    """
    # Pattern: parola con trattino a fine riga seguita da continuazione
    pattern = r'(\w+)-\s*\n\s*(\w+)'

    def fix_match(match):
        return match.group(1) + match.group(2)

    return re.sub(pattern, fix_match, text)


def _normalize_whitespace(text: str) -> str:
    """Normalizza spazi bianchi e linee vuote."""
    # Rimuovi spazi a fine riga
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    # Riduci linee vuote multiple a massimo 2
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Rimuovi spazi multipli (ma non all'inizio riga per indentazione)
    text = re.sub(r'(?<!^)[ \t]{2,}', ' ', text, flags=re.MULTILINE)

    return text


def _normalize_headings(text: str) -> str:
    """Normalizza le intestazioni Markdown."""
    # Assicura spazio dopo # nelle intestazioni
    text = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', text, flags=re.MULTILINE)

    # Rimuovi # senza testo
    text = re.sub(r'^#{1,6}\s*$', '', text, flags=re.MULTILINE)

    return text


def _clean_image_references(text: str) -> str:
    """
    Pulisce i riferimenti alle immagini.
    Mantiene il formato ![alt](path) ma normalizza.
    """
    # Normalizza riferimenti immagine
    # Pattern: ![qualsiasi](path)
    def clean_image(match):
        alt = match.group(1).strip() or "image"
        path = match.group(2).strip()
        return f"![{alt}]({path})"

    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', clean_image, text)

    return text


def clean_for_chunking(text: str) -> str:
    """
    Pulizia aggiuntiva specifica per preparare il testo al chunking.
    Più aggressiva della pulizia standard.
    """
    text = clean_markdown(text)

    # Rimuovi commenti HTML residui
    text = re.sub(r'<!--[\s\S]*?-->', '', text)

    # Rimuovi link vuoti
    text = re.sub(r'\[([^\]]*)\]\(\s*\)', r'\1', text)

    # Normalizza liste
    text = re.sub(r'^(\s*)[•◦▪▸►]\s*', r'\1- ', text, flags=re.MULTILINE)

    return text.strip()


# Funzioni di utilità per identificare strutture speciali

def find_theorems_definitions(text: str) -> list[dict]:
    """
    Trova teoremi, definizioni, lemmi, ecc. nel testo.
    Utile per il chunking intelligente.
    """
    patterns = [
        (r'(?i)\*?\*?(Teorema|Theorem)\s*[\d.]*\*?\*?[:\s]', 'theorem'),
        (r'(?i)\*?\*?(Definizione|Definition)\s*[\d.]*\*?\*?[:\s]', 'definition'),
        (r'(?i)\*?\*?(Lemma)\s*[\d.]*\*?\*?[:\s]', 'lemma'),
        (r'(?i)\*?\*?(Corollario|Corollary)\s*[\d.]*\*?\*?[:\s]', 'corollary'),
        (r'(?i)\*?\*?(Proposizione|Proposition)\s*[\d.]*\*?\*?[:\s]', 'proposition'),
        (r'(?i)\*?\*?(Esempio|Example)\s*[\d.]*\*?\*?[:\s]', 'example'),
        (r'(?i)\*?\*?(Dimostrazione|Proof)\s*[\d.]*\*?\*?[:\s]', 'proof'),
    ]

    results = []
    for pattern, struct_type in patterns:
        for match in re.finditer(pattern, text):
            results.append({
                'type': struct_type,
                'position': match.start(),
                'text': match.group(0)
            })

    return sorted(results, key=lambda x: x['position'])

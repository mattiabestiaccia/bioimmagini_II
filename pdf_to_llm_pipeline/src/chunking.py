"""
chunking.py - Modulo per chunking del Markdown per LLM/RAG

Funzionalità:
- Chunking intelligente rispettando la struttura del documento
- Preservazione delle formule LaTeX
- Preservazione di teoremi/definizioni
- Output JSON strutturato per RAG

PUNTO DI INTEGRAZIONE TOKENIZER:
Per usare un tokenizer reale invece del proxy parole, sostituire
la funzione `estimate_tokens()` con una che usi tiktoken:

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")  # Per GPT-4/Claude

    def estimate_tokens(text: str) -> int:
        return len(enc.encode(text))

PUNTO DI INTEGRAZIONE VECTOR DB:
Per salvare i chunk in un vector DB (es. ChromaDB):

    import chromadb
    client = chromadb.Client()
    collection = client.create_collection("dispense")

    for chunk in chunks:
        collection.add(
            documents=[chunk['content_markdown']],
            metadatas=[{"source": chunk['source_pdf'], "title": chunk['title']}],
            ids=[chunk['id']]
        )
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


# === CONFIGURAZIONE TOKEN ===

# Fattore di conversione parole → token (approssimativo)
# Per italiano/inglese tecnico, ~1.3-1.5 token per parola
TOKENS_PER_WORD = 1.4

# Fattore caratteri → token (fallback)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """
    Stima il numero di token in un testo.

    Usa un proxy basato sul conteggio parole. Per maggiore precisione,
    sostituire con tiktoken o altro tokenizer reale.
    """
    # Conta parole (split su whitespace)
    words = len(text.split())
    return int(words * TOKENS_PER_WORD)


def estimate_tokens_chars(text: str) -> int:
    """Stima alternativa basata sui caratteri."""
    return len(text) // CHARS_PER_TOKEN


@dataclass
class Chunk:
    """Rappresenta un singolo chunk di testo."""
    id: str
    title: str
    content_markdown: str
    token_estimate: int
    page_range: list[int] = field(default_factory=lambda: [0, 0])
    section_hierarchy: list[str] = field(default_factory=list)


@dataclass
class ChunkedDocument:
    """Documento suddiviso in chunks."""
    source_pdf: str
    total_chunks: int
    total_tokens: int
    chunks: list[Chunk]


def chunk_markdown(
    markdown: str,
    source_name: str,
    min_tokens: int = 500,
    max_tokens: int = 1500,
    token_estimator: Optional[Callable[[str], int]] = None
) -> ChunkedDocument:
    """
    Divide il markdown in chunk ottimizzati per LLM/RAG.

    Args:
        markdown: Testo markdown da dividere
        source_name: Nome del file sorgente (per ID)
        min_tokens: Minimo token per chunk (evita chunk troppo piccoli)
        max_tokens: Massimo token per chunk
        token_estimator: Funzione custom per stima token (default: estimate_tokens)

    Returns:
        ChunkedDocument con tutti i chunks
    """
    if token_estimator is None:
        token_estimator = estimate_tokens

    # Identifica i boundaries naturali
    sections = _split_by_headers(markdown)

    chunks = []
    chunk_counter = 0
    current_hierarchy = []

    for section in sections:
        section_chunks = _chunk_section(
            section['content'],
            section['title'],
            section['level'],
            source_name,
            chunk_counter,
            min_tokens,
            max_tokens,
            token_estimator,
            current_hierarchy.copy()
        )

        # Aggiorna la gerarchia
        if section['level'] > 0:
            # Rimuovi livelli >= al corrente
            current_hierarchy = [h for h in current_hierarchy if h[0] < section['level']]
            current_hierarchy.append((section['level'], section['title']))

        chunks.extend(section_chunks)
        chunk_counter += len(section_chunks)

    # Calcola statistiche
    total_tokens = sum(c.token_estimate for c in chunks)

    return ChunkedDocument(
        source_pdf=source_name,
        total_chunks=len(chunks),
        total_tokens=total_tokens,
        chunks=chunks
    )


def _split_by_headers(markdown: str) -> list[dict]:
    """
    Divide il markdown usando le intestazioni come separatori.
    """
    # Pattern per intestazioni markdown
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    sections = []
    last_end = 0
    current_level = 0
    current_title = "Introduzione"

    for match in header_pattern.finditer(markdown):
        # Salva la sezione precedente
        if match.start() > last_end:
            content = markdown[last_end:match.start()].strip()
            if content:
                sections.append({
                    'level': current_level,
                    'title': current_title,
                    'content': content
                })

        # Aggiorna per la nuova sezione
        current_level = len(match.group(1))
        current_title = match.group(2).strip()
        last_end = match.end()

    # Ultima sezione
    if last_end < len(markdown):
        content = markdown[last_end:].strip()
        if content:
            sections.append({
                'level': current_level,
                'title': current_title,
                'content': content
            })

    # Se non ci sono intestazioni, tratta tutto come una sezione
    if not sections:
        sections.append({
            'level': 0,
            'title': "Contenuto",
            'content': markdown.strip()
        })

    return sections


def _chunk_section(
    content: str,
    title: str,
    level: int,
    source_name: str,
    start_counter: int,
    min_tokens: int,
    max_tokens: int,
    token_estimator: Callable[[str], int],
    hierarchy: list
) -> list[Chunk]:
    """
    Divide una sezione in chunks rispettando i limiti di token.
    """
    chunks = []
    counter = start_counter

    # Se la sezione è abbastanza piccola, restituiscila intera
    total_tokens = token_estimator(content)
    if total_tokens <= max_tokens:
        if total_tokens >= min_tokens or not chunks:  # Accetta piccoli se è l'unico
            chunks.append(Chunk(
                id=f"{_sanitize_name(source_name)}-{counter:04d}",
                title=title,
                content_markdown=content,
                token_estimate=total_tokens,
                section_hierarchy=[h[1] for h in hierarchy] + [title] if title else [h[1] for h in hierarchy]
            ))
        return chunks

    # Sezione troppo grande: divide su boundaries intelligenti
    paragraphs = _split_preserving_structures(content)

    current_chunk_parts = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = token_estimator(para)

        # Se il paragrafo singolo è troppo grande, dividilo
        if para_tokens > max_tokens:
            # Salva chunk corrente se non vuoto
            if current_chunk_parts:
                chunk_content = '\n\n'.join(current_chunk_parts)
                chunks.append(Chunk(
                    id=f"{_sanitize_name(source_name)}-{counter:04d}",
                    title=title,
                    content_markdown=chunk_content,
                    token_estimate=current_tokens,
                    section_hierarchy=[h[1] for h in hierarchy] + [title] if title else [h[1] for h in hierarchy]
                ))
                counter += 1
                current_chunk_parts = []
                current_tokens = 0

            # Dividi il paragrafo grande
            sub_chunks = _split_large_paragraph(para, max_tokens, token_estimator)
            for sub in sub_chunks:
                chunks.append(Chunk(
                    id=f"{_sanitize_name(source_name)}-{counter:04d}",
                    title=f"{title} (cont.)" if counter > start_counter else title,
                    content_markdown=sub,
                    token_estimate=token_estimator(sub),
                    section_hierarchy=[h[1] for h in hierarchy] + [title] if title else [h[1] for h in hierarchy]
                ))
                counter += 1
            continue

        # Controlla se aggiungere al chunk corrente o iniziarne uno nuovo
        if current_tokens + para_tokens > max_tokens and current_chunk_parts:
            # Salva chunk corrente
            chunk_content = '\n\n'.join(current_chunk_parts)
            chunks.append(Chunk(
                id=f"{_sanitize_name(source_name)}-{counter:04d}",
                title=title if counter == start_counter else f"{title} (cont.)",
                content_markdown=chunk_content,
                token_estimate=current_tokens,
                section_hierarchy=[h[1] for h in hierarchy] + [title] if title else [h[1] for h in hierarchy]
            ))
            counter += 1
            current_chunk_parts = []
            current_tokens = 0

        current_chunk_parts.append(para)
        current_tokens += para_tokens

    # Salva ultimo chunk
    if current_chunk_parts:
        chunk_content = '\n\n'.join(current_chunk_parts)
        if current_tokens >= min_tokens or not chunks:
            chunks.append(Chunk(
                id=f"{_sanitize_name(source_name)}-{counter:04d}",
                title=title if counter == start_counter else f"{title} (cont.)",
                content_markdown=chunk_content,
                token_estimate=current_tokens,
                section_hierarchy=[h[1] for h in hierarchy] + [title] if title else [h[1] for h in hierarchy]
            ))
        elif chunks:
            # Unisci al chunk precedente se troppo piccolo
            chunks[-1].content_markdown += '\n\n' + chunk_content
            chunks[-1].token_estimate += current_tokens

    return chunks


def _split_preserving_structures(content: str) -> list[str]:
    """
    Divide il contenuto in paragrafi preservando strutture speciali.
    Non spezza formule LaTeX, teoremi, definizioni.
    """
    # Prima proteggi le formule display
    protected = content
    latex_blocks = []

    # Proteggi formule display $$ ... $$
    def protect_latex(match):
        idx = len(latex_blocks)
        latex_blocks.append(match.group(0))
        return f"__LATEX_BLOCK_{idx}__"

    protected = re.sub(r'\$\$[\s\S]*?\$\$', protect_latex, protected)
    protected = re.sub(r'\\\[[\s\S]*?\\\]', protect_latex, protected)
    protected = re.sub(r'\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}', protect_latex, protected)

    # Dividi su doppi newline (paragrafi)
    paragraphs = re.split(r'\n\s*\n', protected)

    # Ripristina formule
    result = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Ripristina i blocchi LaTeX
        for idx, block in enumerate(latex_blocks):
            para = para.replace(f"__LATEX_BLOCK_{idx}__", block)

        result.append(para)

    return result


def _split_large_paragraph(
    paragraph: str,
    max_tokens: int,
    token_estimator: Callable[[str], int]
) -> list[str]:
    """
    Divide un paragrafo troppo grande in parti più piccole.
    Cerca di spezzare su frasi complete.
    """
    # Prova a dividere su frasi
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)

    if len(sentences) <= 1:
        # Nessuna frase riconoscibile, dividi su spazi
        words = paragraph.split()
        chunks = []
        current = []
        current_tokens = 0

        for word in words:
            word_tokens = token_estimator(word + " ")
            if current_tokens + word_tokens > max_tokens and current:
                chunks.append(' '.join(current))
                current = []
                current_tokens = 0
            current.append(word)
            current_tokens += word_tokens

        if current:
            chunks.append(' '.join(current))

        return chunks

    # Raggruppa frasi
    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = token_estimator(sentence)

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(' '.join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(' '.join(current))

    return chunks


def _sanitize_name(name: str) -> str:
    """Sanitizza un nome per uso come ID."""
    # Rimuovi estensione
    name = Path(name).stem
    # Sostituisci caratteri non alfanumerici
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Rimuovi underscore multipli
    name = re.sub(r'_+', '_', name)
    return name.lower()


def save_chunks(
    chunked_doc: ChunkedDocument,
    output_dir: Path,
    save_individual_files: bool = False
) -> Path:
    """
    Salva i chunk in formato JSON.

    Args:
        chunked_doc: Documento chunked
        output_dir: Cartella output
        save_individual_files: Se True, salva anche singoli file .md per chunk

    Returns:
        Path al file JSON principale
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepara struttura JSON
    json_data = {
        "source_pdf": chunked_doc.source_pdf,
        "total_chunks": chunked_doc.total_chunks,
        "total_tokens": chunked_doc.total_tokens,
        "chunks": [asdict(c) for c in chunked_doc.chunks]
    }

    # Salva JSON principale
    base_name = _sanitize_name(chunked_doc.source_pdf)
    json_path = output_dir / f"{base_name}_chunks.json"
    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    logger.info(f"Salvato: {json_path}")

    # Salva file individuali se richiesto
    if save_individual_files:
        chunks_dir = output_dir / f"{base_name}_chunks"
        chunks_dir.mkdir(exist_ok=True)

        for chunk in chunked_doc.chunks:
            chunk_path = chunks_dir / f"{chunk.id}.md"
            # Aggiungi header con metadati
            header = f"---\nid: {chunk.id}\ntitle: {chunk.title}\ntokens: {chunk.token_estimate}\n---\n\n"
            chunk_path.write_text(header + chunk.content_markdown, encoding='utf-8')

        logger.info(f"Salvati {len(chunked_doc.chunks)} chunk individuali in {chunks_dir}")

    return json_path


def load_chunks(json_path: Path) -> ChunkedDocument:
    """Carica un documento chunked da file JSON."""
    data = json.loads(json_path.read_text(encoding='utf-8'))

    chunks = [
        Chunk(
            id=c['id'],
            title=c['title'],
            content_markdown=c['content_markdown'],
            token_estimate=c['token_estimate'],
            page_range=c.get('page_range', [0, 0]),
            section_hierarchy=c.get('section_hierarchy', [])
        )
        for c in data['chunks']
    ]

    return ChunkedDocument(
        source_pdf=data['source_pdf'],
        total_chunks=data['total_chunks'],
        total_tokens=data['total_tokens'],
        chunks=chunks
    )

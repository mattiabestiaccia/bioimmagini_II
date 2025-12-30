"""
converter.py - Modulo per conversione PDF → Markdown

Utilizza marker-pdf per la conversione, scelto per:
- Eccellente preservazione delle formule LaTeX (inline e block)
- Riconoscimento accurato della struttura (titoli, sezioni, sottosezioni)
- Output JSON opzionale con metadati strutturati
- Ottimo supporto per documenti accademici/scientifici
"""

import logging
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Risultato della conversione di un PDF."""
    source_pdf: str
    markdown_content: str
    structure: Optional[dict] = None
    page_count: int = 0
    success: bool = True
    error_message: Optional[str] = None


def convert_pdf_to_markdown(
    pdf_path: Path,
    output_dir: Path,
    save_json_structure: bool = True
) -> ConversionResult:
    """
    Converte un singolo PDF in Markdown usando marker-pdf.

    Args:
        pdf_path: Percorso al file PDF
        output_dir: Cartella dove salvare gli output
        save_json_structure: Se True, salva anche la struttura JSON

    Returns:
        ConversionResult con markdown e metadati
    """
    pdf_name = pdf_path.stem

    try:
        # Import marker-pdf
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError:
            # Fallback se marker non è installato - usa implementazione base
            logger.warning("marker-pdf non installato, uso fallback PyMuPDF")
            return _fallback_convert(pdf_path, output_dir, save_json_structure)

        logger.info(f"Conversione PDF: {pdf_path.name}")

        # Inizializza i modelli di marker (lazy loading)
        model_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=model_dict)

        # Esegui conversione
        rendered = converter(str(pdf_path))
        markdown_content, _, images = text_from_rendered(rendered)

        # Estrai struttura se disponibile
        structure = None
        if hasattr(rendered, 'metadata'):
            structure = {
                "sections": _extract_sections(markdown_content),
                "metadata": rendered.metadata if hasattr(rendered, 'metadata') else {}
            }

        # Salva file Markdown
        md_path = output_dir / f"{pdf_name}.md"
        md_path.write_text(markdown_content, encoding='utf-8')
        logger.info(f"Salvato: {md_path}")

        # Salva struttura JSON se richiesto
        if save_json_structure and structure:
            json_path = output_dir / f"{pdf_name}_structure.json"
            json_path.write_text(
                json.dumps(structure, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            logger.info(f"Salvato: {json_path}")

        # Salva immagini estratte
        if images:
            img_dir = output_dir / f"{pdf_name}_images"
            img_dir.mkdir(exist_ok=True)
            for img_name, img_data in images.items():
                img_path = img_dir / img_name
                img_path.write_bytes(img_data)
            logger.info(f"Salvate {len(images)} immagini in {img_dir}")

        return ConversionResult(
            source_pdf=pdf_path.name,
            markdown_content=markdown_content,
            structure=structure,
            page_count=_count_pages(pdf_path),
            success=True
        )

    except Exception as e:
        logger.error(f"Errore conversione {pdf_path.name}: {e}")
        return ConversionResult(
            source_pdf=pdf_path.name,
            markdown_content="",
            success=False,
            error_message=str(e)
        )


def _fallback_convert(
    pdf_path: Path,
    output_dir: Path,
    save_json_structure: bool
) -> ConversionResult:
    """
    Fallback usando PyMuPDF se marker-pdf non è disponibile.
    Meno preciso sulle formule ma funzionale.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ConversionResult(
            source_pdf=pdf_path.name,
            markdown_content="",
            success=False,
            error_message="Nessun convertitore PDF disponibile. Installa marker-pdf o PyMuPDF."
        )

    pdf_name = pdf_path.stem

    try:
        doc = fitz.open(str(pdf_path))
        markdown_parts = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            # Aggiungi marcatore di pagina (utile per tracking)
            markdown_parts.append(f"<!-- Page {page_num + 1} -->\n{text}")

        markdown_content = "\n\n".join(markdown_parts)

        # Salva file
        md_path = output_dir / f"{pdf_name}.md"
        md_path.write_text(markdown_content, encoding='utf-8')

        structure = {"sections": _extract_sections(markdown_content)}

        if save_json_structure:
            json_path = output_dir / f"{pdf_name}_structure.json"
            json_path.write_text(
                json.dumps(structure, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )

        return ConversionResult(
            source_pdf=pdf_path.name,
            markdown_content=markdown_content,
            structure=structure,
            page_count=len(doc),
            success=True
        )

    except Exception as e:
        return ConversionResult(
            source_pdf=pdf_path.name,
            markdown_content="",
            success=False,
            error_message=str(e)
        )


def _extract_sections(markdown: str) -> list[dict]:
    """Estrae la struttura delle sezioni dal markdown."""
    import re
    sections = []

    # Pattern per intestazioni markdown
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    for match in header_pattern.finditer(markdown):
        level = len(match.group(1))
        title = match.group(2).strip()
        sections.append({
            "level": level,
            "title": title,
            "position": match.start()
        })

    return sections


def _count_pages(pdf_path: Path) -> int:
    """Conta le pagine di un PDF."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        return len(doc)
    except:
        return 0


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    save_json_structure: bool = True
) -> list[ConversionResult]:
    """
    Converte tutti i PDF in una cartella.

    Args:
        input_dir: Cartella con i PDF
        output_dir: Cartella per gli output
        save_json_structure: Se salvare anche i JSON struttura

    Returns:
        Lista di ConversionResult
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))

    if not pdf_files:
        logger.warning(f"Nessun PDF trovato in {input_dir}")
        return []

    logger.info(f"Trovati {len(pdf_files)} PDF da convertire")

    results = []
    for pdf_path in pdf_files:
        result = convert_pdf_to_markdown(pdf_path, output_dir, save_json_structure)
        results.append(result)

    return results

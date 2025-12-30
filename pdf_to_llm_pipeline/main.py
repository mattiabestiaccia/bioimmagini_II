#!/usr/bin/env python3
"""
main.py - Pipeline PDF → Markdown → Chunks per LLM

Pipeline completa per convertire dispense universitarie PDF
in contenuti pronti per LLM/RAG.

Uso:
    python main.py --input_dir ./input_pdfs --output_dir ./output_md

    python main.py --input_dir ./input_pdfs --output_dir ./output_md \
                   --max_tokens 1000 --save_individual_chunks
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from converter import batch_convert, ConversionResult
from cleaning import clean_markdown, clean_for_chunking
from chunking import chunk_markdown, save_chunks, ChunkedDocument


def setup_logging(verbose: bool = False) -> None:
    """Configura il logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse degli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(
        description='Pipeline PDF → Markdown → Chunks per LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  %(prog)s --input_dir ./input_pdfs --output_dir ./output_md
  %(prog)s -i ./pdfs -o ./output --max_tokens 1000 --verbose
  %(prog)s -i ./pdfs -o ./output --save_individual_chunks
        """
    )

    parser.add_argument(
        '-i', '--input_dir',
        type=Path,
        default=Path('./input_pdfs'),
        help='Cartella contenente i PDF da convertire (default: ./input_pdfs)'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=Path,
        default=Path('./output_md'),
        help='Cartella per gli output (default: ./output_md)'
    )

    parser.add_argument(
        '--min_tokens',
        type=int,
        default=500,
        help='Minimo token per chunk (default: 500)'
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=1500,
        help='Massimo token per chunk (default: 1500)'
    )

    parser.add_argument(
        '--save_individual_chunks',
        action='store_true',
        help='Salva anche i chunk come file .md individuali'
    )

    parser.add_argument(
        '--skip_chunking',
        action='store_true',
        help='Salta la fase di chunking (solo conversione e pulizia)'
    )

    parser.add_argument(
        '--skip_cleaning',
        action='store_true',
        help='Salta la fase di pulizia del markdown'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Output verboso (debug)'
    )

    return parser.parse_args()


def process_pipeline(
    input_dir: Path,
    output_dir: Path,
    min_tokens: int = 500,
    max_tokens: int = 1500,
    save_individual_chunks: bool = False,
    skip_chunking: bool = False,
    skip_cleaning: bool = False
) -> dict:
    """
    Esegue la pipeline completa.

    Args:
        input_dir: Cartella con i PDF
        output_dir: Cartella per output
        min_tokens: Minimo token per chunk
        max_tokens: Massimo token per chunk
        save_individual_chunks: Salva chunk individuali
        skip_chunking: Salta il chunking
        skip_cleaning: Salta la pulizia

    Returns:
        Dict con statistiche del processing
    """
    logger = logging.getLogger(__name__)
    stats = {
        'pdfs_found': 0,
        'pdfs_converted': 0,
        'pdfs_failed': 0,
        'total_chunks': 0,
        'total_tokens': 0,
        'details': []
    }

    # Verifica input
    if not input_dir.exists():
        logger.error(f"Cartella input non trovata: {input_dir}")
        return stats

    # Crea output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # === FASE 1: Conversione PDF → Markdown ===
    logger.info("=" * 50)
    logger.info("FASE 1: Conversione PDF → Markdown")
    logger.info("=" * 50)

    conversion_results = batch_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        save_json_structure=True
    )

    stats['pdfs_found'] = len(conversion_results)

    for result in conversion_results:
        if result.success:
            stats['pdfs_converted'] += 1
            logger.info(f"✓ {result.source_pdf} ({result.page_count} pagine)")
        else:
            stats['pdfs_failed'] += 1
            logger.error(f"✗ {result.source_pdf}: {result.error_message}")

    # === FASE 2: Pulizia Markdown ===
    if not skip_cleaning:
        logger.info("")
        logger.info("=" * 50)
        logger.info("FASE 2: Pulizia Markdown")
        logger.info("=" * 50)

        for result in conversion_results:
            if not result.success:
                continue

            md_path = output_dir / f"{Path(result.source_pdf).stem}.md"
            if md_path.exists():
                original_content = md_path.read_text(encoding='utf-8')
                cleaned_content = clean_for_chunking(original_content)

                # Salva versione pulita
                cleaned_path = output_dir / f"{Path(result.source_pdf).stem}_cleaned.md"
                cleaned_path.write_text(cleaned_content, encoding='utf-8')

                # Aggiorna il contenuto nel result per il chunking
                result.markdown_content = cleaned_content

                original_lines = len(original_content.split('\n'))
                cleaned_lines = len(cleaned_content.split('\n'))
                logger.info(f"✓ {result.source_pdf}: {original_lines} → {cleaned_lines} righe")

    # === FASE 3: Chunking ===
    if not skip_chunking:
        logger.info("")
        logger.info("=" * 50)
        logger.info("FASE 3: Chunking per LLM/RAG")
        logger.info("=" * 50)

        for result in conversion_results:
            if not result.success:
                continue

            # Usa il contenuto pulito se disponibile
            if skip_cleaning:
                md_path = output_dir / f"{Path(result.source_pdf).stem}.md"
                if md_path.exists():
                    content = md_path.read_text(encoding='utf-8')
                else:
                    content = result.markdown_content
            else:
                content = result.markdown_content

            if not content:
                logger.warning(f"Nessun contenuto per {result.source_pdf}, skip chunking")
                continue

            # Esegui chunking
            chunked = chunk_markdown(
                markdown=content,
                source_name=result.source_pdf,
                min_tokens=min_tokens,
                max_tokens=max_tokens
            )

            # Salva chunks
            save_chunks(
                chunked_doc=chunked,
                output_dir=output_dir,
                save_individual_files=save_individual_chunks
            )

            stats['total_chunks'] += chunked.total_chunks
            stats['total_tokens'] += chunked.total_tokens
            stats['details'].append({
                'pdf': result.source_pdf,
                'chunks': chunked.total_chunks,
                'tokens': chunked.total_tokens
            })

            logger.info(f"✓ {result.source_pdf}: {chunked.total_chunks} chunks, ~{chunked.total_tokens} tokens")

    return stats


def print_summary(stats: dict) -> None:
    """Stampa il riepilogo finale."""
    print("\n" + "=" * 50)
    print("RIEPILOGO")
    print("=" * 50)

    print(f"\nPDF processati: {stats['pdfs_converted']}/{stats['pdfs_found']}")

    if stats['pdfs_failed'] > 0:
        print(f"PDF falliti: {stats['pdfs_failed']}")

    if stats['total_chunks'] > 0:
        print(f"\nChunks totali: {stats['total_chunks']}")
        print(f"Token totali (stima): {stats['total_tokens']}")

        if stats['details']:
            print("\nDettaglio per PDF:")
            for detail in stats['details']:
                avg_tokens = detail['tokens'] // detail['chunks'] if detail['chunks'] > 0 else 0
                print(f"  • {detail['pdf']}: {detail['chunks']} chunks (~{avg_tokens} token/chunk)")

    print("")


def main():
    """Entry point principale."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    print("\n" + "=" * 50)
    print("PDF to LLM Pipeline")
    print("=" * 50)
    print(f"\nInput:  {args.input_dir.absolute()}")
    print(f"Output: {args.output_dir.absolute()}")
    print(f"Token range: {args.min_tokens}-{args.max_tokens}")
    print("")

    start_time = datetime.now()

    stats = process_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        save_individual_chunks=args.save_individual_chunks,
        skip_chunking=args.skip_chunking,
        skip_cleaning=args.skip_cleaning
    )

    elapsed = datetime.now() - start_time

    print_summary(stats)
    print(f"Tempo totale: {elapsed.total_seconds():.1f}s")

    # Exit code
    if stats['pdfs_failed'] > 0 and stats['pdfs_converted'] == 0:
        sys.exit(1)
    elif stats['pdfs_found'] == 0:
        logger.warning("Nessun PDF trovato nella cartella input")
        sys.exit(0)

    return 0


if __name__ == '__main__':
    main()

from . import commands
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from pytauri.webview import WebviewWindow

import subprocess
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.pipeline.vlm_pipeline import VlmPipeline
from markdown2 import markdown as md2_markdown
import weasyprint

# === Base Model ===
class _BaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

# === Request Model ===
class PDFProcessRequest(_BaseModel):
    pdf_path: str  # 前端传入 PDF 文件路径
    use_vlm: bool = False  # 是否使用 Granite VLM 处理 PDF

# === Command ===
@commands.command()
async def process_pdf(body: PDFProcessRequest, webview_window: WebviewWindow) -> dict:
    """
    Process a PDF file with optional Granite VLM pipeline:
      1. Export Markdown
      2. Generate tokens
      3. Chunking
      4. Generate PDF with TOC
    """
    pdf_path = body.pdf_path
    webview_window.set_title(f"Processing: {pdf_path}")

    # === 选择转换器 ===
    if body.use_vlm:
        # Granite VLM pipeline
        pipeline_options = VlmPipelineOptions()
        pipeline_options.vlm_options = vlm_model_specs.GRANITEDOCLING_MLX

        converter = DocumentConverter(
         format_options={
            InputFormat.PDF: PdfFormatOption(
              pipeline_cls=VlmPipeline,
              pipeline_options=pipeline_options,
            )
         }
        )
    else:
        # 普通 Docling pipeline
        converter = DocumentConverter()

    # === 转换 PDF ===
    result = converter.convert(pdf_path)
    doc = result.document

    # === 生成 Markdown ===
    markdown_text = doc.export_to_markdown()
    md_file = "output.markdown.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    # === 生成 Token 文件 ===
    doctags = doc.export_to_doctags()
    token_file = "output_tokens.txt"
    with open(token_file, "w", encoding="utf-8") as f:
        f.write(f"Total doctags count: {len(doctags)}\n\n")
        f.write(" ".join(doctags))

    # === 生成 Chunking ===
    chunker = HybridChunker(max_tokens=512)
    chunks = list(chunker.chunk(dl_doc=doc))
    chunk_file = "output_chunks.txt"
    with open(chunk_file, "w", encoding="utf-8") as f:
        f.write(f"Total chunks: {len(chunks)}\n")
        total_tokens = sum(len(chunk.text.split()) for chunk in chunks)
        f.write(f"Total estimated tokens: {total_tokens}\n\n")
        for i, chunk in enumerate(chunks, 1):
            f.write(f"--- Chunk {i} ---\n")
            f.write(f"Estimated tokens: {len(chunk.text.split())}\n")
            page_info = chunk.meta.page_numbers if hasattr(chunk.meta, 'page_numbers') else 'N/A'
            type_info = chunk.meta.type if hasattr(chunk.meta, 'type') else 'N/A'
            f.write(f"Page(s): {page_info}\n")
            f.write(f"Type: {type_info}\n")
            f.write("Text:\n")
            f.write(chunk.text.strip() + "\n")
            f.write("---\n\n")

    # === 生成 PDF with TOC ===
    pdf_file = "output_with_toc.pdf"
    try:
        subprocess.run([
            "pandoc",
            md_file,
            "-o", pdf_file,
            "--toc",
            "--toc-depth=3",
            "--pdf-engine=xelatex"
        ], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pdf_file = None

    webview_window.set_title(f"Completed: {pdf_path}")

    # 返回前端可用信息
    return {
        "markdown_file": md_file,
        "token_file": token_file,
        "chunk_file": chunk_file,
        "pdf_file": pdf_file,
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "used_vlm": body.use_vlm,
    }

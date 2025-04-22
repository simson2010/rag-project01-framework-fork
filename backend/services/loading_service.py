from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
import pdfplumber
import fitz  # PyMuPDF
import logging
import os
from datetime import datetime
import json
from pathlib import Path # Import Path for better path handling

from .loaders.loaders import PDFLoader, TXTLoader, DOCXLoader, BaseLoader # Import the new loaders

logger = logging.getLogger(__name__)
"""
文档加载服务类
    这个服务类提供了多种文档加载方法，支持不同文件类型（PDF, TXT, DOCX）
    和PDF的加载策略及分块选项。
    主要功能：
    1. 支持多种文件类型加载：PDF, TXT, DOCX
    2. 对于PDF，支持多种解析库和unstructured策略。
    3. 文档加载特性：
        - 保持页码信息 (对于TXT/DOCX，页码概念简化)
        - 提供元数据存储
 """
class LoadingService:
    """
    文档加载服务类，提供多种文档加载和处理方法。

    属性:
        active_loader (BaseLoader): 当前使用的loader实例
    """

    def __init__(self):
        # {{ edit_3 }}
        # self.total_pages = 0 # Removed
        # self.current_page_map = [] # Removed
        self.active_loader: BaseLoader | None = None # Store the active loader instance

    # {{ edit_4 }}
    # Renamed from load_pdf to load_document and added file_type parameter
    def load_document(self, file_path: str, file_type: str, method: str = None, strategy: str = None, chunking_strategy: str = None, chunking_options: dict = None) -> list:
        """
        加载文档的主方法，根据文件类型选择合适的loader。

        参数:
            file_path (str): 文档文件路径
            file_type (str): 文件类型，支持 'pdf', 'txt', 'word'
            method (str, optional): PDF加载方法，支持 'pymupdf', 'pypdf', 'pdfplumber', 'unstructured'
            strategy (str, optional): 使用unstructured方法时的策略
            chunking_strategy (str, optional): 文本分块策略
            chunking_options (dict, optional): 分块选项配置

        返回:
            list: 提取的页面/分块数据列表 [{"text": ..., "page": ..., "metadata": {...}}]
        """
        try:
            if file_type.lower() == "pdf":
                self.active_loader = PDFLoader()
                # Pass PDF-specific parameters
                page_map = self.active_loader.load(
                    file_path,
                    method=method,
                    strategy=strategy,
                    chunking_strategy=chunking_strategy,
                    chunking_options=chunking_options
                )
            elif file_type.lower() == "txt":
                self.active_loader = TXTLoader()
                page_map = self.active_loader.load(file_path)
            elif file_type.lower() == "word": # Assuming 'word' type from frontend
                 self.active_loader = DOCXLoader()
                 page_map = self.active_loader.load(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # {{ edit_5 }}
            # self.current_page_map = page_map # Removed, loader stores its own map
            # self.total_pages = self.active_loader.get_total_pages() # Removed, main.py will get this

            return page_map # Return the page map directly

        except Exception as e:
            logger.error(f"Error loading document of type {file_type}: {str(e)}")
            self.active_loader = None # Reset loader on error
            raise

    def get_total_pages(self) -> int:
        """
        获取当前加载文档的总页数。
        Delegates to the active loader.
        """
        # {{ edit_6 }}
        if not self.active_loader:
             logger.warning("get_total_pages called before loading a document.")
             return 0
        return self.active_loader.get_total_pages()

    def get_page_map(self) -> list:
        """
        获取当前文档的页面映射信息。
        Delegates to the active loader.
        """
        # {{ edit_7 }}
        if not self.active_loader:
             logger.warning("get_page_map called before loading a document.")
             return []
        return self.active_loader.get_page_map()

    # {{ edit_8 }}
    # Removed _load_with_pymupdf, _load_with_pypdf, _load_with_pdfplumber, _load_with_unstructured
    # These methods are now in PDFLoader

    def save_document(self, filename: str, chunks: list, metadata: dict, loading_method: str, strategy: str = None, chunking_strategy: str = None) -> str:
        """
        保存处理后的文档数据。

        参数:
            filename (str): 原文件名
            chunks (list): 文档分块列表
            metadata (dict): 文档元数据
            loading_method (str): 使用的加载方法 (e.g., pymupdf, unstructured, txt, word)
            strategy (str, optional): 使用的加载策略 (for unstructured)
            chunking_strategy (str, optional): 使用的分块策略 (for unstructured)

        返回:
            str: 保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            # Use Pathlib to handle extensions generically
            original_path = Path(filename)
            base_name = original_path.stem.split('_')[0] # Get base name before any existing suffixes

            # Construct a more informative document name
            parts = [base_name, loading_method]
            if loading_method == "unstructured":
                 if strategy:
                      parts.append(strategy)
                 if chunking_strategy:
                      parts.append(chunking_strategy)

            doc_name = "_".join(parts) + f"_{timestamp}"

            # 构建文档数据结构，确保所有值都是可序列化的
            document_data = {
                "filename": str(filename),
                "document_name": doc_name, # Add a generated document name
                "total_chunks": int(len(chunks)),
                "total_pages": int(metadata.get("total_pages", 1)), # Use metadata total_pages
                "loading_method": str(loading_method),
                "loading_strategy": str(strategy) if loading_method == "unstructured" and strategy else None,
                "chunking_strategy": str(chunking_strategy) if loading_method == "unstructured" and chunking_strategy else None,
                "chunking_method": "loaded", # This might need refinement depending on how chunking is applied after loading
                "timestamp": datetime.now().isoformat(),
                "chunks": chunks
            }

            # Save to file
            filepath = os.path.join("01-loaded-docs", f"{doc_name}.json")
            os.makedirs("01-loaded-docs", exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)

            return filepath

        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            raise

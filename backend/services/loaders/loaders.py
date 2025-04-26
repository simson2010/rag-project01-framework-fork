# backend/services/loaders.py
from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import pdfplumber
import fitz  # PyMuPDF
import docx # python-docx
import logging
import os

logger = logging.getLogger(__name__)

class BaseLoader:
    """Base class concept for document loaders."""
    def __init__(self):
        self._page_map = []
        self._total_pages = 0

    def load(self, file_path: str, **kwargs) -> list:
        """
        Load the document and return a list of page/chunk data.
        Each item in the list should be a dictionary with at least 'text' and 'page'.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_total_pages(self) -> int:
        """Get the total number of pages/sections loaded."""
        return self._total_pages

    def get_page_map(self) -> list:
        """Get the list of page/chunk data."""
        return self._page_map

class PDFLoader(BaseLoader):
    """Loader for PDF files, supporting multiple methods."""

    def load(self, file_path: str, method: str, strategy: str = None, chunking_strategy: str = None, chunking_options: dict = None) -> list:
        """
        Load PDF using the specified method.
        Returns a list of dictionaries: [{"text": ..., "page": ..., "metadata": {...}}]
        """
        try:
            if method == "pymupdf":
                self._page_map = self._load_with_pymupdf(file_path)
            elif method == "pypdf":
                self._page_map = self._load_with_pypdf(file_path)
            elif method == "pdfplumber":
                self._page_map = self._load_with_pdfplumber(file_path)
            elif method == "unstructured":
                self._page_map = self._load_with_unstructured(
                    file_path,
                    strategy=strategy,
                    chunking_strategy=chunking_strategy,
                    chunking_options=chunking_options
                )
            else:
                raise ValueError(f"Unsupported PDF loading method: {method}")

            self._total_pages = max(page_data['page'] for page_data in self._page_map) if self._page_map else 0
            return self._page_map

        except Exception as e:
            logger.error(f"Error loading PDF with {method}: {str(e)}")
            raise

    def _load_with_pymupdf(self, file_path: str) -> list:
        text_blocks = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                if text.strip():
                    text_blocks.append({
                        "text": text.strip(),
                        "page": page_num,
                        "metadata": {} # Add empty metadata for consistency
                    })
        return text_blocks

    def _load_with_pypdf(self, file_path: str) -> list:
        text_blocks = []
        with open(file_path, "rb") as file:
            pdf = PdfReader(file)
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_blocks.append({
                        "text": page_text.strip(),
                        "page": page_num,
                        "metadata": {} # Add empty metadata for consistency
                    })
        return text_blocks

    def _load_with_pdfplumber(self, file_path: str) -> list:
        text_blocks = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_blocks.append({
                        "text": page_text.strip(),
                        "page": page_num,
                        "metadata": {} # Add empty metadata for consistency
                    })
        return text_blocks

    def _load_with_unstructured(self, file_path: str, strategy: str = "fast", chunking_strategy: str = "basic", chunking_options: dict = None) -> list:
        strategy_params = {
            "fast": {"strategy": "fast"},
            "hi_res": {"strategy": "hi_res"},
            "ocr_only": {"strategy": "ocr_only"}
        }

        chunking_params = {}
        if chunking_strategy == "basic":
            chunking_params = {
                "max_characters": chunking_options.get("maxCharacters", 4000),
                "new_after_n_chars": chunking_options.get("newAfterNChars", 3000),
                "combine_text_under_n_chars": chunking_options.get("combineTextUnderNChars", 2000),
                "overlap": chunking_options.get("overlap", 200),
                "overlap_all": chunking_options.get("overlapAll", False)
            }
        elif chunking_strategy == "by_title":
            chunking_params = {
                "chunking_strategy": "by_title",
                "combine_text_under_n_chars": chunking_options.get("combineTextUnderNChars", 2000),
                "multipage_sections": chunking_options.get("multiPageSections", False)
            }

        params = {**strategy_params.get(strategy, {"strategy": "fast"}), **chunking_params}

        elements = partition_pdf(file_path, **params)

        text_blocks = []
        pages = set()

        for elem in elements:
            metadata = elem.metadata.__dict__
            page_number = metadata.get('page_number')

            if page_number is not None:
                pages.add(page_number)

                cleaned_metadata = {}
                for key, value in metadata.items():
                    if key == '_known_field_names':
                        continue
                    try:
                        json.dumps({key: value})
                        cleaned_metadata[key] = value
                    except (TypeError, OverflowError):
                        cleaned_metadata[key] = str(value)

                cleaned_metadata['element_type'] = elem.__class__.__name__
                cleaned_metadata['id'] = str(getattr(elem, 'id', None))
                cleaned_metadata['category'] = str(getattr(elem, 'category', None))

                text_blocks.append({
                    "text": str(elem),
                    "page": page_number,
                    "metadata": cleaned_metadata
                })

        return text_blocks

class TXTLoader(BaseLoader):
    """Loader for TXT files."""

    def load(self, method: str, file_path: str, **kwargs) -> list:
        """
        Load TXT file. Treats each line as a chunk on page 1.
        Returns a list of dictionaries: [{"text": ..., "page": 1, "metadata": {}}]
        """
        text_blocks = []
        try:
            if method == "puretext":
                text_blocks = self._load_plain_text_using_reader(file_path)
            elif method == "textloader":
                text_blocks = self._load_plain_text_using_textLoader(file_path)
            else:
                logger.error("No loading method found in paramters!")
                raise
            # 更新页面映射和总页数
            self._page_map = text_blocks
            self._total_pages = 1 if text_blocks else 0  # TXT文件在这里被视为一个“页面”
            return self._page_map

        except FileNotFoundError:
            logger.error(f"The file {file_path} was not found.")
            raise
        except IOError as e:
            logger.error(f"IOError occurred while reading the file {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading TXT file {file_path}: {str(e)}")
            raise

    def _load_plain_text_using_textLoader(self, file_path: str) -> list:
        loader = TextLoader(file_path)
        documents = loader.load()
        documents[0].page_content = documents[0].page_content.replace("\n\xa0\n","\n\n")
        text_splitter = CharacterTextSplitter(
            chunk_size=200,  # 每个文本块的大小为100个字符, 中文字要除以2
            chunk_overlap=60,  # 文本块之间没有重叠部分
        )
        chunks = text_splitter.split_documents(documents)
        print(len(chunks), chunks[::-1])
        text_blocks = []
        for chunk in chunks:
            text_blocks.append({
                "text": chunk.page_content,
                "page": 1,
                "metadata": {
                    "line_number": 0,
                    "len_of_text": len(chunk.page_content)
                }
            })

        return text_blocks

    def _load_plain_text_using_reader(self, file_path: str) -> list:
        readlines = 0
        onePage = ""
        text_blocks = []
        # 尝试打开文件，如果文件不存在或无法读取，将抛出异常
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            # 逐行读取文件
            for line_num, line in enumerate(f, 1):
                readlines += 1
                line = line.strip()
                onePage += line
                if readlines == 10 and len(onePage) >= 1:  # 如果行非空
                    text_blocks.append({
                        "text": onePage,
                        "page": 1,  # 假设所有行都在第一页
                        "metadata": {"line_number": line_num}
                    })
                    readlines = 0
                    onePage = ""
        return text_blocks

class DOCXLoader(BaseLoader):
    """Loader for DOCX files."""

    def load(self, file_path: str, **kwargs) -> list:
        """
        Load DOCX file. Treats each paragraph as a chunk on page 1.
        Returns a list of dictionaries: [{"text": ..., "page": 1, "metadata": {}}]
        """
        text_blocks = []
        try:
            doc = docx.Document(file_path)
            # Treat each paragraph as a block, all on page 1
            for para_num, paragraph in enumerate(doc.paragraphs, 1):
                text = paragraph.text.strip()
                if text:
                    text_blocks.append({
                        "text": text,
                        "page": 1, # Treat all paragraphs as page 1 for simplicity
                        "metadata": {"paragraph_number": para_num}
                    })
            self._page_map = text_blocks
            self._total_pages = 1 if text_blocks else 0 # A DOCX file is one "page" conceptually here
            return self._page_map
        except Exception as e:
            logger.error(f"Error loading DOCX file {file_path}: {str(e)}")
            raise

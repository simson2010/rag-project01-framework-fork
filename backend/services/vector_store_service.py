import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
from chromadb.config import Settings
from utils.config import VectorDBProvider, CHROMA_CONFIG
import hashlib

logger = logging.getLogger(__name__)

class VectorDBConfig:
    """
    向量数据库配置类，用于存储和管理向量数据库的配置信息
    """
    def __init__(self, provider="chroma", index_mode: str = "default"):
        """
        初始化向量数据库配置
        """
        self.provider = provider  # 固定为 Chroma
        self.index_mode = index_mode
        self.db_config = CHROMA_CONFIG

    @property
    def uri(self):
        return self.db_config.get("uri")

    def _get_index_type(self, index_mode: str) -> str:
        return self.db_config.get("index_types", {}).get(index_mode, "default")

    def _get_index_params(self, index_mode: str) -> Dict[str, Any]:
        return self.db_config.get("index_params", {}).get(index_mode, {})

class VectorStoreService:
    """
    向量存储服务类，提供向量数据的索引、查询和管理功能（仅支持 ChromaDB）
    """
    def __init__(self):
        os.makedirs("03-vector-store", exist_ok=True)
    
    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到 ChromaDB
        """
        start_time = datetime.now()
        embeddings_data = self._load_embeddings(embedding_file)
        result = self._index_to_chroma(embeddings_data, config)
        
        return {
            "database": config.provider,
            "total_vectors": len(embeddings_data["embeddings"]),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "collection_name": result.get("collection_name", "N/A")
        }

    def _load_embeddings(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format")
                return data
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

    def _index_to_chroma(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        try:
            import chromadb
            from chromadb.config import Settings

            # 客户端配置
            client = chromadb.PersistentClient(
                path=config.uri
            )

            # 生成集合名称
            filename = embeddings_data.get("filename", "").replace('.pdf', '')
            #hash file name here 
            hasher = hashlib.sha256(filename.encode('utf-8') if filename else b'')
            filename = hasher.hexdigest()[:8]  # 使用前8位作为简短哈希
            base_name = f"chroma_{filename}" if filename and not filename[0].isalpha() else filename
            collection_name = f"{base_name}_{embeddings_data.get('embedding_provider', 'unknown')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # 准备数据
            print(f"collection_name: >>> {collection_name}")
            collection = client.get_or_create_collection(collection_name)
            documents, metadatas, ids, vectors = [], [], [], []

            for i, emb in enumerate(embeddings_data["embeddings"]):
                documents.append(str(emb["metadata"].get("content", "")))
                metadatas.append({
                    "document_name": embeddings_data.get("filename", ""),
                    "chunk_id": emb["metadata"].get("chunk_id", 0),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "word_count": emb["metadata"].get("word_count", 0),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),
                    "embedding_model": embeddings_data.get("embedding_model", ""),
                })
                ids.append(f"doc_{base_name}_chunk_{i}")
                vectors.append([float(x) for x in emb.get("embedding", [])])

            # 插入数据
            collection.add(
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            # client.persist()

            return {
                "index_size": collection.count(),
                "collection_name": collection_name
            }

        except Exception as e:
            logger.error(f"ChromaDB indexing error: {str(e)}")
            raise

    def list_collections(self, vector_provider) -> List[str]:
        """列出所有 Chroma 集合"""
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=CHROMA_CONFIG["uri"]
            )
            return [col.name for col in client.list_collections()]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """删除指定集合"""
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=CHROMA_CONFIG["uri"]
            )
            client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False

    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=CHROMA_CONFIG["uri"]
            )
            col = client.get_collection(collection_name)
            return {
                "name": col.name,
                "num_entities": col.count(),
                "metadata": col.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
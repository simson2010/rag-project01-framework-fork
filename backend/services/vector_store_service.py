import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
import hashlib
from pathlib import Path
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG  # Updated import
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# 修改 CHROMA_CONFIG 配置
CHROMA_CONFIG = {
    "persist_directory": "chroma_db"
}

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
    
    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        return config._get_milvus_index_type(config.index_mode)
    
    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        return config._get_milvus_index_params(config.index_mode)
    
    def _init_chroma_client(self):
        client = chromadb.Client(Settings(
            persist_directory=CHROMA_CONFIG["persist_directory"],
            anonymized_telemetry=False
        ))
        return client
    
    def _index_to_chroma(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        try:
            start_time = datetime.now()
            
            # 初始化 Chroma 客户端
            client = self._init_chroma_client()
            
            # 创建或获取集合
            collection_name = f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 创建集合，使用简单的 metadata
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Document embeddings collection",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            # 准备数据
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for idx, emb in enumerate(embeddings_data["embeddings"]):
                ids.append(str(idx))
                embeddings.append(emb["embedding"])
                # 确保 metadata 中只包含简单类型
                metadatas.append({
                    "content": str(emb["metadata"].get("content", "")),
                    "page_number": str(emb["metadata"].get("page_number", "")),
                    "chunk_id": str(emb["metadata"].get("chunk_id", "")),
                    "document_name": str(embeddings_data.get("filename", "")),
                    "embedding_model": str(embeddings_data.get("embedding_model", "")),
                    "embedding_provider": str(embeddings_data.get("embedding_provider", ""))
                })
                documents.append(str(emb["metadata"].get("content", "")))
            
            # 添加数据到集合
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "index_size": len(ids),
                "collection_name": collection_name,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Chroma: {str(e)}")
            raise


    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到 ChromaDB
        """
        start_time = datetime.now()
        embeddings_data = self._load_embeddings(embedding_file)
        
        # 根据不同的数据库进行索引
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings_data, config)
        elif config.provider == VectorDBProvider.CHROMA:
            result = self._index_to_chroma(embeddings_data, config)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
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
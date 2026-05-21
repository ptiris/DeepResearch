"""
Embedding客户端，使用阿里云百炼 text-embedding-v4 API
提供文本编码和相似度计算功能
"""

import numpy as np
from openai import OpenAI
import os
import time

# 使用统一的 provider 配置
_EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', os.getenv('PROVIDER', 'dashscope'))

_PROVIDER_API_CONFIGS = {
    'openrouter': {
        'api_key_env': 'OPENROUTER_API_KEY',
        'base_url_env': 'OPENROUTER_BASE_URL',
        'base_url_default': 'https://openrouter.ai/api/v1',
    },
    'dashscope': {
        'api_key_env': 'DASHSCOPE_API_KEY',
        'base_url_env': 'DASHSCOPE_BASE_URL',
        'base_url_default': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
    'openai': {
        'api_key_env': 'OPENAI_API_KEY',
        'base_url_env': 'OPENAI_API_BASE',
        'base_url_default': 'https://api.openai.com/v1',
    },
}

_config = _PROVIDER_API_CONFIGS.get(_EMBEDDING_PROVIDER, _PROVIDER_API_CONFIGS['dashscope'])
API_KEY = os.getenv(_config['api_key_env'], '')
BASE_URL = os.getenv(_config['base_url_env'], _config['base_url_default'])
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024


def get_client():
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def encode_texts(texts, batch_size=6):
    """
    编码文本列表为embedding向量
    
    Args:
        texts: 文本列表
        batch_size: 批处理大小，默认为6
        
    Returns:
        np.ndarray: shape为(len(texts), EMBEDDING_DIM)的embedding数组
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    
    if batch_size > 10:
        batch_size = 10
    
    client = get_client()
    all_embs = []
    
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIM,
            )
            for item in resp.data:
                all_embs.append(item.embedding)
            print(f"[Embedding] encoded: {end}/{len(texts)}", flush=True)
            
            if end < len(texts):
                time.sleep(0.1)
        except Exception as e:
            print(f"[Embedding] Error encoding batch {start}-{end}: {e}")
            # 用零向量填充失败的batch
            for _ in range(len(batch)):
                all_embs.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
    
    return np.array(all_embs, dtype=np.float32)


def cosine_sim(a, b):
    """
    计算两个向量之间的余弦相似度
    
    Args:
        a: 第一个向量
        b: 第二个向量
        
    Returns:
        float: 余弦相似度，范围[0, 1]
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class EmbeddingClient:
    """Embedding客户端类，方便管理和复用"""
    
    def __init__(self):
        self.client = get_client()
        self.model = EMBEDDING_MODEL
        self.dim = EMBEDDING_DIM
    
    def encode(self, texts, batch_size=6):
        """编码文本列表"""
        return encode_texts(texts, batch_size)
    
    def similarity(self, a, b):
        """计算相似度"""
        return cosine_sim(a, b)
    
    def batch_similarity(self, query_emb, ref_embs):
        """
        批量计算query embedding与多个参考embeddings的相似度
        
        Args:
            query_emb: 查询的embedding向量
            ref_embs: 参考embeddings数组，shape为(n, dim)
            
        Returns:
            list: 相似度列表，长度为n
        """
        if len(ref_embs) == 0:
            return []
        
        similarities = []
        for ref_emb in ref_embs:
            sim = self.similarity(query_emb, ref_emb)
            similarities.append(sim)
        
        return similarities

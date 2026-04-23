"""
查询去重模块，管理查询历史并提供相似度查询功能
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .embedding_client import EmbeddingClient


class QueryHistory:
    """查询历史管理类"""
    
    def __init__(self):
        self.history = []
        self.embedding_client = EmbeddingClient()
    
    def add_query(self, query: str, embedding: np.ndarray, results: str, 
                  success: bool, is_executed: bool, turn_id: int):
        """
        添加查询到历史记录
        
        Args:
            query: 查询文本
            embedding: embedding向量
            results: 搜索结果
            success: 执行是否成功
            is_executed: 是否真正执行了搜索API
            turn_id: 轮次ID
        """
        self.history.append({
            "query": query,
            "embedding": embedding,
            "results": results,
            "success": success,
            "is_executed": is_executed,
            "timestamp": np.datetime64('now'),
            "turn_id": turn_id
        })
    
    def find_similar_queries(self, query_embedding: np.ndarray, threshold: float,
                           scope: str, current_turn_queries: Optional[List[Tuple[str, np.ndarray]]] = None,
                           current_turn_id: Optional[int] = None) -> List[Dict]:
        """
        查找相似查询

        Args:
            query_embedding: 查询的embedding向量
            threshold: 相似度阈值
            scope: "single_turn" 或 "global"
            current_turn_queries: 当前轮次的查询列表（仅用于single_turn scope）
            current_turn_id: 当前轮次ID

        Returns:
            List[Dict]: 相似查询列表，每个元素包含 query, similarity, is_executed, results 等信息
                        按相似度降序排列
        """
        similar_queries = []

        if scope == "single_turn" and current_turn_queries is not None:
            # 只与当前轮次的其他查询比较
            for q_text, q_emb in current_turn_queries:
                # 不与自己比较
                if not np.array_equal(q_emb, query_embedding):
                    sim = self.embedding_client.similarity(query_embedding, q_emb)
                    if sim >= threshold:
                        similar_queries.append({
                            "query": q_text,
                            "similarity": sim,
                            "is_executed": None,  # 当前轮次的查询还未执行，设为None
                            "results": None
                        })
        else:
            # 与所有历史查询比较（global scope）
            for entry in self.history:
                sim = self.embedding_client.similarity(query_embedding, entry["embedding"])
                if sim >= threshold:
                    similar_queries.append({
                        "query": entry["query"],
                        "similarity": sim,
                        "is_executed": entry["is_executed"],
                        "results": entry["results"],
                        "success": entry["success"]
                    })

        # 按相似度降序排序
        similar_queries.sort(key=lambda x: x["similarity"], reverse=True)

        return similar_queries
    
    def find_first_executed_similar_query(self, similar_queries: List[Dict]) -> Optional[Dict]:
        """
        在相似查询中找到第一个真正执行的查询
        
        Args:
            similar_queries: 相似查询列表
            
        Returns:
            Optional[Dict]: 第一个 is_executed=True 的查询，如果没有则返回None
        """
        for query in similar_queries:
            if query.get("is_executed") is True:
                return query
        return None
    
    def get_exact_query_result(self, query: str) -> Optional[Dict]:
        """
        获取精确匹配的查询结果
        
        Args:
            query: 查询文本
            
        Returns:
            Optional[Dict]: 查询的详细信息，如果找不到则返回None
        """
        for entry in reversed(self.history):  # 从最新的开始查找
            if entry["query"] == query:
                return {
                    "query": entry["query"],
                    "results": entry["results"],
                    "success": entry["success"],
                    "is_executed": entry["is_executed"]
                }
        return None
    
    def get_all_queries_in_turn(self, turn_id: int) -> List[Dict]:
        """
        获取指定轮次的所有查询
        
        Args:
            turn_id: 轮次ID
            
        Returns:
            List[Dict]: 该轮次的所有查询
        """
        return [entry for entry in self.history if entry["turn_id"] == turn_id]
    
    def clear_all(self):
        """清空所有历史记录"""
        self.history = []
    
    def get_history_size(self) -> int:
        """获取历史记录数量"""
        return len(self.history)
    
    def get_summary(self) -> Dict:
        """
        获取历史记录摘要
        
        Returns:
            Dict: 包含统计信息的字典
        """
        total = len(self.history)
        executed = sum(1 for e in self.history if e["is_executed"])
        successful = sum(1 for e in self.history if e["success"])
        
        return {
            "total_queries": total,
            "executed_queries": executed,
            "cached_queries": total - executed,
            "successful_queries": successful,
            "failed_queries": total - successful
        }

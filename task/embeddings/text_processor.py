from enum import StrEnum

import psycopg2

from task.embeddings.embeddings_client import EmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"
    COSINE_DISTANCE = "cosine"


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(self, file_name: str, chunk_size: int, overlap: int, should_table_truncated: bool):
        db_connection = self._get_connection()
        cursor = db_connection.cursor()

        if should_table_truncated:
            cursor.execute("TRUNCATE TABLE vectors")

        with open(file_name) as file:
            text: str = file.read()

        chunks: list[str] = chunk_text(text, chunk_size, overlap)
        embeddings: dict[int, list[float]] = self.embeddings_client.get_embeddings(chunks)

        for index, chunk in enumerate(chunks):
            formatted_embeddings = self._to_pgvector(embeddings[index])
            cursor.execute("INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s)",
                           (file_name, chunk, formatted_embeddings))
        db_connection.commit()

    @staticmethod
    def _to_pgvector(embeddings: list[float]) -> str:
        return f"[{', '.join(map(str, embeddings))}]"

    def search(self, search_mode: str, user_request: str, top_k: int, min_score_threshold: float = None,
               dimensions: int = 384) -> list[str]:
        db_connection = self._get_connection()
        cursor = db_connection.cursor()

        user_query_embeddings = self.embeddings_client.get_embeddings([user_request], dimensions)[0]
        formatted_user_query_embeddings = self._to_pgvector(user_query_embeddings)

        if search_mode == SearchMode.EUCLIDIAN_DISTANCE:
            cursor.execute("""
                    SELECT text, embedding <-> %s::vector AS distance
                    FROM vectors
                    WHERE embedding <-> %s::vector <= %s
                    ORDER BY distance
                    LIMIT %s;
                """, (formatted_user_query_embeddings, formatted_user_query_embeddings, min_score_threshold, top_k))

        elif search_mode == SearchMode.COSINE_DISTANCE:
            cursor.execute("""
                    SELECT text, embedding <=> %s::vector AS distance
                    FROM vectors
                    WHERE embedding <=> %s::vector <= %s
                    ORDER BY distance
                    LIMIT %s;
                """, (formatted_user_query_embeddings, formatted_user_query_embeddings, min_score_threshold, top_k))
        results = cursor.fetchall()
        return [result[0] for result in results]

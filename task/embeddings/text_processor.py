from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import EmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


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
            cursor.execute("TRUNCATE TABLE table_name;")

        with open(file_name) as file:
            text: str = file.read()

        chunks: list[str] = chunk_text(text, chunk_size, overlap)
        embeddings: dict[int, list[float]] = self.embeddings_client.get_embeddings(chunks)

        for index, chunk in enumerate(chunks):
            formatted_embeddings = self._to_pgvector(embeddings[index])
            cursor.execute("INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s)",
                           (file_name, chunk, formatted_embeddings))

    @staticmethod
    def _to_pgvector(embeddings: list[float]) -> str:
        return f"[{', '.join(map(str, embeddings))}]"

    #TODO:
    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)


    #TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`

    def search(self, query: str):
        pass


# SELECT text, embedding <->  '[0.23, -0.45, 0.67, ..., 0.12]'::vector AS distance
# FROM microwave_data
# WHERE embedding <->  '[0.23, -0.45, 0.67, ..., 0.12]'::vector <= {score}
# ORDER BY distance
# LIMIT {top_k};

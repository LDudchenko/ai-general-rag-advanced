import requests
import os


class EmbeddingsClient:
    _endpoint: str
    _api_key: str
    _model_name: str

    def __init__(self, model_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = "https://api.openai.com/v1/embeddings"
        self._api_key = "Bearer " + api_key
        self._model_name = model_name

    def get_embeddings(self, texts: list[str], dimensions: int = 384) -> dict[int, list[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._api_key
        }

        payload = {
            "input": texts,
            "model": self._model_name,
            "dimensions": dimensions
        }

        response = requests.post(self._endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        data = response.json()
        result = {item["index"]: item["embedding"] for item in data["data"]}
        return result
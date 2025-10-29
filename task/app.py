import os

from task.chat.chat_completion_client import ChatCompletionClient
from task.embeddings.embeddings_client import EmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""

#TODO:
# - create embeddings client with 'text-embedding-3-small-' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)

def main():
    embeddings_client = EmbeddingsClient("text-embedding-3-small", OPENAI_API_KEY)
    chat_completion_client = ChatCompletionClient("gpt-4o", OPENAI_API_KEY)
    text_processor = TextProcessor(embeddings_client,
                                   {'host': 'localhost', 'port': 5433, 'database': 'vectordb', 'user': 'postgres',
                                    'password': 'postgres'})
    text_processor.process_text_file("embeddings/microwave_manual.txt", 300, 40, False)

    messages = []
    system_message = Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)
    messages.append(system_message)

    while True:
        user_question = input("\n> ").strip()

        context = text_processor.search(user_question)

        augmented_query = USER_PROMPT.format(context=context, query=user_question)
        user_message = Message(role=Role.USER, content=augmented_query)
        messages.append(user_message)

        ai_answer = chat_completion_client.get_completion(messages, print_request=True).content
        ai_message = Message(role=Role.USER, content=ai_answer)
        messages.append(ai_message)


# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml
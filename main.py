from llama_index import GPTSimpleVectorIndex, download_loader
from llama_index.indices.query.query_transform.base import HyDEQueryTransform
from dotenv import load_dotenv
import os

load_dotenv(override=True)

assert (
    os.getenv("OPENAI_API_KEY") is not None
), "Please set the OPENAI_API_KEY environment variable."

GPTRepoReader = download_loader("GPTRepoReader")

loader = GPTRepoReader()
documents = loader.load_data(repo_path="./discord-gpt-bot",
                             preamble_str="This is the repository of the Discord GPT bot")

index = GPTSimpleVectorIndex.from_documents(documents)

print(index.query("what does the getReplyRefMessage2 function do?",
      query_transform=HyDEQueryTransform()))

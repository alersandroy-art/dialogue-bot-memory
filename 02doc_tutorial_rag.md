https://haystack.deepset.ai/tutorials/27_first_rag_pipeline
Tutorial: Creating Your First QA Pipeline with Retrieval-Augmentation

Last Updated: June 17, 2026
Level: Beginner
Time to complete: 10 minutes
Components Used: InMemoryDocumentStore, SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder, InMemoryEmbeddingRetriever, ChatPromptBuilder, and a ChatGenerator such as OpenAIChatGenerator, MistralChatGenerator, or TransformersChatGenerator.
Prerequisites: Access to a large language model, either an API key from a provider or a locally or on-premises hosted model (for example on Colab runtime).
Goal: After completing this tutorial, you’ll have learned the new prompt syntax and how to use ChatPromptBuilder with a ChatGenerator to build a generative question-answering pipeline with retrieval-augmentation.
Overview
This tutorial shows you how to create a generative question-answering pipeline using the retrieval-augmentation ( RAG) approach with Haystack. The process involves four main components: SentenceTransformersTextEmbedder for creating an embedding for the user query, InMemoryEmbeddingRetriever for fetching relevant documents, ChatPromptBuilder for creating a template prompt, and a ChatGenerator for generating the final answer.

The LLM behind that generator can be hosted in the cloud, for example with OpenAI, Anthropic, Google, Mistral, or other providers, usually by setting an API key in the environment or run locally, for example via Ollama or vLLM, or on a Colab VM by loading an open-weight model from Hugging Face. The Initialize a ChatGenerator section shows three concrete options (OpenAI, Mistral, and a local model).

For this tutorial, you’ll use the Wikipedia pages of Seven Wonders of the Ancient World as Documents, but you can replace them with any text you want.

Installing Haystack
Install Haystack and other required packages with pip:

%%bash

pip install haystack-ai mistral-haystack transformers-haystack "datasets>=2.6.1" sentence-transformers-haystack
Fetching and Indexing Documents
You’ll start creating your question answering system by downloading the data and indexing the data with its embeddings to a DocumentStore.

In this tutorial, you will take a simple approach to writing documents and their embeddings into the DocumentStore. For a full indexing pipeline with preprocessing, cleaning and splitting, check out our tutorial on Preprocessing Different File Types.

Initializing the DocumentStore
Initialize a DocumentStore to index your documents. A DocumentStore stores the Documents that the question answering system uses to find answers to your questions. In this tutorial, you’ll be using the InMemoryDocumentStore.

from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
InMemoryDocumentStore is the simplest DocumentStore to get started with. It requires no external dependencies and it’s a good option for smaller projects and debugging. But it doesn’t scale up so well to larger Document collections, so it’s not a good choice for production systems. To learn more about the different types of external databases that Haystack supports, see DocumentStore Integrations.

The DocumentStore is now ready. Now it’s time to fill it with some Documents.

Fetch the Data
You’ll use the Wikipedia pages of Seven Wonders of the Ancient World as Documents. We preprocessed the data and uploaded it to Hugging Face as the Seven Wonders dataset. Thus, you don’t need to perform any additional cleaning or splitting.

Fetch the data and convert it into Haystack Documents:

from datasets import load_dataset
from haystack import Document

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
Initialize a Document Embedder
To store your data in the DocumentStore with embeddings, initialize a SentenceTransformersDocumentEmbedder with the model name.

If you’d like, you can use a different Embedder for your documents.

from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
Write Documents to the DocumentStore
Run the doc_embedder with the Documents. The embedder will create embeddings for each document and store them in that document’s embedding field. Then, write the Documents to the DocumentStore with the write_documents() method.

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])
Building the RAG Pipeline
The next step is to build a Pipeline to generate answers for the user query following the RAG approach. To create the pipeline, you first need to initialize each component, add them to your pipeline, and connect them.

Initialize a Text Embedder
Initialize a text embedder to create an embedding for the user query. The created embedding will later be used by the Retriever to retrieve relevant documents from the DocumentStore.

⚠️ Notice that you used sentence-transformers/all-MiniLM-L6-v2 model to create embeddings for your documents before. This is why you need to use the same model to embed the user queries.

from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
Initialize the Retriever
Initialize an InMemoryEmbeddingRetriever and make it use the InMemoryDocumentStore you initialized earlier in this tutorial. This Retriever will fetch the documents most relevant to the query.

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store)
Define a Template Prompt
Create a ChatMessage object with the from_user method and pass the custom prompt for a question answering task using the RAG approach. The prompt should take in two parameters: documents, which are retrieved from a document store, and a question from the user. Use the Jinja2 looping syntax to combine the content of the retrieved documents in the prompt.

Next, initialize a ChatPromptBuilder instance with your prompt template. The ChatPromptBuilder, when given the necessary values, will automatically fill in the variable values and generate a complete prompt. This approach allows for a more tailored and effective question-answering experience.

By default, all prompt variables are treated as optional. Set required_variables="*" to ensure that all prompt variables are mandatory for the prompt.

from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template, required_variables="*")
Initialize a ChatGenerator
ChatGenerators are the components that call large language models (LLMs) and return chat completions.

Before you run the pipeline, decide how you will access the LLM:

Hosted provider API — Create an API key with a provider. In Colab, you can store it under Secrets tab or set the matching environment variable (OPENAI_API_KEY, MISTRAL_API_KEY, …). The cells below prompt for a key if it is not already set.
Local or self-hosted (including on Colab) — If you prefer not to use a remote API, you can run an open-weight model on your machine or the Colab runtime with TransformersChatGenerator. See the generators documentation for more integrations.
The next three sections show OpenAI, Mistral, and Hugging Face as examples. Run only one of them to define chat_generator.

Use open-weight models from Hugging Face (no API key required for local inference)

Initialize TransformersChatGenerator with an open-weight LLM from Hugging Face, such as Qwen/Qwen3-4B-Instruct-2507. TransformersChatGenerator is provided by the transformers-haystack integration. To call models through the Hugging Face Inference API instead, use HuggingFaceAPIChatGenerator, which requires a Hugging Face API token.

from haystack_integrations.components.generators.transformers import TransformersChatGenerator

chat_generator = TransformersChatGenerator(model="Qwen/Qwen3-4B-Instruct-2507")
Use OpenAI’s GPT models (requires an API key)

Get an OpenAI API key and set it as the OPENAI_API_KEY environment variable. Then initialize OpenAIChatGenerator with the model name you want to use.

import os
from getpass import getpass
from haystack.components.generators.chat import OpenAIChatGenerator

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
    
chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
Use Mistral models (requires a free API key)

Get a Mistral API key (free tier available) and set it as the MISTRAL_API_KEY environment variable. Then initialize MistralChatGenerator with the model name you want to use.

import os
from getpass import getpass
from haystack_integrations.components.generators.mistral import MistralChatGenerator

if "MISTRAL_API_KEY" not in os.environ:
  os.environ["MISTRAL_API_KEY"] = getpass("Enter Mistral API key:")
  
chat_generator = MistralChatGenerator(model="mistral-small-latest")
You can replace the examples above with any Haystack ChatGenerator that fits your setup: another API provider or a local / Colab-hosted backend. See the full list of chat generators here.

Build the Pipeline
To build a pipeline, add all components to your pipeline and connect them. Create connections from text_embedder’s “embedding” output to “query_embedding” input of retriever, from retriever to prompt_builder and from prompt_builder to llm. Explicitly connect the output of retriever with “documents” input of the prompt_builder to make the connection obvious as prompt_builder has two inputs (“documents” and “question”).

For more information on pipelines and creating connections, refer to Creating Pipelines documentation.

from haystack import Pipeline

basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)
# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
That’s it! Your RAG pipeline is ready to generate answers to questions!

Asking a Question
When asking a question, use the run() method of the pipeline. Make sure to provide the question to both the text_embedder and the prompt_builder. This ensures that the {{question}} variable in the template prompt gets replaced with your specific question.

⚠️ If you host the model on the Colab runtime (for example with TransformersChatGenerator), the first pipeline run can take longer as the LLM is loaded and prepared for inference.

question = "What does Rhodes Statue look like?"

response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0].text)
question = "What does Rhodes Statue look like?"

response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0].text)
Here are some other example questions to test:

examples = [
    "Where is Gardens of Babylon?",
    "Why did people build Great Pyramid of Giza?",
    "What does Rhodes Statue look like?",
    "Why did people visit the Temple of Artemis?",
    "What is the importance of Colossus of Rhodes?",
    "What happened to the Tomb of Mausolus?",
    "How did Colossus of Rhodes collapse?",
]
What’s next
🎉 Congratulations! You’ve learned how to create a generative QA system for your documents with the RAG approach.

If you liked this tutorial, you may also enjoy:

Filtering Documents with Metadata
Preprocessing Different File Types
Creating a Hybrid Retrieval Pipeline
To stay up to date on the latest Haystack developments, you can subscribe to our newsletter and join the Haystack Discord community.
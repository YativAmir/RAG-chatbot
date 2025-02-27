
from langchain_core.output_parsers import StrOutputParser
import langchain
import os
from langsmith import utils
from langsmith import wrappers, traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from typing import List
from langchain_core.documents import Document


# ------------------------------------------------------------------------------
# INITIAL CONFIGURATION (Optional)
# ------------------------------------------------------------------------------
# - These environment variables allow you to leverage Langsmith/LangChain tracing,
#   which helps log and analyze your chain executions and LLM requests.
# - Uncomment and set them if you wish to enable advanced tracing/logging analytics.
# ------------------------------------------------------------------------------
# os.environ["LANGCHAIN_API_KEY"] = "API_KEY"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Code"
#

# LANGSMITH_TRACING="true"
# LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
# LANGSMITH_API_KEY="<API_KEY>"
# LANGSMITH_PROJECT="pr-sandy-shingle-52"
# OPENAI_API_KEY="<API_KEY>"

# ------------------------------------------------------------------------------
# 1) CREATE A ChatOpenAI LLM
# ------------------------------------------------------------------------------
# - This creates an LLM (Large Language Model) instance from LangChain's
#   ChatOpenAI wrapper around OpenAI’s chat completions API.
# - We provide the model name and the API key to authenticate.
# ------------------------------------------------------------------------------
llm = ChatOpenAI(
    api_key="API_KEY",
    model="gpt-3.5-turbo"
)

# ------------------------------------------------------------------------------
# 3) CREATE AN OUTPUT PARSER
# ------------------------------------------------------------------------------
# - StrOutputParser just returns the string as is, but can be replaced with
#   more advanced parsers (e.g. JSONOutputParser) for structured data.
# ------------------------------------------------------------------------------
output_parser = StrOutputParser()

# ------------------------------------------------------------------------------
# 4) PIPE THE LLM AND THE PARSER
# ------------------------------------------------------------------------------
# - By using the '|' (pipe) operator, LangChain allows you to chain
#   the output of one Runnable (the LLM) to the next (the parser).
# - This chain can be invoked with a single input, which passes
#   sequentially through each step.
# ------------------------------------------------------------------------------
chain1 = llm | output_parser

#

# ------------------------------------------------------------------------------
# DOCUMENT LOADING
# ------------------------------------------------------------------------------
# - This section demonstrates how to load PDF and DOCX documents into LangChain
#   Document objects. Later, these Documents can be chunked, embedded, and
#   stored in a vector database for retrieval.
# - PyPDFLoader is for PDFs, Docx2txtLoader is for DOCX files.
# - The function load_documents() returns a List[Document] with each file’s text.
# ------------------------------------------------------------------------------



def load_documents(folder_path: str) -> List[Document]:
    """
    Load PDF or DOCX documents from a specified folder and return a list of Document objects.
    """
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # If the file is a PDF, use PyPDFLoader
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        # If the file is a DOCX, use Docx2txtLoader
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue

        # Extend the documents list with all pages from the file
        documents.extend(loader.load())
    return documents


# Here we load PDF and DOCX files from the specified folder
folder_path = "C:/Users/yativ/OneDrive/Desktop/Learning2040/CivicsDoc"
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")
print(documents)

print("///////////////////Embeddings////////////////")

# ------------------------------------------------------------------------------
# CREATING EMBEDDINGS
# ------------------------------------------------------------------------------
# - OpenAIEmbeddings transforms text into a high-dimensional numerical vector.
# - These vectors can then be fed into vector stores (like Chroma) for similarity searches.
# ------------------------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key="API_KEY"
)

# We extract the page_content from each Document and embed them,
# resulting in a list of embeddings (one per doc).
document_embeddings = embeddings.embed_documents([doc.page_content for doc in documents])

print(f"Created embeddings for {len(document_embeddings)} document chunks.")

print("////////////////////vector storing/////////////////")

# ------------------------------------------------------------------------------
# STORING EMBEDDINGS IN A VECTOR STORE (Chroma)
# ------------------------------------------------------------------------------
# - We create a Chroma database from the documents and embeddings.
# - The persist_directory indicates where to store the vector database on disk.
# - collection_name is an optional label for the set of documents.
# ------------------------------------------------------------------------------
from langchain_chroma import Chroma

collection_name = "Civics"
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector store created and persisted to './chroma_db'")


# ------------------------------------------------------------------------------
# # RETRIEVER
# # ------------------------------------------------------------------------------
# # - .as_retriever() converts the vectorstore into a retriever object,
# #   which is commonly used in retrieval-augmented generation.
# # - We specify search_kwargs={"k": 2} to limit returned docs to top 2 matches.
# # ------------------------------------------------------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# retriever_results = retriever.invoke("מה הם תפקידי הכנסת?")
# print(retriever_results)
#

# ------------------------------------------------------------------------------
# BUILDING A RAG (Retrieval-Augmented Generation) CHAIN
# ------------------------------------------------------------------------------
# - We create a prompt template and define a pipeline that:
#     1) uses the retriever to get relevant docs,
#     2) merges them into context,
#     3) feeds them to the LLM,
#     4) parses the response as a string.
# ------------------------------------------------------------------------------
from langchain.schema.runnable import RunnablePassthrough

# from langchain_core.output_parsers import StrOutputParser  # (already imported above)

template = """Answer the question based only on the following context:
{context}
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(template)


def docs2str(docs):
    """Utility to join multiple Document.page_content fields into one string."""
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | docs2str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

question = "מה הם תפקידי הכנסת?"
response = rag_chain.invoke(question)
print(f"Question: {question}")
print(f"Answer: {response}")

# ------------------------------------------------------------------------------
# HISTORY-AWARE RETRIEVAL
# ------------------------------------------------------------------------------
# - Sometimes a user’s question refers to earlier discussion.
#   We can handle this by "contextualizing" or "rephrasing" the new question
#   using the conversation history.
# ------------------------------------------------------------------------------
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# A system prompt used to “contextualize” the user query based on chat history.
contextualize_q_system_prompt = """
Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# This chain takes chat history + new question → returns a “contextualized” question.
contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
print(contextualize_chain.invoke({"input": "מי מעביר את החוק?", "chat_history": []}))

# create_history_aware_retriever uses an LLM to rephrase a question
# based on chat history, then uses the retriever to fetch docs.
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)

print("hereee//////////////////////////////////")
print(history_aware_retriever)
# ------------------------------------------------------------------------------
# CREATE A QA CHAIN
# ------------------------------------------------------------------------------
# - We define a QA prompt that instructs the system to use only the retrieved context.
# - Then we build a chain that merges docs with the LLM using a "stuff" method
#   (basically stuffing the doc content into the prompt).
# ------------------------------------------------------------------------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a school Teacher. Use only following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain: A combined approach that uses the history-aware retriever and the QA chain
# for multi-step conversation. The create_history_aware_retriever method is used
# to re-contextualize queries, then we feed them to the LLM with the QA prompt.

rag_chain = create_history_aware_retriever(history_aware_retriever, llm, qa_prompt)

# ------------------------------------------------------------------------------
# DEMO MULTI-TURN CHAT
# ------------------------------------------------------------------------------
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

# First user question
question1 = "כמה קריאות צריך כדי להעביר חוק?"
print("question1")
answer1_msg = rag_chain.invoke({"input": question1, "chat_history": chat_history})
answer1 = answer1_msg.content  # לחלץ את הטקסט מתוך האובייקט AIMessage

# answer1 = rag_chain.invoke({"input": question1, "chat_history": chat_history})['answer']


# We store the conversation in chat_history for subsequent context
chat_history.extend([
    HumanMessage(content=question1),
    AIMessage(content=answer1)
])

print(f"Human: {question1}")
print(f"AI: {answer1}\n")
# Second user question
question2 = "מי מעביר את החוק?"  # we need to pu the rephrased retriver

con = vectorstore.similarity_search(question2, k=2)
# con = create_stuff_documents_chain(llm, qa_prompt)
answer2_msg = rag_chain.invoke({"input": question2, "chat_history": chat_history})
answer2 = answer2_msg.content  # לחלץ את הטקסט מתוך האובייקט AIMessage

# answer2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})['answer']

chat_history.extend([
    HumanMessage(content=question2),
    AIMessage(content=answer2)
])

print(f"Human: {question2}")
print(f"AI: {answer2}")


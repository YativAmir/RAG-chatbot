from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer




PINECONE_API_KEY = "API_KEY"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "learning2040civics"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

llm = ChatOpenAI(
    api_key="API_KEY",
    model="gpt-3.5-turbo"
   # model = "gpt-4"
)
# 3) Create Embeddings
# --------------------------------------------------------------------

# Create embeddings for each document. Embeddings are vector representations
# of text, which allow semantic similarity searches.
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name)

def query_pinecone(user_query, top_k=3):
    """Retrieve relevant text from Pinecone based on user query."""
    query_embedding = model.encode(user_query).tolist()

    # Search Pinecone
    results = index.query(vector=query_embedding, top_k=top_k,   include_metadata=True)
    retrieved_text = [match["metadata"]["text"] for match in results["matches"]]
    return retrieved_text


# Create a retriever object that you can later query with a question.


def docs2str(docs):
    return "\n\n".join(docs)


contextualize_q_system_prompt = """\
You are a helpful assistant that takes the user’s current question and past conversation (chat history), 
then returns a self-contained question in Hebrew that does not depend on the conversation context.
Follow these rules:
1. If the question already stands on its own (i.e., no pronouns or references to previous messages), leave it mostly unchanged!
2. If the question references earlier parts of the chat (using references like "this", "that", or "he/she/it"), clarify those references explicitly using information from the chat history.
3. Preserve the original meaning as closely as possible while clarfying its meaning, and keep it in the form of a question.
4. Do NOT answer the question.
5. Do NOT add extra information not present in the user’s question or the chat history.
6. Output only the revised question, nothing else.
7. Check yourself at the end if you kept the original meaning of the question
"""





contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_question}"),
    ]
)

qa_system_prompt = """\
You are a kind and knowledgeable teacher. Use ONLY the following context to answer the user's question in Hebrew:
Context:
{context}
"""

qa_final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_question}"),
    ]
)


def ask_question(user_question: str, chat_history: list):
    """
    - Takes the user question and the existing chat_history.
    - 1) Rephrase question if needed using `contextualize_q_prompt`.
    - 2) Use `retriever` to fetch docs relevant to the rephrased Q.
    - 3) Build final prompt that includes the context, chat_history, and new question.
    - 4) Call the LLM to answer, and return the AIMessage.
    """

    # 1) Rephrase question using chat_history
    rephrase_prompt_result = contextualize_q_prompt.format_prompt(
        chat_history=chat_history,
        user_question=user_question
    )

    # Now call the LLM to get the rephrased question
    rephrased_question_msg = llm.invoke(rephrase_prompt_result.to_messages())
    rephrased_question = rephrased_question_msg.content.strip()
    print("rephrased question: ",rephrased_question )

    # 2) Retrieve relevant docs using the rephrased question
    relevant_docs = query_pinecone(rephrased_question)
    context_text = docs2str(relevant_docs)


    # 3) Build final QA prompt with the retrieved context
    final_prompt_result = qa_final_prompt.format_prompt(
        chat_history=chat_history,
        context=context_text,
        user_question=user_question
        #user_question=rephrased_question
    )

    # 4) Call the LLM to get the final answer
    final_answer_msg = llm.invoke(final_prompt_result.to_messages())

    return final_answer_msg


def print_retriever_data(results):
    """
    Nicely print the retriever data from a Pinecone query.
    """
    if not results or "matches" not in results or len(results["matches"]) == 0:
        print("No matching results found.")
        return

    for i, match in enumerate(results["matches"]):
        match_id = match.get("id", "N/A")
        score = match.get("score", 0.0)
        text = match.get("metadata", {}).get("text", "No text provided")

        print(f"--- Result {i + 1} ---")
        print(f"ID: {match_id}")
        print(f"Score: {score:.4f}")
        print("Text:")
        print(text)
        print("\n")


chat_history = []  # Start with an empty conversation history.

# Example 1
question1 = "מה התפקידים של הכנסת?"
answer1_msg = ask_question(question1, chat_history)

# Add the first user question and the AI's response to history
chat_history.append(HumanMessage(content=question1))
chat_history.append(answer1_msg)

print(f"Human: {question1}")
print(f"AI: {answer1_msg.content}\n")

# Example 2
question2 = "איזה רשות היא?"
answer2_msg = ask_question(question2, chat_history)

# Add the second user question and the AI's response to history
chat_history.append(HumanMessage(content=question2))
chat_history.append(answer2_msg)

print(f"Human: {question2}")
print(f"AI: {answer2_msg.content}\n")

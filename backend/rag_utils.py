
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant that helps AuroraStay Hotels staff answer guest questions using the hotel's
    policy documents (booking and cancellation terms, room and property information, refunds and
    additional charges, and guest conduct and accessibility guidelines).

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly stated in the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or general hotel-industry practice.
    3. Include every detail needed to answer the query completely: exact times, fees, limits, deadlines, and the property code when the query names a property or city.
    4. Do not omit conditions, exceptions, approvals, or verification steps that change the answer (for example "subject to availability" or "confirm the live property profile").
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated policies, properties, or rate plans from the retrieved information.
    7. Answer the specific question in the first sentence, then give the supporting conditions.
    8. Keep the answer concise, clear, and easy for front-desk staff to read.
    9. Do not invent fees, times, room features, amenities, eligibility rules, or exceptions.
    10. If the retrieved information answers only part of the query, answer that part and state which part is not covered.
        If it does not address the query at all, say:
        "The available policy documents do not contain enough information to answer this question."
        and add the next step only if the retrieved information names one (for example, contacting the property team).
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 5,                      # defaults match the evaluated configuration
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_tokens: int = 256,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks

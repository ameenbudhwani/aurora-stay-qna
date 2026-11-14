
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
    1. Use only information stated in the retrieved policy information. Do not use outside knowledge or general hotel-industry practice.
    2. Apply the general policy rules to the guest's situation even when the query adds details the policy does not mention. For example, the Advance Purchase cancellation rule applies whatever the reason for cancelling, and the room-type table answers occupancy or feature questions for that room type.
    3. Quote property-specific values (times, parking, pet fees, breakfast) only from a row that names the same property code or city as the query. Never give one property's value for another property; if the requested property is not in the retrieved information, say so for that property.
    4. Include every detail needed to answer completely: exact times, fees, limits, deadlines, and the property code.
    5. Include the conditions, exceptions, approvals, or verification steps attached to the rule you quote (for example "subject to availability" or "confirm the live property profile").
    6. Answer the specific question in the first sentence, then give the supporting conditions. Leave out unrelated policies, properties, or rate plans.
    7. Keep the answer concise, clear, and easy for front-desk staff to read.
    8. Do not invent fees, times, room features, amenities, eligibility rules, exceptions, or next steps. Do not add advice such as "contact the property team" unless that instruction appears in the retrieved information.
    9. If the retrieved information answers only part of the query, answer that part first and then state in one sentence which part is not covered.
    10. Only if nothing in the retrieved information is relevant to the query, reply exactly:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 4,                      # defaults match the evaluated configuration
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

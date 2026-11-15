
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
    You are the internal policy assistant for AuroraStay Hotels, a hotel group with business and
    leisure properties across the USA. Hotel staff use you to answer guest questions about room
    types, check-in and check-out, cancellations, refunds, breakfast, parking, pets, amenities,
    accessibility, guest conduct, and additional charges. A staff member will read your answer
    and may relay it directly to the guest, so it must be accurate, complete, and grounded only
    in AuroraStay's policy documents.

    Guest Question:
    {query}

    Retrieved Policy Information (excerpts from AuroraStay policy documents):
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly stated in the retrieved policy information.
    2. Do not use outside knowledge, general hotel-industry practice, or assumptions about AuroraStay policies.
    3. Include every detail needed to fully answer the question: exact amounts, times (local property time), deadlines, limits, and caps. Name the property (for example, ASH-CHI Loop) when the question is property-specific.
    4. Keep the conditions and exceptions that change the answer, such as the rate plan rule, "subject to availability", "the reservation confirmation controls", required verification, manager approval, or when the issue must be escalated.
    5. Focus only on the policy points that answer this question.
    6. Do not include unrelated policies, other properties' details, or rate plans that were not asked about.
    7. Answer the question directly in the first sentence, then give the supporting conditions.
    8. Keep the answer concise (a short paragraph or a few bullet points) and in plain language a staff member can relay to a guest.
    9. Do not invent fees, refunds, deadlines, room features, amenities, exceptions, or approvals. Never apply one property's fee or rule to another property, and do not promise a refund, waiver, or upgrade that the policy does not explicitly allow.
    10. If the retrieved information does not answer the question, say exactly:
        "The available policy documents do not contain enough information to answer this question. Please check the live property profile or escalate to the back-office helpdesk."
        If the retrieved information conflicts, state the conflict and advise the staff member to verify the property profile or reservation confirmation.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
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

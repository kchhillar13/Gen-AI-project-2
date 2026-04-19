from pathlib import Path
from typing import List

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import HashingVectorizer


# BASE_DIR always points to the project root (one level up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "diabetes_medical_guidelines"


# Hardcoded diabetes guideline chunks. No PDFs or uploaded files are used.
DIABETES_GUIDELINES = [
    {
        "id": "diabetes_glucose_001",
        "source": "American Diabetes Association Standards of Care",
        "text": (
            "Glucose management is central to diabetes risk assessment. Elevated fasting "
            "plasma glucose, impaired glucose tolerance, or consistently high random "
            "glucose readings should prompt clinical follow-up, repeat testing, and "
            "individualized lifestyle counseling. People with high glucose values should "
            "be assessed for symptoms, cardiovascular risk factors, and barriers to diet "
            "and physical activity changes."
        ),
    },
    {
        "id": "diabetes_bmi_002",
        "source": "CDC Diabetes Prevention Program Guidance",
        "text": (
            "Weight management and regular physical activity are recommended for people "
            "at elevated risk of type 2 diabetes. A structured lifestyle program should "
            "emphasize gradual weight loss when appropriate, balanced nutrition, reduced "
            "intake of refined carbohydrates and sugar-sweetened beverages, and at least "
            "150 minutes per week of moderate-intensity activity when medically safe."
        ),
    },
    {
        "id": "diabetes_blood_pressure_003",
        "source": "American Diabetes Association Cardiovascular Risk Guidance",
        "text": (
            "Blood pressure should be reviewed as part of diabetes risk care because "
            "hypertension increases cardiovascular and kidney risk. Patients with elevated "
            "blood pressure should receive counseling on sodium reduction, physical "
            "activity, weight management, medication adherence when prescribed, and timely "
            "follow-up with a qualified healthcare professional."
        ),
    },
    {
        "id": "diabetes_followup_004",
        "source": "American Diabetes Association Screening Guidance",
        "text": (
            "Patients with risk factors such as higher age, elevated BMI, family history, "
            "history of gestational diabetes, or abnormal glucose measurements should "
            "receive periodic screening. Follow-up testing may include fasting plasma "
            "glucose, A1C, or oral glucose tolerance testing, depending on clinician "
            "judgment and local guidelines."
        ),
    },
    {
        "id": "diabetes_medication_005",
        "source": "Clinical Diabetes Care Medication Guidance",
        "text": (
            "Medication decisions for diabetes prevention or treatment must be made by a "
            "licensed clinician. Metformin may be considered for selected high-risk adults "
            "in some guidelines, especially when glucose and BMI are elevated, but this "
            "requires individualized assessment of kidney function, contraindications, "
            "current medications, and patient preferences."
        ),
    },
    {
        "id": "diabetes_patient_education_006",
        "source": "Diabetes Self-Management Education Guidance",
        "text": (
            "Patient education should be clear, practical, and culturally appropriate. "
            "Recommendations should include recognizing symptoms of hyperglycemia, "
            "understanding risk factors, improving meal patterns, increasing safe physical "
            "activity, attending follow-up visits, and seeking urgent medical care for "
            "severe symptoms."
        ),
    },
]


class LocalHashEmbeddingFunction(EmbeddingFunction):
    """
    Small local embedding function for ChromaDB.

    HashingVectorizer avoids external model downloads and keeps this RAG layer
    deterministic for demos and coursework environments.
    """

    def __init__(self):
        self.vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self.vectorizer.transform(input)
        return vectors.toarray().astype("float32").tolist()


embedding_function = LocalHashEmbeddingFunction()


def get_guideline_collection():
    """
    Creates or loads the persistent diabetes guideline vector collection.
    The store is saved in ./chroma_db/ at the project root.
    """

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "Hardcoded diabetes medical guideline chunks"},
    )

    existing_count = collection.count()
    if existing_count == 0:
        collection.add(
            ids=[item["id"] for item in DIABETES_GUIDELINES],
            documents=[item["text"] for item in DIABETES_GUIDELINES],
            metadatas=[
                {
                    "source": item["source"],
                    "chunk_id": item["id"],
                }
                for item in DIABETES_GUIDELINES
            ],
        )

    return collection


def retrieve_guidelines(query: str, top_k: int = 3) -> List[str]:
    """
    Retrieves the most relevant hardcoded diabetes guideline chunks.

    Parameters
    ----------
    query : str
        Patient-specific question or risk profile summary.
    top_k : int
        Number of guideline chunks to return. Defaults to 3.

    Returns
    -------
    list[str]
        Relevant guideline chunks with source labels.
    """

    collection = get_guideline_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    guideline_chunks = []
    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Medical guideline")
        guideline_chunks.append(f"{source}: {document}")

    return guideline_chunks


if __name__ == "__main__":
    test_query = (
        "High diabetes risk patient with elevated glucose, high BMI, and need "
        "for lifestyle follow-up recommendations."
    )
    chunks = retrieve_guidelines(test_query)

    print("Top 3 relevant diabetes guideline chunks:\n")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}\n")

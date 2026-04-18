import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from server.modules.embeddings import get_embeddings_model, get_embeddings_dimension
except ModuleNotFoundError:
    from modules.embeddings import get_embeddings_model, get_embeddings_dimension

load_dotenv()

PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _validate_env() -> tuple[str, str]:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    missing = []

    if not google_api_key:
        missing.append("GOOGLE_API_KEY")
    if not pinecone_api_key:
        missing.append("PINECONE_API_KEY")

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return google_api_key, pinecone_api_key


def _extract_index_dimension(index_description) -> int:
    if isinstance(index_description, dict):
        dimension = index_description.get("dimension")
        if dimension is not None:
            return int(dimension)

    dimension = getattr(index_description, "dimension", None)
    if dimension is not None:
        return int(dimension)

    raise RuntimeError(
        f"Unable to detect dimension for Pinecone index '{PINECONE_INDEX_NAME}'."
    )


def _get_index(pinecone_api_key: str, embedding_dimension: int):
    pc = Pinecone(api_key=pinecone_api_key)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    listed_indexes = pc.list_indexes()
    if hasattr(listed_indexes, "names"):
        existing_indexes = listed_indexes.names()
    else:
        existing_indexes = [i["name"] for i in listed_indexes]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric="dotproduct",
            spec=spec,
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        existing_dimension = _extract_index_dimension(
            pc.describe_index(PINECONE_INDEX_NAME)
        )
        if existing_dimension != embedding_dimension:
            raise RuntimeError(
                f"Pinecone index '{PINECONE_INDEX_NAME}' has dimension "
                f"{existing_dimension}, but the selected embedding model outputs "
                f"{embedding_dimension}. Use a new PINECONE_INDEX_NAME in server/.env "
                "or recreate the existing index."
            )

    return pc.Index(PINECONE_INDEX_NAME)

# load,split,embed and upsert pdf docs content

def _normalize_label(value: Optional[str], fallback: str) -> str:
    clean = (value or "").strip().lower().replace(" ", "-")
    return clean or fallback


def load_vectorstore(uploaded_files, specialty: Optional[str] = None, namespace: Optional[str] = None):
    google_api_key, pinecone_api_key = _validate_env()
    os.environ["GOOGLE_API_KEY"] = google_api_key

    embed_model = get_embeddings_model()
    embedding_dimension = get_embeddings_dimension()
    index = _get_index(pinecone_api_key, embedding_dimension)
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    ingest_namespace = _normalize_label(namespace, "medical-guidelines")
    ingest_specialty = _normalize_label(specialty, "general")

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        file_specialty = ingest_specialty
        if ingest_specialty == "general":
            file_specialty = _normalize_label(Path(file_path).stem, "general")

        metadatas = [
            {
                "text": chunk.page_content,
                "source_name": Path(file_path).name,
                "source_url": str(file_path),
                "doc_type": "guideline",
                "category": "medical-knowledge",
                "specialty": file_specialty,
                "namespace": ingest_namespace,
                **chunk.metadata,
            }
            for chunk in chunks
        ]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(
                vectors=list(zip(ids, embeddings, metadatas)),
                namespace=ingest_namespace,
            )
            progress.update(len(embeddings))

        print(f"Upload complete for {file_path}")

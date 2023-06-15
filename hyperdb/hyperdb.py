import pickle
import numpy as np
import gzip
from sentence_transformers import SentenceTransformer
from hyperdb.galaxy_brain_math_shit import cosine_similarity, euclidean_metric, derridaean_similarity, adams_similarity, hyper_SVM_ranking_algorithm_sort

EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 'cpu')

def get_embedding(documents, key=None):
    texts = None
    if isinstance(documents, list):
        if isinstance(documents[0], dict):
            texts = []
            if isinstance(key, str):
                if "." in key:
                    key_chain = key.split(".")
                else: 
                    key_chain = [key]
                for doc in documents:
                    for key in key_chain:
                        doc = doc[key]
                    texts.append(doc.replace("\n", " "))
            elif key is None:
                for doc in documents:
                    text = ", ".join([f"{key}: {value}" for key, value in doc.items()])
                    texts.append(text)
        elif isinstance(documents[0], str):
            texts = documents
            
    embeddings = EMBEDDING_MODEL.encode(texts)
    return embeddings


class HyperDB:
    def __init__(
        self,
        documents=None,
        vectors=None,
        embedding_function=None,
        similarity_metric="cosine",
    ):
        documents = documents or []
        self.documents = []
        self.vectors = []
        self.embedding_function = embedding_function or get_embedding
        self.add_documents(documents)
        if similarity_metric.__contains__("cosine"):
            self.similarity_metric = cosine_similarity
        elif similarity_metric.__contains__("euclidean"):
            self.similarity_metric = euclidean_metric
        elif similarity_metric.__contains__("derrida"):
            self.similarity_metric = derridaean_similarity
        elif similarity_metric.__contains__("adams"):
            self.similarity_metric = adams_similarity
        else:
            raise Exception("Similarity metric not supported. Please use either 'cosine', 'euclidean', 'adams', or 'derrida'.")

    def dict(self, vectors=False):
        if vectors:
            return [{
                "document": document,
                "vector": vector.tolist(),
                "index": index
            } for index, (document, vector) in enumerate(zip(self.documents, self.vectors))]
        return [{"document": document, "index": index} for index, document in enumerate(self.documents)]

    def add(self, documents, vectors=None):
        if not isinstance(documents, list):
            self.add_document(documents, vectors)
        else:
            self.add_documents(documents, vectors)


    def add_document(self, document, vector=None):
        if vector is None:
            vectors = self.embedding_function([document])
            vector = vectors[0]
        else:
            vector = np.array(vector)
        if len(self.vectors) == 0:
            self.vectors = np.empty((0, vector.shape[0]), dtype=np.float32)
        elif vector.shape[0] != self.vectors.shape[1]:
            raise ValueError("All vectors must have the same length.")
        self.vectors = np.vstack([self.vectors, vector]).astype(np.float32)
        self.documents.append(document)



    def remove_document(self, index):
        self.vectors = np.delete(self.vectors, index, axis=0)
        self.documents.pop(index)

    def add_documents(self, documents, vectors=None):
        if not documents:
            return
        vectors = vectors or self.embedding_function(documents)
        for document, vector in zip(documents, vectors):
            self.add_document(document, vector)


    def save(self, storage_file):
        data = {
            "vectors": self.vectors,
            "documents": self.documents
        }
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(storage_file, "wb") as f:
                pickle.dump(data, f)

    def load(self, storage_file):
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "rb") as f:
                data = pickle.load(f)
        else:
            with open(storage_file, "rb") as f:
                data = pickle.load(f)
        self.vectors = data["vectors"].astype(np.float32)
        self.documents = data["documents"]

    def query(self, query_text, top_k=5, return_similarities=True):
        query_vector = self.embedding_function([query_text])[0]
        ranked_results, similarities = hyper_SVM_ranking_algorithm_sort(
            self.vectors,
            query_vector,
            top_k=top_k,
            metric=self.similarity_metric
        )
        if return_similarities:
            return list(zip([self.documents[index] for index in ranked_results], similarities))
        return [self.documents[index] for index in ranked_results]

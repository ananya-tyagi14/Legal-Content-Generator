from LegalKnowledgeIndexer import LegalKnowledgeIndexer
from BM25Preprocessor      import BM25Preprocessor
import re


class HybridRetrieval:

    """
    Combines semantic retrieval (via a FAISS index) and BM25 ranking to return the most relevant text chunks for a legal query.
    """

    def __init__(self, input_query):

        self.query = input_query  # Store the query string

        #Set up Semantic Buffer
        self.sb = LegalKnowledgeIndexer(
            index_path="./legal_chunks.faiss",
            chunks_path="./legal_chunks.chunks.pkl",
            id_meta_path="./legal_chunks.ids.pkl"
        )

        self.sb.load()  # Load the FAISS index and metadata into memory

        #Set up the BM25 preprocessor on the JSON files of chunks:
        self.bm25 = BM25Preprocessor(json_dir="./legal resources/json_files2", k1=1.5, b=0.75)
        # Build the BM25 index
        self.bm25.load_and_prepare()

        

    def run(self, top_k_sem = 5, top_k_bm25 = 3):

        """
        Perform hybrid retrieval:
          1. Fetch top_k_sem semantic hits (chunk IDs + scores) via the FAISS index.
          2. From those semantic hits, extract their internal integer IDs for BM25.
          3. Score those candidate IDs with BM25 and return the top_k_bm25 chunks.
        Args:
            top_k_sem (int): Number of semantic neighbors to retrieve.
            top_k_bm25 (int): Number of BM25-ranked chunks to return.
        Returns:
            List[str]: Cleaned text strings of the top BM25 chunks.
        """

        #Query the semantic index: returns a list of (score, chunk_id) tuples
        sem_hits = self.sb.query(self.query, top_k=top_k_sem)

        #Convert each chunk_id (metadata ID) into its integer index inside BM25Preprocessor
        candidate_int_ids = [self.sb.ids.index(chunk_id) for _, chunk_id in sem_hits]

        #Run BM25 scoring on only those candidate document indices
        bm25_hits = self.bm25.score_subset(self.query, candidate_int_ids, top_k=top_k_bm25)

        result_chunks = []
        for doc_id, score in bm25_hits:
            chunk = self.bm25.raw_chunks[doc_id]

            lines = chunk.splitlines()
            
            while lines and lines[0].startswith("Section:"):
                lines.pop(0)

            
            clean_chunk = "\n".join(lines).strip()
            result_chunks.append(clean_chunk)

        return result_chunks




    
        
    


    


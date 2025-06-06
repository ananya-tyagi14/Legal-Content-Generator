import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class LegalKnowledgeIndexer:

    """
    Builds and queries a FAISS index over “flattened” legal document chunks.

    """

    def __init__(self, model_name = "all-MiniLM-L12-v2", index_path = None, chunks_path=None, id_meta_path = None):

            # Directory containing JSON files of structured chunks
            self.json_dir = r"C:\Users\User\OneDrive\Documents\UNI work\SCC\year 4\placement\Final-Project-Code\legal resources\json_files2"

            # Paths for saving/loading index and metadata
            self.index_path = index_path
            self.chunks_path = chunks_path
            self.id_meta_path = id_meta_path

            # file_chunks: maps filename → list of flattened chunk strings
            self.file_chunks = {}
             # chunk_meta: maps filename → list of metadata dicts (section/subsection/topic) per chunk
            self.chunk_meta = {}

            # After flattening, we combine all chunks across files into a single list:
            self.ids = []
            self.all_chunks = []

            # Load the SentenceTransformer model for embedding text
            self.model = SentenceTransformer(model_name)
            self.index = None


    def flatten_content(self, entry):

        """
        Flatten a single JSON “entry” into one string:
        """
        
        lines = []

        # Iterate over each content block in this entry
        for block in entry.get("Content", []):
            text = block.get("text", "").strip()
            if text:
                lines.append(text)

            # Recursively process any nested lists in this block
            self.process_list(block.get("list", []), depth=0, lines=lines)

        # Now we insert blank lines in out_lines when:
        #  - A heading (Section:/Subsection:) is followed by a bullet
        #  - Or any line is followed by a non-bullet
        out_lines = []
        for i, ln in enumerate(lines):
            out_lines.append(ln)
            if i + 1 < len(lines):
                next_ln = lines[i + 1]
                next_is_bullet = next_ln.lstrip().startswith(("•", "-"))
                is_heading = ln.startswith("Section:") or ln.startswith("Subsection:")

                # If current is a heading and next is a bullet, or next is not a bullet, insert a blank line
                if (is_heading and next_is_bullet) or (not next_is_bullet):
                    out_lines.append("")

        return "\n".join(out_lines)


    def process_list(self, items, depth, lines):

        """
        Recursively traverse a nested list of items and append their text to `lines`.
        Args:
            items (list of dict): Each dict has "text" and optional nested "list".
            depth (int): Current nesting depth (0 = top-level).
            lines (list of str): Accumulator list for flattened text lines.
        """

        for item in items:
            raw = item.get("text", "").strip()
            if depth == 0:
                # Top-level list item: keep raw text
                lines.append(raw)

            else:
                # Nested list: remove leading “• ”, indent by (depth * 8) spaces, and prefix with “- ”
                content = raw.lstrip("• ").strip()
                indent = ' ' * (depth * 8)
                lines.append(f"{indent}- {content}")

            # If there is a further nested list under this item, recurse
            nested = item.get("list", [])
            if nested:
                self.process_list(nested, depth + 1, lines)
                

    def load_and_flatten(self):

        """
        Read every .json file in self.json_dir, flatten each entry, and store:
         - self.file_chunks[filename]: list of flattened chunk strings
         - self.chunk_meta[filename]: list of metadata dicts for each chunk
        The metadata dict contains "section", "subsection", and "topic".
        """

        # Create a directory for any output if needed
        output_dir = os.path.join(self.json_dir, "flattened_chunks")
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over all JSON files sorted alphabetically
        for file_idx, fname in enumerate(sorted(os.listdir(self.json_dir))):
            if not fname.endswith(".json"):
                continue

            path = os.path.join(self.json_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = []
            meta_for_file = []
            for entry_idx, entry in enumerate(data):
                chunk = self.flatten_content(entry)
                if chunk:
                    chunks.append(chunk)

                    # Record metadata for this chunk
                    meta_for_file.append({
                        "section":  entry.get("Section"),
                        "subsection": entry.get("Subsection"),
                        "topic": entry.get("topic")
                    })

            # Store the flattened chunks and metadata under this filename
            self.file_chunks[fname] = chunks
            self.chunk_meta[fname] = meta_for_file
            

    def embed(self, texts):

        """
        Embed a list of text strings into normalized dense vectors using SentenceTransformer.
        Raises an error if any embedding is not unit-norm or contains NaN/Inf.
        Args:
            texts (list of str): Texts to embed.
        Returns:
            np.ndarray: 2D array of shape (len(texts), embedding_dim), unit-normalized.
        """
        
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # shape, norms, nan/inf checks
        assert embs.ndim == 2, f"Embeddings should be 2D, got {embs.ndim}D"
        norms = np.linalg.norm(embs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("Some embeddings deviate from unit norm")
        if np.any(np.isnan(embs)) or np.any(np.isinf(embs)):
            raise ValueError("Invalid values in embeddings")
        return embs

    
    def build_index(self):

        """
        Construct the FAISS index over all flattened chunks:
          
        """

        #Gather all chunks and assign them a unique ID
        for fname, chunks in self.file_chunks.items():
            metas = self.chunk_meta.get(fname, [])
            for i, (chunk, meta) in enumerate(zip(chunks, metas)):
                self.all_chunks.append(chunk)
                # Create an ID string combining filename and chunk index
                self.ids.append(f"{fname}_chunk_{i}")

        #Embed all chunks into a 2D array
        embeddings = self.embed(self.all_chunks)
        dim = embeddings.shape[1]

        #Create a FAISS IndexFlatIP (inner-product) for nearest-neighbor search
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # write index and metadata to disk
        if self.index_path:
            faiss.write_index(self.index, self.index_path)
            
        if self.chunks_path:
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.all_chunks, f)
                
        if self.id_meta_path:
            with open(self.id_meta_path, 'wb') as f:
                pickle.dump(self.ids, f)
                         
        print(f"Built index over {len(self.all_chunks)} chunks.")


    def load(self):

        """
        Load a pre-built FAISS index and associated chunk lists/IDs from disk.
        
        """

        if self.index_path:
            self.index = faiss.read_index(self.index_path)

        if self.chunks_path:
            with open(self.chunks_path, "rb") as f:
                self.all_chunks = pickle.load(f)

        if self.id_meta_path:
            with open(self.id_meta_path, "rb") as f:
                self.ids = pickle.load(f)

    def query(self, text, top_k = 5):

        """
        Query the FAISS index for the nearest neighbors of the input text.

        Args:
            text (str): The query string to embed and search for.
            top_k (int): Number of nearest neighbors to return.
        Returns:
            List[ (float, str) ]: Each tuple is (score, chunk_id).
        """
        # Compute a single embedding for the query
        q_emb = self.embed([text])
        # Search the index: D (scores), I (indices into self.ids)
        D, I = self.index.search(q_emb, top_k)
        # Convert each result to (float_score, chunk_id) using self.ids
        return [(float(score), self.ids[idx]) for score, idx in zip(D[0], I[0])]



    
if __name__ == "__main__":

    indexer = LegalKnowledgeIndexer(
        index_path="./legal_chunks.faiss",
        chunks_path="./legal_chunks.chunks.pkl",
        id_meta_path="./legal_chunks.ids.pkl"
    )
    indexer.load_and_flatten()
    indexer.build_index()

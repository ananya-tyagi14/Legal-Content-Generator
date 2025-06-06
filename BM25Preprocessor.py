import os
import json
import re
import unicodedata
from collections import defaultdict
import math

class BM25Preprocessor:

    """
    Preprocess a directory of JSON files (each containing document “chunks”) to build
    the data structures needed for BM25 scoring:
     - raw_chunks: original unnormalized text of each chunk
     - normalised_chunks: token‐lowercased, punctuation‐stripped text
     - df: document frequency of each term across all chunks
     - postings: term → { doc_id: term_frequency_in_that_doc, … }
     - idf: inverse document frequency per term
     - doc_len: list of token counts for each chunk
     - avgdl: average document length across all chunks
    """

    def __init__(self, json_dir, k1=1.5, b=0.75):

        self.json_dir = json_dir
        self.k1 = k1
        self.b = b
        
        self.raw_chunks = []                 # orginal text chunks
        self.normalised_chunks = []          # processed text chunks

        self.N = 0                           # total number of documents
        self.avgdl = 0                       # average document length (mean number of tokens per chunk)
        self.doc_len = []                    # each entry is the length of the corresponding normalised chunk
        self.df = defaultdict(int)           # document frequency per term
        self.postings = defaultdict(dict)    # mapping of term-frequency for that term
        self.idf = {}                        # inverse document frequency


    def flatten_content(self, entry):

        """
        flatten JSON entry into a single text blob with headings and list items in order.
        Args:
            entry (dict): A JSON object representing one “chunk” with nested content.
        Returns:
            str: A single string combining section/subsection lines and content lines.
        """

        lines = []

        # If the entry has a non-empty "Section", prepend "Section: <section>."
        if entry.get("Section"):
            lines.append(f"Section: {entry['Section']}.")

        # If the entry has a non-empty "Subsection",  prepend "Subsection: <subsection>."
        if entry.get("Subsection"):
            lines.append(f"Subsection: {entry['Subsection']}.")

        # Process each block inside "Content" in order
        for block in entry.get("Content", []):
            text = block.get("text", "").strip()
            if text:
                lines.append(text)

            # Recursively process any nested list items inside this block
            self.process_list(block.get("list", []), depth=0, lines=lines)

        return "\n".join(lines)


    def process_list(self, items, depth, lines):

        """
        Recursively traverse a nested list of items and append their text lines to `lines`.
        Args:
            items (list of dict): Each dict has "text" (str) and optional "list" (list of items).
            depth (int): Current nesting depth (0 for top-level list).
            lines (list of str): Accumulator list for the flattened text lines.
        """

        for item in items:
            raw = item.get("text", "").strip()
            if depth == 0:
                 # Top-level list item
                lines.append(raw)

            else:
                # Deeper nested list: strip leading “• ” and indent by 8 spaces per depth
                content = raw.lstrip("• ").strip()
                indent = ' ' * (depth * 8)
                lines.append(f"{indent}* {content}")

            # If this item has a nested sublist, recurse with depth + 1
            nested = item.get("list", [])
            nested = item.get("list", [])
            if nested:
                self.process_list(nested, depth + 1, lines)


    def normalize(self, text):

        text = unicodedata.normalize('NFKC', text)  #1. Unicode normalization
        text = text.lower()  # 2. Lowercase
        text = re.sub(r"\b(\w+)'s\b", r"\1", text)  # 3. Remove "'s" possessives
        text = re.sub(r"[^\w\s]", " ", text)  # 4. Remove punctuation (anything that isn’t a word character or whitespace
        text = re.sub(r"\s+", " ", text).strip()   # 5. Collapse multiple spaces
        return text


    def load_and_prepare(self):

        """
        Load every JSON file in `self.json_dir`, flatten each entry into a text chunk,
        normalize it, and build the BM25 index structures.
        """

        # Iterate over all files in the JSON directory in sorted order
        for file_idx, fname in enumerate(sorted(os.listdir(self.json_dir))):
            if not fname.lower().endswith('.json'):
                continue

            path = os.path.join(self.json_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)   # data is a list of “entry” dicts

             # For each entry (chunk) in this JSON file
            for entry_idx, entry in enumerate(data):
                chunk = self.flatten_content(entry)
                if chunk:
                    # Store the raw text chunk
                    self.raw_chunks.append(chunk)
                    # Store its normalized form
                    self.normalised_chunks.append(self.normalize(chunk))

            

        self.N = len(self.normalised_chunks)  #Compute total number of chunks N

        # Sum of token counts across all chunks
        total_len = 0

        # Iterate over each normalized chunk to compute term frequencies and document lengths
        for idx, doc in enumerate(self.normalised_chunks):
            tokens = doc.split()
            dl = len(tokens)
            total_len += dl
            self.doc_len.append(dl)
            tf_counts = defaultdict(int)

            # Count term frequencies in this document
            for tok in tokens:
                tf_counts[tok] += 1

            # For each unique token in this document, update df and postings
            for tok, freq in tf_counts.items():
                self.df[tok] += 1
                self.postings[tok][idx] = freq
                
        # Compute average document length (avgdl)
        self.avgdl = total_len / self.N if self.N else 0
        for tok, df in self.df.items():
            
            # Compute IDF for each term using BM25’s IDF formula:
            self.idf[tok] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        


    def score_subset(self, query, candidate_ids, top_k = 3):

        """
        Given a search query and a subset of document IDs (candidate_ids),
        compute BM25 scores for each candidate document.
        Args:
            query (str): The user’s raw query string.
            candidate_ids (Iterable[int]): Document indices to consider.
            top_k (int): Number of top-scoring docs to return.
        Returns:
            List[(doc_id, score)]: Ranked list of (doc_id, BM25_score) tuples.
        """

        # Normalize the query the same way we normalized documents
        norm_q = self.normalize(query)
        q_tokens = norm_q.split()
        scores = {}   # Accumulate BM25 scores per doc_id

        # For each token in the query
        for tok in q_tokens:
            if tok not in self.postings:
                continue
            idf = self.idf.get(tok, 0.0) # IDF for this term

            # For every document that contains this term
            for doc_id, freq in self.postings[tok].items():
                if doc_id not in candidate_ids:
                    continue
                
                dl  = self.doc_len[doc_id]  # Document length for this candidate

                # BM25 denominator
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)

                # Accumulate the score
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * freq * (self.k1 + 1) / denom

        # Sort documents by descending BM25 score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Return top_k results
        return ranked[:top_k]
                
    

    



        

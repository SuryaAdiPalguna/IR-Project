import pandas as pd
import pickle
import gzip
import math
from collections import defaultdict

def spimi_invert_for_inverted_index(df: pd.DataFrame, output_file_name: str) -> str:
    def new_file():
        return output_file_name
    def new_hash():
        return defaultdict(list)
    def add_to_dictionary(dictionary, term):
        dictionary[term] = []
        return dictionary[term]
    def get_postings_list(dictionary, term):
        return dictionary[term]
    def add_to_postings_list(postings_list, doc_id):
        if doc_id not in postings_list:
            postings_list.append(doc_id)
    def sort_terms(dictionary):
        return sorted(dictionary.keys())
    def write_block_to_disk(sorted_terms, dictionary, output_file):
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(dictionary, f)

    output_file = new_file()
    dictionary = new_hash()
    for _, row in df.iterrows():
        doc_id = row['id']
        tokens = row['text_preprocessed']
        for token in tokens:
            if token not in dictionary:
                postings_list = add_to_dictionary(dictionary, token)
            else:
                postings_list = get_postings_list(dictionary, token)
            add_to_postings_list(postings_list, doc_id)
    sorted_terms = sort_terms(dictionary)
    write_block_to_disk(sorted_terms, dictionary, output_file)
    return output_file

def spimi_for_dictionary_index(inverted_index: defaultdict, output_file_name: str) -> str:
    def new_file():
        return output_file_name
    def sort_terms(dictionary):
        return sorted(dictionary.keys())
    def write_block_to_disk(sorted_terms, dictionary, output_file):
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(dictionary, f)

    output_file = new_file()
    dictionary_index = {term_id: term for term_id, term in enumerate(inverted_index.keys(), start=1)}
    sorted_terms = sort_terms(dictionary_index)
    write_block_to_disk(sorted_terms, dictionary_index, output_file)
    return output_file

def spimi_for_tfidf_index(inverted_index: defaultdict, dictionary_index: defaultdict, output_file_name: str) -> str:
    def new_file():
        return output_file_name
    def new_hash():
        return defaultdict(list)
    def sort_terms(dictionary):
        return sorted(dictionary.keys())
    def write_block_to_disk(sorted_terms, dictionary, output_file):
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(dictionary, f)

    output_file = new_file()
    tfidf_index = new_hash()
    N = len(set(doc_id for postings in inverted_index.values() for doc_id in postings))
    for term, postings in inverted_index.items():
        term_id = next((key for key, value in dictionary_index.items() if value == term), None)
        df_term = len(postings)
        idf = math.log(N / (1 + df_term))
        for doc_id in postings:
            tf = 1
            tfidf = tf * idf
            tfidf_index[term_id].append((doc_id, tfidf))
    sorted_terms = sort_terms(tfidf_index)
    write_block_to_disk(sorted_terms, tfidf_index, output_file)
    return output_file

def spimi_for_kgram_index(dictionary_index: defaultdict, output_file_name: str, k: int = 2) -> str:
    def new_file():
        return output_file_name
    def new_hash():
        return defaultdict(set)
    def sort_terms(dictionary):
        return sorted(dictionary.keys())
    def write_block_to_disk(sorted_terms, dictionary, output_file):
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(dictionary, f)

    output_file = new_file()
    kgram_index = new_hash()
    for term_id, term in dictionary_index.items():
        padded_term = f'${term}$'
        for i in range(len(padded_term) - k + 1):
            kgram = padded_term[i:i + k]
            kgram_index[kgram].add(term_id)
    kgram_index = {kgram: list(term_ids) for kgram, term_ids in kgram_index.items()}
    sorted_terms = sort_terms(kgram_index)
    write_block_to_disk(sorted_terms, kgram_index, output_file)
    return output_file

# ====================================================================================================

# SPIMI-INVERT(token_stream)
#   output_file = NEWFILE()
#   dictionary = NEWHASH()
#   while (free memory available)
#   do token ← next(token_stream)
#     if term(token) ∈/ dictionary
#       then postings_list = ADDTODICTIONARY(dictionary, term(token))
#       else postings_list = GETPOSTINGSLIST(dictionary, term(token))
#     if full(postings_list)
#       then postings_list = DOUBLEPOSTINGSLIST(dictionary, term(token))
#     ADDTOPOSTINGSLIST(postings_list, docID(token))
#   sorted_terms ← SORTTERMS(dictionary)
#   WRITEBLOCKTODISK(sorted_terms, dictionary, output_file)
#   return output_file

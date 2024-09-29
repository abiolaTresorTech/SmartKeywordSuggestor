# import streamlit as st
# import pandas as pd
# import numpy as np

# st.title('Uber pickups in NYC')

import os

import streamlit as st

import streamlit_ext as ste

from st_files_connection import FilesConnection

from sentence_transformers import SentenceTransformer

import faiss

#import numpy as np


import pandas as pd

from google.cloud import storage
# from google import cloud

from io import BytesIO

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ds-research-playground-d3bb9f4b9084.json"

bucket_name = "datasets-datascience-team"

list_of_keywords_blob_name =  "datasets-datascience-team/Streamlit_apps_datasets/Keyword_suggestor_datasets/extracted_list_of_keywords.pkl"
index_blob_name = "datasets-datascience-team/Streamlit_apps_datasets/Keyword_suggestor_datasets/extracted_keywords_index.bin"

# client = storage.Client()

import faiss
from io import BytesIO

conn = st.connection('gcs', type=FilesConnection)

# Custom callback reader for reading from bytes
class BytesReader:
    def __init__(self, data_bytes):
        self.data = BytesIO(data_bytes)
    
    def read(self, size):
        return self.data.read(size)
    
def read_gcp_excel_csv_into_pd_df(bucket_name, blob_name, client):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    data_bytes = blob.download_as_bytes()

    if blob_name.endswith(".csv"):
        df = pd.read_csv(BytesIO(data_bytes))
    elif blob_name.endswith(".xlsx"):
        df = pd.read_csv(BytesIO(data_bytes))
    elif blob_name.endswith(".pkl"):
        df = pd.read_pickle(BytesIO(data_bytes))
    else:
        raise("The data you're trying to fetch should either be excel file or csv.")
    return df

def read_gcp_file_into_bytes(bucket_name, blob_name, client):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    data_bytes = blob.download_as_bytes()
    return data_bytes


def read_gcp_bin_into_faiss_index(bucket_name, blob_name, client):
    ind_as_bytes = read_gcp_file_into_bytes(bucket_name, blob_name, client)

    # Creating an instance of the custom reader
    byte_reader = BytesReader(ind_as_bytes)

    # Creating a FAISS callback IO reader with the custom reader
    callback_reader = faiss.PyCallbackIOReader(byte_reader.read)

    # Reading the FAISS index from the callback reader
    index = faiss.read_index(callback_reader)

    return index


# st.image("/Users/abiolatresordjigui/DM/streamlit-apps/Data/logo_dm.png", width=100)

st.title("Smart Keyword Suggestor")

if "init" not in st.session_state or not st.session_state.init:
    with st.spinner("Setting everything up..."):
        # index = read_gcp_bin_into_faiss_index(bucket_name, index_blob_name, client)
        index = conn.read(bucket_name+"/"+index_blob_name, input_format="binary", ttl=600)
        st.session_state.index = index
        # extracted_keywords = read_gcp_excel_csv_into_pd_df(bucket_name, list_of_keywords_blob_name, client)["Keyword"].to_list()
        extracted_keywords = conn.read(bucket_name+"/"+list_of_keywords_blob_name, input_format="pickle", ttl=600)
        st.session_state.extracted_keywords = extracted_keywords
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.model = model

        st.session_state.init = True
else:
    index  =st.session_state.index
    extracted_keywords  =st.session_state.extracted_keywords
    model  =st.session_state.model
#if 'initialized' not in st.session_state or not st.session_state.initialized:
#with st.spinner("Setting things up..."):
    #st.write("ohhhh 1")
    #keywords_embeddings = torch.from_numpy(np.load('/Users/abiolatresordjigui/DM/streamlit-apps/extracted_keywords_normalized_embeddings.npy'))
st.success("Everything was set up successfully!")

#@st.cache_data
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")



# def build_faiss_index(embeddings):
#     d = embeddings.shape[1]
#     index = faiss.IndexFlatL2(d)
#     faiss.normalize_L2(embeddings.numpy())
#     index.add(embeddings.numpy())
#     return index

def retrieve_documents(query, model,   k=20):
    query_embedding = model.encode(query, convert_to_tensor=True)
    faiss.normalize_L2(query_embedding.cpu().numpy().reshape(1, -1))
    D, I = index.search(query_embedding.cpu().numpy().reshape(1, -1), k)
    retrieved_docs = [extracted_keywords[i] for i in I[0]]
    scores = [d for d in D[0]]
    return retrieved_docs, scores

# def generate_response(retrieved_docs, query):
#     generator = pipeline("text-generation", model="gpt-2")
#     context = " ".join(retrieved_docs)
#     prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
#     response = generator(prompt, max_length=150, num_return_sequences=1)
#     return response[0]['generated_text']


def main():

    query = st.text_input("Enter a topic here:", placeholder="My Topic")
    if query:
        retrieved_docs, scores = retrieve_documents(query, model)
        result_data = pd.DataFrame({"Keyword":retrieved_docs, "Score":scores})

        st.write("Response")
        st.dataframe(result_data, use_container_width = True)

        result_data_as_csv = convert_df_to_csv(result_data)
        ste.download_button(
        label="Download results as CSV",
        data=result_data_as_csv,
        file_name="suggested_keywords_for_{}.csv".format(query),
        mime="text/csv",
    )

if __name__ == "__main__":
    main()

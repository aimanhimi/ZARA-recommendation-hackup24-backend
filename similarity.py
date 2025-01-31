import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize Pinecone
pc = Pinecone(api_key=api_key)
index_name = 'caption-vectors'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,  # Ensure this matches your vectorizer output
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(index_name)

df = pd.read_csv('captions.csv')
tfidf = TfidfVectorizer(max_features=512)
tfidf.fit(df['caption'])

def vectorize_and_pad(caption):
    vector = tfidf.transform([caption]).toarray().flatten()
    if len(vector) < 512:
        vector = np.pad(vector, (0, 512 - len(vector)), 'constant')
    return vector

def search_similar_captions(input_sentence, top_k=5):
    input_vector = vectorize_and_pad(input_sentence)
    print(f"Input vector: {input_vector}")
    results = index.query(vector=input_vector.tolist(), top_k=top_k, include_metadata=True)

    res = []
    
    print(f"Search results for '{input_sentence}':")
    for idx, match in enumerate(results['matches']):
        image_id = match['id']
        score = match['score']
        caption = match['metadata']['caption']  # Retrieve caption from metadata
        print(f"ID: {image_id}, Score: {score:.4f}, Caption: {caption}")
        res.append((image_id, score, caption))

    return res
    
    


def get_image_caption_from_id(id):
    row = df.iloc[id-1]
    return row['caption']

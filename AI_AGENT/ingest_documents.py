"""
Simple TXT File Ingestion Script for Rooman Knowledge Base
Only processes .txt files using Gemini embeddings
"""

import os
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import time

# ========== CONFIGURATION ==========
GEMINI_API_KEY = "AIzaSyDfaZrIBQPiQX3k1LIULsru1no_FRh3XHA"
PINECONE_API_KEY = "pcsk_58nZ62_UMekrnN77cyQzCL5Nm8R2dqxgpHATAKPyzpeCPeybqhYhKmUs6auMihQSKC239f"
INDEX_NAME = "rooman-kb"
DOCUMENTS_FOLDER = "documents"

# Configure
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ========== FUNCTIONS ==========

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_embedding(text):
    """Create embedding using Gemini"""
    try:
        result = genai.embed_content(
            model="models/embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def process_txt_files(folder_path):
    """Process all TXT files in folder"""
    all_chunks = []
    chunk_id = 0
    
    if not os.path.exists(folder_path):
        print(f"Creating {folder_path} folder...")
        os.makedirs(folder_path)
        print(f"📁 Add your .txt files to '{folder_path}' and run again!")
        return None
    
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"⚠️ No .txt files found in {folder_path}")
        return None
    
    print(f"📄 Found {len(txt_files)} .txt files")
    
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        print(f"\n🔄 Processing: {filename}")
        
        text = extract_text_from_txt(file_path)
        
        if not text.strip():
            print(f"⚠️ Empty file: {filename}")
            continue
        
        chunks = chunk_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        for chunk in chunks:
            all_chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': chunk,
                'source': filename
            })
            chunk_id += 1
    
    print(f"\n✅ Total chunks: {len(all_chunks)}")
    return all_chunks

def create_or_connect_index():
    """Create or connect to Pinecone index"""
    existing = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME not in existing:
        print(f"🔨 Creating index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("⏳ Waiting for index...")
        time.sleep(10)
    else:
        print(f"✅ Index exists: {INDEX_NAME}")
    
    return pc.Index(INDEX_NAME)

def upload_to_pinecone(chunks, index):
    """Upload chunks to Pinecone"""
    print("\n🚀 Uploading to Pinecone...")
    batch_size = 50
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        vectors = []
        for chunk in batch:
            embedding = create_embedding(chunk['text'])
            
            if embedding:
                vectors.append({
                    'id': chunk['id'],
                    'values': embedding,
                    'metadata': {
                        'text': chunk['text'][:1000],
                        'source': chunk['source']
                    }
                })
        
        if vectors:
            index.upsert(vectors=vectors)
            print(f"✅ Uploaded batch {i//batch_size + 1}")
        
        time.sleep(1)  # Rate limiting
    
    print("\n🎉 Upload complete!")

# ========== MAIN ==========

def main():
    print("=" * 60)
    print("📚 ROOMAN KB - TXT INGESTION (Gemini Embeddings)")
    print("=" * 60)
    
    chunks = process_txt_files(DOCUMENTS_FOLDER)
    
    if not chunks:
        return
    
    index = create_or_connect_index()
    upload_to_pinecone(chunks, index)
    
    stats = index.describe_index_stats()
    print(f"\n📊 Total vectors in index: {stats['total_vector_count']}")
    print("\n✅ Done! Run: streamlit run app.py")

if __name__ == "__main__":
    main()

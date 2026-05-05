from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

print("Memuat model dan dataset...")
embeddings = np.load('multimodal_embeddings_5k.npy')
df = pd.read_csv('filtered_metadata_5k.csv')

# Sesuaikan path ini dengan letak folder gambar aslimu
IMAGES_DIR = 'images'

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/', methods=['GET'])
def index():
    # Mengambil nilai query dari URL (misal web.com/?query=15)
    query_index = request.args.get('query', type=int)
    
    # 1. GENERATE KATALOG AWAL
    # Mengambil 10 data pertama yang kategorinya unik (1 dari setiap kategori)
    katalog_df = df.drop_duplicates(subset=['articleType']).head(10)
    katalog = katalog_df.to_dict('records')
    for item in katalog:
        item['image_name'] = f"{item['id']}.jpg"
        # Kita simpan index aslinya agar bisa dikirim saat gambar diklik
        item['original_index'] = df.index[df['id'] == item['id']].tolist()[0]
        
    query_image = None
    recommendations = []
    
    # 2. LOGIKA REKOMENDASI (Berjalan HANYA JIKA ada gambar yang diklik)
    if query_index is not None and 0 <= query_index < len(df):
        query_embedding = embeddings[query_index].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        similar_indices = similarities.argsort()[::-1][1:6]
        
        query_image = df.iloc[query_index].to_dict()
        query_image['image_name'] = f"{query_image['id']}.jpg"
        
        recommendations = df.iloc[similar_indices].to_dict('records')
        for i, idx in enumerate(similar_indices):
            recommendations[i]['score'] = round(similarities[idx] * 100, 2)
            recommendations[i]['image_name'] = f"{recommendations[i]['id']}.jpg"
            # Menambahkan index ke rekomendasi agar hasilnya bisa diklik lagi!
            recommendations[i]['original_index'] = int(idx)

    return render_template('index.html', katalog=katalog, query_image=query_image, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
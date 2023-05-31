import pandas as pd
import numpy as np
from unidecode import unidecode
import chardet
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Setting display options for pandas dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Loading the dataset
df = pd.read_csv("GoodReads_100k_books.csv", low_memory=False)

# Renaming and reordering the columns
df.columns = map(str.upper, df.columns)
df = df.rename(columns={"DESC": "DESCRIPTION",
                        "IMG": "BOOK_IMAGE",
                        "LINK": "BOOK_LINK",
                        "TOTALRATINGS": "TOTAL_RATINGS",
                        "BOOKFORMAT": "BOOK_FORMAT"})

df = df[['ISBN', 'TITLE', 'AUTHOR', 'DESCRIPTION', 'GENRE', 'PAGES', 'BOOK_FORMAT', 'BOOK_IMAGE', 'BOOK_LINK',
         'ISBN13', 'REVIEWS', 'RATING', 'TOTAL_RATINGS']]

# Handling missing values
df["ISBN"] = df["ISBN"].fillna("unknown")
df["GENRE"] = df["GENRE"].fillna("unknown")
df["BOOK_FORMAT"] = df["BOOK_FORMAT"].fillna("unknown")
df["BOOK_IMAGE"] = df["BOOK_IMAGE"].fillna("unknown")
df.dropna(subset="DESCRIPTION", inplace=True)
df.drop("ISBN13", axis=1, inplace=True)

# Splitting the genre into a list of genres
# df['GENRE'] = df['GENRE'].apply(lambda x: x.split(', ') if isinstance(x, str) else ['unknown'])


def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


df['DESCRIPTION'] = df['DESCRIPTION'].apply(clean_text)

df['GENRE'] = df['GENRE'].apply(lambda x: x.split(', ') if isinstance(x, str) else ['unknown'])


# Encoding check for titles
def detect_encoding(data: str) -> str:
    result = chardet.detect(data[:50].encode())  # checking the first 50 characters
    return result['encoding']


df["IS_VALID_UTF8"] = df["TITLE"].apply(detect_encoding) == "utf-8"
invalid_utf8_titles = df[df['IS_VALID_UTF8'] == True]['TITLE']

# Dropping rows with invalid utf-8 titles
df = df.drop(invalid_utf8_titles.index).reset_index(drop=True)
df = df.drop(columns=['IS_VALID_UTF8'])

# Encoding check for descriptions
df["IS_VALID_UTF8_CONTROL_desc"] = df["DESCRIPTION"].apply(detect_encoding) == "utf-8"
invalid_utf8_titles_desc = df[df['IS_VALID_UTF8_CONTROL_desc'] == True]['DESCRIPTION']

# Dropping rows with invalid utf-8 descriptions
df = df.drop(invalid_utf8_titles_desc.index).reset_index(drop=True)
df = df.drop(columns=["IS_VALID_UTF8_CONTROL_desc"])

df = df.loc[df["TOTAL_RATINGS"] >= 700].reset_index()

# Veriyi pickle dosyası olarak kaydetme
df.to_pickle("preprocessed_books.pkl")
df.columns
df.groupby('AUTHOR')["RATING"].sum().sort_values(ascending=False).head(20)
df.groupby(['TITLE', "AUTHOR"])["TOTAL_RATINGS"].sum().sort_values(ascending=False).head(20)

df[df["AUTHOR"].str.contains("Agatha Christie")].sort_values(by="RATING", ascending=False).head(10)

# Merging description and genre into one column
df['DESCRIPTION_AND_GENRE'] = df['DESCRIPTION'] + df['GENRE'].apply(', '.join)

# Content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
df['DESCRIPTION_AND_GENRE'] = df['DESCRIPTION_AND_GENRE'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['DESCRIPTION_AND_GENRE'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['TITLE']).drop_duplicates()

# Model bileşenlerini pickle dosyalarına kaydetme
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("indices.pkl", "wb") as f:
    pickle.dump(indices, f)
np.save('cosine_sim.npy', cosine_sim)


def recommend_books_t(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:2]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices]


df.head()

df.columns

print("The Kite Runner" in df['TITLE'].values)
print("Khaled Hosseini" in df['AUTHOR'].values)

df[df["AUTHOR"].str.contains("Friedrich Nietzsche")].sort_values(by="TOTAL_RATINGS", ascending=False)
df[df["TITLE"].str.contains("Stephen Hawking")]

recommend_books_t("Harry Potter and the Chamber of Secrets")

df["BOOK_IMAGE_URL"][0]
df.head()

#### APP.PY FOUR YOUTUBE ###
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from PIL import Image
from youtubesearchpython import VideosSearch
import base64

# Veriyi ve model bileşenlerini yükleme
df = pd.read_pickle("preprocessed_books.pkl")
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)
cosine_sim = np.load('cosine_sim.npy')


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


add_bg_from_local('backgr.jpg')


def search_youtube_videos(query):
    videos_search = VideosSearch(query, limit=1)
    results = videos_search.result()["result"]

    if results:
        video_url = f"https://www.youtube.com/watch?v={results[0]['id']}"
        return {"title": results[0]['title'], "url": video_url}
    else:
        return None


def recommend_books_t(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:2]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices]


# Streamlit uygulamasını başlatma
st.title("📚MIUULIB📚")
st.write(
    "MIUULIB is a content-based book recommendation system. If you enjoyed the content of the book you're reading and"
    " would like to explore related books, simply choose the last book you read from the list below. Additionally, "
    "we will provide you with a link to access the book and share video content from YouTube users discussing the book."
    "Keep on reading!!!")

# Önceden belirlenmiş kitap listesi
book_list = sorted(df['TITLE'].tolist())

# Kullanıcıdan kitap seçimi için bir selectbox oluştur
book_title = st.selectbox("Select the Last Book You Read:", book_list)

# Giriş varsa, kitap önerilerini ve ilgili YouTube videosunu gösterme
if st.button('Recommend'):
    if book_title:
        recommended_books = recommend_books_t(book_title)
    for i in range(len(recommended_books)):
        st.write("Book:")
        image = Image.open(requests.get(recommended_books.iloc[i]['BOOK_IMAGE'], stream=True).raw)
        st.image(image, width=150)
        st.write("Title:", recommended_books.iloc[i]['TITLE'])
        st.write("Author:", recommended_books.iloc[i]['AUTHOR'])
        st.write("---------------------------------------")
        st.write("Description:", recommended_books.iloc[i]['DESCRIPTION'])
        st.write("Genre:", ', '.join(recommended_books.iloc[i]['GENRE']))

        st.markdown(f'See on GoodReads: [Show Me The Book🔥]({recommended_books.iloc[i]["BOOK_LINK"]})')

        st.write("---------------------------------------")

        # YouTube'da ilgili video araması yapma
        search_term = recommended_books.iloc[i]['TITLE'] + " " + recommended_books.iloc[i]['AUTHOR'] + " Book Analysis"
        video = search_youtube_videos(search_term)
        if video:
            try:
                st.video(video['url'])
            except Exception as e:
                st.write(
                    f"Error: {e}. Can't play the video here, but you can watch it on YouTube [here]({video['url']}).")
        else:
            st.write("No related YouTube video found.")


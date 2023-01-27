# Holy Verse Search App

### [try the app here](https://bible-verse-search.streamlitapp.com/)

The Holy Bible has 31,102 verses. It is difficult to remember the references of many verses, this app solves it using Natural Language Processing.

This app allows the user to semantically search for Bible verses with just the verse text they remember to instantly get the matching verses along with their references.

![screenshot of app](./screenshot.png)

#### Working

- Transformer model to embed all Bible verses into a vector space using sentence-transformers
- compare input embeddings with embeddings of each Bible verse using cosine similarity
- fetch top N verses with high cosine similarity
- KJV Bible dataset provided by: [@scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases) in public domain.

---

This app is hosted on streamlit

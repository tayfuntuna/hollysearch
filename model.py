from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


class Evaluate:
    def __init__(self):
        self.model_path = Path('./model/EncoderModel')
        self._dataset_path = Path('./dataset')
        self.embeddings_path = self._dataset_path / 'verse_embeddings.pkl'
        self.embeddings_path2 = self._dataset_path / 'embeds.pickle.npy'
        self.bible_path = self._dataset_path / 'kjv_bible.csv'
        self.qr_path = self._dataset_path / 'qe.csv'
        self.input_embeddings = ''
        try:
            self._get_bible()
            self._get_embeddings()
            self._prepare_model()
        except:
            raise IOError

    def _get_embeddings(self):
        with open(self.embeddings_path,'rb') as f:
            data:dict = pickle.load(f)

        embeddings:np.ndarray = data['embeddings']
        
        self.embeddings = embeddings
        
        embeddings2:np.ndarray = np.load(self.embeddings_path2, allow_pickle=True)
        
        self.embeddings2 = embeddings2

    def _get_bible(self):
        self.bible = pd.read_csv(self.bible_path)
        self.qr = pd.read_csv(self.qr_path)

    def _prepare_model(self):
        self.model:SentenceTransformer = SentenceTransformer(str(self.model_path))
        self.model.eval()

    def _evaluate(self,text):
        text_embeddings = self.model.encode([text])
        return text_embeddings
    def get_embed(self,text):
        self.input_embeddings = self._evaluate(text)
    def get_verses(self,text,top=10):
        get_embed(self,text)
        similarities = cosine_similarity(self.embeddings, self.input_embeddings)
        similarities = similarities.reshape(-1)
        
        indices = similarities.argsort()
        top_indices = [idx for idx in indices][::-1][:top]

        verses = self.bible.iloc[top_indices,:]
        response = {
            'reference': verses.loc[:,'reference'].tolist(),
            'verse': verses.loc[:,'text'].tolist()
        }
        return response
    
    def get_verses2(self,text, text_embeddings, top=10):
        similarities = cosine_similarity(self.embeddings,  self.input_embeddings)
        similarities = similarities.reshape(-1)
        
        indices = similarities.argsort()
        top_indices = [idx for idx in indices][::-1][:top]

        verses = self.qe.iloc[top_indices,:]
        response = {
            'reference': verses.loc[:,'Name'].tolist(),
            'verse': verses.loc[:,'Verse'].tolist()
        }
        return response

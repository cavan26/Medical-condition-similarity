'''
Created on 5 Dec 2018

@author: camille.vanassel
'''

import requests
import numpy as np
from fasttext import FastVector
import subprocess
from Extract_cond_THIN import load_dict
import urllib
import json


def get_token():
    CMD = 'echo $(source /Users/camille.vanassel/scripts/token.sh; )' 
    p = subprocess.Popen(CMD, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    return p.stdout.readlines()[0].strip()


def unique(list):
    unique_list = []
    for x in list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between vec_a and vec_b"""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def sentence_encoder(sentences):
    Token = get_token()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': "Bearer {}".format(Token)
    }
    body = {
        "signature_name": "serving_default",
        "instances": sentences
    }
    url = "https://services-uk.dev.babylontech.co.uk/tensorflow-serving/v1/models/universal-sentence-encoder-large/versions/3:predict"
    response = requests.post(url, json=body, headers=headers).json()
    embeddings = np.array(response['predictions'])
    return embeddings


def IRI_encoder(IRIs):
    # loading word vectors
    vec = FastVector(vector_file='thin_2018-11-23_d100_e5.bin.vec')
    wordvector = []
    for IRI in IRIs:
        if IRI in vec.word2id.keys():
            idx = vec.word2id[IRI]
            wordvector.append(vec.embed[idx])
        else:
            print(IRI)
            wordvector.append([0]*100)
    wordvector = np.array(wordvector)
    return wordvector


def parse_json(json_data):
    Concept = []
    IRI = []
    Confidence = []
    for item in json_data:
        IRI.append(item.get("iri"))
        Concept.append(item.get("text"))
        Confidence.append(item.get("confidence"))
    return (IRI, Concept, Confidence)
    

def concept_embedding(condition):
    Token = get_token()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': "Bearer {}".format(Token)
    }
    body = {
        "signature_name": "serving_default"
    }
    url = "https://services-uk.dev.babylontech.co.uk/concept-embeddings/concept-embedding/bbl_alt?text=" + urllib.quote(condition) + "&n=20"
    response = requests.get(url, body, headers=headers)
    json_data = json.loads(response.text)
    return parse_json(json_data)


def get_parent(IRI):
    Token = get_token()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': "Bearer {}".format(Token)
    }
    body = {
        "signature_name": "serving_default"
    }
    
    url = "https://services-global1.dev.babylontech.co.uk/clinical-knowledge/v2/paths/parents/immediate?iri=" + urllib.quote(IRI)
    response = requests.get(url, body, headers=headers)
    json_data = json.loads(response.text)
    return json_data[0]['iri']

    
def extract_closest_vectors(VEC, embedding):
    # Extract the 5 closest conditions present in the VEC model
    # and with similarity coefficient >0.75
    new_vec = []
    i = 0
    while i < len(embedding[0]) and len(new_vec) < 6:
        print(embedding[1][i])
        IRI = embedding[0][i]
        if IRI in VEC.word2id.keys() and embedding[2][i] > 0.7:
            print(embedding[1][i])
            idx = VEC.word2id[IRI]
            new_vec.append(VEC.embed[idx])
        i +=1
    return np.array(new_vec)


def close_concept(vec, condition):
    embedding = concept_embedding(condition)
    Close_IRIs = extract_closest_vectors(vec, embedding)
    if len(Close_IRIs)>1:
        return np.mean(Close_IRIs, axis=0)
    else:
        return np.zeros([100])


def get_most_similar_concepts(word_vector, sentence_vectors, thresh):
    N = len(sentence_vectors)
    index = []
    sim = []
    Similarity = [0] * N
    for i in range(0, N):
        Similarity[i] = cosine_similarity(word_vector, sentence_vectors[i])
    Max_sim = max(Similarity)
    while Max_sim > thresh:
        idx = Similarity.index(max(Similarity))
        index.append(idx)
        sim.append(Max_sim)
        Similarity[idx] = 0
        Max_sim = max(Similarity)
    return (index, sim)


class WordVector:

    def __init__(self, iri='', condition='', vec=''):
        self.IRI = unique(load_dict(iri))
        self.condition = unique(load_dict(condition))
        self.word_vector = []
        self.encode(vec)

    def encode(self, vec):
        for i in range(0, len(self.IRI)):
            IRI = self.IRI[i]
            Condition = self.condition[i]
            if IRI in vec.word2id.keys():
                self.add_vector(IRI, vec)
            else:
                iri_parent = get_parent(IRI)
                if iri_parent in vec.word2id.keys():
                    self.add_vector(iri_parent, vec)
                else:
                    self.add_vector(close_concept(vec, Condition))

    def add_vector(self, iri, vec= None):
        if vec is None:
            self.word_vector.append(iri)
        else:
            idx = vec.word2id[iri]
            self.word_vector.append(vec.embed[idx])

    def patient_similar_concept(self, word, vec):
        word_vector = close_concept(vec, word)
        if np.count_nonzero(word_vector) == 0:
            print('Unknown word')
        else:
            (Ind, Sim) = get_most_similar_concepts(word_vector, self.word_vector, 0.7)
            if len(Ind) > 0:
                print('The conditions reported in that category are: ')
                for i in range(0, len(Ind)):
                    print(str(self.condition[Ind[i]]) + ' : ' + str(Sim[i]))
                print(' - - - - - - ')
            else:
                print("No condition reported in that category")
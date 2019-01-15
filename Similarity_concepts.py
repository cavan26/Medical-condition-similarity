'''
Created on 4 Dec 2018

@author: camille.vanassel
'''
from __future__ import print_function
import requests
import numpy as np
from Extract_cond_THIN import load_obj
from sklearn.cluster import KMeans
from Word_embedding import IRI_encoder, sentence_encoder


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between vec_a and vec_b"""
    return np.dot(vec_a, vec_b) / \
        (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        
        
def similarity_matrix(embeddings):
    nb_sentences = len(embeddings)
    Similarity = np.zeros([nb_sentences, nb_sentences])
    for i in range(0,nb_sentences):
        for j in range(i+1,nb_sentences):
            Similarity[i,j] = cosine_similarity(embeddings[i],embeddings[j])
            Similarity[j,i] = Similarity[i,j]
    return Similarity


def kmeans_clustering(num_clusters, embeddings):
    km = KMeans(n_clusters=num_clusters)
    km.fit(embeddings)
    clusters = km.labels_.tolist()
    return km


def sumRow(matrix, i):
    return np.sum(matrix[i,:])
 
 
def determineRow(matrix):
    maxNumOfOnes = -1
    row = -1
    for i in range(len(matrix)):
        if maxNumOfOnes < sumRow(matrix, i):
            maxNumOfOnes = sumRow(matrix, i)
            row = i
    return row

 
def addIntoGroup(matrix, ind):
    change = True
    indexes = []
    for col in range(len(matrix)):
        if matrix[ind, col] == 1:
            indexes.append(col)
    while change == True:
        change = False
        numIndexes = len(indexes)
        for i in indexes:
            for col in range(len(matrix)):
                if matrix[i, col] == 1:
                    if col not in indexes:
                        indexes.append(col)
        numIndexes2 = len(indexes)
        if numIndexes != numIndexes2:
            change = True
    return indexes
 
 
def deleteChosenRowsAndCols(matrix, indexes):
    for i in indexes:
        matrix[i,:] = 0
        matrix[:,i] = 0
    return matrix


def categorizeIntoClusters(matrix):
    groups = []
    while np.sum(matrix) > 0:
        group = []
        row = determineRow(matrix)
        indexes = addIntoGroup(matrix, row)
        groups.append(indexes)
        matrix = deleteChosenRowsAndCols(matrix, indexes)
    return groups


def buildSimilarityMatrix(similarity):
    numOfSamples = len(similarity)
    matrix = np.zeros(shape=(numOfSamples, numOfSamples))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i==j:
                matrix[i,j] = 1
            if similarity[i,j] > 0.75:
                matrix[i,j] = 1
    return matrix


def print_in_groups(terms, groups):
    nb = 1
    for group in groups:
        print('Group ' + str(nb))
        for i in group:
            print(terms[i],end=',')
        print(' ')
        nb+=1
        
        
def get_cond_from_IRIs(IRIs):
    Sentences = []
    for iri in IRIs:
        response = requests.get('https://services-global1.dev.babylontech.co.uk/clinical-knowledge/v2/labels?iri=%s&onlyPrefLabels=true' % iri)
        if response.status_code == 200:
            cond = response.json()[0]['text']
        else:
            cond = "UNKNOWN"
        Sentences.append(cond)
    return Sentences


def main(method):
    if method =='IRIs':
        IRIs = load_obj('CondIRL.text')
        embeddings = IRI_encoder(IRIs)   
        Sim = similarity_matrix(embeddings)
        matrix = buildSimilarityMatrix(Sim)
        groups = categorizeIntoClusters(matrix)
        sentences = get_cond_from_IRIs(IRIs)
        print_in_groups(sentences, groups)
    elif method == 'SentenceEmbedding':
        sentences = load_obj('Cond.text')
        embeddings = sentence_encoder(sentences)   
        Sim = similarity_matrix(embeddings)
        matrix = buildSimilarityMatrix(Sim)
        groups = categorizeIntoClusters(matrix)
        print_in_groups(sentences, groups)
    else:
        print('Not an acceptable method')


if __name__ == "__main__":
    # Method = IRIs or SentenceEmbedding
    method = 'SentenceEmbedding'
    main(method)          
            
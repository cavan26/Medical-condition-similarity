
'''
Created on 5 Dec 2018

@author: camille.vanassel
'''

import requests
import subprocess
import numpy as np
import urllib
from fasttext import FastVector


def get_cond_from_iris(iri):
    header = {
        'Content-Type': 'application/json',
        'Content-Language': 'EN'}

    url = 'https://services-global1.dev.babylontech.co.uk/clinical-knowledge/v2/labels?iri=%s&onlyPrefLabels=true' \
          % urllib.quote(iri).replace('/', '%2F')

    response = requests.get(url, headers=header)
    if response.status_code == 200:
        cond = response.json()[0]['text']
    else:
        cond = ""
    return cond


def get_token():
    cmd = 'echo $(source /Users/camille.vanassel/scripts/token.sh; )'
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    return p.stdout.readlines()[0].strip()


def get_vector_from_cond(condition):
    # Using the word2vec model from google
    token = get_token()

    headers = {
        'Content-Type': 'application/json',
        'Authorization': "Bearer {}".format(token)
    }
    body = {
        "signature_name": "serving_default",
        "instances": condition
    }

    url = "https://services-uk.dev.babylontech.co.uk/tensorflow-serving/v1/models/universal-sentence-encoder-large/versions/3:predict"

    response = requests.post(url, json=body, headers=headers).json()
    embeddings = np.array(response['predictions'])
    return embeddings


class MedicalCond2Vector:

    def __init__(self, iri):
        """
        Constructor for MedicalCond2Vector
        :param iri: iri can be the name of the file where the model
        is stored, or a list of the iris to create a new model
        """
        if type(iri) is str:
            self.iri = []
            self.cond = []
            self.load_model(iri)
        if type(iri) is list:
            self.iri = iri
            self.cond = self.convert_iri2cond()
            self.vector = self.convert_cond2vector()
            self.n_words = len(self.vector)
            self.n_dim = len(self.vector[0])

    def convert_iri2cond(self):
        """
        Take as an input the IRIs a list of strings
        :return: The associated medical conditions as a list of strings
        """
        condition = []
        if len(self.iri) == 0:
            print('The list of iri is empty')
            return None
        elif len(self.iri) == 1:
            condition = get_cond_from_iris(self.iri)
        else:
            for item in self.iri:
                condition.append(get_cond_from_iris(item))
        print('IRIs successfully converted to condition names')
        return condition

    def convert_cond2vector(self):
        """
        Take as input the condition as a list of strings
        :return: The associated vectors in the word2vec space as an array
        """
        if len(self.cond) == 0:
            print('The list of condition is empty')
            return None
        else:
            vector = get_vector_from_cond(self.cond)
            print('Condition successfully converted to vectors')
            return vector

    def save_model(self, vector_file):
        """
        Save the MedicalCond2Vec object
        :param vector_file: name of the file
        """
        print('writing word vectors in %s' % vector_file)
        f = open(vector_file,"w+")
        f.write(str(len(self.iri)) + ' ' + str(len(self.vector[0])))
        for i in range(len(self.iri)):
            f.write(self.iri[i] + ' ' + self.cond[i] + ' ' + " ".join(str(item) for item in self.vector[i]))
        f.close()
        print('The model is saved to %s' % vector_file)

    def load_model(self, vector_file):
        """
        Load a previously savec model
        :param vector_file: name of the file where the model is stored
        """
        with open(vector_file, 'r') as f:
            (self.n_words, self.n_dim) = (int(x) for x in f.readline().rstrip('\n').split(' '))
            self.vector = np.zeros((self.n_words, self.n_dim))
            for i, line in enumerate(f):
                elems = line.rstrip('\n').split(' ')
                self.iri.append(elems[0])
                self.cond.append(elems[1])
                self.vector[i] = elems[2:self.ndims+2]


def main():
    vec = FastVector('thin_2018-11-23_d100_e5.bin.vec')
    iris = vec.word2id.keys()
    cond2vec = MedicalCond2Vector(iris)
    cond2vec.save_model('thin_to_word2vec.bin.vec')
    print(cond2vec.vector)


if __name__ == "__main__":
    main()

#https://www.kdnuggets.com/2019/10/easily-deploy-machine-learning-models-using-flask.html

import numpy as np
import pickle, dill
import string
import re
from flask import Flask
from flask import request
from flask import render_template
from werkzeug import secure_filename

app = Flask(__name__)
rs = dill.load(open('read_sinopsis.pkl', 'rb'))
s2w = dill.load(open('sent_to_words.pkl', 'rb'))
stop = pickle.load(open('stopwords.pkl', 'rb'))
rsw = dill.load(open('remove_stopwords.pkl', 'rb'))
lm = dill.load(open('lemmatization.pkl', 'rb'))
vec = pickle.load(open('vectorizer.pkl', 'rb'))
mod = pickle.load(open('lda_model.pkl', 'rb'))
dtgw = dill.load(open('document_topic_genre_words.pkl', 'rb'))
met = dill.load(open('metrics.pkl', 'rb'))
plot = dill.load(open('lda_plot.pkl', 'rb'))


@app.route('/')
def home():    
    return render_template("index.html")  

@app.route('/getfile', methods=['GET','POST'])
def predict():
    
    file = request.files['myfile']
    filename = secure_filename(file.filename) 
    data = file.read().decode('utf-8')
    
    data = rs(data)
    data_words = list(s2w(data))
    data_words_nonstop = rsw(data_words, stop)
    data_lemmatized = lm(data_words_nonstop, allowed_postags=['NOUN', 'ADJ', 'ADV', 'VERB'])
    data_vectorized = vec.transform(data_lemmatized)
    lda_output = mod.transform(data_vectorized)
    df_relevance, genre, df_topic_keywords = dtgw(mod, lda_output, vec)
    loglikelihood, perplexity = met(mod, data_vectorized)

    return render_template('results.html', \
                           tables=[df_relevance.to_html(classes='relevance'), \
                                   df_topic_keywords.to_html(classes='words'), \
                                   genre.capitalize(), loglikelihood, perplexity], \
                           titles=['na', 'Relevancia', 'Palabras en el Tópico', 'Género: ', 'LogLikelihood: ', 'Perplexity: '])



if __name__ == "__main__":
    app.run(debug=True)
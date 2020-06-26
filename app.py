import pandas as pd
import re
import string
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request
from gevent.pywsgi import WSGIServer
from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()


new_model = pickle.load(open('svm_model.pkl','rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pred', methods=['POST'])
def pred():
    sent = request.form['message']
#     sent = sent.lower()
#     pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#     sent = pattern.sub('', sent)
#     sent = re.sub(r'[,.\"!@#$%^&*(){}?/;`~:<>+=-]', '', sent)
#     tokens = nltk.word_tokenize(sent)
#     table = str.maketrans('', '', string.punctuation)
#     stripped = [w.translate(table) for w in tokens]
#     words = [word for word in stripped if word.isalpha()]
#     stop_words = set(nltk.corpus.stopwords.words('english'))
#     stop_words.discard('not')
#     words = [Lemmatizer.lemmatize(word) for word in words if word not in stop_words]
#     words = ' '.join(words)
    words = [sent]
    x = cv.transform(words).toarray()
    p = new_model.predict(x)
    if p >= 0.5:
        val = 'Postive'
    else:
        val = 'Negative'    

    return render_template('two.html', pred_text = "That's a {b} review".format(b=val))

if __name__ == '__main__':
    app.run(debug=True)

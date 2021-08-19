from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, create_engine
from prediction import QA
import pandas as pd
import random
import nltk
import string
from nltk.corpus import stopwords 
import gensim
from gensim import corpora, models, similarities
import os


nltk.download("wordnet")
nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///previously_asked.sqlite3'
engine = create_engine('sqlite:///previously_asked.sqlite3',echo=True)
db = SQLAlchemy(app)
run_with_ngrok(app)
model = QA("/content/gdrive/My Drive/finalYearProject/model")
max_sequence_len = 512
average_sequence_len = 15
threshold_sequence_len = max_sequence_len-average_sequence_len

class RAQ(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    question = db.Column(db.String(100))
    answer = db.Column(db.String(50))
    context = db.Column(db.String(600))
    def __init__(self,question, answer, context):
        self.question=question
        self.answer=answer
        self.context = context

#Create Stop Word
newstopwords = set(stopwords.words('english'))
#define Wordnet Lemmatizer 
WNlemma = nltk.WordNetLemmatizer()
totalDocs =26


def prep_model():
    docs = []
    orig_docs = []
    file_text_list = []
    file_text = ""
    doc_names = os.listdir(os.getcwd()+'/documents/')

    for i in doc_names:
        file = open(os.getcwd()+'/documents/'+i,"r")
        for line in file:
          file_text = file_text+line
          if(len(file_text))>threshold_sequence_len:
            file_text_list.append(file_text)
            file_text = ""
    
    docs = [pre_process(doc) for doc in file_text_list]
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(a) for a in docs]
    tfidf = models.TfidfModel(corpus)
        
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=800) # Threshold A
    corpus_lsi = lsi[corpus_tfidf]
    index = similarities.MatrixSimilarity(corpus_lsi)
    return dictionary,tfidf,lsi,index,docs,file_text_list

def find_doc(test_set_sentence):        
    dictionary,tfidf,lsi,index,docs,orig_docs= prep_model()
    tokens = pre_process(test_set_sentence)
    texts = " ".join(tokens)    
    
    vec_bow = dictionary.doc2bow(texts.lower().split())
    vec_tfidf = tfidf[vec_bow]
    vec_lsi = lsi[vec_tfidf]
    #print(vec_lsi)
    #If not in the topic trained.
    if not (vec_lsi):
        
        not_understood = "Cannot understand question ?"
        return not_understood, 999
    
    else: 
        # sort similarity
        sims = index[vec_lsi]
        #print("sims ",sims)
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        index_s =[]
        score_s = []
        for i in range(len(sims)):
            x = sims[i][1] 
            index_s.append(str(sims[i][0]))
            score_s.append(str(sims[i][1]))
            reply_indexes = pd.DataFrame({'index': index_s,'score': score_s})
            #print(reply_indexes)
            r_index = int(reply_indexes['index'].loc[0])
            r_score = float(reply_indexes['score'].loc[0])
            reply = (docs[r_index])
 
            return  orig_docs[r_index]

def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[WNlemma.lemmatize(t) for t in tokens]
    tokens= [ t for t in tokens if t not in string.punctuation ]
    tokens=[word for word in tokens if word.lower() not in newstopwords]
    # bigr = nltk.bigrams(tokens[:10])
    # trigr = nltk.trigrams(tokens[:10])
    return(tokens)



@app.route("/",methods = ['GET', 'POST'])
def home():
    questions = db.session.query(RAQ.question.distinct()).all()
    ques = []
    for i in range(len(questions)):
      ques.append(questions[i][0])
    return render_template("form.html",questions = ques)





@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    
    #doc = 'Serious financial damage has been caused by security breaches, but because there is no standard model for estimating the cost of an incident, the only data available is that which is made public by the organizations involved. "Several computer security consulting firms produce estimates of total worldwide losses attributable to virus and worm attacks and to hostile digital acts in general. The 2003 loss estimates by these firms range from $13 billion (worms and viruses only) to $226 billion (for all forms of covert attacks). The reliability of these estimates is often challenged; the underlying methodology is basically anecdotal'
    q = request.form['question']
    if len(q)>100:
      return "Enter a Shorter Question"
    ques_q = RAQ.query.filter_by(question=q).all()
    print("question in predict",ques_q)
    if len(ques_q) != 0:
      ans_q = db.session.query(RAQ.answer, RAQ.context).filter_by(question=q).all()
      return render_template("answer.html",answer= ans_q[0][0],context=ans_q[0][1])

    doc = find_doc(q)

    try:
        out = model.predict(doc,q)
        answer  = out['answer']
        context = " ".join(out['document']) 
        raq = RAQ(q,answer,context)
        db.session.add(raq)
        db.session.commit()
        return render_template("answer.html",answer=answer,context=context)
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})
    

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    db.create_all()
    app.run()


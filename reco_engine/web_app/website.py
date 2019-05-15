from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from reco_engine.Lib_User import User, User_Helper
from reco_engine.Lib_Content import Content, Content_Helper
import pandas as pd
import json
import sys
from reco_engine.recommenders.content_based import Cosine_Recommender
from reco_engine.recommenders.coll_filt_based import CosSim_Recommender as W_CosSim_Recommender

#db_connect = create_engine('sqlite:///chinook.db')
app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/users')
def view_users():
    users = User_Helper.get_user_list()
    users = json.loads(users.to_json(orient='records'))
    return render_template("users.html", users=users)

@app.route('/user/<int:id>')
def view_user(id):
    Usr = User(id)
    user = json.loads(Usr.user.to_json(orient='records'))[0]
    ratings = json.loads(Usr.get_rating_history().reset_index().to_json(orient='records'))
    CS_Rec = W_CosSim_Recommender("user")
    pre_ratings = json.loads(CS_Rec.make_recommendation_by_user(id, 10).to_json(orient='records'))
    sim_users = json.loads(Usr.get_similar_users(10).to_json(orient='records'))
    return render_template("user.html", user=user, ratings=ratings, pre_ratings=pre_ratings, sim_users=sim_users)

@app.route('/contents')
def view_contents():
    lst_cnt = Content_Helper.get_content_list().reset_index()
    print(lst_cnt)
    contents = json.loads(lst_cnt.to_json(orient='records'))
    print(contents)
    return render_template("contents.html", contents=contents)

@app.route('/content/<int:id>')
def view_content(id):
    Cnt = Content(id)
    content = json.loads(Cnt.content.to_json(orient='records'))[0]
    #print(content)
    CR = Cosine_Recommender(["overview"])
    descsimcontents = CR.make_recommendation(id, 10).reset_index()
    descsimcontents = json.loads(descsimcontents.to_json(orient='records'))
    CR = Cosine_Recommender(["leads","genres"])
    genressimcontents = CR.make_recommendation(id, 10).reset_index()
    genressimcontents = json.loads(genressimcontents.to_json(orient='records'))
    CR = Cosine_Recommender(["cast"])
    castsimcontents = CR.make_recommendation(id, 10).reset_index()
    castsimcontents = json.loads(castsimcontents.to_json(orient='records'))
    CR = Cosine_Recommender(["keywords"])
    keywordsimcontents = CR.make_recommendation(id,  10).reset_index()
    keywordsimcontents = json.loads(keywordsimcontents.to_json(orient='records'))

    WCSR = W_CosSim_Recommender("movie")
    watchhistsimcontents = WCSR.make_recommendation_by_movie(id,10).reset_index()
    watchhistsimcontents = json.loads(watchhistsimcontents.to_json(orient='records'))
    return render_template("content.html", content=content, descsimcontents=descsimcontents, genressimcontents=genressimcontents,
                           castsimcontents=castsimcontents, keywordsimcontents=keywordsimcontents, watchhistsimcontents=watchhistsimcontents)

if __name__ == '__main__':
    app.run(port='5003')
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
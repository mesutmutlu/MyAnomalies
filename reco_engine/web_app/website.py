from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from reco_engine.Users import User
from reco_engine.Contents import Content
import pandas as pd
import json
import sys
from reco_engine.recommenders.content_based import CosSim_Recommender

#db_connect = create_engine('sqlite:///chinook.db')
app = Flask(__name__)
api = Api(app)

@app.route('/users')
def view_users():
    Usr = User()
    Usr.load_user_list()
    users = json.loads(Usr.users.to_json(orient='records'))
    print(users)
    return render_template("users.html", users=users)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/contents')
def view_contents():
    Cnt = Content()
    Cnt.load_content_list()
    contents = json.loads(Cnt.movielist.to_json(orient='records'))
    print(contents)
    return render_template("contents.html", contents=contents)

@app.route('/content/<int:id>')
def get_content(id):
    Cnt = Content()
    c = Cnt.get_content_by_id(id)
    print(c)
    content = json.loads(c.to_json(orient='records'))[0]
    #print(content)
    CSR = CosSim_Recommender()
    descsimcontents = CSR.make_recommendation(content["title"], "overview", 10).reset_index()
    descsimcontents = json.loads(descsimcontents.to_json(orient='records'))
    genressimcontents = CSR.make_recommendation(content["title"], ["leads","genres"], 10).reset_index()
    genressimcontents = json.loads(genressimcontents.to_json(orient='records'))
    castsimcontents = CSR.make_recommendation(content["title"], "cast", 10).reset_index()
    castsimcontents = json.loads(castsimcontents.to_json(orient='records'))
    keywordsimcontents = CSR.make_recommendation(content["title"], "keywords", 10).reset_index()
    keywordsimcontents = json.loads(keywordsimcontents.to_json(orient='records'))

    return render_template("content.html", content=content, descsimcontents=descsimcontents, genressimcontents=genressimcontents,
                           castsimcontents=castsimcontents, keywordsimcontents=keywordsimcontents)

if __name__ == '__main__':
    app.run(port='5003')
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
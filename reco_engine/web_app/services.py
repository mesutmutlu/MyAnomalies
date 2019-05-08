from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from reco_engine.Users import User
from reco_engine.Contents import Content
import pandas as pd
import json
import sys
#db_connect = create_engine('sqlite:///chinook.db')
app = Flask(__name__)
api = Api(app)


@app.route('/getuserlist')
def getuserlist():
    Usr = User()
    Usr.load_user_list()
    #print(Usr.users.to_json(orient='records'))
    return Usr.users.to_json(orient='records')

@app.route('/getcontentlist')
def get():
    Cnt = Content()
    Cnt.load_content_list()
    return Cnt.movielist.to_json(orient="records")

def loadratings():
    ratings = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_ratings.csv")
    #print(ratings)
    Cnt = Content()
    Cnt.load_content_list()
    Usr = User()
    Usr.load_user_list()
    #print(Usr.users)
    ratings = ratings.merge(Usr.users, left_on ="userId", right_on= "userid", how="inner")
    ratings = ratings.merge(Cnt.movielist, on=["id"], how="inner")[["userId", "username", "id", "title", "rating"]]
    return ratings

@app.route('/getratings')
def getratings():
    return loadratings().to_json(orient="records")

@app.route('/getuserratings/<username>/')
def getuserratings(username):
    ratings = loadratings()
    return ratings[ratings["username"]==username][["id", "title", "rating"]].to_json(orient="records")

@app.route('/gettitleratings/<title>/')
def gettitleratings(title):
    ratings = loadratings()
    return ratings[ratings["title"]==title][["userId", "username", "rating"]].to_json(orient="records")

@app.route('/users')
def view_users():
    Usr = User()
    Usr.load_user_list()
    render_template()

if __name__ == '__main__':
    app.run(port='5002')
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(loadratings())
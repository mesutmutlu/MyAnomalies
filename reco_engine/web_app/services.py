from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from reco_engine.Users import User
from reco_engine.Contents import Content
import pandas as pd

#db_connect = create_engine('sqlite:///chinook.db')
app = Flask(__name__)
api = Api(app)


class Users(Resource):
    def get(self):
        Usr = User()
        Usr.load_user_list()
        return jsonify(Usr.users)

class Contents(Resource):
    def get(self):
        Cnt = Content()
        Cnt.load_content_list()
        return jsonify(Cnt.movielist)


class Ratings(Resource):

    def __loadratings(self):
        ratings = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_ratings")
        Cnt = Content()
        Cnt.load_content_list()
        Usr = User()
        Usr.load_user_list()
        ratings = ratings.merge(Usr.users, on=["userId", "userid"], how="inner")
        ratings = ratings.merge(Cnt.movielist, on=["id"], how="inner")
        return ratings

    def getall(self):
        return jsonify(self.__loadratings())

    def get_by_username(self, username):
        ratings = self.__loadratings()
        return jsonify(ratings[ratings["username"]==username])

    def get_by_title(self, title):
        ratings = self.__loadratings()
        return jsonify(ratings[ratings["title"]==title])



api.add_resource(Users, '/users')  # Route_1
api.add_resource(Contents, '/contents')  # Route_2
api.add_resource(Ratings, '/ratings/<employee_id>')  # Route_3

if __name__ == '__main__':
    app.run(port='5002')
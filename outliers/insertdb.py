import psycopg2
import time
import csv
import os
import datetime

def insert_from_file(type):
    conn = psycopg2.connect("host=localhost dbname=anomaly user=postgres")
    cur = conn.cursor()


    for filename in os.listdir("C:/D/kibanalogs/feature/"):
        if os.path.isfile("C:/D/kibanalogs/feature/"+filename):

            with open("C:/D/kibanalogs/feature/"+filename, "r") as f:
                reader = csv.reader(f)
                next(reader)
                if filename[-3:] == "err" and 'err' in type:
                    print(datetime.datetime.now(), filename)
                    for row in reader:
                        cur.execute(
                            "INSERT INTO err VALUES (to_timestamp(%s, 'YYYYMMDDhh24mi')::timestamp, %s, %s, %s, %s, %s, %s)",
                            row
                        )
                if filename[-3:] == "hit" and "hit" in type:
                    print(datetime.datetime.now(), filename, datetime.datetime)
                    for row in reader:
                        cur.execute(
                            "INSERT INTO hit VALUES (to_timestamp(%s, 'YYYYMMDDhh24mi')::timestamp, %s, %s, %s, %s, %s)",
                            row
                        )
            conn.commit()

if __name__ == "__main__":
    pass
    insert_from_file(['hit'])
#print(datetime.datetime.now())
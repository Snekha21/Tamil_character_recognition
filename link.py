from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import json 
# from json import jsonify
from werkzeug.utils import secure_filename
import pathlib
import pickle
from PIL import Image
import cv2 
import numpy as np
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask'
 
# mysql = MySQL(app)
 
# #Creating a connection cursor
# cursor = mysql.connection.cursor()
 
# #Executing SQL Statements
# cursor.execute(''' CREATE TABLE patient_data(name,age,gender,profession,mailid,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12) ''')
# # cursor.execute(''' INSERT INTO table_name VALUES(v1,v2...) ''')
# # cursor.execute(''' DELETE FROM table_name WHERE condition ''')
 
# #Saving the Actions performed on the DB
# mysql.connection.commit()
 
# #Closing the cursor
# cursor.close()

subject = "Tamil pdf"
body = "Hey there! Here is your pdf generated from Blue's Meter"
sender_email = "snekhasuresh2777@gmail.com"
receiver_email = "bluesmeter@gmail.com"
password = "ffomorgrzxnezxdj"
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject
message["Bcc"] = receiver_email
message.attach(MIMEText(body, "plain"))
@app.route('/')
# def home():
#     return render_template('home.html')

@app.route('/home', methods=['GET', 'POST'])
def index_func():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('home'))
    # show the form, it wasn't submitted
    return render_template('home.html')

@app.route('/about', methods=['GET', 'POST'])
def about_func():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('About'))
    # show the form, it wasn't submitted
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login_func():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Login'))
    # show the form, it wasn't submitted
    return render_template('login.html')


@app.route('/main', methods=['GET', 'POST'])
def main_func():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Main'))
    # show the form, it wasn't submitted
    return render_template('Main.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact_func():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Contact'))
    # show the form, it wasn't submitted
    return render_template('contact.html')

# # @app.route('/result', methods=['GET', 'POST'])
# # def result_func():
# #     if request.method == 'POST':
# #         # do stuff when the form is submitted
# #         # redirect to end the POST handling
# #         # the redirect can be to the same route or somewhere else
# #         return redirect(url_for('result'))
# #     # show the form, it wasn't submitted
# #     return render_template('result.html')

# @app.route('/input', methods=['GET', 'POST'])
# def test_func():
#     if request.method == 'POST':
#         # data = request.form.get("q1")
#         # for key, value in request.form.items():
#         #     print("key: {0}, value: {1}".format(key, value))

#         # data=console.log(JSON.stringify(input))
#         # data1 =json.loads(request.json)
#         # data = request.form.get("q1")
#         # data = request.json
#         data= request.form
#         # data = json.dump(request.get_json(force=True))
#         # with open('file.json', 'w') as f:
#         #   json.dump(request.get_json(force=True), f)
#         # do stuff when the form is submitted
#         # redirect to end the POST handling
#         # the redirect can be to the same route or somewhere else
#         print(data)
#         # print(data1)
#         convertJsonToCsv(data)
#         return redirect(url_for('result'))
#         # return render_template('result.html')
#     return render_template('index.html')
#     # show the form, it wasn't submitted
#     # return render_template('index.html')

@app.route("/image")
def image():
#   if request.method == "POST":
#     image = request.files.get('img', '')
#     image = request.form["img"]

#     print("img ",image)
#     return "<h1> yo! </h1>"
#   else:
    return render_template('login.html') 

@app.route('/upload', methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['File']
        f.save(secure_filename("image.pdf"))
        filename="tamil_pdff.txt"
        
        # if mri()==1:
        #     html_data= "You are depressed"
        #     result = "You are showing symptoms of depression"
        #     cmt= "You are adviced to contact your doctor and get treated asap"
        #     filename = "depressed.pdf"
        # else:
        #     html_data = "You are not depressed"
        #     result = "Congrats! You are not showing symptoms of depression"
        #     cmt = "Stay happy and healthy"
        #     filename="healthy.pdf"
        with open(filename, "rb") as attachment:
             part = MIMEBase("application", "octet-stream")
             part.set_payload(attachment.read())
             encoders.encode_base64(part)
             part.add_header(
            "Content-Disposition",
             f"attachment; filename= {filename}",
             )
             message.attach(part)
             text = message.as_string()
             context = ssl.create_default_context()
             with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
            # msg = Message('Hello', sender = 'snekhasuresh2777@gmail.com', recipients = ['bluesmeter@gmail.com'])
            # msg.body = "Hello Flask message sent from Flask-Mail"
            # mail.send(msg)
                server.sendmail(sender_email, receiver_email, text)
        return render_template("about.html")
# @app.route("/test", methods = ['POST','GET'])
# def login():
#     if request.method == 'POST':
#         return redirect(url_for('test'))
    
#     return render_template('index.html')


# def convertJsonToCsv(jsonData):
#     dfs = []
#     dfs.append(pd.DataFrame([jsonData]))
#     df = pd.concat(dfs, ignore_index=True, sort=False)
#     # df.drop(['userName'], 
#     #     axis = 1, inplace = True)
#     df.to_csv('input.csv',index=False)

# @app.route("/result")
# def result():
#     data_dir = pathlib.Path("input.csv")
#     df=pd.read_csv(data_dir)
#     data1= df.transpose()
#     # try x=data1[0].value_counts()[0]
#     x=data1[0].value_counts()[0]
#     y=data1[0].value_counts()[1]
#     z=data1[0].value_counts()[2]
#     n= (x/24)*100
#     s= (y/24)*100
#     a= (z/24)*100
#     d=s+a
#     # data1['Average'] = data1.mean(axis=1)
#     # data1.drop(['Monday','Tuesday','Wednesday','Thursday','Friday'],axis = 1, inplace = True)
#     # fmodel = pickle.load(open('fin_model.pkl','rb'))
#     # model = pickle.load(open('fin_model.pkl','rb'))
#     # if(model.predict(data1) == 0):
#     #     return  {"output":"You are depressed"}
#     # else:
#     #     return {"output":"You are not depressed"}
#     # data1 = data1.astype(float)
#     model = pickle.load(open('lr_model.pkl','rb'))
#     if(model.predict(df) == 1):
#         if (d<5):
#             html_data= "No depression"
#             result = "Congrats! You are living a healthy life!"
#             cmt= "Stay happy and healthy"
#         elif (d<25):
#             html_data= "Beginner Stage"
#             result = "You had just started showing the symptoms of depression"
#             cmt= "Just wake up! It's not too late..."
#         elif (d<65):
#             html_data= "Moderate Stage"
#             result = "You are showing mild symptoms of depression and adviced to consult a psychiatrist asap"
#             cmt= "There is hope, even when your brain tells you there isn't.."
#         elif (d>65):
#             html_data= "Severe Stage"
#             result = "You are adviced to visit a psychiatrist asap and get necessary treatment, contact help-desk to reach out to doctors"
#             cmt= "To confirm the severity of depression, kindly take a DTI-MRI scan"
#         filename = "depressed.pdf"
#     elif(model.predict(df) == 0):
#         html_data = "You are not depressed"
#         result = "Congrats! You are living a healthy life!"
#         cmt = "Stay happy and healthy"
#         filename="healthy.pdf"

#     with open(filename, "rb") as attachment:
#              part = MIMEBase("application", "octet-stream")
#              part.set_payload(attachment.read())
#              encoders.encode_base64(part)
#              part.add_header(
#             "Content-Disposition",
#              f"attachment; filename= {filename}",
#              )
#              message.attach(part)
#              text = message.as_string()
#              context = ssl.create_default_context()
#              with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
#                 server.login(sender_email, password)
#             # msg = Message('Hello', sender = 'snekhasuresh2777@gmail.com', recipients = ['bluesmeter@gmail.com'])
#             # msg.body = "Hello Flask message sent from Flask-Mail"
#             # mail.send(msg)
#                 server.sendmail(sender_email, receiver_email, text)
#     print(model.predict(df))
#     return render_template("result.html", html_data = html_data , result=result, cmt=cmt)

# @app.route("/img")
# def mri():
#         # img = Image.open("image.jpg")
#         # img = img.convert("L")
#         img_array=cv2.imread("image.jpg")
#         image = cv2.resize(img_array, (100,100))
#         X= np.array(image).reshape(1,-1)
#         # svc.predict(X)
# #       test=[]
# #       img_array=cv2.imread("image.jpg")
# #       CATEGORIES = ['healthy','depressed']
# #       class_num=CATEGORIES.index(category)
# #       new_array=cv2.resize(img_array,(100,100))
# #       test.append([new_array,class_num])
# #     # data1['Average'] = data1.mean(axis=1)
# #     # data1.drop(['Monday','Tuesday','Wednesday','Thursday','Friday'],axis = 1, inplace = True)
# #     # fmodel = pickle.load(open('fin_model.pkl','rb'))
# #     # model = pickle.load(open('fin_model.pkl','rb'))
# #     # if(model.predict(data1) == 0):
# #     #     return  {"output":"You are depressed"}
# #     # else:
# #     #     return {"output":"You are not depressed"}
# #     # data1 = data1.astype(float)
#         model = pickle.load(open('image.pkl','rb'))
#         # if(model.predict(X) == 1):
#         #     html_data= "You are depressed"
#         #     result = "You are showing symptoms of depression"
#         #     cmt= "You are adviced to contact your doctor and get treated asap"
#         # elif(model.predict(X) == 0):
#         #     html_data = "You are not depressed"
#         #     result = "Congrats! You are not showing symptoms of depression"
#         #     cmt = "Stay happy and healthy"
#         # print(model.predict(X))
#         # return (html_data , result, cmt)
#         return(model.predict(X))

# english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = ChatterBotCorpusTrainer(english_bot)
# trainer.train("chatterbot.corpus.english")
# response = english_bot.get_response('What is depression?')
# print(response)
 
# response = english_bot.get_response('Who are you?')
# print(response)
 
# @app.route("/chat")
# def chat():
#     return render_template("chat.html")
 
# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     return str(english_bot.get_response(userText))

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000,debug=True)


    
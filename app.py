# I am using mediapipe as a hand landmark processing and prediction and landmark detector and a Random Forest classifier as sign classifier.

import requests
from socket import SocketIO
from wsgiref.simple_server import WSGIServer
from google.oauth2 import id_token

from flask import Flask, jsonify, render_template, url_for, redirect, flash, session, request, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
import socketio
# sio = socketio.Server(async_mode='gevent')
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, Email, EqualTo
from flask_bcrypt import Bcrypt
from datetime import datetime
from itsdangerous import URLSafeTimedSerializer as Serializer
from flask_mail import Message, Mail
import random
import re
import pickle
import cv2
import mediapipe as mp
import numpy as np


app = Flask(__name__)





CORS(app)  # Allow cross-origin requests for all routes

# Encrypt Password using Hash Func
bcrypt = Bcrypt(app)

# Database Model Setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
serializer = Serializer(app.config['SECRET_KEY'])
db = SQLAlchemy(app)
app.app_context().push()


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# mail configuration
app.config["MAIL_SERVER"] = 'smtp.gmail.com'
app.config["MAIL_PORT"] = 587
app.config["MAIL_USERNAME"] = 'handssignify@gmail.com'
app.config["MAIL_PASSWORD"] = 'ttbylakctxvvvnxe'
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
mail = Mail(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# -------------------Database Model-------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), nullable=False, unique=True)
    email = db.Column(db.String(30), nullable=False) 
    password = db.Column(db.String(80), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
# ----------------------------------------------------

# -------------------Welcome or Home Page-------------

@app.route('/', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('home.html')
# ----------------------------------------------------

# -------------------feed back Page-----------------------
@app.route('/feed', methods=['GET', 'POST'])
@login_required
def feed():
    return render_template('feed.html')
# ----------------------------------------------------


# -------------------Discover More Page---------------
@app.route('/discover_more', methods=['GET', 'POST']) 
def discover_more():
    return render_template('discover_more.html')
# ----------------------------------------------------

# -------------------Guide Page-----------------------
@app.route('/guide', methods=['GET', 'POST'])
def guide():
    return render_template('guide.html')
# ----------------------------------------------------


# -------------------Login Page-------------------
class LoginForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    password = PasswordField(label='password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    # Check if the user has registered before showing the login form
    if 'registered' in session and session['registered']:
        session.pop('registered', None)
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data) and User.query.filter_by(email=form.email.data).first():
            login_user(user)
            flash('Login successfully.', category='success')
            name = form.username.data
            session['name'] = name
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash(f'Login unsuccessful for {form.username.data}.', category='danger')
    return render_template('login.html', form=form)

@app.route('/google_signin', methods=['POST'])
def google_signin():
    token = request.json.get('id_token')
    try:
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), '')
        userid = idinfo['sub']
        # Check if user exists in your database and create a new user if necessary
        # Log the user in
        return jsonify({'status': 'success'}), 200
    except ValueError:
        # Invalid token
        return jsonify({'status': 'error', 'message': 'Invalid token'}), 400

# ----------------------------------------------------


# -------------------Dashboard or Logged Page-------------------
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if 'logged_in' in session and session['logged_in']:
        name = session.get('name')
        # character = session.get('character')
        # templs = ['detect_characters.html', 'dashboard.html']
        return render_template('dashboard.html', name=name)
    return redirect(url_for('login'))
# ----------------------------------------------------

# -------------------About Page-----------------------
@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')
# ----------------------------------------------------

# -------------------Logged Out Page-------------------

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    session.clear()
    session.pop("github_user", None)
    logout_user()
    flash('Account Logged out successfully.', category='success')
    return redirect(url_for('login'))
# ----------------------------------------------------

# -------------------Register Page-------------------

class RegisterForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    password = PasswordField(label='password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    confirm_password = PasswordField(label='confirm_password', validators=[InputRequired(), EqualTo('password')], render_kw={"placeholder": "Confirm Password"})
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            flash('That Username already exists. Please choose a different one.', 'danger')
            raise ValidationError('That username already exists. Please choose a different one.')


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data,email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        # Set a session variable to indicate successful registration
        session['registered'] = True
        flash(f'Account Created for {form.username.data} successfully.', category='success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)
# ----------------------------------------------------

# -------------------Update or reset Email Page-------------------


class ResetMailForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Old Email"})
    new_email = StringField(label='new_email', validators=[InputRequired(), Email()], render_kw={"placeholder": "New Email"})
    password = PasswordField(label='password', validators=[InputRequired()], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login', validators=[InputRequired()])


@app.route('/reset_email', methods=['GET', 'POST'])
@login_required
def reset_email():
    form = ResetMailForm()
    if 'logged_in' in session and session['logged_in']:
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if user and bcrypt.check_password_hash(user.password, form.password.data) and User.query.filter_by(email=form.email.data).first():
                user.email = form.new_email.data  # Replace old email with new email
                db.session.commit()
                flash('Email reset successfully.', category='success')
                session.clear()
                return redirect(url_for('login'))
            else:
                flash('Invalid email, password, or combination.', category='danger')

        return render_template('reset_email.html', form=form)
    return redirect(url_for('login'))
# --------------------------------------------------------------

# -------------------Forgot Password With OTP-------------------

class ResetPasswordForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    submit = SubmitField('Submit', validators=[InputRequired()])


class ForgotPasswordForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    new_password = PasswordField(label='new_password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "New Password"})
    confirm_password = PasswordField(label='confirm_password', validators=[InputRequired(), EqualTo('new_password')], render_kw={"placeholder": "Confirm Password"})
    otp = StringField(label='otp', validators=[InputRequired(), Length(min=6, max=6)], render_kw={"placeholder": "Enter OTP"})
    submit = SubmitField('Submit', validators=[InputRequired()])


@staticmethod
def send_mail(name, email, otp):
    msg = Message('Reset Email OTP Password',sender='handssignify@gmail.com', recipients=[email])
    msg.body = "Hii " + name + "," + "\nYour email OTP is :"+str(otp)
    mail.send(msg)


    # Generate your OTP logic here
def generate_otp():
    return random.randint(100000, 999999)


@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    otp = generate_otp()
    session['otp'] = otp
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and User.query.filter_by(email=form.email.data).first():
            send_mail(form.username.data, form.email.data, otp)
            flash('Reset Request Sent. Check your mail.', 'success')
            return redirect(url_for('forgot_password'))
        else:
            flash('Email and username combination is not exist.', 'danger')
    return render_template('reset_password_request.html', form=form)


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        otp = request.form['otp']
        valid = (otp == request.form['otp'])

        if valid:
            user = User.query.filter_by(username=form.username.data).first()
            if user and User.query.filter_by(email=form.email.data).first():
                user.password = bcrypt.generate_password_hash(form.new_password.data).decode('utf-8')
                db.session.commit()
                flash('Password Changed Successfully.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Email and username combination is not exist.', 'danger')
        else:
            flash("OTP verification failed.", 'danger')
    return render_template('forgot_password.html', form=form)
# ---------------------------------------------------------------

# ------------------------- Update Password ---------------------

class UpdatePasswordForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    new_password = PasswordField(label='new_password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "New Password"})
    confirm_password = PasswordField(label='confirm_password', validators=[InputRequired(), EqualTo('new_password')], render_kw={"placeholder": "Confirm Password"})
    submit = SubmitField('Submit', validators=[InputRequired()])


@app.route('/update_password', methods=['GET', 'POST'])
@login_required
def update_password():
    form = UpdatePasswordForm()
    if form.validate_on_submit() and 'logged_in' in session and session['logged_in']:

            user = User.query.filter_by(username=form.username.data).first()
            if user and User.query.filter_by(email=form.email.data).first():
                user.password = bcrypt.generate_password_hash(form.new_password.data).decode('utf-8')
                db.session.commit()
                flash('Password Changed Successfully.', 'success')
                session.clear()
                return redirect(url_for('login'))
            else:
                flash("Username and email combination is not exist.", 'danger')
    return render_template('update_password.html', form=form)
# -----------------------------  end  ---------------------------


# --------------------------- Machine Learning ------------------
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading the model:", e)
    model = None
@app.route('/generate_frames', methods=['POST'])
def generate_frames():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

    # labels_dict = {0: 'A', 1: 'B', 2: 'C'}
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.' }

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # ... Rest of the hand landmark processing and prediction code ...
            # data_aux = []
            # x_ = []
            # y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                # response_data = {'characters': predicted_character}
                print("Predicted character : ",predicted_character)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)
                # flash(f'Predicted Character is {predicted_character}.', category='success')

            except Exception as e:
                print("Prediction error:", e)
                   #print(e)
                   # Handle prediction error
                   
        else:
            print("No hand landmarks detected")            

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')






# GitHub OAuth Credentials
GITHUB_CLIENT_ID = "Ov23liIxOiVVsWo9mDuJ"
GITHUB_CLIENT_SECRET = "44f356374a98c70dd8421b76075e78566fda6d30"
GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_API = "https://api.github.com/user"


@app.route("/github/login")
def github_login():
    github_redirect_url = f"{GITHUB_AUTH_URL}?client_id={GITHUB_CLIENT_ID}&scope=user"
    return redirect(github_redirect_url)

@app.route("/github/callback")
def github_callback():
    code = request.args.get("code")
    if not code:
        flash("GitHub authorization failed!", category="danger")
        return redirect(url_for("login"))

    # Exchange code for access token
    token_response = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={"client_id": GITHUB_CLIENT_ID, "client_secret": GITHUB_CLIENT_SECRET, "code": code},
    )
    token_json = token_response.json()
    access_token = token_json.get("access_token")

    if not access_token:
        flash("Failed to retrieve access token!", category="danger")
        return redirect(url_for("login"))

    # Get user details from GitHub API
    user_response = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"token {access_token}"}
    )
    user_data = user_response.json()

    github_username = user_data.get("login")
    github_email = user_data.get("email")  # Might be None if private
    avatar_url = user_data.get("avatar_url")

    # If GitHub email is None, fetch emails separately
    if github_email is None:
        email_response = requests.get(
            "https://api.github.com/user/emails",
            headers={"Authorization": f"token {access_token}"}
        )
        emails = email_response.json()
        for email in emails:
            if email.get("primary") and email.get("verified"):
                github_email = email.get("email")
                break

    # If email is still None, generate a placeholder
    if github_email is None:
        github_email = f"{github_username}@github.com"

    # Check if the user already exists in the database
    existing_user = User.query.filter_by(email=github_email).first()

    if existing_user:
        login_user(existing_user)
        session["name"] = existing_user.username
        session["logged_in"] = True
        flash("Login successful!", category="success")
    else:
        # Create a new user
        new_user = User(username=github_username, email=github_email, password="")
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        session["name"] = github_username
        session["logged_in"] = True
        flash("Account created & logged in successfully!", category="success")

    return redirect(url_for("dashboard"))  # ✅ Redirect to dashboard




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
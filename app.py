from flask import Flask, render_template, request, jsonify, redirect
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
import pickle, os, requests
from bs4 import BeautifulSoup

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# MySQL Config
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')

mysql = MySQL(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Model load karo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    if user:
        return User(user[0], user[1], user[2])
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
        mysql.connection.commit()
        cur.close()
        return render_template('login.html', msg='Registration successful! Please login.')
    return redirect('/detector')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        if user and bcrypt.check_password_hash(user[3], password):
            login_user(User(user[0], user[1], user[2]))
            return render_template('detector.html')
        return render_template('login.html', msg='Invalid credentials!')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
@login_required
def detect():
    text = request.form.get('news_text', '')
    url = request.form.get('news_url', '')

    if url:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
        except:
            return jsonify({'error': 'URL se content nahi mila!'})

    if not text.strip():
        return jsonify({'error': 'Kuch text enter karo!'})

    prediction = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0]) * 100
    credibility_score = int(confidence) if prediction == 'REAL' else int(100 - confidence)

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO analysis_history (user_id, input_text, verdict, credibility_score) VALUES (%s, %s, %s, %s)",
                (current_user.id, text[:500], prediction, credibility_score))
    mysql.connection.commit()
    cur.close()

    return jsonify({
        'verdict': prediction,
        'credibility_score': credibility_score,
        'confidence': round(confidence, 2)
    })

@app.route('/detector')
@login_required
def detector():
    return render_template('detector.html')

@app.route('/history')
@login_required
def history():
    cur = mysql.connection.cursor()
    cur.execute("SELECT input_text, verdict, credibility_score, created_at FROM analysis_history WHERE user_id = %s ORDER BY created_at DESC", (current_user.id,))
    records = cur.fetchall()
    cur.close()
    return render_template('history.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
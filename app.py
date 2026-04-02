from flask import Flask, render_template, request, jsonify, redirect
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import pickle, os, requests
from bs4 import BeautifulSoup

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def get_db():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        dbname=os.getenv('DB_NAME'),
        sslmode='require'
    )
    return conn

def dict_cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    try:
        db = get_db()
        cur = dict_cursor(db)
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        db.close()
        if user:
            return User(user['id'], user['username'], user['email'])
    except:
        pass
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            email = request.form['email']
            password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
            db = get_db()
            cur = dict_cursor(db)
            cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
            db.commit()
            db.close()
            return render_template('login.html', msg='Registration successful! Please login.')
        except Exception as e:
            return render_template('register.html', msg='Username or email already exists!')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']
            db = get_db()
            cur = dict_cursor(db)
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            db.close()
            if user and bcrypt.check_password_hash(user['password'], password):
                login_user(User(user['id'], user['username'], user['email']))
                return redirect('/detector')
            return render_template('login.html', msg='Invalid credentials!')
        except Exception as e:
            return render_template('login.html', msg='Error: ' + str(e))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')

@app.route('/detector')
@login_required
def detector():
    return render_template('detector.html')

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

    try:
        db = get_db()
        cur = dict_cursor(db)
        cur.execute("INSERT INTO analysis_history (user_id, input_text, verdict, credibility_score) VALUES (%s, %s, %s, %s)",
                    (current_user.id, text[:500], prediction, credibility_score))
        db.commit()
        db.close()
    except:
        pass

    return jsonify({
        'verdict': prediction,
        'credibility_score': credibility_score,
        'confidence': round(confidence, 2)
    })

@app.route('/history')
@login_required
def history():
    try:
        db = get_db()
        cur = dict_cursor(db)
        cur.execute("SELECT input_text, verdict, credibility_score, created_at FROM analysis_history WHERE user_id = %s ORDER BY created_at DESC", (current_user.id,))
        records = cur.fetchall()
        db.close()
    except:
        records = []
    return render_template('history.html', records=records)

@app.route('/admin')
@login_required
def admin():
    try:
        db = get_db()
        cur = dict_cursor(db)
        
        cur.execute("SELECT COUNT(*) as total FROM users")
        total_users = cur.fetchone()['count']
        
        cur.execute("SELECT COUNT(*) as total FROM analysis_history")
        total_analyses = cur.fetchone()['count']
        
        cur.execute("SELECT COUNT(*) as total FROM analysis_history WHERE verdict = 'FAKE'")
        total_fake = cur.fetchone()['count']
        
        cur.execute("SELECT COUNT(*) as total FROM analysis_history WHERE verdict = 'REAL'")
        total_real = cur.fetchone()['count']
        
        cur.execute("SELECT u.username, u.email, u.created_at FROM users u ORDER BY u.created_at DESC")
        all_users = cur.fetchall()
        
        cur.execute("SELECT u.username, a.input_text, a.verdict, a.credibility_score, a.created_at FROM analysis_history a JOIN users u ON a.user_id = u.id ORDER BY a.created_at DESC LIMIT 20")
        recent_analyses = cur.fetchall()
        
        db.close()
    except Exception as e:
        return str(e)
    
    return render_template('admin.html', 
        total_users=total_users,
        total_analyses=total_analyses,
        total_fake=total_fake,
        total_real=total_real,
        all_users=all_users,
        recent_analyses=recent_analyses
    )

if __name__ == '__main__':
    app.run(debug=True)
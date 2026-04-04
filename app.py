from flask import Flask, render_template, request, jsonify, redirect
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect as detect_language
from urllib.parse import urlparse
from indian_facts import check_indian_facts, get_credibility_boost
from groq import Groq
import psycopg2
import psycopg2.extras
import pickle, os, requests, json
from bs4 import BeautifulSoup

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# -- GROQ SETUP --
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# -- RAPIDAPI SETUP --
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')

def analyze_with_groq(text):
    try:
        prompt = f"""You are a highly accurate fact-verification expert with complete knowledge of India and the world up to 2026.

Your job is to verify whether the given statement/news is REAL (factually correct) or FAKE (factually incorrect or misleading).

STATEMENT TO VERIFY:
"{text}"

IMPORTANT RULES:
1. If the statement contains CORRECT, VERIFIABLE FACTS -> verdict must be "REAL" with credibility_score 85-100
2. If the statement is PARTIALLY correct -> verdict "REAL" with score 60-80
3. If the statement contains CLEARLY FALSE information -> verdict "FAKE" with score 0-40
4. If the statement is unverifiable opinion -> score 45-65
5. DO NOT be biased toward FAKE - most factual statements are REAL

INDIAN KNOWLEDGE BASE:
- Capitals: Ranchi=Jharkhand, Patna=Bihar, Lucknow=UP, Mumbai=Maharashtra, Delhi=India, Bhopal=MP, Jaipur=Rajasthan, Chennai=TN, Hyderabad=Telangana, Bengaluru=Karnataka
- PM: Narendra Modi (BJP), President: Droupadi Murmu
- States: India has 28 states and 8 UTs
- Operation Sindoor: Indian military operation 2025
- Government schemes: PM Kisan, Ayushman Bharat, Jan Dhan, etc.

EXAMPLES:
- "Ranchi is the capital of Jharkhand" -> REAL, score: 98
- "Patna is the capital of Bihar" -> REAL, score: 98
- "Delhi is capital of India" -> REAL, score: 99
- "Mumbai is capital of India" -> FAKE, score: 5
- "Modi is PM of India" -> REAL, score: 97

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{
  "verdict": "REAL",
  "credibility_score": 95,
  "confidence": 95,
  "reason": "Brief explanation in 1-2 sentences",
  "facts": ["fact1", "fact2"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )

        response_text = response.choices[0].message.content.strip()

        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            response_text = response_text[start:end]

        result = json.loads(response_text)
        return {
            'verdict': result.get('verdict', 'REAL'),
            'credibility_score': int(result.get('credibility_score', 70)),
            'confidence': float(result.get('confidence', 70)),
            'reason': result.get('reason', ''),
            'gemini_facts': result.get('facts', [])
        }
    except Exception as e:
        print(f"Groq error: {e}")
        return None

# -- CRICKET KEYWORDS --
CRICKET_KEYWORDS = [
    'cricket', 'ipl', 'match', 'wicket', 'batting', 'bowling',
    'test match', 'odi', 't20', 'bcci', 'runs', 'over', 'innings',
    'six', 'four', 'stumped', 'lbw', 'yorker', 'bouncer',
    'dc', 'mi', 'csk', 'rcb', 'kkr', 'srh', 'rr', 'pbks', 'lsg', 'gt',
    'delhi capitals', 'mumbai indians', 'chennai', 'bangalore',
    'kohli', 'rohit', 'dhoni', 'bumrah', 'pandya'
]

def is_cricket_news(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRICKET_KEYWORDS)

def get_cricket_scores():
    try:
        url = "https://cricbuzz-cricket.p.rapidapi.com/matches/v1/live"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        matches = []
        for match_type in data.get("typeMatches", []):
            for series in match_type.get("seriesMatches", []):
                series_data = series.get("seriesAdWrapper", {})
                for m in series_data.get("matches", []):
                    info = m.get("matchInfo", {})
                    score = m.get("matchScore", {})
                    t1 = info.get("team1", {})
                    t2 = info.get("team2", {})
                    t1i = score.get("team1Score", {}).get("inngs1", {})
                    t2i = score.get("team2Score", {}).get("inngs1", {})
                    matches.append({
                        "team1": t1.get("teamSName", t1.get("teamName", "Team 1")),
                        "team2": t2.get("teamSName", t2.get("teamName", "Team 2")),
                        "status": info.get("status", "Live"),
                        "seriesName": series_data.get("seriesName", ""),
                        "t1score": f"{t1i.get('runs','-')}/{t1i.get('wickets','-')} ({t1i.get('overs','-')} ov)" if t1i else "-",
                        "t2score": f"{t2i.get('runs','-')}/{t2i.get('wickets','-')} ({t2i.get('overs','-')} ov)" if t2i else "-",
                    })
        return matches[:5]
    except:
        return []

# -- LANGUAGE SETUP --
LANGUAGE_DISPLAY_NAMES = {
    'en': 'English', 'hi': 'Hindi / Haryanvi', 'ta': 'Tamil',
    'te': 'Telugu', 'mr': 'Marathi', 'gu': 'Gujarati',
    'pa': 'Punjabi', 'bn': 'Bengali', 'ml': 'Malayalam',
    'kn': 'Kannada', 'ur': 'Urdu', 'or': 'Odia',
}

def translate_to_english(text):
    try:
        lang_code = detect_language(text)
        lang_name = LANGUAGE_DISPLAY_NAMES.get(lang_code, lang_code)
        if lang_code != 'en':
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated, lang_name
        return text, 'English'
    except:
        return text, 'English'

# -- FACT CHECK API --
FACT_CHECK_API_KEY = os.getenv('FACT_CHECK_API_KEY')

def check_facts(text):
    try:
        short_query = ' '.join(text.split()[:10])
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {'query': short_query, 'key': FACT_CHECK_API_KEY, 'languageCode': 'en'}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        results = []
        for claim in data.get('claims', [])[:3]:
            review = claim.get('claimReview', [{}])[0]
            results.append({
                'text': claim.get('text', '')[:120],
                'rating': review.get('textualRating', 'Unknown'),
                'source': review.get('publisher', {}).get('name', ''),
                'url': review.get('url', '')
            })
        return results
    except:
        return []

# -- SOURCE REPUTATION --
SOURCE_REPUTATION = {
    'thehindu.com':       {'tier': 1, 'label': 'Highly Credible', 'desc': 'Major Indian newspaper'},
    'hindustantimes.com': {'tier': 1, 'label': 'Highly Credible', 'desc': 'Major Indian newspaper'},
    'ndtv.com':           {'tier': 1, 'label': 'Highly Credible', 'desc': 'Indian news channel'},
    'aajtak.in':          {'tier': 1, 'label': 'Highly Credible', 'desc': 'Indian news channel'},
    'bbc.com':            {'tier': 1, 'label': 'Highly Credible', 'desc': 'International broadcaster'},
    'reuters.com':        {'tier': 1, 'label': 'Highly Credible', 'desc': 'International wire service'},
    'indianexpress.com':  {'tier': 1, 'label': 'Highly Credible', 'desc': 'Major Indian newspaper'},
    'timesofindia.com':   {'tier': 1, 'label': 'Highly Credible', 'desc': 'Major Indian newspaper'},
    'theonion.com':       {'tier': 2, 'label': 'Satire', 'desc': 'Satirical website'},
    'postcard.news':      {'tier': 3, 'label': 'Known Misinformation', 'desc': 'Flagged by AltNews'},
}

def check_source_reputation(url):
    try:
        domain = urlparse(url).netloc.replace('www.', '')
        if domain in SOURCE_REPUTATION:
            return {'found': True, **SOURCE_REPUTATION[domain]}
        return {'found': False}
    except:
        return {'found': False}

# -- DATABASE --
def get_db():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
        dbname=os.getenv('DB_NAME'), sslmode='require'
    )

def dict_cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# -- USER MODEL --
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

# -- ROUTES --
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
        except:
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
    url  = request.form.get('news_url', '')

    if url:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
        except:
            return jsonify({'error': 'Could not fetch content from URL!'})

    if not text.strip():
        return jsonify({'error': 'Please enter some text!'})

    english_text, detected_lang = translate_to_english(text)
    groq_result = analyze_with_groq(english_text)

    if groq_result:
        prediction = groq_result['verdict']
        credibility_score = groq_result['credibility_score']
        confidence = groq_result['confidence']
        gemini_reason = groq_result['reason']
        gemini_facts = groq_result['gemini_facts']
        indian_facts_matched = check_indian_facts(english_text)
    else:
        prediction = model.predict([english_text])[0]
        confidence = max(model.predict_proba([english_text])[0]) * 100
        credibility_score = int(confidence) if prediction == 'REAL' else int(100 - confidence)
        gemini_reason = ''
        gemini_facts = []
        indian_facts_matched = check_indian_facts(english_text)
        facts_boost = get_credibility_boost(english_text)
        if facts_boost >= 40:
            credibility_score = min(100, 70 + (facts_boost - 40))
            prediction = 'REAL'
        elif facts_boost >= 20:
            credibility_score = min(100, credibility_score + facts_boost)
            if credibility_score >= 45:
                prediction = 'REAL'
        else:
            credibility_score = min(100, credibility_score + facts_boost)

    reputation = {}
    if url:
        reputation = check_source_reputation(url)
        if reputation.get('tier') == 3:
            credibility_score = max(0, credibility_score - 20)

    fact_results = check_facts(english_text)

    # -- Cricket Live Scores --
    cricket_scores = get_cricket_scores() if is_cricket_news(text) else []

    try:
        db = get_db()
        cur = dict_cursor(db)
        cur.execute(
            "INSERT INTO analysis_history (user_id, input_text, verdict, credibility_score) VALUES (%s, %s, %s, %s)",
            (current_user.id, text[:500], prediction, credibility_score)
        )
        db.commit()
        db.close()
    except:
        pass

    return jsonify({
        'verdict': prediction,
        'credibility_score': credibility_score,
        'confidence': round(confidence, 2),
        'detected_lang': detected_lang,
        'reputation': reputation,
        'fact_results': fact_results,
        'indian_facts': indian_facts_matched,
        'gemini_reason': gemini_reason,
        'gemini_facts': gemini_facts,
        'cricket_scores': cricket_scores
    })

@app.route('/history')
@login_required
def history():
    try:
        db = get_db()
        cur = dict_cursor(db)
        cur.execute(
            "SELECT input_text, verdict, credibility_score, created_at FROM analysis_history WHERE user_id = %s ORDER BY created_at DESC",
            (current_user.id,)
        )
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
        cur = db.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM analysis_history")
        total_analyses = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM analysis_history WHERE verdict = 'FAKE'")
        total_fake = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM analysis_history WHERE verdict = 'REAL'")
        total_real = cur.fetchone()[0]
        cur2 = dict_cursor(db)
        cur2.execute("SELECT username, email, created_at FROM users ORDER BY created_at DESC")
        all_users = cur2.fetchall()
        cur2.execute("""SELECT u.username, a.input_text, a.verdict, a.credibility_score, a.created_at
                        FROM analysis_history a JOIN users u ON a.user_id = u.id
                        ORDER BY a.created_at DESC LIMIT 20""")
        recent_analyses = cur2.fetchall()
        db.close()
    except Exception as e:
        return str(e)
    return render_template('admin.html',
        total_users=total_users, total_analyses=total_analyses,
        total_fake=total_fake, total_real=total_real,
        all_users=all_users, recent_analyses=recent_analyses)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
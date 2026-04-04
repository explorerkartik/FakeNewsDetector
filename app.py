from flask import Flask, render_template, request, jsonify, redirect
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect as detect_language
from urllib.parse import urlparse
import psycopg2
import psycopg2.extras
import pickle, os, requests, json
from bs4 import BeautifulSoup
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ── GROK SETUP ───────────────────────────────────────────
GROK_API_KEY = os.getenv('GROK_API_KEY')

# ── ABBREVIATIONS DICTIONARY ─────────────────────────────
ABBREVIATIONS = {
    # IPL Teams
    "dc": "Delhi Capitals",
    "mi": "Mumbai Indians",
    "csk": "Chennai Super Kings",
    "rcb": "Royal Challengers Bengaluru",
    "kkr": "Kolkata Knight Riders",
    "srh": "Sunrisers Hyderabad",
    "rr": "Rajasthan Royals",
    "pbks": "Punjab Kings",
    "lsg": "Lucknow Super Giants",
    "gt": "Gujarat Titans",
    # Cricket
    "bcci": "Board of Control for Cricket in India",
    "icc": "International Cricket Council",
    "odi": "One Day International cricket",
    "t20": "Twenty20 cricket",
    "ipl": "Indian Premier League",
    # Politics
    "pm": "Prime Minister",
    "cm": "Chief Minister",
    "bjp": "Bharatiya Janata Party",
    "inc": "Indian National Congress",
    "aap": "Aam Aadmi Party",
    "sp": "Samajwadi Party",
    "bsp": "Bahujan Samaj Party",
    "nda": "National Democratic Alliance",
    "upa": "United Progressive Alliance",
    "mla": "Member of Legislative Assembly",
    "mp": "Member of Parliament",
    # Sports
    "isl": "Indian Super League football",
    "pkl": "Pro Kabaddi League",
    "hil": "Hockey India League",
    # Others
    "isro": "Indian Space Research Organisation",
    "rbi": "Reserve Bank of India",
    "cbi": "Central Bureau of Investigation",
    "ed": "Enforcement Directorate",
    "upi": "Unified Payments Interface",
    "gst": "Goods and Services Tax",
    "cji": "Chief Justice of India",
    "nsa": "National Security Advisor",
    "cds": "Chief of Defence Staff",
    "iit": "Indian Institute of Technology",
    "iim": "Indian Institute of Management",
    "upsc": "Union Public Service Commission",
    "ssc": "Staff Selection Commission",
    "neet": "National Eligibility cum Entrance Test",
    "jee": "Joint Entrance Examination",
}

def expand_abbreviations(text):
    """Expand abbreviations in text before sending to Grok"""
    words = text.split()
    expanded = []
    for word in words:
        clean_word = word.lower().strip('.,!?')
        if clean_word in ABBREVIATIONS:
            expanded.append(f"{word} ({ABBREVIATIONS[clean_word]})")
        else:
            expanded.append(word)
    return ' '.join(expanded)

def analyze_with_grok(text):
    try:
        # Expand abbreviations first
        expanded_text = expand_abbreviations(text)
        today = datetime.now().strftime("%d %B %Y")

        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-3-latest",
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a fake news detection expert. Today's date is {today}.

You have complete knowledge about:
- Indian politics: PM Narendra Modi, all state CMs, Ministers, BJP, Congress, AAP, SP, BSP
- IPL 2025: All teams — MI (Mumbai Indians), DC (Delhi Capitals), CSK (Chennai Super Kings), RCB (Royal Challengers Bengaluru), KKR (Kolkata Knight Riders), SRH (Sunrisers Hyderabad), RR (Rajasthan Royals), PBKS (Punjab Kings), LSG (Lucknow Super Giants), GT (Gujarat Titans)
- Cricket: Virat Kohli, Rohit Sharma, MS Dhoni, Hardik Pandya, Jasprit Bumrah
- Chess: D Gukesh (World Champion 2024), Viswanathan Anand
- Olympics, Asian Games, Commonwealth Games
- Football: ISL, FIFA, Sunil Chhetri
- Badminton: PV Sindhu, Saina Nehwal, Lakshya Sen
- Hockey, Wrestling, Boxing, Tennis, Kabaddi
- Indian geography: all state capitals, major cities
- Government schemes, science, technology, AI
- Current affairs up to {today}

Always expand abbreviations and understand context fully before judging."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this news/text and determine if it is REAL or FAKE.
Today's date: {today}

Original text: "{text}"
Expanded text: "{expanded_text}"

Important: 
- If asking about recent match results or live events, say you cannot verify real-time results but assess based on known facts
- Expand all abbreviations (DC=Delhi Capitals, MI=Mumbai Indians, etc.)
- Be accurate about Indian facts

Respond ONLY in this exact JSON format:
{{
  "verdict": "REAL" or "FAKE",
  "credibility_score": (number 0-100),
  "confidence": (number 0-100),
  "reason": "Brief explanation in 1-2 sentences",
  "facts": ["relevant fact 1", "relevant fact 2"]
}}

Rules:
- Correct verifiable facts → REAL (75-100)
- False information → FAKE (0-35)
- Recent unverifiable events → score 45-65
- Opinion → score 40-60"""
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }

        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )

        data = response.json()
        response_text = data['choices'][0]['message']['content'].strip()

        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        result = json.loads(response_text)
        return {
            'verdict': result.get('verdict', 'REAL'),
            'credibility_score': int(result.get('credibility_score', 70)),
            'confidence': float(result.get('confidence', 70)),
            'reason': result.get('reason', ''),
            'grok_facts': result.get('facts', [])
        }
    except Exception as e:
        return None

# ── LANGUAGE SETUP ───────────────────────────────────────
LANGUAGE_DISPLAY_NAMES = {
    'en': 'English',
    'hi': 'Hindi / Haryanvi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'bn': 'Bengali',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'ur': 'Urdu',
    'or': 'Odia',
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

# ── FACT CHECK API ───────────────────────────────────────
FACT_CHECK_API_KEY = os.getenv('FACT_CHECK_API_KEY')

def check_facts(text):
    try:
        short_query = ' '.join(text.split()[:10])
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            'query': short_query,
            'key': FACT_CHECK_API_KEY,
            'languageCode': 'en'
        }
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

# ── SOURCE REPUTATION DATABASE ───────────────────────────
SOURCE_REPUTATION = {
    'thehindu.com':       {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'Major Indian newspaper'},
    'hindustantimes.com': {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'Major Indian newspaper'},
    'ndtv.com':           {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'Indian news channel'},
    'aajtak.in':          {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'Indian news channel'},
    'bbc.com':            {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'International broadcaster'},
    'reuters.com':        {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'International wire service'},
    'indianexpress.com':  {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'Major Indian newspaper'},
    'timesofindia.com':   {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'Major Indian newspaper'},
    'ptinews.com':        {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'Press Trust of India'},
    'ani.co.in':          {'tier': 1, 'label': '✅ Highly Credible', 'desc': 'ANI News Agency'},
    'theonion.com':       {'tier': 2, 'label': '😄 Satire', 'desc': 'Satirical website'},
    'fakingnews.com':     {'tier': 2, 'label': '😄 Satire', 'desc': 'Indian satire website'},
    'postcard.news':      {'tier': 3, 'label': '❌ Known Misinformation', 'desc': 'Flagged by AltNews'},
}

def check_source_reputation(url):
    try:
        domain = urlparse(url).netloc.replace('www.', '')
        if domain in SOURCE_REPUTATION:
            info = SOURCE_REPUTATION[domain]
            return {'found': True, **info}
        return {'found': False}
    except:
        return {'found': False}

# ── DATABASE ─────────────────────────────────────────────
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

# ── USER MODEL ───────────────────────────────────────────
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

# ── ROUTES ───────────────────────────────────────────────
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
            cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                        (username, email, password))
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
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
        except:
            return jsonify({'error': 'Could not fetch content from URL!'})

    if not text.strip():
        return jsonify({'error': 'Please enter some text!'})

    # ── Translate to English ──
    english_text, detected_lang = translate_to_english(text)

    # ── Grok Analysis ──
    grok_result = analyze_with_grok(english_text)

    if grok_result:
        prediction = grok_result['verdict']
        credibility_score = grok_result['credibility_score']
        confidence = grok_result['confidence']
        grok_reason = grok_result['reason']
        grok_facts = grok_result['grok_facts']
    else:
        # Fallback — old ML model
        prediction = model.predict([english_text])[0]
        confidence = max(model.predict_proba([english_text])[0]) * 100
        credibility_score = int(confidence) if prediction == 'REAL' else int(100 - confidence)
        grok_reason = ''
        grok_facts = []

    # ── Source Reputation ──
    reputation = {}
    if url:
        reputation = check_source_reputation(url)
        if reputation.get('tier') == 3:
            credibility_score = max(0, credibility_score - 20)

    # ── Fact Check API ──
    fact_results = check_facts(english_text)

    # ── Save to DB ──
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
        'grok_reason': grok_reason,
        'grok_facts': grok_facts
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
        total_users=total_users,
        total_analyses=total_analyses,
        total_fake=total_fake,
        total_real=total_real,
        all_users=all_users,
        recent_analyses=recent_analyses)

if __name__ == '__main__':
    app.run(debug=True)
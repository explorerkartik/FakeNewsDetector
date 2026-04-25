from flask import Flask, render_template, request, jsonify, redirect, url_for, session, make_response
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect as detect_language
from urllib.parse import urlparse
from indian_facts import check_indian_facts, get_credibility_boost, fetch_and_store_current_affairs
from groq import Groq
from authlib.integrations.flask_client import OAuth
import psycopg2
import psycopg2.extras
import pickle, os, requests, json, uuid, secrets
from bs4 import BeautifulSoup
from datetime import datetime

# ── PDF ──────────────────────────────────────────────────────────────────────
from fpdf import FPDF
import io

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ── FLASK-MAIL CONFIG (Gmail SMTP – FREE) ─────────────────────────────────────
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_APP_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
mail = Mail(app)

# ── MODEL LOADING (safe fallback if pkl corrupted) ───────────────────────────
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ model.pkl loaded successfully")
except Exception as e:
    print(f"⚠️  model.pkl load failed ({e}), building fallback model...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion
    import re as _re

    def _clean(text):
        if not isinstance(text, str): return ""
        text = _re.sub(r'http\S+|<[^>]+>', ' ', text)
        text = _re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
        return _re.sub(r'\s+', ' ', text).strip().lower()

    _REAL = [
        "narendra modi is the prime minister of india",
        "chandrayaan 3 landed on moon south pole successfully",
        "india won t20 world cup 2024 defeating south africa",
        "rcb won ipl 2025 championship title",
        "operation sindoor india military operation 2025",
        "ram mandir inaugurated ayodhya january 2024",
        "neeraj chopra silver medal paris olympics 2024",
        "india gdp fifth largest economy world",
        "chandrayaan 3 ne chand par landing ki",
        "india ne t20 world cup 2024 jeeta",
        "modi ji pradhan mantri hain bharat ke",
        "rcb ne ipl 2025 jita",
        "operation sindoor india ki military operation thi",
        "bharat ki jansankhya 1 arab 44 crore hai",
        "gst 1 july 2017 ko lagu hua",
        "article 370 hataya gaya 2019 mein",
        "droupadi murmu president of india since 2022",
        "isro chandrayaan mission successful moon landing",
        "india 28 states 8 union territories",
        "pm kisan yojana farmers 6000 rupees annually",
    ]
    _FAKE = [
        "modi fled country went to pakistan secret sources",
        "free recharge for all indians government scheme register now",
        "whatsapp banned india next week government order",
        "5g towers spreading coronavirus proven research",
        "bill gates microchipped indians covid vaccines",
        "india secretly sold states to china",
        "rbi ran out of gold reserves hidden crisis",
        "chandrayaan landing video filmed in studio fake",
        "india china nuclear war started breaking news",
        "modi ne desh chhod diya pakistan gaye",
        "free recharge milega sarkar ki taraf se",
        "vaccine mein chip laga hai government spy",
        "india china war nuclear bomb gira",
        "free iphone milega sarkar ki taraf se register",
        "whatsapp band ho raha hai india mein",
        "5g towers corona failate hain scientific proof",
        "modi ji resign kar diye aaj raat",
        "rupee zero hone wala hai next month",
        "army ne delhi mein coup kiya hai",
        "sarkar ne bank accounts freeze karne ka order diya",
    ]
    _texts  = [_clean(t) for t in _REAL + _FAKE]
    _labels = ['REAL'] * len(_REAL) + ['FAKE'] * len(_FAKE)

    _vec = FeatureUnion([
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=50000, sublinear_tf=True)),
        ('word', TfidfVectorizer(analyzer='word',    ngram_range=(1,2), max_features=30000, sublinear_tf=True)),
    ])
    model = Pipeline([('tfidf', _vec), ('clf', LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42))])
    model.fit(_texts, _labels)
    print("✅ Fallback model built successfully")

# ── GROQ ──────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# ── EXTERNAL KEYS ─────────────────────────────────────────────────────────────
RAPIDAPI_KEY       = os.getenv('RAPIDAPI_KEY')
SIGHTENGINE_USER   = os.getenv('SIGHTENGINE_USER')
SIGHTENGINE_SECRET = os.getenv('SIGHTENGINE_SECRET')

# ── WHATSAPP (Meta Cloud API – FREE) ─────────────────────────────────────────
WHATSAPP_TOKEN    = os.getenv('WHATSAPP_TOKEN')
WHATSAPP_VERIFY   = os.getenv('WHATSAPP_VERIFY_TOKEN', 'fakenews_verify_123')
WHATSAPP_PHONE_ID = os.getenv('WHATSAPP_PHONE_ID')

# ── GOOGLE OAUTH ──────────────────────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# ─────────────────────────────────────────────────────────────────────────────
#  GROQ ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
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
- IPL 2025 Winner: Royal Challengers Bengaluru (RCB)
- Government schemes: PM Kisan, Ayushman Bharat, Jan Dhan, etc.

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
        end   = response_text.rfind('}') + 1
        if start != -1 and end > start:
            response_text = response_text[start:end]
        result = json.loads(response_text)
        return {
            'verdict':           result.get('verdict', 'REAL'),
            'credibility_score': int(result.get('credibility_score', 70)),
            'confidence':        float(result.get('confidence', 70)),
            'reason':            result.get('reason', ''),
            'gemini_facts':      result.get('facts', [])
        }
    except Exception as e:
        print(f"Groq error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  CRICKET
# ─────────────────────────────────────────────────────────────────────────────
CRICKET_KEYWORDS = [
    'cricket','ipl','match','wicket','batting','bowling',
    'test match','odi','t20','bcci','runs','over','innings',
    'six','four','stumped','lbw','yorker','bouncer',
    'dc','mi','csk','rcb','kkr','srh','rr','pbks','lsg','gt',
    'delhi capitals','mumbai indians','chennai','bangalore',
    'kohli','rohit','dhoni','bumrah','pandya'
]

def is_cricket_news(text):
    return any(kw in text.lower() for kw in CRICKET_KEYWORDS)

def get_cricket_scores():
    try:
        url = "https://cricbuzz-cricket.p.rapidapi.com/matches/v1/live"
        headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        matches = []
        for match_type in data.get("typeMatches", []):
            for series in match_type.get("seriesMatches", []):
                series_data = series.get("seriesAdWrapper", {})
                for m in series_data.get("matches", []):
                    info  = m.get("matchInfo", {})
                    score = m.get("matchScore", {})
                    t1    = info.get("team1", {})
                    t2    = info.get("team2", {})
                    t1i   = score.get("team1Score", {}).get("inngs1", {})
                    t2i   = score.get("team2Score", {}).get("inngs1", {})
                    matches.append({
                        "team1":      t1.get("teamSName", t1.get("teamName", "Team 1")),
                        "team2":      t2.get("teamSName", t2.get("teamName", "Team 2")),
                        "status":     info.get("status", "Live"),
                        "seriesName": series_data.get("seriesName", ""),
                        "t1score":    f"{t1i.get('runs','-')}/{t1i.get('wickets','-')} ({t1i.get('overs','-')} ov)" if t1i else "-",
                        "t2score":    f"{t2i.get('runs','-')}/{t2i.get('wickets','-')} ({t2i.get('overs','-')} ov)" if t2i else "-",
                    })
        return matches[:5]
    except:
        return []

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def analyze_image_sightengine(image_file):
    try:
        response = requests.post(
            'https://api.sightengine.com/1.0/check.json',
            files={'media': image_file},
            data={'models': 'genai', 'api_user': SIGHTENGINE_USER, 'api_secret': SIGHTENGINE_SECRET},
            timeout=30
        )
        data = response.json()
        if data.get('status') == 'success':
            genai_score = data.get('type', {}).get('ai_generated', 0)
            is_ai = genai_score > 0.5
            return {
                'source':             'Sightengine',
                'ai_generated_score': round(genai_score * 100, 1),
                'is_fake':            is_ai,
                'verdict':            'AI GENERATED / FAKE' if is_ai else 'LIKELY REAL',
                'confidence':         round(genai_score * 100 if is_ai else (1 - genai_score) * 100, 1)
            }
        return None
    except Exception as e:
        print(f"Sightengine image error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def analyze_video_sightengine(video_file):
    try:
        import cv2, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name
        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_positions = [int(total_frames * p) for p in [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]]
        frame_scores = []
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as img_tmp:
                cv2.imwrite(img_tmp.name, frame)
                img_path = img_tmp.name
            with open(img_path, 'rb') as img_file:
                response = requests.post(
                    'https://api.sightengine.com/1.0/check.json',
                    files={'media': img_file},
                    data={'models': 'deepfake', 'api_user': SIGHTENGINE_USER, 'api_secret': SIGHTENGINE_SECRET},
                    timeout=20
                )
            os.unlink(img_path)
            data = response.json()
            if data.get('status') == 'success':
                frame_scores.append(data.get('face', {}).get('deepfake', 0))
        cap.release()
        os.unlink(tmp_path)
        if not frame_scores:
            return None
        avg_score = sum(frame_scores) / len(frame_scores)
        max_score = max(frame_scores)
        is_deepfake = avg_score > 0.5
        return {
            'frames_analyzed':    len(frame_scores),
            'avg_deepfake_score': round(avg_score * 100, 1),
            'max_deepfake_score': round(max_score * 100, 1),
            'is_deepfake':        is_deepfake,
            'verdict':            'DEEPFAKE DETECTED' if is_deepfake else 'LIKELY AUTHENTIC',
            'confidence':         round(avg_score * 100 if is_deepfake else (1 - avg_score) * 100, 1)
        }
    except Exception as e:
        print(f"Video analysis error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  LANGUAGE
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_DISPLAY_NAMES = {
    'en':'English','hi':'Hindi','ta':'Tamil','te':'Telugu',
    'mr':'Marathi','gu':'Gujarati','pa':'Punjabi','bn':'Bengali',
    'ml':'Malayalam','kn':'Kannada','ur':'Urdu','or':'Odia',
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

# ─────────────────────────────────────────────────────────────────────────────
#  FACT CHECK API
# ─────────────────────────────────────────────────────────────────────────────
FACT_CHECK_API_KEY = os.getenv('FACT_CHECK_API_KEY')

def check_facts(text):
    try:
        short_query = ' '.join(text.split()[:10])
        url    = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {'query': short_query, 'key': FACT_CHECK_API_KEY, 'languageCode': 'en'}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        results = []
        for claim in data.get('claims', [])[:3]:
            review = claim.get('claimReview', [{}])[0]
            results.append({
                'text':   claim.get('text', '')[:120],
                'rating': review.get('textualRating', 'Unknown'),
                'source': review.get('publisher', {}).get('name', ''),
                'url':    review.get('url', '')
            })
        return results
    except:
        return []

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE REPUTATION
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_REPUTATION = {
    'thehindu.com':       {'tier': 1, 'label': 'Highly Credible',      'desc': 'Major Indian newspaper'},
    'hindustantimes.com': {'tier': 1, 'label': 'Highly Credible',      'desc': 'Major Indian newspaper'},
    'ndtv.com':           {'tier': 1, 'label': 'Highly Credible',      'desc': 'Indian news channel'},
    'aajtak.in':          {'tier': 1, 'label': 'Highly Credible',      'desc': 'Indian news channel'},
    'bbc.com':            {'tier': 1, 'label': 'Highly Credible',      'desc': 'International broadcaster'},
    'reuters.com':        {'tier': 1, 'label': 'Highly Credible',      'desc': 'International wire service'},
    'indianexpress.com':  {'tier': 1, 'label': 'Highly Credible',      'desc': 'Major Indian newspaper'},
    'timesofindia.com':   {'tier': 1, 'label': 'Highly Credible',      'desc': 'Major Indian newspaper'},
    'theonion.com':       {'tier': 2, 'label': 'Satire',               'desc': 'Satirical website'},
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

# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
        dbname=os.getenv('DB_NAME'), sslmode='require'
    )

def dict_cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# ── DB INIT ───────────────────────────────────────────────────────────────────
def init_db_extras():
    """Add new columns & tables (safe to re-run)."""
    try:
        db  = get_db()
        cur = db.cursor()
        # email verification columns
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified   BOOLEAN DEFAULT FALSE")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS verify_token  TEXT")
        # shared results table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shared_results (
                id             SERIAL PRIMARY KEY,
                share_id       UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                input_text     TEXT,
                verdict        VARCHAR(10),
                credibility_score INT,
                reason         TEXT,
                created_at     TIMESTAMP DEFAULT NOW()
            )
        """)
        # chatbot history table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chatbot_history (
                id         SERIAL PRIMARY KEY,
                user_id    INT REFERENCES users(id) ON DELETE CASCADE,
                role       VARCHAR(10),
                content    TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        # ── NEW: user feedback table ──────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detection_feedback (
                id                SERIAL PRIMARY KEY,
                user_id           INT REFERENCES users(id) ON DELETE SET NULL,
                share_id          UUID,
                feedback          VARCHAR(15) NOT NULL,
                verdict           VARCHAR(10),
                credibility_score INT,
                input_text        TEXT,
                created_at        TIMESTAMP DEFAULT NOW()
            )
        """)
        # ── NEW: dynamic current affairs table ───────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_facts (
                id         SERIAL PRIMARY KEY,
                keyword    TEXT UNIQUE NOT NULL,
                fact       TEXT NOT NULL,
                source     VARCHAR(50) DEFAULT 'auto',
                active     BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        db.commit()
        db.close()
    except Exception as e:
        print(f"init_db_extras error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  USER MODEL
# ─────────────────────────────────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, username, email, is_verified=False):
        self.id          = id
        self.username    = username
        self.email       = email
        self.is_verified = is_verified

@login_manager.user_loader
def load_user(user_id):
    try:
        db  = get_db()
        cur = dict_cursor(db)
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        db.close()
        if user:
            return User(user['id'], user['username'], user['email'], user.get('is_verified', False))
    except:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_fill_color(108, 99, 255)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(8)
    pdf.cell(0, 10, 'FakeNews Detector - Analysis Report', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}  |  Powered by Groq AI + LLaMA 3.3", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    verdict = data.get('verdict', 'UNKNOWN')
    score   = data.get('credibility_score', 0)
    is_real = verdict == 'REAL'
    if is_real:
        pdf.set_fill_color(40, 167, 69)
    else:
        pdf.set_fill_color(220, 53, 69)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 14)
    label = 'REAL NEWS' if is_real else 'FAKE NEWS'
    pdf.cell(0, 12, f'Verdict: {label}', align='C', fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_text_color(50, 50, 50)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, f'Credibility Score: {score}/100', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_draw_color(108, 99, 255)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_text_color(50, 50, 50)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'Analyzed Content:', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 9)
    input_text = data.get('input_text', '')[:600]
    input_text = input_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.set_fill_color(245, 245, 250)
    pdf.multi_cell(0, 6, input_text, fill=True)
    pdf.ln(4)

    reason = data.get('reason', '')
    if reason:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 7, 'AI Reasoning:', new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('Helvetica', '', 9)
        reason = reason.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, reason)
        pdf.ln(4)

    facts = data.get('facts', [])
    if facts:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 7, 'Key Facts:', new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('Helvetica', '', 9)
        for f in facts[:5]:
            f = f.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 6, f'  - {f}', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 7, 'Summary:', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    rows = [
        ('Detected Language', data.get('detected_lang', 'English')),
        ('Confidence',        f"{data.get('confidence', 0):.1f}%"),
        ('Source Reputation', data.get('source_label', 'N/A')),
        ('Fact-Check Results', str(len(data.get('fact_results', []))) + ' found'),
    ]
    pdf.set_font('Helvetica', '', 9)
    for i, (k, v) in enumerate(rows):
        if i % 2 == 0:
            pdf.set_fill_color(235, 232, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(70, 7, k, border=1, fill=True)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(120, 7, str(v), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, 'FakeNews Detector | Built by Kartik Kumar Tiwari | MCA Final Year | Doranda College, Ranchi', align='C', new_x="LMARGIN", new_y="NEXT")

    buf = io.BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
#  EMAIL VERIFICATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def send_verification_email(email, token):
    verify_url = url_for('verify_email', token=token, _external=True)
    msg = Message(
        subject="✅ Verify your FakeNews Detector account",
        recipients=[email],
        html=f"""
        <div style="font-family:Arial,sans-serif;max-width:500px;margin:auto;padding:30px;
                    border-radius:12px;background:#1a1a2e;color:#fff;">
          <h2 style="color:#6c63ff;">🔍 FakeNews Detector</h2>
          <p>Click the button below to verify your email address:</p>
          <a href="{verify_url}"
             style="display:inline-block;padding:12px 28px;background:#6c63ff;
                    color:#fff;border-radius:8px;text-decoration:none;font-weight:bold;margin:16px 0;">
            Verify My Email
          </a>
          <p style="color:#aaa;font-size:12px;">
            If you didn't register, ignore this email.<br>
            Built by Kartik Kumar Tiwari • MCA Final Year • Doranda College, Ranchi
          </p>
        </div>"""
    )
    try:
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Mail error: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── REGISTER ──────────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            email    = request.form['email']
            password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
            token    = secrets.token_urlsafe(32)
            db  = get_db()
            cur = dict_cursor(db)
            cur.execute(
                "INSERT INTO users (username, email, password, is_verified, verify_token) VALUES (%s,%s,%s,%s,%s)",
                (username, email, password, False, token)
            )
            db.commit()
            db.close()
            send_verification_email(email, token)
            return render_template('login.html',
                                   msg='Registration successful! Check your email to verify your account.')
        except:
            return render_template('register.html', msg='Username or email already exists!')
    return render_template('register.html')

# ── EMAIL VERIFY ──────────────────────────────────────────────────────────────
@app.route('/verify-email/<token>')
def verify_email(token):
    try:
        db  = get_db()
        cur = dict_cursor(db)
        cur.execute("SELECT * FROM users WHERE verify_token = %s", (token,))
        user = cur.fetchone()
        if user:
            cur.execute("UPDATE users SET is_verified=TRUE, verify_token=NULL WHERE id=%s", (user['id'],))
            db.commit()
            db.close()
            return render_template('login.html', msg='✅ Email verified! You can now login.')
        db.close()
        return render_template('login.html', msg='❌ Invalid or expired verification link.')
    except Exception as e:
        return render_template('login.html', msg='Error: ' + str(e))

# ── LOGIN ─────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email    = request.form['email']
            password = request.form['password']
            db  = get_db()
            cur = dict_cursor(db)
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            db.close()
            if user and bcrypt.check_password_hash(user['password'], password):
                if not user.get('is_verified', True):
                    return render_template('login.html',
                                           msg='Please verify your email first. Check your inbox.')
                login_user(User(user['id'], user['username'], user['email'], user.get('is_verified', False)))
                return redirect('/detector')
            return render_template('login.html', msg='Invalid credentials!')
        except Exception as e:
            return render_template('login.html', msg='Error: ' + str(e))
    return render_template('login.html')

# ── GOOGLE LOGIN ──────────────────────────────────────────────────────────────
@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    try:
        token     = google.authorize_access_token()
        user_info = token.get('userinfo') or google.userinfo()
        email     = user_info['email']
        name      = user_info.get('name', email.split('@')[0])
        db  = get_db()
        cur = dict_cursor(db)
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        if not user:
            cur.execute(
                "INSERT INTO users (username, email, password, is_verified) VALUES (%s,%s,%s,%s) RETURNING *",
                (name, email, bcrypt.generate_password_hash('google_oauth_user').decode('utf-8'), True)
            )
            db.commit()
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
        db.close()
        login_user(User(user['id'], user['username'], user['email'], True))
        return redirect('/detector')
    except Exception as e:
        print(f"Google OAuth error: {e}")
        return redirect('/login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')

@app.route('/detector')
@login_required
def detector():
    return render_template('detector.html')

@app.route('/free')
def free_detector():
    return render_template('detector.html')

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DETECT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
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
        prediction        = groq_result['verdict']
        credibility_score = groq_result['credibility_score']
        confidence        = groq_result['confidence']
        gemini_reason     = groq_result['reason']
        gemini_facts      = groq_result['gemini_facts']
        indian_facts_matched = check_indian_facts(english_text)
    else:
        prediction        = model.predict([english_text])[0]
        confidence        = max(model.predict_proba([english_text])[0]) * 100
        credibility_score = int(confidence) if prediction == 'REAL' else int(100 - confidence)
        gemini_reason     = ''
        gemini_facts      = []
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

    fact_results   = check_facts(english_text)
    cricket_scores = get_cricket_scores() if is_cricket_news(text) else []

    share_id = None
    try:
        db  = get_db()
        cur = dict_cursor(db)
        if current_user.is_authenticated:
            cur.execute(
                "INSERT INTO analysis_history (user_id, input_text, verdict, credibility_score) VALUES (%s,%s,%s,%s)",
                (current_user.id, text[:500], prediction, credibility_score)
            )
        cur.execute(
            """INSERT INTO shared_results (input_text, verdict, credibility_score, reason)
               VALUES (%s,%s,%s,%s) RETURNING share_id""",
            (text[:500], prediction, credibility_score, gemini_reason)
        )
        share_id = str(cur.fetchone()['share_id'])
        db.commit()
        db.close()
    except:
        pass

    return jsonify({
        'verdict':           prediction,
        'credibility_score': credibility_score,
        'confidence':        round(confidence, 2),
        'detected_lang':     detected_lang,
        'reputation':        reputation,
        'fact_results':      fact_results,
        'indian_facts':      indian_facts_matched,
        'gemini_reason':     gemini_reason,
        'gemini_facts':      gemini_facts,
        'cricket_scores':    cricket_scores,
        'share_id':          share_id,
        'share_url':         f"/result/{share_id}" if share_id else None,
        'input_text':        text[:300]
    })

# ─────────────────────────────────────────────────────────────────────────────
#  SHAREABLE RESULT PAGE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/result/<share_id>')
def shared_result(share_id):
    try:
        db  = get_db()
        cur = dict_cursor(db)
        cur.execute("SELECT * FROM shared_results WHERE share_id = %s", (share_id,))
        result = cur.fetchone()
        db.close()
        if result:
            return render_template('shared_result.html', result=result)
        return "Result not found", 404
    except:
        return "Error loading result", 500

# ─────────────────────────────────────────────────────────────────────────────
#  ★ NEW: USER FEEDBACK (Thumbs Up / Down)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Save user feedback on detection result."""
    data     = request.get_json()
    feedback = data.get('feedback')        # 'helpful' or 'not_helpful'
    share_id = data.get('share_id')
    verdict  = data.get('verdict')
    score    = data.get('credibility_score')
    text     = data.get('input_text', '')[:300]

    if feedback not in ('helpful', 'not_helpful'):
        return jsonify({'error': 'Invalid feedback type'}), 400

    try:
        db      = get_db()
        cur     = dict_cursor(db)
        user_id = current_user.id if current_user.is_authenticated else None
        cur.execute(
            """INSERT INTO detection_feedback
               (user_id, share_id, feedback, verdict, credibility_score, input_text)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (user_id, share_id, feedback, verdict, score, text)
        )
        db.commit()
        db.close()
        return jsonify({'status': 'ok', 'message': 'Thank you for your feedback!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
#  PDF DOWNLOAD ROUTE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    data = request.get_json()
    pdf_bytes = generate_pdf_report(data)
    response = make_response(pdf_bytes)
    response.headers['Content-Type']        = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=fakenews_report.pdf'
    return response

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE / VIDEO DETECTION ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/detect-image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded!'})
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected!'})
    allowed = {'png','jpg','jpeg','gif','webp','bmp'}
    ext = image_file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Invalid image format!'})
    sightengine_result = analyze_image_sightengine(image_file)
    final_verdict  = 'UNKNOWN'
    overall_score  = 50
    if sightengine_result:
        final_verdict = sightengine_result['verdict']
        overall_score = sightengine_result['ai_generated_score']
    return jsonify({'type':'image','sightengine':sightengine_result,'final_verdict':final_verdict,'overall_score':overall_score})

@app.route('/detect-video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded!'})
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected!'})
    allowed = {'mp4','avi','mov','mkv','webm','flv'}
    ext = video_file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Invalid video format!'})
    video_file.seek(0, 2)
    size = video_file.tell()
    video_file.seek(0)
    if size > 50 * 1024 * 1024:
        return jsonify({'error': 'Video too large! Max 50MB'})
    sightengine_result = analyze_video_sightengine(video_file)
    if not sightengine_result:
        return jsonify({'error': 'Video analysis failed!'})
    return jsonify({'type':'video','sightengine':sightengine_result,'final_verdict':sightengine_result['verdict'],'overall_score':sightengine_result['avg_deepfake_score']})

# ─────────────────────────────────────────────────────────────────────────────
#  HISTORY
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    try:
        db  = get_db()
        cur = dict_cursor(db)
        cur.execute(
            "SELECT input_text, verdict, credibility_score, created_at FROM analysis_history WHERE user_id=%s ORDER BY created_at DESC",
            (current_user.id,)
        )
        records = cur.fetchall()
        db.close()
    except:
        records = []
    return render_template('history.html', records=records)

# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/admin')
def admin():
    try:
        db  = get_db()
        cur = db.cursor()
        cur.execute("SELECT COUNT(*) FROM users");          total_users    = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM analysis_history"); total_analyses = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM analysis_history WHERE verdict='FAKE'"); total_fake = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM analysis_history WHERE verdict='REAL'"); total_real = cur.fetchone()[0]

        # Feedback counts
        cur.execute("SELECT COUNT(*) FROM detection_feedback WHERE feedback='helpful'")
        total_helpful = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM detection_feedback WHERE feedback='not_helpful'")
        total_not_helpful = cur.fetchone()[0]

        # Dynamic facts count
        try:
            cur.execute("SELECT COUNT(*) FROM dynamic_facts WHERE active=TRUE")
            total_dynamic_facts = cur.fetchone()[0]
        except:
            total_dynamic_facts = 0

        cur2 = dict_cursor(db)
        cur2.execute("SELECT username, email, created_at FROM users ORDER BY created_at DESC")
        all_users = cur2.fetchall()
        cur2.execute("""
            SELECT u.username, a.input_text, a.verdict, a.credibility_score, a.created_at
            FROM analysis_history a JOIN users u ON a.user_id = u.id
            ORDER BY a.created_at DESC LIMIT 20
        """)
        recent_analyses = cur2.fetchall()

        # Recent dynamic facts
        try:
            cur2.execute("SELECT keyword, fact, source, updated_at FROM dynamic_facts ORDER BY updated_at DESC LIMIT 10")
            recent_facts = cur2.fetchall()
        except:
            recent_facts = []

        db.close()
    except Exception as e:
        return str(e)

    return render_template('admin.html',
        total_users=total_users, total_analyses=total_analyses,
        total_fake=total_fake, total_real=total_real,
        total_helpful=total_helpful, total_not_helpful=total_not_helpful,
        total_dynamic_facts=total_dynamic_facts,
        all_users=all_users, recent_analyses=recent_analyses,
        recent_facts=recent_facts)

# ── Admin Analytics API ───────────────────────────────────────────────────────
@app.route('/admin/analytics')
@login_required
def admin_analytics():
    try:
        db  = get_db()
        cur = dict_cursor(db)

        cur.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as count
            FROM analysis_history
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY day ORDER BY day
        """)
        daily = cur.fetchall()

        cur.execute("""
            SELECT verdict, COUNT(*) as count
            FROM analysis_history GROUP BY verdict
        """)
        ratio = cur.fetchall()

        cur.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as count
            FROM users
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY day ORDER BY day
        """)
        signups = cur.fetchall()

        cur.execute("""
            SELECT DATE(created_at) as day, ROUND(AVG(credibility_score),1) as avg_score
            FROM analysis_history
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY day ORDER BY day
        """)
        avg_scores = cur.fetchall()

        # ★ NEW: Feedback stats
        try:
            cur.execute("""
                SELECT feedback, COUNT(*) as count
                FROM detection_feedback GROUP BY feedback
            """)
            feedback_stats = cur.fetchall()
        except:
            feedback_stats = []

        # ★ NEW: Dynamic facts count
        try:
            cur.execute("SELECT COUNT(*) as count FROM dynamic_facts WHERE active = TRUE")
            facts_row   = cur.fetchone()
            facts_count = facts_row['count'] if facts_row else 0
        except:
            facts_count = 0

        db.close()
        return jsonify({
            'daily':      [{'day': str(r['day']), 'count': r['count']} for r in daily],
            'ratio':      [{'verdict': r['verdict'], 'count': r['count']} for r in ratio],
            'signups':    [{'day': str(r['day']), 'count': r['count']} for r in signups],
            'avg_scores': [{'day': str(r['day']), 'avg_score': float(r['avg_score'])} for r in avg_scores],
            'feedback':   [{'type': r['feedback'], 'count': r['count']} for r in feedback_stats],
            'facts_count': facts_count,
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ─────────────────────────────────────────────────────────────────────────────
#  ★ NEW: ADMIN — UPDATE CURRENT AFFAIRS
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/admin/update-facts', methods=['POST'])
def admin_update_facts():
    """
    Admin button: scrape PIB + Wikipedia, process with Groq,
    store in dynamic_facts table.
    """
    try:
        db = get_db()
        added, error = fetch_and_store_current_affairs(groq_client, db)
        db.close()
        if error:
            return jsonify({'status': 'error', 'message': error})
        return jsonify({
            'status':  'ok',
            'added':   added,
            'message': f'✅ {added} current affairs updated successfully!'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ─────────────────────────────────────────────────────────────────────────────
#  AI CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    data    = request.get_json()
    message = data.get('message', '').strip()
    history = data.get('history', [])

    if not message:
        return jsonify({'reply': 'Kuch toh likhiye! 😊'})

    system_prompt = """You are FakeBot — the friendly AI assistant of FakeNews Detector.
You help users:
1. Understand if news is fake or real
2. Learn media literacy tips
3. Navigate the website features
4. Understand AI/deepfake concepts

Rules:
- Reply in the same language the user writes (Hindi or English)
- Keep replies short and helpful (2-4 sentences max unless asked for detail)
- Be friendly and use emojis occasionally
- If asked to check specific news, give a brief analysis
- Do NOT make up URLs or statistics"""

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-10:]:
        messages.append({"role": turn['role'], "content": turn['content']})
    messages.append({"role": "user", "content": message})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = "Sorry, abhi server busy hai. Thodi der baad try karein. 🙏"
        print(f"Chatbot error: {e}")

    if current_user.is_authenticated:
        try:
            db  = get_db()
            cur = dict_cursor(db)
            cur.execute("INSERT INTO chatbot_history (user_id, role, content) VALUES (%s,'user',%s)",
                        (current_user.id, message))
            cur.execute("INSERT INTO chatbot_history (user_id, role, content) VALUES (%s,'assistant',%s)",
                        (current_user.id, reply))
            db.commit()
            db.close()
        except:
            pass

    return jsonify({'reply': reply})

# ─────────────────────────────────────────────────────────────────────────────
#  WHATSAPP BOT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/webhook/whatsapp', methods=['GET'])
def whatsapp_verify():
    mode      = request.args.get('hub.mode')
    token     = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    if mode == 'subscribe' and token == WHATSAPP_VERIFY:
        return challenge, 200
    return 'Forbidden', 403

@app.route('/webhook/whatsapp', methods=['POST'])
def whatsapp_webhook():
    data = request.get_json(silent=True)
    try:
        entry   = data['entry'][0]
        changes = entry['changes'][0]
        value   = changes['value']
        msg     = value['messages'][0]
        from_   = msg['from']
        body    = msg['text']['body']

        english_text, _ = translate_to_english(body)
        result = analyze_with_groq(english_text)

        if result:
            verdict = result['verdict']
            score   = result['credibility_score']
            reason  = result['reason']
            icon    = '✅' if verdict == 'REAL' else '❌'
            reply_text = (
                f"{icon} *{verdict} NEWS*\n"
                f"📊 Credibility Score: *{score}/100*\n\n"
                f"💡 {reason}\n\n"
                f"_Powered by FakeNews Detector 🔍_\n"
                f"_Visit: https://fakenewsdetector-aew9.onrender.com_"
            )
        else:
            reply_text = "❓ Analysis failed. Please try again or visit our website."

        requests.post(
            f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"},
            json={"messaging_product": "whatsapp", "to": from_, "type": "text", "text": {"body": reply_text}},
            timeout=10
        )
    except Exception as e:
        print(f"WhatsApp webhook error: {e}")

    return jsonify({'status': 'ok'})

# ─────────────────────────────────────────────────────────────────────────────
#  VOICE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/voice', methods=['POST'])
def voice_assistant():
    data         = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"reply": "Kuch suna nahi, dobara boliye"})
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Tum ek Fake News Detector assistant ho. User jo bhi news bolega, tum analyze karke batao ki wo fake hai ya real. Chhota aur clear jawab do Hindi mein."},
            {"role": "user", "content": user_message}
        ],
        max_tokens=150
    )
    reply = response.choices[0].message.content
    return jsonify({"reply": reply})

@app.route('/voice')
def voice_page():
    return render_template('voice.html')

# ─────────────────────────────────────────────────────────────────────────────
#  HINDI UI – language preference API
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/set-language', methods=['POST'])
def set_language():
    lang = request.get_json().get('lang', 'en')
    session['ui_lang'] = lang
    return jsonify({'status': 'ok', 'lang': lang})

@app.route('/get-translations')
def get_translations():
    lang = request.args.get('lang', session.get('ui_lang', 'en'))
    translations = {
        'en': {
            'title': 'FakeNews Detector',
            'detect_btn': 'Detect Now',
            'paste_placeholder': 'Paste news text here…',
            'url_placeholder': 'Or paste URL here…',
            'verdict_real': 'REAL NEWS',
            'verdict_fake': 'FAKE NEWS',
            'credibility': 'Credibility Score',
            'download_pdf': 'Download PDF Report',
            'share': 'Share Result',
            'history': 'My History',
            'logout': 'Logout',
            'chatbot': 'Ask FakeBot',
        },
        'hi': {
            'title': 'फेक न्यूज़ डिटेक्टर',
            'detect_btn': 'अभी जांचें',
            'paste_placeholder': 'यहाँ खबर का टेक्स्ट पेस्ट करें…',
            'url_placeholder': 'या यहाँ URL पेस्ट करें…',
            'verdict_real': 'सच्ची खबर ✅',
            'verdict_fake': 'झूठी खबर ❌',
            'credibility': 'विश्वसनीयता स्कोर',
            'download_pdf': 'PDF रिपोर्ट डाउनलोड करें',
            'share': 'परिणाम शेयर करें',
            'history': 'मेरा इतिहास',
            'logout': 'लॉग आउट',
            'chatbot': 'फेकबॉट से पूछें',
        }
    }
    return jsonify(translations.get(lang, translations['en']))

# ─────────────────────────────────────────────────────────────────────────────
#  BROWSER EXTENSION
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/extension/manifest.json')
def extension_manifest():
    manifest = {
        "manifest_version": 3,
        "name": "FakeNews Detector",
        "version": "1.0",
        "description": "Check any news for fake/real with one click!",
        "permissions": ["contextMenus", "activeTab", "scripting"],
        "host_permissions": ["https://fakenewsdetector-aew9.onrender.com/*"],
        "background": {"service_worker": "background.js"},
        "action": {
            "default_popup": "popup.html",
            "default_icon": {"16": "icon16.png", "48": "icon48.png"}
        },
        "icons": {"16": "icon16.png", "48": "icon48.png"}
    }
    return jsonify(manifest)

# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP — runs on both gunicorn and direct python
# ─────────────────────────────────────────────────────────────────────────────
with app.app_context():
    init_db_extras()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)

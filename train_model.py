"""
train_model.py — Upgraded ML Model with Hindi/Hinglish Support
==============================================================
Author : Kartik Kumar Tiwari | MCA Final Year | Doranda College, Ranchi
Model  : TF-IDF + Logistic Regression (multilingual)
Dataset: ISOT (English) + Hindi/Hinglish synthetic + Indian news patterns

Run:
    python train_model.py

Output:
    model.pkl  — drop-in replacement for existing model.pkl
"""

import os, re, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Universal cleaner for English, Hindi (Devanagari), and Hinglish text.
    Keeps Devanagari Unicode range so Hindi words are preserved.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Keep: English letters, Devanagari (Hindi), digits, spaces
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — HINDI / HINGLISH SYNTHETIC DATASET
#  (covers common WhatsApp forwards, political rumours, health myths in India)
# ─────────────────────────────────────────────────────────────────────────────

HINDI_REAL = [
    # Politics — verified facts
    "narendra modi bharat ke pradhan mantri hain",
    "droupadi murmu bharat ki rashtrapati hain",
    "bharat mein 28 rajya aur 8 kendra shasit pradesh hain",
    "lok sabha mein 543 seaten hain",
    "bharat ka samvidhan 26 january 1950 ko lagu hua tha",
    "dr br ambedkar ne bharat ka samvidhan banaya tha",
    "independence day 15 august ko manaya jata hai",
    "republic day 26 january ko manaya jata hai",
    "chandrayaan 3 ne 23 august 2023 ko chand ke dakshin dhruv par safaltapurvak landing ki",
    "india ne t20 world cup 2024 jita south africa ko final mein haraya",
    "rcb ne ipl 2025 jita",
    "kkr ne ipl 2024 jita",
    "neeraj chopra ne paris olympics 2024 mein silver medal jita",
    "operation sindoor bharat ki military operation thi 2025 mein",
    "ram mandir ayodhya mein 22 january 2024 ko pratishthit hua",
    "gst 1 july 2017 ko lagu hua",
    "article 370 5 august 2019 ko hataya gaya",
    "bharat ki jansankhya lagbhag 1 arab 44 crore hai",
    "rbi ke governor sanjay malhotra hain december 2024 se",
    "india gdp duniya ki 5vi sabse badi economy hai",
    "pm kisan yojana mein kisanon ko 6000 rupye milte hain saal mein",
    "ayushman bharat yojana mein 5 lakh tak ka ilaj muft milta hai",
    "india ne 5g october 2022 mein launch kiya",
    "bharat mein 23 iit hain",
    "virat kohli ne test cricket se retirement li 2024 mein",
    "rohit sharma ne t20 international se sanyas liya after t20 wc 2024",
    "mahatma gandhi ki hatya 30 january 1948 ko hui thi",
    "jawaharlal nehru bharat ke pehle pradhan mantri the",
    "sardar vallabhbhai patel ko loh purush kaha jata hai",
    "bharat ka rashtriya pashu bagh sher hai",
    "bharat ka rashtriya pakshi mor hai",
    "jana gana mana bharat ka rashtriya gaan hai",
    "rajasthan bharat ka sabse bada rajya hai kshetra ke hisab se",
    "goa bharat ka sabse chhota rajya hai",
    "ranchi jharkhand ki rajdhani hai",
    "patna bihar ki rajdhani hai",
    "lucknow uttar pradesh ki rajdhani hai",
    "bhopal madhya pradesh ki rajdhani hai",
    "jaipur rajasthan ki rajdhani hai",
    "amit shah home minister hain 2019 se",
    "nirmala sitharaman finance minister hain",
    "isro ka mukhyalay bengaluru mein hai",
    "upi se har mahine 15 arab se adhik transactions hote hain",
    "bharat ratna india ka sabse bada nagrik samman hai",
    "bns ipc ki jagah 1 july 2024 se lagu hua",
    "india ne chandrayaan 3 se chand ke south pole par pahla desh bana",
    "aditya l1 india ka pehla solar mission hai",
    "brahmos india aur russia ki sanyukt missile hai",
    "ins vikrant india ka swadeshi aircraft carrier hai",
    "india ka area lagbhag 32 lakh 87 hazar varg kilometre hai",
]

HINDI_FAKE = [
    # Common WhatsApp fake forwards in Hindi
    "modi ne desh chhod diya aur pakistan chale gaye",
    "rahul gandhi arrested for treason by cbi last night",
    "india china war shuru ho gayi hai aaj raat",
    "free recharge milega sabko sarkar ne ghoshana ki hai",
    "whatsapp band ho raha hai india mein next week",
    "pm modi ne free laptop dene ka plan banaya hai sabhi students ke liye",
    "5g towers se corona virus failta hai scientific proof ke saath",
    "bill gates ne india mein chips lagaye corona vaccine mein",
    "haldi aur doodh pine se cancer 100 percent theek hota hai",
    "onion in pocket cures all diseases scientific fact",
    "neem leaves cure diabetes completely in 3 days guaranteed",
    "government ne sabka bank account freeze karne ka order diya",
    "rupee value zero hone wali hai next month rbi ne bataya",
    "petrol free hone wala hai india mein modi sarkar ka faisla",
    "isro ne alien contact confirm kiya secret file leak",
    "india pakistan nuclear war shuru ho gayi breaking news",
    "army chief ne coup kiya hai delhi mein sena ka kabza",
    "modi ji ne resign kar diya puri cabinet ke saath",
    "bjp sarkar ne 10 lakh crore ka ghota kiya taxpayers ka paisa",
    "congress ne desh bech diya china ko secret deal mein",
    "virat kohli retire ho gaye aaj is team se forever",
    "ipl fix hota hai sare players ko pehle se pata hota hai kaun jitega",
    "neeraj chopra ne doping test fail kiya medal vapas hoga",
    "chandrayaan 3 asli nahi tha studio mein bana fake video tha",
    "isro scientists ne resign kar diya mass protest ke baad",
    "india ki gdp actually negative hai government chupa rahi hai",
    "rbi ne rupee print karna band kar diya gold standard laega",
    "aadhaar card se government sabka phone sun rahi hai spy karne ke liye",
    "covid vaccine se 5 saal mein maut ho jaegi proven research",
    "electricity free hogi sabke liye next month se sarkar ka aadesh",
    "sarkar sabka savings account band kar degi new law ke tahat",
    "india ab america ka colony ban gaya secret treaty ke baad",
    "modi ji ki net worth trillion dollars hai swiss bank mein",
    "supreme court ne election results cancel kar diye secret order",
    "india china border par nuclear bomb gira china ne",
    "pakistan ne india ka ek shehar capture kar liya",
    "america ne india par sanctions laga diye new order se",
    "rbi governor ne resign kiya modi se ladai ke baad",
    "farmers protest mein 10000 log mare gaye government ne chhupaaya",
    "whatsapp aur facebook india mein permanently banned ho rahe hain",
    "jio recharge free milega sabko sarkari scheme mein register karein",
    "aapka phone hack ho gaya hai is number se call aaya to mat uthao",
    "india ne un se resign kar liya koi nahi bataa raha",
    "modi ne apni beti ki shaadi secret mein karvai crores kharch karke",
    "rahul gandhi pakka PM banega court ke order se election hoga phir",
    "china ne india ke 5 state le liye koi rok nahi paya",
    "sarkar ne internet band karne ka plan banaya hai 2025 mein",
    "army ne delhi mein curfew laga diya khabar dabai ja rahi hai",
    "petrol 10 rupye litre ho jaega sarkar ka bada faisla aane wala hai",
    "free ration band ho raha hai modi ji ne cancel kar diya",
]

HINGLISH_REAL = [
    # Hinglish (mixed Hindi-English) — real facts
    "modi ji ne operation sindoor ka order diya 2025 mein",
    "rcb ne ipl 2025 ka title jeeta bangalore mein",
    "india ki economy 5th largest hai world mein",
    "chandrayaan successfully land kiya moon ke south pole par",
    "neeraj chopra ne paris olympics mein silver medal jeeta",
    "t20 world cup india ne jeeta 2024 mein barbados mein",
    "ram mandir inauguration january 2024 mein hua ayodhya mein",
    "gst implementation 2017 mein hui thi india mein",
    "article 370 remove hua jammu kashmir se 2019 mein",
    "bharat ratna lk advani ko mila 2024 mein",
    "india ne paris olympics 2024 mein 6 medals jeete",
    "upi payments india mein bahut popular ho gayi hain",
    "5g india mein 500 plus cities mein available hai",
    "isro ka aditya l1 mission sun ko study kar raha hai",
    "pm kisan yojana se crores farmers ko paisa milta hai",
    "ayushman bharat se garib logo ko 5 lakh tak health cover milta hai",
    "india population wise duniya mein number one hai",
    "startup india 2016 mein launch hua tha pm modi ne",
    "digital india initiative se internet access badha hai",
    "ins vikrant india ka pehla indigenous aircraft carrier hai",
    "india mein 28 states hain aur 8 union territories hain",
    "supreme court of india ka chief justice sanjiv khanna hain",
    "lok sabha speaker om birla hain 2019 se",
    "india ka national flower lotus hai",
    "india ka national animal tiger hai bengal tiger",
    "india ne moon pe pahle country ban gayi south pole touch karne wali",
]

HINGLISH_FAKE = [
    # Hinglish fake forwards
    "modi ji ka account hack ho gaya aur unka secret message leak hua",
    "free mein iphone milega sarkar ki taraf se register karo abhi",
    "whatsapp new feature aaya hai agar forward nahi kiya to account delete hoga",
    "india mein alien spaceship land ki hai army ne chhupaaya",
    "corona virus 5g towers se create hua hai proven science",
    "vaccine mein chip laga hai government track karti hai sab ko",
    "india china war officially start ho gayi hai breaking news abhi",
    "rupee zero hone wala hai dollar ke muqabale mein next week",
    "free bijli milegi sabko modi government ka naya plan",
    "kkr ne ipl fix kiya tha 2024 mein proof saamne aaya",
    "virat kohli aur anushka ka divorce ho gaya secret mein",
    "modi ji ne pakistan ko nuclear bomb se dhamki di war possible",
    "rahul gandhi ne country ko secret mein sell karne ki koshish ki",
    "isro ka chandrayaan video fake tha studio mein shoot kiya",
    "india pakistan ke 10 sheher capture kar liye koi nahi bata raha",
    "government sabka data bech rahi hai companies ko secretly",
    "neem ka pani peene se cancer theek ho jaata hai guaranteed",
    "onion pocket mein rakhne se corona nahi hoga proven method",
    "haldi doodh se sugar diabetes permanently cure ho jaati hai",
    "america ne india ko secretly threaten kiya war ki",
    "modi ki real age 90 saal hai chhupaaya ja raha hai",
    "ipl teams sab ek hi malik ke hain fix match hota hai always",
    "india ka asli gdp bahut kam hai government jhooth bol rahi hai",
    "rbi ke paas gold khatam ho gaya hai secret news",
    "free recharge 84 din ka milega jio airtel ko sarkaar ka order",
]

def build_hindi_hinglish_dataset():
    """Combine all Hindi/Hinglish samples into a DataFrame."""
    texts  = HINDI_REAL + HINGLISH_REAL + HINDI_FAKE + HINGLISH_FAKE
    labels = (['REAL'] * (len(HINDI_REAL) + len(HINGLISH_REAL)) +
              ['FAKE'] * (len(HINDI_FAKE) + len(HINGLISH_FAKE)))
    df = pd.DataFrame({'text': texts, 'label': labels})
    # Augment: duplicate & slightly vary
    aug = df.copy()
    aug['text'] = aug['text'].apply(lambda t: t + ' ' + t.split()[0] if len(t.split()) > 3 else t)
    return pd.concat([df, aug], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — ISOT DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_isot_dataset(data_dir: str = 'Data') -> pd.DataFrame:
    """
    Load ISOT dataset from CSV files.
    Expected files: Data/True.csv, Data/Fake.csv
    Columns expected: title, text
    """
    frames = []
    for filename, label in [('True.csv', 'REAL'), ('Fake.csv', 'FAKE')]:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"  ⚠️  {path} not found — skipping")
            continue
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        # Use title + text if both available, else whichever exists
        if 'title' in df.columns and 'text' in df.columns:
            df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'text' in df.columns:
            df['combined'] = df['text'].fillna('')
        elif 'title' in df.columns:
            df['combined'] = df['title'].fillna('')
        else:
            print(f"  ⚠️  No usable columns in {filename} — skipping")
            continue
        df = df[['combined']].rename(columns={'combined': 'text'})
        df['label'] = label
        # Use up to 12,000 rows per file to keep training fast
        frames.append(df.head(12000))
        print(f"  ✅ Loaded {min(len(df), 12000):,} rows from {filename} [{label}]")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=['text', 'label'])


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — EXTRA INDIAN ENGLISH PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

INDIAN_REAL_EN = [
    "India's Chandrayaan-3 successfully landed on the Moon's south pole on August 23 2023",
    "Prime Minister Narendra Modi inaugurated Ram Mandir in Ayodhya on January 22 2024",
    "India won the ICC T20 World Cup 2024 defeating South Africa in the final",
    "Royal Challengers Bengaluru won IPL 2025 title",
    "Kolkata Knight Riders won IPL 2024 defeating Sunrisers Hyderabad",
    "Neeraj Chopra won a silver medal at Paris Olympics 2024 in javelin throw",
    "India won 6 medals at Paris Olympics 2024 including one silver and five bronze",
    "Operation Sindoor was launched by India in May 2025 against terrorist camps",
    "India's GDP is the fifth largest in the world at approximately 3.9 trillion dollars",
    "Article 370 was abrogated on August 5 2019 removing special status of Jammu Kashmir",
    "GST was implemented in India on July 1 2017",
    "Demonetization was announced on November 8 2016 scrapping 500 and 1000 rupee notes",
    "India has 28 states and 8 Union Territories",
    "ISRO's Aditya-L1 is India's first solar mission launched in September 2023",
    "India became the first country to land near the Moon's south pole",
    "Droupadi Murmu is the 15th President of India and first tribal woman president",
    "The Bharat Ratna 2024 was awarded to LK Advani among others",
    "India's UPI processes over 15 billion transactions per month",
    "India launched 5G services in October 2022",
    "INS Vikrant is India's first indigenously built aircraft carrier commissioned in 2022",
    "BNS replaced IPC from July 1 2024 as India's new criminal law",
    "India's population crossed 1.44 billion making it the most populous country",
    "PM Kisan Samman Nidhi provides 6000 rupees annually to farmers",
    "Ayushman Bharat provides health cover of 5 lakh rupees per family to poor families",
    "India has 23 IITs as of 2024",
    "Virat Kohli retired from Test cricket in 2024",
    "Rohit Sharma retired from T20 internationals after winning T20 World Cup 2024",
    "Sanjiv Khanna is the Chief Justice of India since November 2024",
    "Sanjay Malhotra is the RBI Governor since December 2024",
    "Omar Abdullah is the Chief Minister of Jammu Kashmir since October 2024",
]

INDIAN_FAKE_EN = [
    "India and China have started a nuclear war breaking news",
    "PM Modi has fled the country and gone to Switzerland secret sources",
    "Free recharge for all Indians government new scheme register now",
    "WhatsApp will be permanently banned in India next week government order",
    "5G towers are spreading coronavirus proven by secret government research",
    "Bill Gates microchipped Indians through COVID vaccines secret document leaked",
    "India has secretly sold three states to China bilateral agreement",
    "RBI has run out of gold reserves secret crisis hidden from public",
    "Chandrayaan-3 landing video was filmed in a studio fake moon landing",
    "ISRO scientists have all resigned in protest against government",
    "India's real GDP is negative government hiding data from public",
    "Modi government has ordered all bank accounts to be frozen next month",
    "Petrol will be free for all Indians from next month government announcement",
    "Army has taken over Delhi government in secret military coup",
    "America has imposed sanctions on India secret diplomatic crisis",
    "Virat Kohli tested positive for doping medal to be taken back",
    "IPL matches are all fixed owners decide result before season starts",
    "India has officially declared war on Pakistan nuclear strike imminent",
    "Supreme Court has secretly cancelled 2024 election results new election coming",
    "Rahul Gandhi arrested for anti-national activities by NIA last night",
    "Modi's real age is 90 being hidden from public records",
    "India has left the United Nations secret letter sent",
    "Neeraj Chopra's Paris medal revoked due to nationality issues",
    "COVID vaccine causes death within five years proven international study",
    "Onion kept in pocket prevents all viral diseases scientific fact",
    "Turmeric milk cures cancer completely in 30 days guaranteed natural remedy",
    "Neem water permanently cures diabetes no medicine needed proven",
    "Aliens landed in Rajasthan army hiding the spaceship from public",
    "Government selling citizen data to foreign companies secret deal exposed",
    "Free iPhone being given by Modi government to all citizens register now",
]

def build_indian_english_dataset():
    texts  = INDIAN_REAL_EN + INDIAN_FAKE_EN
    labels = (['REAL'] * len(INDIAN_REAL_EN) + ['FAKE'] * len(INDIAN_FAKE_EN))
    df = pd.DataFrame({'text': texts, 'label': labels})
    # Augment x3 with slight variations
    aug_frames = [df]
    for _ in range(2):
        aug = df.copy()
        aug['text'] = aug['text'].apply(
            lambda t: ' '.join(t.split() + t.split()[:2]) if len(t.split()) > 5 else t
        )
        aug_frames.append(aug)
    return pd.concat(aug_frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — BUILD PIPELINE & TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline():
    """
    TF-IDF with char + word n-grams for multilingual support.
    Char n-grams help with Hindi/Hinglish words that aren't in English vocab.
    """
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',           # character n-grams → handles Hindi/Hinglish
        ngram_range=(2, 5),
        max_features=150_000,
        sublinear_tf=True,
        min_df=1,
        strip_accents=None,           # keep Devanagari accents
        lowercase=True,
    )
    # We'll combine with word-level TF-IDF via FeatureUnion
    from sklearn.pipeline import FeatureUnion
    from sklearn.base import BaseEstimator, TransformerMixin

    class TextSelector(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X): return X

    word_vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        min_df=2,
        strip_accents=None,
        lowercase=True,
    )

    combined = FeatureUnion([
        ('char', vectorizer),
        ('word', word_vec),
    ])

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ('tfidf', combined),
        ('clf',   clf),
    ])


def train():
    print("\n" + "="*60)
    print("  FakeNews Detector — Model Training (Multilingual)")
    print("  Hindi + Hinglish + English + ISOT Dataset")
    print("="*60 + "\n")

    # ── 1. Load ISOT English dataset ──────────────────────────────────────────
    print("📂 Loading ISOT dataset...")
    df_isot = load_isot_dataset('Data')

    # ── 2. Hindi/Hinglish synthetic data ──────────────────────────────────────
    print("🇮🇳 Loading Hindi/Hinglish dataset...")
    df_hindi = build_hindi_hinglish_dataset()
    print(f"  ✅ {len(df_hindi):,} Hindi/Hinglish samples")

    # ── 3. Indian English patterns ─────────────────────────────────────────────
    print("📰 Loading Indian English patterns...")
    df_indian = build_indian_english_dataset()
    print(f"  ✅ {len(df_indian):,} Indian English samples")

    # ── 4. Combine all datasets ───────────────────────────────────────────────
    frames = [df_hindi, df_indian]
    if not df_isot.empty:
        frames.insert(0, df_isot)
    df = pd.concat(frames, ignore_index=True)
    df = shuffle(df, random_state=42).reset_index(drop=True)

    print(f"\n📊 Total dataset: {len(df):,} samples")
    print(f"   REAL: {(df.label=='REAL').sum():,}")
    print(f"   FAKE: {(df.label=='FAKE').sum():,}")

    # ── 5. Clean text ─────────────────────────────────────────────────────────
    print("\n🧹 Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 10].reset_index(drop=True)
    print(f"   After cleaning: {len(df):,} samples")

    # ── 6. Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"\n✂️  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 7. Build & train pipeline ─────────────────────────────────────────────
    print("\n🏋️  Training model (this may take 2-5 minutes)...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    print("\n📈 Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])
    print(f"           REAL  FAKE")
    print(f"  REAL  :  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  FAKE  :  {cm[1][0]:5d}  {cm[1][1]:5d}")

    # ── 9. Quick Hindi/Hinglish sanity check ─────────────────────────────────
    print("\n🇮🇳 Hindi/Hinglish quick test:")
    test_cases = [
        ("modi ne bharat chhod diya pakistan gaye", "FAKE"),
        ("chandrayaan 3 ne chand par landing ki", "REAL"),
        ("free recharge milega sarkar ki taraf se", "FAKE"),
        ("india ne t20 world cup 2024 jeeta", "REAL"),
        ("rcb ne ipl 2025 jita", "REAL"),
        ("vaccine mein chip laga hai government spy karti hai", "FAKE"),
        ("operation sindoor india ki military operation thi 2025 mein", "REAL"),
        ("india china war nuclear bomb gira", "FAKE"),
    ]
    passed = 0
    for text, expected in test_cases:
        cleaned  = clean_text(text)
        pred     = pipeline.predict([cleaned])[0]
        proba    = pipeline.predict_proba([cleaned])[0]
        conf     = max(proba) * 100
        status   = "✅" if pred == expected else "❌"
        if pred == expected: passed += 1
        print(f"  {status} [{pred:4s} {conf:5.1f}%] {text[:55]}")
    print(f"\n  Passed: {passed}/{len(test_cases)}")

    # ── 10. Save model ────────────────────────────────────────────────────────
    model_path = 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f, protocol=4)

    size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"\n💾 Model saved → {model_path}  ({size_mb:.1f} MB)")
    print("\n✅ Training complete!")
    print("   Replace the old model.pkl in your project root with this file.")
    print("   No changes needed in app.py — the pipeline interface is identical.")
    print("="*60 + "\n")

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — PREDICT HELPER (for app.py compatibility test)
# ─────────────────────────────────────────────────────────────────────────────

def predict(pipeline, text: str):
    """Same interface as existing model.predict() used in app.py"""
    cleaned = clean_text(text)
    pred    = pipeline.predict([cleaned])[0]
    proba   = pipeline.predict_proba([cleaned])
    conf    = float(max(proba[0])) * 100
    return pred, conf


if __name__ == '__main__':
    trained_model = train()

    # Final demo
    print("🎯 Demo predictions:\n")
    demos = [
        "Narendra Modi is the Prime Minister of India",
        "India won T20 World Cup 2024 in Barbados",
        "modi ne desh chhod diya aaj raat secret sources",
        "free iphone milega sarkar ki taraf se register karo",
        "RCB won IPL 2025 championship",
        "5g towers corona failate hain scientific proof",
        "chandrayaan 3 successfully landed on moon south pole",
        "india pakistan nuclear war shuru ho gayi hai",
    ]
    for text in demos:
        pred, conf = predict(trained_model, text)
        icon = "✅" if pred == "REAL" else "❌"
        print(f"  {icon} {pred} ({conf:.1f}%) — {text[:60]}")

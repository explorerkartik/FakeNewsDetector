"""
train_model.py — Upgraded ML Model v2 (Bigger Data + Real Augmentation + CV)
=============================================================================
Author : Kartik Kumar Tiwari | MCA Final Year | Doranda College, Ranchi
Model  : TF-IDF (char + word n-grams) + Logistic Regression (multilingual)
Dataset: ISOT (English) + Hindi/Hinglish synthetic (expanded) +
         Indian news patterns (2024-26 updated) + paraphrase augmentation

What changed vs v1:
  1. ~3-4x more Hindi/Hinglish/Indian-English synthetic examples
  2. Real paraphrase-style augmentation (word shuffle / synonym swap /
     punctuation & filler variation) instead of "repeat first word"
  3. Newer Indian topics added (2025-26 events)
  4. 5-fold cross-validation reported, not just one train/test split
  5. Per-class precision/recall + confusion matrix printed clearly

Run:
    python train_model.py

Output:
    model.pkl — drop-in replacement for existing model.pkl
"""

import os
import re
import pickle
import random
import warnings

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Universal cleaner for English, Hindi (Devanagari), and Hinglish text."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # Keep: English letters, Devanagari (Hindi), digits, spaces
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — REAL PARAPHRASE-STYLE AUGMENTATION
# (replaces the old "repeat first word" trick with actual variation)
# ─────────────────────────────────────────────────────────────────────────────

# Small bank of natural filler/connector words people use in
# Hindi/Hinglish/English WhatsApp-style writing — inserted/removed to vary
# sentence shape without changing meaning or label.
HINGLISH_FILLERS = [
    "bhai", "yaar", "dekho", "suno", "abhi abhi", "abhi", "aaj",
    "sach mein", "pakka", "vaise", "actually", "by the way", "fact",
    "breaking", "update", "news hai ki",
]

SENTENCE_ENDERS = [
    "", " sach hai ye", " ye sach hai", " confirm hai", " pata chala hai",
    " bataya gaya hai", " khabar hai", " news aayi hai",
]


def synonym_light_swap(text: str) -> str:
    """Swap a few very common words with natural equivalents (label-safe)."""
    swaps = {
        " hai ": [" hai ", " hota hai ", " hai bilkul "],
        " kiya ": [" kiya ", " kar diya ", " kar di "],
        " milega ": [" milega ", " milta hai ", " mil raha hai "],
        " jeeta ": [" jeeta ", " jeet liya ", " jeet gaya "],
        " hua ": [" hua ", " ho gaya ", " hua tha "],
    }
    out = text
    for key, options in swaps.items():
        if key in out and random.random() < 0.5:
            out = out.replace(key, random.choice(options), 1)
    return out


def shuffle_clauses(text: str) -> str:
    """If text has 2+ comma/clause-like chunks, lightly reorder them."""
    parts = re.split(r"(,| aur | ke baad | jabki | lekin )", text)
    if len(parts) >= 5:  # has actual separators
        # keep first chunk fixed (subject), shuffle the rest a bit
        head, rest = parts[0], parts[1:]
        if len(rest) >= 4 and random.random() < 0.4:
            # swap two adjacent clause pairs
            i = random.randrange(0, len(rest) - 3, 2)
            rest[i:i+2], rest[i+2:i+4] = rest[i+2:i+4], rest[i:i+2]
        return head + "".join(rest)
    return text


def paraphrase_augment(text: str, n_variants: int = 2) -> list:
    """Generate n_variants paraphrase-style variations of a sentence.
    Unlike v1 (which just repeated the first word), this actually
    changes sentence shape/wording while preserving meaning + label.
    """
    variants = []
    words = text.split()
    if len(words) < 3:
        return [text] * n_variants

    for _ in range(n_variants):
        v = text

        # 1. maybe add a natural filler at the start
        if random.random() < 0.5:
            v = random.choice(HINGLISH_FILLERS) + " " + v

        # 2. light synonym swap
        v = synonym_light_swap(v)

        # 3. maybe shuffle clauses
        v = shuffle_clauses(v)

        # 4. maybe add a sentence ender
        if random.random() < 0.4:
            v = v + random.choice(SENTENCE_ENDERS)

        # 5. occasionally drop a non-critical middle word (simulate typos/omissions)
        w = v.split()
        if len(w) > 6 and random.random() < 0.3:
            drop_idx = random.randrange(2, len(w) - 2)
            del w[drop_idx]
            v = " ".join(w)

        variants.append(v.strip())

    return variants


def augment_dataframe(df: pd.DataFrame, n_variants: int = 2) -> pd.DataFrame:
    """Apply paraphrase augmentation to every row, n_variants times."""
    aug_rows = []
    for _, row in df.iterrows():
        for v in paraphrase_augment(row["text"], n_variants=n_variants):
            aug_rows.append({"text": v, "label": row["label"]})
    aug_df = pd.DataFrame(aug_rows)
    return pd.concat([df, aug_df], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — HINDI / HINGLISH SYNTHETIC DATASET (EXPANDED)
# ─────────────────────────────────────────────────────────────────────────────

HINDI_REAL = [
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
    # ── newer additions ──
    "delhi mein assembly chunav 2025 mein bjp ne jeet hasil ki",
    "maharashtra mein 2024 vidhan sabha chunav hue the",
    "supreme court ne 2024 mein electoral bonds scheme ko radd kar diya",
    "bharat ne 2025 mein g20 summit ki adhyakshta poori ki",
    "isro ne gaganyaan mission ki taiyari 2025 mein tez ki",
    "reserve bank of india repo rate set karta hai monetary policy ke through",
    "election commission of india lok sabha aur vidhan sabha chunav karwata hai",
    "bharat ka chief election commissioner gyanesh kumar hain",
    "income tax ka naya bill parliament mein pass hua 2025 mein",
    "bharat sarkar ne digital personal data protection act 2023 mein pass kiya",
    "una recent g20 summit bharat mein september 2023 mein hua tha",
    "indian railways desh ka sabse bada employer hai",
    "iit aur nit jaise sansthano mein jee ke through admission hota hai",
    "neet exam medical colleges mein admission ke liye hota hai",
    "supreme court of india ki sthapna 1950 mein hui thi",
    "bharat ka high court har rajya mein alag hota hai",
]

HINDI_FAKE = [
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
    # ── newer additions ──
    "supreme court ne modi sarkar ko desh chhodne ka order diya",
    "election commission ne 2025 ke chunav cancel kar diye secretly",
    "delhi chunav fix tha results pehle se decide the",
    "rbi ne sabka paisa freeze kar diya digital currency ke naam par",
    "aadhaar card band ho raha hai sabko naya card lena hoga paid",
    "gaganyaan mission fail ho gaya isro ne chhupaaya hai",
    "g20 summit mein bharat ne secretly china se deal ki",
    "neet exam paper leak hua tha sabko pata hai government chhupa rahi",
    "income tax sabka double ho jaega naya bill ke through",
    "modi sarkar ne whatsapp data foreign companies ko becha",
]

HINGLISH_REAL = [
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
    # ── newer additions ──
    "delhi assembly election 2025 mein bjp ne majority paayi",
    "electoral bonds scheme supreme court ne unconstitutional declare kiya",
    "digital personal data protection act 2023 mein pass hua india mein",
    "gaganyaan mission ki testing 2025 mein continue hai isro dwara",
    "g20 presidency india ne successfully complete ki september 2023 mein",
    "neet aur jee exams se medical aur engineering colleges mein admission milta hai",
    "income tax naya bill parliament mein discuss hua 2025 mein",
]

HINGLISH_FAKE = [
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
    # ── newer additions ──
    "delhi election results fake the EVM hack hui thi",
    "electoral bonds case mein supreme court judge ko bribe mili",
    "data protection act se government sabka whatsapp padh sakti hai",
    "gaganyaan astronaut space mein lost ho gaya isro ne chhupaaya",
    "g20 summit mein modi ne secretly loan liya china se",
    "neet paper telegram pe leak hua tha sabko mil gaya tha",
]

# ── SCAM / PHISHING + AWARENESS SAMPLES (Hindi/Hinglish) ──────────────────────

SCAM_REAL_HI = [
    "kyc update karne ke liye bank kabhi otp nahi mangta hai",
    "rbi kabhi phone par bank account details nahi mangta",
    "bank kabhi bhi sms ya call par otp ya pin nahi mangta",
    "cyber fraud report karne ke liye 1930 helpline number hai india mein",
    "income tax refund sirf official portal se process hota hai",
    "upi pin kisi ke saath share nahi karna chahiye",
    "kyc update hamesha bank branch ya official app se hota hai",
    "lottery jeetne ke liye pehle ticket kharidna padta hai bina ticket lottery fake hoti hai",
    "pib fact check sarkari news verify karta hai",
    "electricity bill official app ya website se check karna chahiye",
    "digital arrest jaisa koi kanooni concept india mein exist nahi karta",
    "police kabhi video call par paise nahi mangti",
    "police complaint cyber crime portal cybercrime gov in par hoti hai",
    "sarkari yojana ki jankari official website par milti hai",
    "covid vaccines ne lakhs of lives bachayi hain worldwide",
    "who ne covid vaccine ko safe bataya hai",
    "election commission evm ki suraksha ke liye kai jaanch karta hai",
    "bank fraud hone par 1930 helpline par call karna chahiye",
    "vaccine se polio india mein khatam hua 2014 mein",
    "ayushman bharat card official website se banta hai",
    # ── newer additions ──
    "koi bhi bank officer phone par aapka card pin nahi puchega",
    "share market mein guaranteed return ka koi scheme legal nahi hota",
    "sebi registered advisor se hi investment advice leni chahiye",
    "asli courier company kabhi otp maang kar parcel release nahi karti",
]

SCAM_FAKE_HI = [
    # Lottery / prize scams
    "aapne 25 lakh ki lottery jiti hai claim karne ke liye link par click karein",
    "kbc lottery mein aapka number select hua hai 25 lakh jeete",
    "congratulations aapne kbc lottery mein 25 lakh jeeta hai whatsapp number par contact karo",
    "congratulations aapko free scooty mili hai sarkar ki yojana mein form bharo",
    # KYC / bank fraud
    "aapka bank account band ho jaega aaj hi kyc update karein is link se",
    "aapka kyc expire ho gaya hai 24 hours mein update karo warna account freeze",
    "pan card update nahi kiya to account block ho jaega turant link kholo",
    "sbi alert aapka account block ho gaya hai pan card link karne ke liye click karein",
    "bank manager bol raha hoon aapka atm card block ho gaya number batao",
    "sbi ki taraf se aapko 50000 ka loan approved hai link par click karein",
    # Electricity scam
    "bijli ka bill nahi bhara aaj raat connection kat jaega is number par call karo",
    "aapka electricity connection disconnect ho raha hai turant officer ko call karein",
    "electricity bill pending hai tonight disconnection hoga is number pe call karo urgent",
    # OTP fraud
    "otp batao aapka order cancel karna hai refund milega",
    "otp share karo apna refund claim karne ke liye amazon se call hai",
    "army officer hoon paytm se advance payment bhej raha hoon otp batao",
    "paytm cashback 5000 ka mila hai claim karne ke liye pin dalein",
    # Fake jobs
    "ghar baithe roz 5000 kamao bina kisi investment ke abhi join karo",
    "amazon mein job ke liye selection hua hai registration fee 2000 bharein",
    "work from home job 50000 monthly sirf 2 ghante kaam telegram pe join karo",
    "instagram pe paid task karo roz 3000 kamao registration free hai",
    # Digital arrest / courier scam
    "aapka parcel customs mein pakda gaya hai fine bharne ke liye otp batao",
    "aapke aadhaar se illegal sim nikli hai digital arrest hoga paise bharo",
    "cbi ne aapke naam warrant nikala hai video call par turant aao",
    "aapka parcel customs ne seize kiya hai drugs mile hain paise bharo warna arrest",
    # Fake govt schemes
    "modi yojana mein sabko 4000 mahina milega yahan register karein",
    "berojgari bhatta 3500 rupye sabko milega form bharo whatsapp par",
    "tax refund 15000 approved hai claim karne ke liye bank details bhejein",
    # Health misinformation
    "garam pani aur nimbu se corona khatam ho jata hai",
    "gomutra pine se cancer theek hota hai research mein saabit",
    "raat ko phone charge karne se cancer hota hai doctor ne bataya",
    "tulsi ke patte se heart blockage khul jata hai 100 percent",
    # Deepfake / viral
    "deepfake video mein neta ne desh bechne ki baat kabool ki",
    "deepfake video viral ho raha hai actor ne secret confession kiya",
    "whatsapp par yeh message 10 logo ko bhejo warna account band",
    # Financial misinformation
    "share market tip 1 din mein paisa double guaranteed scheme join karo",
    "crypto investment mein 1 lakh ka 10 lakh guaranteed 30 days mein",
    "free recharge 239 ka sabko mil raha hai jio ki taraf se link share karo",
    "flipkart anniversary offer free iphone sirf 99 rupye mein link par jao",
    "aapke phone mein virus hai turant yeh app download karo bank details daalo",
    # ── newer additions ──
    "trading app se 1 lakh laga ke 3 din mein 10 lakh banao guaranteed",
    "sim card band ho jaega aadhaar verify karo is link par turant",
    "income tax department se call hai notice bheja hai bank details do",
    "stock market expert telegram group join karo free tips milengi guaranteed profit",
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — INDIAN ENGLISH (EXPANDED)
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
    # ── newer additions ──
    "BJP won the Delhi assembly election held in early 2025",
    "Supreme Court declared the electoral bonds scheme unconstitutional in 2024",
    "India completed its G20 presidency in September 2023",
    "ISRO continued Gaganyaan mission testing through 2025",
    "The Digital Personal Data Protection Act was passed in India in 2023",
    "NEET and JEE exams determine admission to medical and engineering colleges in India",
    "Indian Railways is one of the largest employers in the world",
    "RBI sets the repo rate as part of its monetary policy decisions",
    "Every state in India has its own High Court",
    "Gyanesh Kumar is the Chief Election Commissioner of India",
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
    # ── newer additions ──
    "Delhi election results were rigged EVMs were hacked say insiders",
    "Electoral bonds judge received bribes from corporates leaked report",
    "New data protection law allows government to read all citizens WhatsApp",
    "Gaganyaan astronaut went missing in space ISRO covering it up",
    "NEET exam paper was leaked and sold on Telegram to thousands",
    "India secretly took a massive loan from China during G20 presidency",
]

# ── SCAM / PHISHING + AWARENESS SAMPLES (English) ────────────────────────────

SCAM_REAL_EN = [
    "Banks never ask for OTP PIN or CVV over phone calls or SMS",
    "India's national cyber crime helpline number is 1930",
    "RBI advises customers to report fraud transactions immediately to their bank",
    "There is no legal concept called digital arrest in Indian law",
    "PIB Fact Check is the official fact checking arm of the Government of India",
    "Income tax refunds are processed only through the official income tax portal",
    "KYC updates are done through official bank branches or verified apps",
    "Cyber fraud complaints can be filed at the cybercrime portal of India",
    "WHO confirms COVID vaccines are safe and effective after rigorous trials",
    "Genuine lotteries require purchasing a ticket no real lottery contacts random people",
    "UPI transactions require a PIN that should never be shared with anyone",
    "Election Commission of India conducts multiple security checks on EVMs",
    # ── newer additions ──
    "No genuine bank representative will ever ask for your card PIN over a call",
    "Guaranteed returns on stock market investments are a major red flag for fraud",
    "Only SEBI registered advisors should be trusted for investment advice",
    "Legitimate courier companies never ask for OTP to release a parcel",
]

SCAM_FAKE_EN = [
    # Lottery / prize scams
    "Congratulations you have won 25 lakh rupees in KBC lottery click link to claim",
    "Amazon is giving free gifts on its anniversary spin the wheel and pay delivery charge",
    "Free recharge of 239 rupees for all users from Jio share this message to claim",
    # KYC / bank fraud
    "Your bank account will be blocked today update KYC immediately through this link",
    "Your PAN card will be deactivated tomorrow click here to link with Aadhaar",
    "Share your OTP to receive the refund initiated by the bank",
    "Your SIM card will be blocked in 24 hours press 9 to talk to TRAI officer",
    # Electricity scam
    "Your electricity connection will be disconnected tonight call this officer number urgently",
    # Digital arrest / courier scam
    "Your parcel has been seized by customs pay fine immediately or face arrest",
    "You are under digital arrest by CBI transfer money to verification account now",
    # Fake jobs
    "Earn 50000 rupees monthly working from home just 2 hours pay registration fee now",
    "Selected for online part time job complete telegram tasks and earn 3000 daily",
    # Fake govt schemes
    "Government giving 4000 rupees monthly to all unemployed youth register on this link",
    "Your income tax refund of 15000 is approved share bank details to claim",
    # Health misinformation
    "Drinking hot lemon water kills coronavirus within minutes proven remedy",
    "Cow urine cures cancer completely proven by international research",
    "Charging phone overnight causes brain cancer doctors warn",
    # Deepfake / viral
    "Deepfake video shows minister confessing to selling national secrets",
    "WhatsApp will start charging users unless this message is forwarded to 10 contacts",
    # Financial misinformation
    "Invest 1 lakh in this crypto scheme get 10 lakh guaranteed in 30 days",
    "Army officer wants to buy your furniture will send advance payment share OTP",
    # ── newer additions ──
    "Your SIM will be deactivated verify Aadhaar immediately through this link",
    "Income Tax Department has sent you a notice share your bank details now",
    "Join this telegram group for guaranteed stock tips and double your money",
]


def build_hindi_hinglish_dataset() -> pd.DataFrame:
    texts = HINDI_REAL + HINGLISH_REAL + SCAM_REAL_HI + HINDI_FAKE + HINGLISH_FAKE + SCAM_FAKE_HI
    labels = (
        ["REAL"] * (len(HINDI_REAL) + len(HINGLISH_REAL) + len(SCAM_REAL_HI))
        + ["FAKE"] * (len(HINDI_FAKE) + len(HINGLISH_FAKE) + len(SCAM_FAKE_HI))
    )
    df = pd.DataFrame({"text": texts, "label": labels})
    # Real paraphrase augmentation (3 variants per sentence instead of 1 weak repeat)
    return augment_dataframe(df, n_variants=3)


def build_indian_english_dataset() -> pd.DataFrame:
    texts = INDIAN_REAL_EN + SCAM_REAL_EN + INDIAN_FAKE_EN + SCAM_FAKE_EN
    labels = (
        ["REAL"] * (len(INDIAN_REAL_EN) + len(SCAM_REAL_EN))
        + ["FAKE"] * (len(INDIAN_FAKE_EN) + len(SCAM_FAKE_EN))
    )
    df = pd.DataFrame({"text": texts, "label": labels})
    return augment_dataframe(df, n_variants=3)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ISOT DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_isot_dataset(data_dir: str = "Data") -> pd.DataFrame:
    """Load ISOT dataset from CSV files. Expected: Data/True.csv, Data/Fake.csv"""
    frames = []
    for filename, label in [("True.csv", "REAL"), ("Fake.csv", "FAKE")]:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"  ⚠️  {path} not found — skipping")
            continue
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        if "title" in df.columns and "text" in df.columns:
            df["combined"] = df["title"].fillna("") + " " + df["text"].fillna("")
        elif "text" in df.columns:
            df["combined"] = df["text"].fillna("")
        elif "title" in df.columns:
            df["combined"] = df["title"].fillna("")
        else:
            print(f"  ⚠️  No usable columns in {filename} — skipping")
            continue
        df = df[["combined"]].rename(columns={"combined": "text"})
        df["label"] = label
        frames.append(df.head(12000))
        print(f"  ✅ Loaded {min(len(df), 12000):,} rows from {filename} [{label}]")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["text", "label"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — BUILD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline():
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=150_000,
        sublinear_tf=True,
        min_df=1,
        strip_accents=None,
        lowercase=True,
    )
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        min_df=2,
        strip_accents=None,
        lowercase=True,
    )
    combined = FeatureUnion([("char", char_vec), ("word", word_vec)])
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", combined), ("clf", clf)])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — TRAIN + CROSS-VALIDATE + EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 60)
    print(" FakeNews Detector — Model Training v2 (Multilingual)")
    print(" Hindi + Hinglish + English + ISOT + Paraphrase Augmentation")
    print("=" * 60 + "\n")

    print("📂 Loading ISOT dataset...")
    df_isot = load_isot_dataset("Data")

    print("🇮🇳 Loading Hindi/Hinglish dataset (with paraphrase augmentation)...")
    df_hindi = build_hindi_hinglish_dataset()
    print(f"  ✅ {len(df_hindi):,} Hindi/Hinglish samples (after augmentation)")

    print("📰 Loading Indian English patterns (with paraphrase augmentation)...")
    df_indian = build_indian_english_dataset()
    print(f"  ✅ {len(df_indian):,} Indian English samples (after augmentation)")

    frames = [df_hindi, df_indian]
    if not df_isot.empty:
        frames.insert(0, df_isot)
    df = pd.concat(frames, ignore_index=True)
    df = shuffle(df, random_state=42).reset_index(drop=True)

    print(f"\n📊 Total dataset: {len(df):,} samples")
    print(f"  REAL: {(df.label=='REAL').sum():,}")
    print(f"  FAKE: {(df.label=='FAKE').sum():,}")

    print("\n🧹 Cleaning text...")
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  After cleaning + dedup: {len(df):,} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"\n✂️  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 5-fold cross-validation on the training set ──────────────────────────
    print("\n🔁 Running 5-fold cross-validation on training data...")
    pipeline = build_pipeline()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"  CV Accuracy : {cv_acc.mean()*100:.2f}% (+/- {cv_acc.std()*100:.2f}%)")
    print(f"  CV F1-macro : {cv_f1.mean()*100:.2f}% (+/- {cv_f1.std()*100:.2f}%)")
    print(f"  Fold scores : {[round(s*100,2) for s in cv_acc]}")

    print("\n🏋️  Training final model on full training set...")
    pipeline.fit(X_train, y_train)

    print("\n📈 Evaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test F1-macro : {f1*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"])
    print("              Pred REAL  Pred FAKE")
    print(f"  True REAL : {cm[0][0]:9d}  {cm[0][1]:9d}")
    print(f"  True FAKE : {cm[1][0]:9d}  {cm[1][1]:9d}")

    # ── Hindi/Hinglish sanity check ───────────────────────────────────────────
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
        ("delhi assembly election 2025 mein bjp ne jeet hasil ki", "REAL"),
        ("aapka sim card band ho jaega aadhaar verify karo link par", "FAKE"),
    ]
    passed = 0
    for text, expected in test_cases:
        cleaned = clean_text(text)
        pred = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned])[0]
        conf = max(proba) * 100
        status = "✅" if pred == expected else "❌"
        if pred == expected:
            passed += 1
        print(f"  {status} [{pred:4s} {conf:5.1f}%] {text[:55]}")
    print(f"\n  Passed: {passed}/{len(test_cases)}")

    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f, protocol=4)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n💾 Model saved → {model_path} ({size_mb:.1f} MB)")
    print("\n✅ Training complete!")
    print("   Replace the old model.pkl in your project root with this file.")
    print("   No changes needed in app.py — the pipeline interface is identical.")
    print("=" * 60 + "\n")

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PREDICT HELPER (for app.py compatibility test)
# ─────────────────────────────────────────────────────────────────────────────

def predict(pipeline, text: str):
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])
    conf = float(max(proba[0])) * 100
    return pred, conf


if __name__ == "__main__":
    trained_model = train()

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

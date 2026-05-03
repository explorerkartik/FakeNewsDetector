# ── INDIAN FACTS DATABASE (Updated 2026) ─────────────────
# Features:
#   1. Expanded INDIAN_FACTS dict
#   2. DYNAMIC_FACTS — loaded from DB at runtime (current affairs)
#   3. PIB + Wikipedia scraper for auto-update
#   4. Admin trigger: /admin/update-facts

import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime

INDIAN_FACTS = {

    # ── PRIME MINISTERS ──
    "pm of india": "Narendra Modi is the Prime Minister of India since 2014.",
    "prime minister of india": "Narendra Modi is the Prime Minister of India since 2014.",
    "india pm": "Narendra Modi is the Prime Minister of India.",
    "narendra modi": "Narendra Modi is the 14th Prime Minister of India, serving since May 2014. He is a member of BJP.",
    "modi": "Narendra Modi is the Prime Minister of India since 2014, member of BJP.",
    "pm modi": "Narendra Modi (PM Modi) is the Prime Minister of India since 2014.",
    "prime minister modi": "Narendra Modi is the Prime Minister of India since 2014.",

    # ── PRESIDENT ──
    "president of india": "Droupadi Murmu is the 15th President of India since July 2022.",
    "droupadi murmu": "Droupadi Murmu is the President of India since 2022, first tribal woman president.",
    "india president": "Droupadi Murmu is the President of India since July 2022.",

    # ── VICE PRESIDENT ──
    "vice president of india": "Jagdeep Dhankhar is the Vice President of India since August 2022.",
    "jagdeep dhankhar": "Jagdeep Dhankhar is the Vice President of India since 2022.",

    # ── CHIEF MINISTERS ──
    "cm of uttar pradesh": "Yogi Adityanath is the Chief Minister of Uttar Pradesh since 2017.",
    "cm of up": "Yogi Adityanath is the Chief Minister of Uttar Pradesh.",
    "uttar pradesh cm": "Yogi Adityanath is the CM of Uttar Pradesh since 2017.",
    "yogi adityanath": "Yogi Adityanath is the Chief Minister of Uttar Pradesh, member of BJP.",
    "yogi": "Yogi Adityanath is the Chief Minister of Uttar Pradesh.",
    "cm of maharashtra": "Devendra Fadnavis is the Chief Minister of Maharashtra since December 2024.",
    "maharashtra cm": "Devendra Fadnavis is the CM of Maharashtra since December 2024.",
    "devendra fadnavis": "Devendra Fadnavis is the Chief Minister of Maharashtra since 2024.",
    "cm of delhi": "Rekha Gupta is the Chief Minister of Delhi since February 2025.",
    "delhi cm": "Rekha Gupta is the Chief Minister of Delhi since February 2025.",
    "rekha gupta": "Rekha Gupta is the Chief Minister of Delhi since February 2025, member of BJP.",
    "cm of rajasthan": "Bhajanlal Sharma is the Chief Minister of Rajasthan since December 2023.",
    "rajasthan cm": "Bhajanlal Sharma is the CM of Rajasthan since December 2023.",
    "cm of madhya pradesh": "Mohan Yadav is the Chief Minister of Madhya Pradesh since December 2023.",
    "mp cm": "Mohan Yadav is the CM of Madhya Pradesh since December 2023.",
    "madhya pradesh cm": "Mohan Yadav is the CM of Madhya Pradesh.",
    "cm of gujarat": "Bhupendra Patel is the Chief Minister of Gujarat since 2021.",
    "gujarat cm": "Bhupendra Patel is the CM of Gujarat since 2021.",
    "cm of karnataka": "Siddaramaiah is the Chief Minister of Karnataka since May 2023.",
    "karnataka cm": "Siddaramaiah is the CM of Karnataka since 2023.",
    "cm of telangana": "A. Revanth Reddy is the Chief Minister of Telangana since December 2023.",
    "telangana cm": "Revanth Reddy is the CM of Telangana since December 2023.",
    "cm of andhra pradesh": "N. Chandrababu Naidu is the Chief Minister of Andhra Pradesh since June 2024.",
    "andhra pradesh cm": "Chandrababu Naidu is the CM of Andhra Pradesh since June 2024.",
    "ap cm": "N. Chandrababu Naidu is the Chief Minister of Andhra Pradesh since 2024.",
    "cm of tamil nadu": "M.K. Stalin is the Chief Minister of Tamil Nadu since May 2021.",
    "tamil nadu cm": "MK Stalin is the CM of Tamil Nadu since 2021.",
    "cm of kerala": "Pinarayi Vijayan is the Chief Minister of Kerala since 2016.",
    "kerala cm": "Pinarayi Vijayan is the CM of Kerala since 2016.",
    "cm of west bengal": "Mamata Banerjee is the Chief Minister of West Bengal since 2011.",
    "west bengal cm": "Mamata Banerjee is the CM of West Bengal since 2011.",
    "mamata banerjee": "Mamata Banerjee is the Chief Minister of West Bengal since 2011.",
    "cm of bihar": "Nitish Kumar is the Chief Minister of Bihar since 2005.",
    "bihar cm": "Nitish Kumar is the CM of Bihar.",
    "nitish kumar": "Nitish Kumar is the Chief Minister of Bihar.",
    "cm of punjab": "Bhagwant Mann is the Chief Minister of Punjab since March 2022.",
    "punjab cm": "Bhagwant Mann is the CM of Punjab since 2022, member of AAP.",
    "bhagwant mann": "Bhagwant Mann is the Chief Minister of Punjab since 2022.",
    "cm of haryana": "Nayab Singh Saini is the Chief Minister of Haryana since March 2024.",
    "haryana cm": "Nayab Singh Saini is the CM of Haryana since 2024.",
    "cm of himachal pradesh": "Sukhvinder Singh Sukhu is the Chief Minister of Himachal Pradesh since December 2022.",
    "himachal pradesh cm": "Sukhvinder Singh Sukhu is the CM of Himachal Pradesh since 2022.",
    "cm of uttarakhand": "Pushkar Singh Dhami is the Chief Minister of Uttarakhand since 2021.",
    "uttarakhand cm": "Pushkar Singh Dhami is the CM of Uttarakhand since 2021.",
    "cm of jharkhand": "Hemant Soren is the Chief Minister of Jharkhand since November 2024.",
    "jharkhand cm": "Hemant Soren is the CM of Jharkhand since November 2024.",
    "cm of chhattisgarh": "Vishnu Deo Sai is the Chief Minister of Chhattisgarh since December 2023.",
    "chhattisgarh cm": "Vishnu Deo Sai is the CM of Chhattisgarh since 2023.",
    "cm of odisha": "Mohan Charan Majhi is the Chief Minister of Odisha since June 2024.",
    "odisha cm": "Mohan Charan Majhi is the CM of Odisha since June 2024.",
    "cm of assam": "Himanta Biswa Sarma is the Chief Minister of Assam since May 2021.",
    "assam cm": "Himanta Biswa Sarma is the CM of Assam since 2021.",
    "cm of goa": "Pramod Sawant is the Chief Minister of Goa since 2019.",
    "goa cm": "Pramod Sawant is the CM of Goa since 2019.",
    "cm of manipur": "N. Biren Singh is the Chief Minister of Manipur since 2017.",
    "manipur cm": "N. Biren Singh is the CM of Manipur since 2017.",
    "cm of tripura": "Manik Saha is the Chief Minister of Tripura since May 2022.",
    "tripura cm": "Manik Saha is the CM of Tripura since 2022.",
    "cm of meghalaya": "Conrad Sangma is the Chief Minister of Meghalaya since 2018.",
    "meghalaya cm": "Conrad Sangma is the CM of Meghalaya since 2018.",
    "cm of sikkim": "Prem Singh Tamang is the Chief Minister of Sikkim since 2019.",
    "sikkim cm": "Prem Singh Tamang is the CM of Sikkim since 2019.",
    "cm of arunachal pradesh": "Pema Khandu is the Chief Minister of Arunachal Pradesh since 2016.",
    "arunachal pradesh cm": "Pema Khandu is the CM of Arunachal Pradesh since 2016.",
    "cm of nagaland": "Neiphiu Rio is the Chief Minister of Nagaland since 2018.",
    "nagaland cm": "Neiphiu Rio is the CM of Nagaland since 2018.",
    "cm of mizoram": "Lalduhoma is the Chief Minister of Mizoram since December 2023.",
    "mizoram cm": "Lalduhoma is the CM of Mizoram since 2023.",
    "cm of jammu and kashmir": "Omar Abdullah is the Chief Minister of Jammu & Kashmir since October 2024.",
    "j&k cm": "Omar Abdullah is the CM of Jammu & Kashmir since October 2024.",
    "jammu kashmir cm": "Omar Abdullah is the CM of J&K since 2024.",
    "omar abdullah": "Omar Abdullah is the Chief Minister of Jammu & Kashmir since October 2024.",

    # ── ABBREVIATIONS ──
    "pm": "PM stands for Prime Minister. PM of India is Narendra Modi since 2014.",
    "cm": "CM stands for Chief Minister. Each state of India has its own Chief Minister.",
    "bjp": "BJP stands for Bharatiya Janata Party, ruling party at centre led by PM Modi.",
    "inc": "INC stands for Indian National Congress, major opposition party in India.",
    "congress": "Indian National Congress (INC) is a major opposition party in India.",
    "aap": "AAP stands for Aam Aadmi Party, ruling Punjab. Led by Arvind Kejriwal.",
    "arvind kejriwal": "Arvind Kejriwal is the national convener of AAP party.",
    "sp": "SP stands for Samajwadi Party, led by Akhilesh Yadav in Uttar Pradesh.",
    "akhilesh yadav": "Akhilesh Yadav is the President of Samajwadi Party and former CM of UP.",
    "bsp": "BSP stands for Bahujan Samaj Party, led by Mayawati.",
    "mayawati": "Mayawati is the President of Bahujan Samaj Party (BSP).",
    "nda": "NDA stands for National Democratic Alliance, political coalition led by BJP.",
    "upa": "UPA stands for United Progressive Alliance, political coalition led by Congress.",
    "rahul gandhi": "Rahul Gandhi is the Leader of Opposition in Lok Sabha and Congress leader.",
    "sonia gandhi": "Sonia Gandhi is a senior Congress leader and former President of INC.",
    "amit shah": "Amit Shah is the Home Minister of India since 2019, member of BJP.",
    "mla": "MLA stands for Member of Legislative Assembly.",
    "lok sabha": "Lok Sabha is the lower house of Indian Parliament. Speaker is Om Birla.",
    "rajya sabha": "Rajya Sabha is the upper house of Indian Parliament.",
    "parliament": "Indian Parliament consists of Lok Sabha and Rajya Sabha.",
    "cbi": "CBI stands for Central Bureau of Investigation, India's premier investigation agency.",
    "ias": "IAS stands for Indian Administrative Service, premier civil service of India.",
    "ips": "IPS stands for Indian Police Service.",
    "isro": "ISRO stands for Indian Space Research Organisation, headquartered in Bengaluru.",
    "rbi": "RBI stands for Reserve Bank of India. Governor is Sanjay Malhotra since December 2024.",
    "upi": "UPI stands for Unified Payments Interface, India's digital payment system by NPCI.",
    "ed": "ED stands for Enforcement Directorate, India's financial investigation agency.",
    "nia": "NIA stands for National Investigation Agency, India's counter-terrorism agency.",
    "raw": "RAW stands for Research and Analysis Wing, India's external intelligence agency.",
    "ib": "IB stands for Intelligence Bureau, India's internal intelligence agency.",

    # ── IMPORTANT POSTS 2025-26 ──
    "chief justice of india": "Sanjiv Khanna is the Chief Justice of India since November 2024.",
    "cji": "CJI Sanjiv Khanna is the Chief Justice of India since November 2024.",
    "rbi governor": "Sanjay Malhotra is the Governor of RBI since December 2024.",
    "army chief": "General Upendra Dwivedi is the Chief of Army Staff since June 2024.",
    "navy chief": "Admiral Dinesh K Tripathi is the Chief of Naval Staff since April 2024.",
    "air force chief": "Air Chief Marshal A.P. Singh is the Chief of Air Staff since September 2024.",
    "cds": "General Anil Chauhan is the Chief of Defence Staff since September 2022.",
    "nsa": "Ajit Doval is the National Security Advisor of India since 2014.",
    "foreign minister": "S. Jaishankar is the External Affairs Minister of India since 2019.",
    "s jaishankar": "S. Jaishankar is the External Affairs Minister of India since 2019.",
    "home minister": "Amit Shah is the Home Minister of India since 2019.",
    "finance minister": "Nirmala Sitharaman is the Finance Minister of India since 2019.",
    "nirmala sitharaman": "Nirmala Sitharaman is the Finance Minister of India since 2019.",
    "defence minister": "Rajnath Singh is the Defence Minister of India since 2019.",
    "rajnath singh": "Rajnath Singh is the Defence Minister of India since 2019.",
    "lok sabha speaker": "Om Birla is the Speaker of Lok Sabha since 2019.",
    "election commissioner": "The Chief Election Commissioner of India is Rajiv Kumar since 2022.",
    "upsc chairman": "Preeti Sudan is the UPSC Chairperson since 2023.",

    # ── INDIAN HISTORY ──
    "independence day": "India got independence on 15 August 1947 from British rule.",
    "republic day": "India became a republic on 26 January 1950 when the Constitution came into effect.",
    "indian constitution": "Constitution of India came into effect on 26 January 1950. Dr. B.R. Ambedkar was the chief architect.",
    "br ambedkar": "Dr. B.R. Ambedkar was the chief architect of Indian Constitution and first Law Minister of India.",
    "ambedkar": "Dr. B.R. Ambedkar was the chief architect of Indian Constitution.",
    "mahatma gandhi": "Mahatma Gandhi, Father of the Nation, led India's independence movement. Born 2 Oct 1869, assassinated 30 Jan 1948.",
    "gandhi": "Mahatma Gandhi is the Father of the Nation who led India's freedom struggle.",
    "jawaharlal nehru": "Jawaharlal Nehru was the first Prime Minister of India from 1947 to 1964.",
    "nehru": "Jawaharlal Nehru was the first Prime Minister of India from 1947 to 1964.",
    "indira gandhi": "Indira Gandhi was the first female Prime Minister of India, served 1966-1977 and 1980-1984.",
    "sardar patel": "Sardar Vallabhbhai Patel was the first Deputy PM and Home Minister, known as Iron Man of India.",
    "subhas chandra bose": "Netaji Subhas Chandra Bose was a freedom fighter who founded the Indian National Army (INA).",
    "netaji": "Netaji Subhas Chandra Bose founded the Indian National Army (INA).",
    "bhagat singh": "Bhagat Singh was a revolutionary freedom fighter, hanged on 23 March 1931.",
    "emergency india": "The Emergency in India was declared by PM Indira Gandhi from 1975 to 1977.",
    "ram mandir": "Ram Mandir in Ayodhya was inaugurated by PM Narendra Modi on 22 January 2024.",
    "ayodhya": "Ram Mandir in Ayodhya was inaugurated by PM Modi on 22 January 2024.",
    "article 370": "Article 370 was abrogated on 5 August 2019, removing special status of Jammu & Kashmir.",
    "gst": "GST (Goods and Services Tax) was implemented in India on 1 July 2017.",
    "demonetization": "Demonetization was announced on 8 November 2016 by PM Modi, scrapping Rs 500 and Rs 1000 notes.",
    "demonetisation": "Demonetisation was announced on 8 November 2016 by PM Modi.",

    # ── GEOGRAPHY ──
    "capital of india": "New Delhi is the capital of India.",
    "largest state india": "Rajasthan is the largest state of India by area.",
    "smallest state india": "Goa is the smallest state of India by area.",
    "most populous state": "Uttar Pradesh is the most populous state of India.",
    "india population": "India's population is approximately 1.44 billion (2024), most populous country in the world.",
    "india gdp": "India's GDP is approximately $3.9 trillion (2024), 5th largest economy in the world.",
    "national anthem": "Jana Gana Mana is the National Anthem of India, written by Rabindranath Tagore.",
    "national song": "Vande Mataram is the National Song of India, written by Bankim Chandra Chatterjee.",
    "national animal": "Bengal Tiger is the National Animal of India.",
    "national bird": "Indian Peacock is the National Bird of India.",
    "national flower": "Lotus is the National Flower of India.",
    "national fruit": "Mango is the National Fruit of India.",

    # ── CAPITALS OF STATES ──
    "capital of rajasthan": "Jaipur is the capital of Rajasthan.",
    "capital of maharashtra": "Mumbai is the capital of Maharashtra.",
    "capital of uttar pradesh": "Lucknow is the capital of Uttar Pradesh.",
    "capital of bihar": "Patna is the capital of Bihar.",
    "capital of jharkhand": "Ranchi is the capital of Jharkhand.",
    "capital of west bengal": "Kolkata is the capital of West Bengal.",
    "capital of gujarat": "Gandhinagar is the capital of Gujarat.",
    "capital of karnataka": "Bengaluru is the capital of Karnataka.",
    "capital of tamil nadu": "Chennai is the capital of Tamil Nadu.",
    "capital of kerala": "Thiruvananthapuram is the capital of Kerala.",
    "capital of andhra pradesh": "Amaravati is the capital of Andhra Pradesh.",
    "capital of telangana": "Hyderabad is the capital of Telangana.",
    "capital of odisha": "Bhubaneswar is the capital of Odisha.",
    "capital of assam": "Dispur is the capital of Assam.",
    "capital of punjab": "Chandigarh is the capital of Punjab.",
    "capital of haryana": "Chandigarh is the capital of Haryana.",
    "capital of himachal pradesh": "Shimla is the capital of Himachal Pradesh.",
    "capital of uttarakhand": "Dehradun is the capital (interim) of Uttarakhand.",
    "capital of chhattisgarh": "Raipur is the capital of Chhattisgarh.",
    "capital of madhya pradesh": "Bhopal is the capital of Madhya Pradesh.",
    "capital of delhi": "New Delhi is the capital of India and Delhi NCT.",
    "capital of goa": "Panaji is the capital of Goa.",

    # ── RECENT EVENTS 2024-2026 ──
    "lok sabha election 2024": "In 2024 Lok Sabha elections, BJP won 240 seats, NDA won 293 seats. Modi became PM for 3rd term.",
    "election 2024": "2024 General Elections: NDA won majority. Modi became PM for third consecutive term.",
    "chandrayaan 3": "Chandrayaan-3 successfully landed on Moon's south pole on 23 August 2023.",
    "chandrayaan": "Chandrayaan-3 landed on Moon's south pole on 23 August 2023. India was first country to do so.",
    "g20 india": "India hosted G20 Summit in New Delhi in September 2023 under PM Modi's presidency.",
    "ucc": "UCC (Uniform Civil Code) - Uttarakhand became first state to implement it in 2024.",
    "budget 2025": "Union Budget 2025-26 was presented by FM Nirmala Sitharaman on 1 February 2025.",
    "pahalgam attack": "Terrorist attack in Pahalgam, Kashmir in April 2025 killed several tourists.",
    "operation sindoor": "Operation Sindoor was launched by India in May 2025 against terrorist camps in Pakistan and PoK.",
    "caa": "CAA (Citizenship Amendment Act) was passed in 2019 and implemented in 2024.",
    "one nation one election": "One Nation One Election proposes simultaneous elections for Lok Sabha and State Assemblies.",
    "india pakistan war 2025": "After Operation Sindoor in May 2025, India conducted precision strikes on terrorist infrastructure in Pakistan and PoK.",
    "viksit bharat": "Viksit Bharat 2047 is PM Modi's vision to make India a developed nation by 2047.",
    "make in india": "Make in India is a government initiative launched in 2014 to boost domestic manufacturing.",
    "upi one world": "UPI One World was launched for international visitors to use UPI without Indian bank accounts.",

    # ── SPORTS ──
    "t20 world cup 2024": "India won T20 World Cup 2024, defeating South Africa in final in Barbados on 29 June 2024.",
    "cricket world cup 2024": "India won T20 World Cup 2024 defeating South Africa.",
    "ipl 2024": "Kolkata Knight Riders (KKR) won IPL 2024, defeating Sunrisers Hyderabad in final.",
    "ipl 2025": "Royal Challengers Bengaluru (RCB) won IPL 2025.",
    "virat kohli": "Virat Kohli is a legendary Indian cricketer. He retired from Test cricket in 2024.",
    "rohit sharma": "Rohit Sharma is captain of Indian cricket team. Retired from T20Is after winning T20 WC 2024.",
    "ms dhoni": "MS Dhoni is former Indian captain, won 2011 ODI WC, 2007 T20 WC, multiple IPL titles with CSK.",
    "dhoni": "MS Dhoni is former Indian cricket captain, known as Captain Cool.",
    "neeraj chopra": "Neeraj Chopra won Gold at Tokyo 2020 Olympics and Silver at Paris 2024 Olympics in javelin.",
    "paris olympics 2024": "India won 6 medals at Paris Olympics 2024: 1 Silver (Neeraj Chopra) and 5 Bronze.",
    "pv sindhu": "PV Sindhu is Indian badminton player, won Silver at Rio 2016 and Bronze at Tokyo 2020 Olympics.",

    # ── TECHNOLOGY & SPACE ──
    "isro missions": "ISRO's missions include Chandrayaan-3 (2023), Aditya-L1 (2023), Gaganyaan (upcoming).",
    "aditya l1": "Aditya-L1 is India's first solar mission launched by ISRO in September 2023.",
    "gaganyaan": "Gaganyaan is India's first crewed space mission planned by ISRO.",
    "digital india": "Digital India is a government initiative to transform India into a digitally empowered society.",
    "upi payments": "UPI processed over 15 billion transactions per month in 2024.",
    "5g india": "India launched 5G services in October 2022 by PM Modi. Currently available in 500+ cities.",
    "ai india": "India launched National AI Mission (NAIM) with Rs 10,372 crore budget in 2024.",

    # ── AWARDS ──
    "bharat ratna 2024": "Bharat Ratna 2024: LK Advani, Karpoori Thakur, PV Narasimha Rao, MS Swaminathan, Chaudhary Charan Singh.",
    "bharat ratna": "Bharat Ratna is India's highest civilian award.",
    "padma awards 2025": "Padma Awards 2025 were announced on Republic Day 2025 by the Government of India.",

    # ── LAWS ──
    "bns": "BNS (Bharatiya Nyaya Sanhita) replaced IPC from 1 July 2024.",
    "ipc": "IPC (Indian Penal Code) was replaced by Bharatiya Nyaya Sanhita (BNS) from July 2024.",
    "pocso": "POCSO (Protection of Children from Sexual Offences Act) was enacted in 2012.",
    "nrc": "NRC stands for National Register of Citizens.",
    "crpc": "CrPC was replaced by Bharatiya Nagarik Suraksha Sanhita (BNSS) from July 2024.",

    # ── ECONOMY ──
    "india economy": "India is the 5th largest economy in the world with GDP of approximately $3.9 trillion (2024).",
    "sensex": "Sensex is India's premier stock market index on BSE (Bombay Stock Exchange).",
    "nifty": "Nifty 50 is the benchmark index of NSE (National Stock Exchange) of India.",
    "inflation india": "India's retail inflation (CPI) target is 4% with ±2% band, set by RBI.",
    "repo rate": "RBI's repo rate as of 2025 is 6.25% after cuts to support economic growth.",

    # ── GOVERNMENT SCHEMES ──
    "pm kisan": "PM Kisan Samman Nidhi provides Rs 6,000/year to farmers directly in 3 installments.",
    "ayushman bharat": "Ayushman Bharat provides health cover of Rs 5 lakh per family per year to poor families.",
    "jan dhan": "PM Jan Dhan Yojana is a financial inclusion scheme. Over 50 crore accounts opened.",
    "ujjwala yojana": "PM Ujjwala Yojana provides free LPG connections to BPL families.",
    "swachh bharat": "Swachh Bharat Mission was launched on 2 October 2014 to achieve open defecation free India.",
    "smart cities": "Smart Cities Mission aims to develop 100 smart cities across India.",
    "atal pension yojana": "Atal Pension Yojana provides pension of Rs 1,000-5,000/month to unorganized sector workers.",
    "mudra yojana": "PM Mudra Yojana provides loans up to Rs 10 lakh to small businesses.",
    "startup india": "Startup India was launched on 16 January 2016 to promote entrepreneurship.",
    "skill india": "Skill India Mission was launched in 2015 to train 40 crore people by 2022.",

    # ── DEFENCE ──
    "indian army": "Indian Army is the world's second largest army by active personnel.",
    "rafale": "India has 36 Rafale fighter jets from France in Indian Air Force.",
    "ins vikrant": "INS Vikrant is India's first indigenously built aircraft carrier, commissioned in 2022.",
    "agni missile": "Agni-V is India's intercontinental ballistic missile with range over 5,000 km.",
    "brahmos": "BrahMos is a supersonic cruise missile jointly developed by India and Russia.",

    # ── EDUCATION ──
    "iit": "India has 23 IITs (Indian Institutes of Technology) as of 2024.",
    "iim": "India has 20 IIMs (Indian Institutes of Management) as of 2024.",
    "nep 2020": "National Education Policy 2020 was approved by Cabinet, replacing 34-year-old policy.",
    "ugc": "UGC (University Grants Commission) regulates higher education in India.",
}

# ── HIGH CONFIDENCE KEYWORDS ─────────────────────────────
HIGH_CONFIDENCE_KEYS = [
    "pm modi", "narendra modi", "modi", "prime minister",
    "cm of", "chief minister", "president of india", "droupadi murmu",
    "parliament", "lok sabha", "rajya sabha",
    "isro", "chandrayaan", "operation sindoor", "pahalgam",
    "bjp", "congress", "aap", "nda",
    "ram mandir", "article 370", "gst", "demonetization",
    "virat kohli", "rohit sharma", "neeraj chopra",
    "ipl 2024", "ipl 2025", "t20 world cup",
    "budget 2025", "election 2024",
    "amit shah", "rajnath singh", "nirmala sitharaman",
    "yogi", "yogi adityanath", "mamata", "kejriwal",
    "rahul gandhi", "sonia gandhi",
    "rbi governor", "chief justice", "army chief",
    "ins vikrant", "rafale", "brahmos", "agni",
    "pm kisan", "ayushman bharat", "jan dhan",
    "viksit bharat", "make in india", "startup india",
]


# ── DYNAMIC FACTS (from DB) ───────────────────────────────────────────────────

def get_dynamic_facts(db_conn=None):
    """Load current affairs from DB (dynamic_facts table)."""
    if db_conn is None:
        return {}
    try:
        cur = db_conn.cursor()
        cur.execute("SELECT keyword, fact FROM dynamic_facts WHERE active = TRUE")
        rows = cur.fetchall()
        return {row[0].lower().strip(): row[1] for row in rows}
    except Exception:
        return {}


# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────────

def check_indian_facts(text, db_conn=None):
    """Check if text contains known Indian facts and return relevant info."""
    text_lower = text.lower()
    matched_facts = []
    seen = set()

    # Merge static + dynamic facts
    all_facts = dict(INDIAN_FACTS)
    all_facts.update(get_dynamic_facts(db_conn))

    for key, fact in all_facts.items():
        if key in text_lower and fact not in seen:
            matched_facts.append(fact)
            seen.add(fact)
    return matched_facts[:5]


def get_credibility_boost(text, db_conn=None):
    """
    Boost credibility score ONLY when the statement closely matches
    a known verified fact — not just because a keyword appears.
    This prevents false boosts on incorrect statements.
    """
    text_lower = text.lower()
    boost = 0

    all_facts = dict(INDIAN_FACTS)
    all_facts.update(get_dynamic_facts(db_conn))

    for key, fact in all_facts.items():
        if key in text_lower:
            # Extract the core claim from the fact (name/value after "is")
            fact_lower = fact.lower()

            # Check if text contradicts the fact
            # e.g., fact says "mamata banerjee is cm of west bengal"
            # but text says "hemant soren is cm of west bengal" — no boost
            fact_words = set(fact_lower.split())
            text_words  = set(text_lower.split())

            # Find key entities in the fact
            # If text has the key but contradicts main entity — skip boost
            common = fact_words & text_words
            overlap_ratio = len(common) / max(len(fact_words), 1)

            if overlap_ratio >= 0.4:
                # Good overlap — text likely matches the fact
                boost += 10
            # else: keyword present but content doesn't match — no boost

    return min(boost, 40)


# ── PIB SCRAPER ───────────────────────────────────────────────────────────────

def scrape_pib_headlines():
    """
    Scrape latest press releases from PIB (Press Information Bureau).
    Returns list of dicts: {title, summary, date}
    """
    facts = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
        resp = requests.get("https://pib.gov.in/allRel.aspx", headers=headers, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")

        # PIB press release links
        links = soup.select("div.content-area ul li a, .release-content a")[:20]
        for link in links:
            title = link.get_text(strip=True)
            if len(title) > 30:
                # Clean title into a fact statement
                clean = re.sub(r'\s+', ' ', title).strip()
                facts.append({
                    "title": clean,
                    "keyword": clean[:50].lower(),
                    "fact": f"[PIB] {clean}",
                    "source": "PIB"
                })
    except Exception as e:
        print(f"PIB scrape error: {e}")
    return facts[:10]


def scrape_wikipedia_current_events():
    """
    Scrape Wikipedia's 'Current events' portal for India-related facts.
    Returns list of dicts: {keyword, fact}
    """
    facts = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/Portal:Current_events",
            headers=headers, timeout=15
        )
        soup = BeautifulSoup(resp.content, "html.parser")

        # Wikipedia current events list items
        items = soup.select("div.current-events-content li")[:30]
        for item in items:
            text = item.get_text(strip=True)
            # Filter India-related
            india_keywords = ["india", "indian", "modi", "delhi", "mumbai", "pakistan",
                              "bharat", "isro", "bcci", "ipl", "rupee", "rbi"]
            if any(kw in text.lower() for kw in india_keywords) and len(text) > 40:
                # Create a short keyword from first 4 words
                words = text.split()[:4]
                keyword = " ".join(words).lower().strip(".,;:")
                fact = f"[Current Events] {text[:200]}"
                facts.append({"keyword": keyword, "fact": fact, "source": "Wikipedia"})
    except Exception as e:
        print(f"Wikipedia scrape error: {e}")
    return facts[:10]


def fetch_and_store_current_affairs(groq_client, db_conn):
    """
    ek click = saare static INDIAN_FACTS + PIB + Wikipedia sab DB mein store.
    Returns: (added_count, error_message)
    """
    import json as _json
    try:
        cur = db_conn.cursor()
        # Ensure table exists
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
        db_conn.commit()

        added = 0

        # ── STEP 1: Store ALL static INDIAN_FACTS into DB ─────────────────────
        for kw, fact in INDIAN_FACTS.items():
            kw   = kw.lower().strip()
            fact = fact.strip()
            if kw and fact:
                try:
                    cur.execute("""
                        INSERT INTO dynamic_facts (keyword, fact, source, updated_at)
                        VALUES (%s, %s, 'static', NOW())
                        ON CONFLICT (keyword) DO UPDATE
                            SET fact = EXCLUDED.fact, updated_at = NOW()
                    """, (kw, fact))
                    added += 1
                except Exception:
                    pass
        db_conn.commit()

        # ── STEP 2: Scrape PIB + Wikipedia ────────────────────────────────────
        pib_data  = scrape_pib_headlines()
        wiki_data = scrape_wikipedia_current_events()
        raw_items = pib_data + wiki_data

        if raw_items:
            raw_text = "\n".join([f"- {item['fact']}" for item in raw_items[:20]])
            prompt = f"""You are an Indian fact-checker. Convert these headlines into verified fact statements.

Headlines:
{raw_text}

Rules:
1. Return ONLY a JSON array, no other text
2. Each item: {{"keyword": "2-4 word lowercase key", "fact": "One clear factual sentence under 150 chars"}}
3. Skip opinions, only verifiable facts
4. Maximum 20 items

JSON array only:"""

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            response_text = response.choices[0].message.content.strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            s = response_text.find('[')
            e = response_text.rfind(']') + 1
            if s != -1 and e > s:
                facts_list = _json.loads(response_text[s:e])
                for item in facts_list:
                    kw   = item.get('keyword', '').lower().strip()
                    fact = item.get('fact', '').strip()
                    if kw and fact and len(kw) > 3 and len(fact) > 10:
                        try:
                            cur.execute("""
                                INSERT INTO dynamic_facts (keyword, fact, source, updated_at)
                                VALUES (%s, %s, 'pib_wiki', NOW())
                                ON CONFLICT (keyword) DO UPDATE
                                    SET fact = EXCLUDED.fact, updated_at = NOW()
                            """, (kw, fact))
                            added += 1
                        except Exception:
                            pass
                db_conn.commit()

        return added, None

    except Exception as e:
        return 0, str(e)

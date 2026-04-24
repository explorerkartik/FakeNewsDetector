# ── INDIAN FACTS DATABASE (Updated 2026) ─────────────────
import requests
from bs4 import BeautifulSoup

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

    # ── TECHNOLOGY & SPACE ──
    "isro missions": "ISRO's missions include Chandrayaan-3 (2023), Aditya-L1 (2023), Gaganyaan (upcoming).",
    "aditya l1": "Aditya-L1 is India's first solar mission launched by ISRO in September 2023.",
    "gaganyaan": "Gaganyaan is India's first crewed space mission planned by ISRO.",
    "digital india": "Digital India is a government initiative to transform India into a digitally empowered society.",
    "upi payments": "UPI processed over 15 billion transactions per month in 2024.",

    # ── AWARDS ──
    "bharat ratna 2024": "Bharat Ratna 2024: LK Advani, Karpoori Thakur, PV Narasimha Rao, MS Swaminathan, Chaudhary Charan Singh.",
    "bharat ratna": "Bharat Ratna is India's highest civilian award.",

    # ── LAWS ──
    "bns": "BNS (Bharatiya Nyaya Sanhita) replaced IPC from 1 July 2024.",
    "ipc": "IPC (Indian Penal Code) was replaced by Bharatiya Nyaya Sanhita (BNS) from July 2024.",
    "pocso": "POCSO (Protection of Children from Sexual Offences Act) was enacted in 2012.",
    "nrc": "NRC stands for National Register of Citizens.",
    "crpc": "CrPC was replaced by Bharatiya Nagarik Suraksha Sanhita (BNSS) from July 2024.",

    # ── STATE CAPITALS ──
    "capital of jharkhand": "Ranchi is the capital of Jharkhand.",
    "capital of bihar": "Patna is the capital of Bihar.",
    "capital of up": "Lucknow is the capital of Uttar Pradesh.",
    "capital of maharashtra": "Mumbai is the capital of Maharashtra.",
    "capital of rajasthan": "Jaipur is the capital of Rajasthan.",
    "capital of mp": "Bhopal is the capital of Madhya Pradesh.",
    "capital of gujarat": "Gandhinagar is the capital of Gujarat.",
    "capital of karnataka": "Bengaluru (Bangalore) is the capital of Karnataka.",
    "capital of tamil nadu": "Chennai is the capital of Tamil Nadu.",
    "capital of telangana": "Hyderabad is the capital of Telangana.",
    "capital of andhra pradesh": "Amaravati is the capital of Andhra Pradesh.",
    "capital of kerala": "Thiruvananthapuram is the capital of Kerala.",
    "capital of west bengal": "Kolkata is the capital of West Bengal.",
    "capital of punjab": "Chandigarh is the capital of Punjab.",
    "capital of haryana": "Chandigarh is the capital of Haryana.",
    "capital of odisha": "Bhubaneswar is the capital of Odisha.",
    "capital of assam": "Dispur is the capital of Assam.",
    "capital of himachal pradesh": "Shimla is the capital of Himachal Pradesh.",
    "capital of uttarakhand": "Dehradun is the capital of Uttarakhand.",
    "capital of chhattisgarh": "Raipur is the capital of Chhattisgarh.",
    "capital of goa": "Panaji is the capital of Goa.",

    # ── GOVERNMENT SCHEMES ──
    "pm kisan": "PM Kisan Samman Nidhi gives Rs 6000/year to farmers in 3 installments.",
    "ayushman bharat": "Ayushman Bharat provides Rs 5 lakh health cover per family per year.",
    "jan dhan": "PM Jan Dhan Yojana provides zero-balance bank accounts to all Indians.",
    "ujjwala yojana": "PM Ujjwala Yojana provides free LPG connections to BPL households.",
    "pm awas yojana": "PM Awas Yojana aims to provide affordable housing to all by 2024.",
    "swachh bharat": "Swachh Bharat Abhiyan was launched on 2 October 2014 by PM Modi for sanitation.",
    "startup india": "Startup India was launched on 16 January 2016 to promote entrepreneurship.",
    "make in india": "Make in India was launched on 25 September 2014 to boost manufacturing.",
    "atmanirbhar bharat": "Atmanirbhar Bharat (Self-Reliant India) was announced in 2020 during COVID-19.",
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
]


def check_indian_facts(text):
    """Check if text contains known Indian facts and return relevant info"""
    text_lower = text.lower()
    matched_facts = []
    seen = set()
    for key, fact in INDIAN_FACTS.items():
        if key in text_lower and fact not in seen:
            matched_facts.append(fact)
            seen.add(fact)
    return matched_facts[:4]


def get_credibility_boost(text):
    """Boost credibility score if text matches known Indian facts"""
    text_lower = text.lower()
    boost = 0
    for key in HIGH_CONFIDENCE_KEYS:
        if key in text_lower:
            boost += 20
    for key in INDIAN_FACTS.keys():
        if key in text_lower:
            boost += 8
    return min(boost, 50)


# ── PIB SCRAPER ──────────────────────────────────────────
def scrape_pib_facts(limit=10):
    """
    PIB (Press Information Bureau) se latest government news scrape karo.
    Returns list of fact strings.
    """
    facts = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        url = "https://pib.gov.in/AllRelease.aspx"
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')

        # PIB ke press release links
        links = soup.select('div.content-area a, .release-content a, ul li a')[:limit]
        for link in links:
            title = link.get_text(strip=True)
            if len(title) > 30:
                facts.append(title[:200])
    except Exception as e:
        print(f"PIB scrape error: {e}")

    return facts


def scrape_wikipedia_current_events(limit=15):
    """
    Wikipedia Current Events page se latest facts scrape karo.
    Returns list of fact strings.
    """
    facts = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        url = "https://en.wikipedia.org/wiki/Portal:Current_events"
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Current events ke bullet points
        items = soup.select('div.current-events-content li, .vevent li')
        for item in items[:limit]:
            text = item.get_text(strip=True)
            # Sirf India-related facts
            india_keywords = ['india', 'indian', 'modi', 'bjp', 'delhi', 'mumbai',
                              'pakistan', 'china', 'kashmir', 'isro', 'rupee']
            if any(kw in text.lower() for kw in india_keywords) and len(text) > 20:
                facts.append(text[:200])

    except Exception as e:
        print(f"Wikipedia scrape error: {e}")

    return facts


def generate_facts_with_groq(groq_client, raw_texts):
    """
    Groq AI se raw scraped text ko proper facts format mein convert karo.
    Returns list of (key, fact) tuples.
    """
    if not raw_texts:
        return []

    combined = "\n".join(f"- {t}" for t in raw_texts[:20])
    prompt = f"""Convert these news headlines into factual key-value pairs for an Indian fact-checker database.

Headlines:
{combined}

Rules:
1. Each fact must be a single clear sentence
2. Key should be 2-5 words (lowercase)
3. Fact should be specific and verifiable
4. Skip opinions or unclear items
5. Focus on India-related facts

Respond ONLY in this JSON format (no markdown):
[
  {{"key": "topic keyword", "fact": "Clear factual sentence about it."}},
  {{"key": "another topic", "fact": "Another factual sentence."}}
]"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        import json
        text = response.choices[0].message.content.strip()
        # Clean JSON
        if '```' in text:
            text = text.split('```')[1].replace('json', '').strip()
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end > start:
            text = text[start:end]
        pairs = json.loads(text)
        return [(p['key'].lower().strip(), p['fact']) for p in pairs if 'key' in p and 'fact' in p]
    except Exception as e:
        print(f"Groq fact generation error: {e}")
        return []


def auto_update_facts(groq_client, db_connection=None):
    """
    Main function: PIB + Wikipedia se scrape karo, Groq se format karo,
    database mein save karo.

    Returns: dict with status and count of new facts added
    """
    result = {
        'status': 'success',
        'pib_scraped': 0,
        'wiki_scraped': 0,
        'facts_generated': 0,
        'facts_saved': 0,
        'new_facts': []
    }

    # Step 1: Scrape sources
    pib_facts = scrape_pib_facts(limit=15)
    wiki_facts = scrape_wikipedia_current_events(limit=20)

    result['pib_scraped'] = len(pib_facts)
    result['wiki_scraped'] = len(wiki_facts)

    all_raw = pib_facts + wiki_facts
    if not all_raw:
        result['status'] = 'no_data'
        return result

    # Step 2: Convert to facts using Groq
    new_pairs = generate_facts_with_groq(groq_client, all_raw)
    result['facts_generated'] = len(new_pairs)

    if not new_pairs:
        result['status'] = 'generation_failed'
        return result

    # Step 3: Save to database if connection provided
    if db_connection:
        try:
            cur = db_connection.cursor()
            saved = 0
            for key, fact in new_pairs:
                # Duplicate check
                cur.execute(
                    "INSERT INTO current_affairs (fact_key, fact_text) VALUES (%s, %s) ON CONFLICT (fact_key) DO UPDATE SET fact_text = EXCLUDED.fact_text, updated_at = NOW()",
                    (key, fact)
                )
                saved += 1
            db_connection.commit()
            result['facts_saved'] = saved
        except Exception as e:
            print(f"DB save error: {e}")
            result['status'] = 'db_error'
            result['error'] = str(e)

    # Step 4: Also update in-memory INDIAN_FACTS
    for key, fact in new_pairs:
        INDIAN_FACTS[key] = fact
        result['new_facts'].append({'key': key, 'fact': fact})

    return result


def load_facts_from_db(db_connection):
    """
    Database se saved current affairs load karke INDIAN_FACTS mein add karo.
    App startup pe call karo.
    """
    try:
        import psycopg2.extras
        cur = db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT fact_key, fact_text FROM current_affairs ORDER BY updated_at DESC")
        rows = cur.fetchall()
        for row in rows:
            INDIAN_FACTS[row['fact_key']] = row['fact_text']
        print(f"Loaded {len(rows)} facts from database")
    except Exception as e:
        print(f"Load facts from DB error: {e}")

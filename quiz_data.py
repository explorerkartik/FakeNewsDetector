# ── QUIZ QUESTION BANK (Fake News Awareness) ────────────────────────────────
# Local question bank used as primary fallback when AI quiz generation fails.
# Format matches /api/quiz/generate response: text, answer (REAL/FAKE),
# category, explanation. Each question also has a difficulty tag.
#
# Usage:
#   from quiz_data import get_quiz_questions
#   questions = get_quiz_questions(count=10, difficulty='medium')

import random

QUIZ_QUESTIONS = [

    # ── EASY ──
    {'text': 'Banks call customers and ask for OTP to keep their account safe.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'easy',
     'explanation': 'Banks NEVER ask for OTP, PIN or CVV. Anyone asking for OTP is a fraudster.'},

    {'text': 'India ka national cyber crime helpline number 1930 hai.',
     'answer': 'REAL', 'category': 'Scam Awareness', 'difficulty': 'easy',
     'explanation': 'Dial 1930 or visit cybercrime.gov.in to report online financial fraud in India.'},

    {'text': 'WhatsApp forwards are always true if they have been forwarded many times.',
     'answer': 'FAKE', 'category': 'Media Literacy', 'difficulty': 'easy',
     'explanation': 'Forward count says nothing about truth. Viral messages are often misinformation.'},

    {'text': "Chandrayaan-3 successfully landed on the Moon's south pole in August 2023.",
     'answer': 'REAL', 'category': 'Science', 'difficulty': 'easy',
     'explanation': 'India became the first country to land near the lunar south pole on 23 August 2023.'},

    {'text': 'Free iPhone milta hai sirf ek link share karne se.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'easy',
     'explanation': 'Free gift offers spread through links are phishing scams to steal data or money.'},

    {'text': 'India won the ICC T20 World Cup 2024.',
     'answer': 'REAL', 'category': 'Sports', 'difficulty': 'easy',
     'explanation': 'India defeated South Africa in the final in Barbados in June 2024.'},

    {'text': 'Drinking hot water with lemon cures coronavirus.',
     'answer': 'FAKE', 'category': 'Health', 'difficulty': 'easy',
     'explanation': 'No home remedy cures COVID-19. This was a widely debunked viral forward.'},

    {'text': 'Narendra Modi is the Prime Minister of India.',
     'answer': 'REAL', 'category': 'Politics', 'difficulty': 'easy',
     'explanation': 'Narendra Modi has been the Prime Minister of India since May 2014.'},

    {'text': 'Agar message 10 logo ko forward nahi kiya to WhatsApp account band ho jayega.',
     'answer': 'FAKE', 'category': 'Viral Forwards', 'difficulty': 'easy',
     'explanation': 'WhatsApp never deletes accounts for not forwarding messages. Classic chain-message hoax.'},

    {'text': 'A news article without any source, date or author should be verified before sharing.',
     'answer': 'REAL', 'category': 'Media Literacy', 'difficulty': 'easy',
     'explanation': 'Missing source, date and author are classic red flags of unreliable news.'},

    # ── MEDIUM ──
    {'text': 'KBC lottery randomly selects WhatsApp numbers and gives 25 lakh rupees prizes.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'medium',
     'explanation': 'The "KBC lottery" is a well-known scam. KBC never runs WhatsApp lotteries.'},

    {'text': 'PIB Fact Check is the official fact-checking unit of the Government of India.',
     'answer': 'REAL', 'category': 'Media Literacy', 'difficulty': 'medium',
     'explanation': 'PIB Fact Check verifies claims about government policies and schemes.'},

    {'text': 'Police can arrest you over a video call and demand money. This is called digital arrest.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'medium',
     'explanation': '"Digital arrest" is not a legal concept in India. It is a scam tactic — hang up and report to 1930.'},

    {'text': 'Cyber fraud complaints can be filed online at cybercrime.gov.in.',
     'answer': 'REAL', 'category': 'Scam Awareness', 'difficulty': 'medium',
     'explanation': 'cybercrime.gov.in is the official national cyber crime reporting portal.'},

    {'text': '5G mobile towers spread the COVID-19 virus.',
     'answer': 'FAKE', 'category': 'Health', 'difficulty': 'medium',
     'explanation': 'Viruses cannot travel on radio waves. This conspiracy theory was debunked worldwide.'},

    {'text': 'Neeraj Chopra won a silver medal at the Paris Olympics 2024.',
     'answer': 'REAL', 'category': 'Sports', 'difficulty': 'medium',
     'explanation': 'Neeraj Chopra won silver in javelin throw at Paris 2024.'},

    {'text': 'Electricity department SMS me personal mobile number bhejta hai disconnection rokne ke liye.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'medium',
     'explanation': 'Disconnection threats with personal numbers are scams. Always check dues on the official app.'},

    {'text': 'Reverse image search se viral photo ki asli source check kar sakte hain.',
     'answer': 'REAL', 'category': 'Media Literacy', 'difficulty': 'medium',
     'explanation': 'Google Lens / reverse image search reveals where an image originally appeared.'},

    {'text': 'COVID vaccines contain microchips to track people.',
     'answer': 'FAKE', 'category': 'Health', 'difficulty': 'medium',
     'explanation': 'Vaccines contain no microchips. This conspiracy theory has been thoroughly debunked.'},

    {'text': 'Article 370 was abrogated in August 2019.',
     'answer': 'REAL', 'category': 'Politics', 'difficulty': 'medium',
     'explanation': 'Article 370 was abrogated on 5 August 2019, removing the special status of Jammu & Kashmir.'},

    {'text': 'Ghar baithe 2 ghante kaam karke 50,000 mahina guaranteed — aisi jobs hamesha genuine hoti hain.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'medium',
     'explanation': 'Unrealistic salary promises with registration fees are classic fake job scams.'},

    {'text': "ISRO's Aditya-L1 is India's first solar mission.",
     'answer': 'REAL', 'category': 'Science', 'difficulty': 'medium',
     'explanation': 'Aditya-L1, launched in September 2023, is India\'s first dedicated solar observatory mission.'},

    # ── HARD ──
    {'text': 'Genuine lotteries require you to have purchased a ticket before you can win.',
     'answer': 'REAL', 'category': 'Scam Awareness', 'difficulty': 'hard',
     'explanation': 'You cannot win a lottery you never entered. "Winning" without a ticket is always a scam.'},

    {'text': 'Customs department phone karke parcel chhudane ke liye online fine transfer karne ko kehta hai.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'hard',
     'explanation': 'Customs never demands payment via phone calls. This is the courier/digital-arrest scam.'},

    {'text': "Deepfake videos can mimic a person's face and voice, so video alone is not always proof.",
     'answer': 'REAL', 'category': 'Media Literacy', 'difficulty': 'hard',
     'explanation': 'AI deepfakes are increasingly realistic. Verify videos with trusted news sources before believing.'},

    {'text': 'If a screenshot of a news channel is viral, the news must be true.',
     'answer': 'FAKE', 'category': 'Media Literacy', 'difficulty': 'hard',
     'explanation': 'Screenshots are easily morphed. Check the channel\'s official website or handle to verify.'},

    {'text': 'TRAI calls users and threatens to block their SIM within 2 hours for illegal activity.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'hard',
     'explanation': 'TRAI does not call individuals about SIM blocking. These automated calls are scams.'},

    {'text': 'Checking URL spelling (like amaz0n.in instead of amazon.in) helps identify phishing websites.',
     'answer': 'REAL', 'category': 'Scam Awareness', 'difficulty': 'hard',
     'explanation': 'Phishing sites use lookalike domains with swapped characters. Always check the address bar.'},

    {'text': 'OLX par army officer advance payment bhejne ke liye aapka UPI PIN maangta hai.',
     'answer': 'FAKE', 'category': 'Scam Awareness', 'difficulty': 'hard',
     'explanation': 'Receiving money NEVER requires entering your UPI PIN. PIN is only needed to SEND money.'},

    {'text': "BNS replaced IPC as India's criminal law from July 2024.",
     'answer': 'REAL', 'category': 'Politics', 'difficulty': 'hard',
     'explanation': 'Bharatiya Nyaya Sanhita (BNS) replaced the Indian Penal Code from 1 July 2024.'},
]


def get_quiz_questions(count=10, difficulty='medium', category='mixed'):
    """
    Return quiz questions from the local bank.
    Filters by difficulty when enough questions exist, otherwise uses the full bank.
    Output format matches the AI-generated quiz API response.
    """
    count = max(1, min(int(count), len(QUIZ_QUESTIONS)))

    pool = [q for q in QUIZ_QUESTIONS if q['difficulty'] == difficulty]
    if len(pool) < count:
        pool = QUIZ_QUESTIONS[:]

    if category and category != 'mixed':
        filtered = [q for q in pool if q['category'].lower() == category.lower()]
        if len(filtered) >= count:
            pool = filtered

    selected = random.sample(pool, min(count, len(pool)))
    return [{'text': q['text'], 'answer': q['answer'],
             'category': q['category'], 'explanation': q['explanation']}
            for q in selected]

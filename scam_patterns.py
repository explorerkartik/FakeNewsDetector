# ── SCAM / PHISHING PATTERN DETECTOR ────────────────────────────────────────
# Detects common Indian scam patterns in news text / forwarded messages:
#   lottery wins, KYC fraud, electricity disconnection, free gifts,
#   OTP fraud, fake bank alerts, digital arrest, fake jobs, courier scams
#
# Usage:
#   from scam_patterns import detect_scam_patterns
#   result = detect_scam_patterns(text)  # → {'is_scam', 'risk_score', 'matched', 'warning'}

import re

SCAM_PATTERNS = {

    'lottery_scam': {
        'label':  'Lottery / Prize Scam',
        'weight': 35,
        'patterns': [
            r'\bkbc\b.*\b(lottery|lucky|winner|jeet\w*|prize)\b',
            r'\blottery\b.*\b(won|winner|jeet\w*|jita|jiti|claim)\b',
            r'\b(won|jeeta|jeete|jiti|jita)\b.*\b\d+\s*(lakh|crore|lakhs|crores)\b',
            r'\blucky\s+draw\b',
            r'\bclaim\s+(your|the)\s+prize\b',
        ],
        'advice': 'Genuine lotteries never contact random people. Never pay a fee or share details to claim a prize.',
    },

    'kyc_fraud': {
        'label':  'KYC / Account Fraud',
        'weight': 35,
        'patterns': [
            r'\bkyc\b.*\b(update|expire\w*|block\w*|suspend\w*|pending)\b',
            r'\b(account|khata)\b.*\b(block\w*|freeze|suspend\w*|band)\b',
            r'\bpan\s*card\b.*\b(link|block\w*|deactivat\w*|update)\b',
        ],
        'advice': 'Banks never ask to update KYC via links or calls. Use only your bank branch or official app.',
    },

    'otp_fraud': {
        'label':  'OTP / PIN Fraud',
        'weight': 40,
        'patterns': [
            r'\botp\b.*\b(share|send|batao|bhejo|tell|bata)\b',
            r'\b(share|send|batao|bhejo)\b.*\botp\b',
            r'\b(cvv|upi\s*pin|atm\s*pin)\b',
        ],
        'advice': 'Never share OTP, PIN or CVV with anyone. Banks and companies never ask for them.',
    },

    'electricity_scam': {
        'label':  'Electricity Bill Scam',
        'weight': 30,
        'patterns': [
            r'\b(electricity|bijli|power)\b.*\b(disconnect\w*|cut|kat\w*|band)\b',
            r'\bbill\b.*\b(pending|due)\b.*\b(call|contact|number)\b',
        ],
        'advice': 'Electricity boards never send disconnection threats with personal mobile numbers. Check dues on the official app.',
    },

    'digital_arrest': {
        'label':  'Digital Arrest / Courier Scam',
        'weight': 45,
        'patterns': [
            r'\bdigital\s+arrest\b',
            r'\b(cbi|police|customs|narcotics|crime\s+branch)\b.*\b(arrest|warrant|case)\b.*\b(call|video|pay|transfer|paise)\b',
            r'\bparcel\b.*\b(seized|seize|customs|drugs|illegal)\b',
        ],
        'advice': '"Digital arrest" is NOT a real legal process in India. Hang up immediately and report at 1930.',
    },

    'free_gift_scam': {
        'label':  'Free Gift / Recharge Scam',
        'weight': 30,
        'patterns': [
            r'\bfree\b.*\b(recharge|iphone|laptop|scooty|gift|mobile|data)\b',
            r'\b(spin|scratch)\b.*\b(win|gift|prize)\b',
            r'\banniversary\s+offer\b',
        ],
        'advice': 'Free gift offers asking you to click links or share messages are scams.',
    },

    'fake_job_scam': {
        'label':  'Fake Job / Work-from-Home Scam',
        'weight': 30,
        'patterns': [
            r'\b(work\s+from\s+home|ghar\s+baithe)\b.*\b(earn|kamao|salary|\d+)\b',
            r'\bregistration\s+fee\b',
            r'\b(telegram|whatsapp)\b.*\b(task|job|earn|paid)\b',
            r'\bearn\b.*\b\d{3,}\b.*\b(daily|monthly|per\s+day|roz)\b',
        ],
        'advice': 'Genuine employers never ask for registration fees. Telegram/WhatsApp "paid task" jobs are scams.',
    },

    'fake_bank_alert': {
        'label':  'Fake Bank Alert',
        'weight': 35,
        'patterns': [
            r'\b(account|card|sim)\b.*\b(blocked|suspended|band)\b.*\b(click|link|call|press)\b',
            r'\bbank\s+manager\b.*\b(bol|speaking|calling)\b',
            r'\bverify\b.*\b(account|card|details)\b.*\b(link|click)\b',
        ],
        'advice': 'Banks never send block warnings with links. Call only the official number printed on your card.',
    },

    'govt_scheme_fraud': {
        'label':  'Fake Government Scheme',
        'weight': 25,
        'patterns': [
            r'\b(yojana|scheme|sarkar|government)\b.*\b(free|\d+\s*(rupye|rupees|rs))\b.*\b(register|link|form|apply)\b',
            r'\bberojgari\s+bhatta\b',
        ],
        'advice': 'Verify schemes only on official .gov.in websites or PIB Fact Check.',
    },

    'urgency_pressure': {
        'label':  'Urgency / Pressure Tactic',
        'weight': 15,
        'patterns': [
            r'\b(within|sirf|only)\s+24\s+hours?\b',
            r'\b(turant|immediately|urgent|urgently|abhi|aaj\s+hi)\b',
            r'\b(warna|otherwise|or\s+else)\b.*\b(block\w*|band|arrest|cancel|delete)\b',
            r'\bforward\b.*\b\d+\s*(logo|log|people|contacts|groups)\b',
        ],
        'advice': 'Scammers create false urgency. Genuine institutions give proper notice through official channels.',
    },

    'suspicious_link': {
        'label':  'Suspicious Link',
        'weight': 15,
        'patterns': [
            r'\b(bit\.ly|tinyurl|t\.co|cutt\.ly|rb\.gy|short\.link)\b',
            r'\bclick\s+(here|this|kare|karein|karo)\b',
        ],
        'advice': 'Shortened or unknown links can be phishing. Type official website addresses manually.',
    },
}


def detect_scam_patterns(text):
    """
    Scan text for common Indian scam/phishing patterns.
    Returns: {'is_scam': bool, 'risk_score': 0-100, 'matched': [...], 'warning': str}
    """
    if not isinstance(text, str) or not text.strip():
        return {'is_scam': False, 'risk_score': 0, 'matched': [], 'warning': ''}

    lowered = text.lower()
    matched = []
    score   = 0

    for key, info in SCAM_PATTERNS.items():
        for pattern in info['patterns']:
            if re.search(pattern, lowered):
                matched.append({'type': key, 'label': info['label'], 'advice': info['advice']})
                score += info['weight']
                break  # one match per scam type is enough

    score   = min(100, score)
    is_scam = score >= 30

    warning = ''
    if is_scam:
        labels  = ', '.join(m['label'] for m in matched)
        warning = (f"⚠️ Possible scam detected ({labels}). "
                   f"Do not click links, share OTP/PIN, or send money. "
                   f"Report fraud at helpline 1930 or cybercrime.gov.in.")

    return {'is_scam': is_scam, 'risk_score': score, 'matched': matched, 'warning': warning}

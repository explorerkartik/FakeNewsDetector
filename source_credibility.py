# ── SOURCE CREDIBILITY DATABASE ──────────────────────────────────────────────
# Features:
#   1. SOURCE_CREDIBILITY — domain → score (0-100) + description
#   2. get_source_credibility(url) — domain lookup with subdomain support
#   3. FACT_CHECK_SOURCES — Indian & global fact-checking organisations
#   4. suggest_fact_checkers() — list to show 'verify on these fact-checkers'

from urllib.parse import urlparse

SOURCE_CREDIBILITY = {

    # ── GOVERNMENT / OFFICIAL (95-100) ──
    'pib.gov.in':            {'score': 98, 'desc': 'Press Information Bureau — official Government of India source'},
    'india.gov.in':          {'score': 98, 'desc': 'National Portal of India'},
    'rbi.org.in':            {'score': 98, 'desc': 'Reserve Bank of India — official'},
    'isro.gov.in':           {'score': 98, 'desc': 'ISRO — official Indian space agency'},
    'eci.gov.in':            {'score': 98, 'desc': 'Election Commission of India'},
    'mygov.in':              {'score': 95, 'desc': 'Government citizen engagement platform'},
    'who.int':               {'score': 95, 'desc': 'World Health Organization'},

    # ── INTERNATIONAL WIRE SERVICES / BROADCASTERS (80-97) ──
    'reuters.com':           {'score': 97, 'desc': 'International wire service'},
    'apnews.com':            {'score': 96, 'desc': 'Associated Press — international wire service'},
    'afp.com':               {'score': 95, 'desc': 'Agence France-Presse'},
    'bbc.com':               {'score': 94, 'desc': 'British public broadcaster'},
    'bbc.co.uk':             {'score': 94, 'desc': 'British public broadcaster'},
    'theguardian.com':       {'score': 88, 'desc': 'Major UK newspaper'},
    'nytimes.com':           {'score': 88, 'desc': 'Major US newspaper'},
    'washingtonpost.com':    {'score': 87, 'desc': 'Major US newspaper'},
    'bloomberg.com':         {'score': 88, 'desc': 'International business news'},
    'ft.com':                {'score': 89, 'desc': 'Financial Times'},
    'dw.com':                {'score': 87, 'desc': 'Deutsche Welle — German public broadcaster'},
    'aljazeera.com':         {'score': 84, 'desc': 'International broadcaster'},
    'cnn.com':               {'score': 80, 'desc': 'US news channel'},
    'cnbc.com':              {'score': 82, 'desc': 'Business news channel'},

    # ── INDIAN WIRE SERVICES ──
    'ptinews.com':           {'score': 92, 'desc': 'Press Trust of India — wire service'},
    'aninews.in':            {'score': 82, 'desc': 'Asian News International — wire service'},

    # ── FACT CHECKERS (85-95) ──
    'altnews.in':            {'score': 93, 'desc': 'Independent Indian fact-checker'},
    'boomlive.in':           {'score': 91, 'desc': 'IFCN-certified Indian fact-checker'},
    'factly.in':             {'score': 90, 'desc': 'Indian data journalism & fact-checking'},
    'factchecker.in':        {'score': 88, 'desc': 'Indian fact-checking initiative'},
    'vishvasnews.com':       {'score': 87, 'desc': 'IFCN-certified Hindi fact-checker'},
    'snopes.com':            {'score': 88, 'desc': 'Global fact-checking website'},
    'politifact.com':        {'score': 88, 'desc': 'US fact-checking website'},

    # ── MAJOR INDIAN NEWSPAPERS (80-92) ──
    'thehindu.com':                  {'score': 90, 'desc': 'Major Indian newspaper'},
    'indianexpress.com':             {'score': 89, 'desc': 'Major Indian newspaper'},
    'livemint.com':                  {'score': 87, 'desc': 'Indian business newspaper'},
    'hindustantimes.com':            {'score': 86, 'desc': 'Major Indian newspaper'},
    'economictimes.indiatimes.com':  {'score': 86, 'desc': 'Indian business newspaper'},
    'business-standard.com':         {'score': 86, 'desc': 'Indian business newspaper'},
    'thehindubusinessline.com':      {'score': 86, 'desc': 'Indian business newspaper'},
    'timesofindia.indiatimes.com':   {'score': 84, 'desc': 'Major Indian newspaper'},
    'deccanherald.com':              {'score': 84, 'desc': 'Major Indian newspaper'},
    'telegraphindia.com':            {'score': 84, 'desc': 'Major Indian newspaper'},
    'tribuneindia.com':              {'score': 83, 'desc': 'Major Indian newspaper'},

    # ── INDIAN TV / DIGITAL NEWS (70-85) ──
    'ndtv.com':                          {'score': 85, 'desc': 'Indian news channel'},
    'indiatoday.in':                     {'score': 83, 'desc': 'Indian news channel & magazine'},
    'theprint.in':                       {'score': 79, 'desc': 'Indian digital news outlet'},
    'thequint.com':                      {'score': 78, 'desc': 'Indian digital news outlet'},
    'scroll.in':                         {'score': 78, 'desc': 'Indian digital news outlet'},
    'newslaundry.com':                   {'score': 78, 'desc': 'Indian media-watch & news outlet'},
    'aajtak.in':                         {'score': 78, 'desc': 'Hindi news channel'},
    'jagran.com':                        {'score': 78, 'desc': 'Major Hindi newspaper'},
    'bhaskar.com':                       {'score': 76, 'desc': 'Major Hindi newspaper'},
    'news18.com':                        {'score': 76, 'desc': 'Indian news network'},
    'abplive.com':                       {'score': 76, 'desc': 'Hindi news channel'},
    'thewire.in':                        {'score': 76, 'desc': 'Indian digital news outlet'},
    'amarujala.com':                     {'score': 75, 'desc': 'Major Hindi newspaper'},
    'firstpost.com':                     {'score': 74, 'desc': 'Indian digital news outlet'},
    'wionews.com':                       {'score': 74, 'desc': 'Indian international news channel'},
    'navbharattimes.indiatimes.com':     {'score': 74, 'desc': 'Hindi newspaper'},
    'zeenews.india.com':                 {'score': 72, 'desc': 'Hindi news channel'},
    'indiatvnews.com':                   {'score': 70, 'desc': 'Hindi news channel'},

    # ── MIXED RELIABILITY (40-69) ──
    'timesnownews.com':      {'score': 65, 'desc': 'Indian news channel — verify sensational claims'},
    'oneindia.com':          {'score': 60, 'desc': 'News aggregator — verify with original source'},
    'republicworld.com':     {'score': 58, 'desc': 'Indian news channel — verify sensational claims'},
    'dailyhunt.in':          {'score': 55, 'desc': 'News aggregator — credibility depends on original source'},

    # ── SATIRE (15-25) — NOT REAL NEWS ──
    'theonion.com':          {'score': 20, 'desc': 'SATIRE — not real news'},
    'babylonbee.com':        {'score': 20, 'desc': 'SATIRE — not real news'},
    'fakingnews.com':        {'score': 20, 'desc': 'SATIRE — Indian satirical website, not real news'},
    'theunrealtimes.com':    {'score': 20, 'desc': 'SATIRE — Indian satirical website, not real news'},

    # ── KNOWN MISINFORMATION / UNRELIABLE (0-35) ──
    'opindia.com':               {'score': 30, 'desc': 'Repeatedly flagged by fact-checkers; IFCN certification rejected'},
    'postcard.news':             {'score': 10, 'desc': 'Known misinformation — flagged by AltNews and BoomLive'},
    'naturalnews.com':           {'score': 15, 'desc': 'Known health misinformation website'},
    'beforeitsnews.com':         {'score': 10, 'desc': 'Known conspiracy/misinformation website'},
    'infowars.com':              {'score': 10, 'desc': 'Known conspiracy/misinformation website'},
    'worldnewsdailyreport.com':  {'score': 5,  'desc': 'Fabricated news website'},
    'empirenews.net':            {'score': 5,  'desc': 'Fabricated news website'},
    'huzlers.com':               {'score': 5,  'desc': 'Fabricated news website'},
    'react365.com':              {'score': 5,  'desc': 'User-generated fake news prank website'},
}


def _label_for_score(score):
    if score >= 70:
        return 'Trusted'
    if score >= 40:
        return 'Mixed'
    return 'Unreliable'


def get_source_credibility(url):
    """Extract domain from a URL and return credibility score + label."""
    try:
        domain = urlparse(url).netloc.lower().replace('www.', '')
        if not domain:
            return {'found': False, 'label': 'Unknown'}
        # Exact match
        if domain in SOURCE_CREDIBILITY:
            entry = SOURCE_CREDIBILITY[domain]
            return {'found': True, 'domain': domain, 'score': entry['score'],
                    'label': _label_for_score(entry['score']), 'desc': entry['desc']}
        # Subdomain match (e.g. sports.ndtv.com → ndtv.com)
        parts = domain.split('.')
        for i in range(1, len(parts) - 1):
            parent = '.'.join(parts[i:])
            if parent in SOURCE_CREDIBILITY:
                entry = SOURCE_CREDIBILITY[parent]
                return {'found': True, 'domain': parent, 'score': entry['score'],
                        'label': _label_for_score(entry['score']), 'desc': entry['desc']}
        return {'found': False, 'domain': domain, 'score': None, 'label': 'Unknown',
                'desc': 'Source not in credibility database — verify independently.'}
    except Exception:
        return {'found': False, 'label': 'Unknown'}


# ── FACT-CHECKING ORGANISATIONS ──────────────────────────────────────────────
FACT_CHECK_SOURCES = [
    {'name': 'PIB Fact Check',             'url': 'https://pib.gov.in/factcheck.aspx',                  'desc': 'Official Government of India fact-checking unit'},
    {'name': 'AltNews',                    'url': 'https://www.altnews.in',                             'desc': 'Independent Indian fact-checker'},
    {'name': 'BoomLive',                   'url': 'https://www.boomlive.in',                            'desc': 'IFCN-certified Indian fact-checker'},
    {'name': 'Factly',                     'url': 'https://factly.in',                                  'desc': 'Indian data journalism & fact-checking portal'},
    {'name': 'Vishvas News',               'url': 'https://www.vishvasnews.com',                        'desc': 'IFCN-certified fact-checker (Hindi + 11 languages)'},
    {'name': 'India Today Fact Check',     'url': 'https://www.indiatoday.in/fact-check',               'desc': 'Fact-check desk of India Today'},
    {'name': 'The Quint WebQoof',          'url': 'https://www.thequint.com/news/webqoof',              'desc': 'Fact-check desk of The Quint'},
    {'name': 'Google Fact Check Explorer', 'url': 'https://toolbox.google.com/factcheck/explorer',      'desc': 'Search fact-checks from publishers worldwide'},
]


def suggest_fact_checkers(limit=5):
    """Return top fact-checking organisations to suggest for manual verification."""
    return FACT_CHECK_SOURCES[:limit]

"""
Auroic Router Dataset Generator v4
====================================
9300 samples | 4 action types | H5+C1-3 input format
New spec: text/react/media/ignore only
No translate, no acknowledge, no canonical labels for text
React TITLE = emoji, Media TITLE = search query (max 3 words)
@BOT windows, filler windows, variable candidates
Think tier stamping for annotation pipeline

Output format:
  R: TYPE=text | TARGET=C2 | EFFORT=low/medium/high
  R: TYPE=react | TARGET=C1 | TITLE=🔥
  R: TYPE=media | TARGET=C3 | TITLE=crying laughing
  R: TYPE=ignore
"""

import json, random, re, hashlib, string
from collections import Counter, defaultdict
from difflib import SequenceMatcher

try:
    import emoji as emoji_lib
    _HAS_EMOJI = True
except ImportError:
    _HAS_EMOJI = False

try:
    import markovify
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False

try:
    from faker import Faker
    _fake_hi = Faker('hi_IN')
    _fake_en = Faker('en_IN')
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION CONFIG — v4 spec
# ═══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Total base samples before filler/bot buckets
    "normal_samples": 7000,
    "bot_samples": 1500,
    "filler_samples": 800,

    # Within normal 7000
    "type_dist": {
        "text":   0.40,   # 2800
        "react":  0.175,  # 1225
        "media":  0.175,  # 1225
        "ignore": 0.25,   # 1750
    },

    # Think tier distribution (applied during annotation, stamped in generator)
    # hard=full think, medium=short think, easy=no think
    "think_tiers": {
        "hard":   0.40,
        "medium": 0.35,
        "easy":   0.25,
    },

    "lang_dist": {
        "hinglish": 0.72,
        "english":  0.28,
    },

    "dedup_threshold": 0.69,
}

SYSTEM = (
    "You are the Auroic Router. "
    "Given history messages H1-H5 and candidate messages C1-C3, "
    "output exactly one routing decision."
)

# ═══════════════════════════════════════════════════════════════════════════════
# MASSIVE VOCABULARY — Indian GC language, WhatsApp style
# ═══════════════════════════════════════════════════════════════════════════════

# Core Hinglish fillers — raw WhatsApp/Instagram GC energy
HI_FILLERS = [
    # Single words
    "hm","ok","lol","bro","haan","nahi","kya","yaar","ugh","ohhh","nice",
    "accha","theek hai","haha","hmm","okay","sure","chal","dekh","sun","bhai",
    "arre","seriously","fr","ngl","imo","sach mein","bata","acha","haan haan",
    "bas","chalo","sun na","dekh na","bol","kya yaar","pagal hai kya","sahi bola",
    "bilkul","nope","yup","nahi re","wahi toh","mujhe pata tha","sahi hai",
    "chal theek","aur bata","kya scene hai","bata na","sun bhai","acha acha",
    "bol na","kya hua phir","seriously yaar","arre haan","matlab","wait kya",
    "huh","abey","oye","arrey","achi baat","noted","got it","samajh gaya",
    "pata hai","toh","phir","kaise","kab","kidhar","dikha","batao na","suno bhai",
    "haan ji","nahi ji","ekdum","pakka","zaroor","koi baat nahi","ho jayega",
    "chhodo","jaane do","rehne do","chup","sab theek","tension mat le",
    "kya farak padta","mujhe kya","bhai sahab","boss","bhai please","yaar please",
    "ek sec","ruk ja","baith","chal hat","jhooth mat bol","kasam se",
    "sach bol raha","maan le","sun toh sahi","dekh le","kar le","hota hai",
    "normal hai","aisa hota hai","koi nahi","arey wah","kya baat","mast",
    "zabardast","badhiya","shandar","kamaal","bakwas","faltu","bekaar","wahiyat",
    "ghatiya","chalo koi nahi","yaar soch","bhai wait","ek min","bas kar",
    "apna kaam kar","chal na yaar","haha true","lol sahi","bhai pagal","yaar 💀",
    "kuch nahi","bas yahi","theek theek","haan bhai","nahi yaar","okay okay",
    "acchi baat hai","wah wah","dang","bhai bruh","yaar bro","oye hoye",
    "seedha baat","sidhi baat","lagta hai","lagnaa nahi","feel ho raha",
    "bhai seriously","yaar honestly","matlab kya","kya matlab","scene kya hai",
    "kya toh","bhai kya","yaar kya","aur kya","bas yaar","chal bhai",
    # Longer fillers
    "bhai kuch nahi bas","yaar mood nahi hai aaj","okay noted bhai",
    "haan haan samajh gaya","chal koi nahi yaar","bhai theek hai",
    "acha go on bhai","lol bhai sahi bola","yaar same here tbh",
    "bhai wahi soch raha tha","okay fair enough yaar","bhai lol",
    "haan yaar wahi toh","bhai ek sec ruk","yaar dekh toh",
    "accha bhai sahi bola","lmao bhai true","yaar okay okay",
    "bhai hmm samajh gaya","yaar chalo koi nahi",
]

EN_FILLERS = [
    "lol","okay","haha","nice","cool","gotcha","makes sense","yeah","sure",
    "no way","interesting","and then","wait what","okay and","right","really",
    "fair enough","i see","hmm","got it","exactly","true","same","fair",
    "i mean","well","anyway","so","oh","ah","wait","nope","yep","mhm",
    "alright","honestly","literally","actually","basically","lowkey","fr",
    "ngl","imo","tbh","rn","omg","wtf","bruh","oof","lmao","lmfao","xd",
    "i guess","whatever","sure thing","totally","absolutely","def","prolly",
    "legit","straight up","on god","thats crazy","wild","insane","nuts",
    "tell me about it","same here","mood","big mood","relatable","felt that",
    "cant even","im done","im dead","no shot","cap","valid","facts","period",
    "say less","bet","idk","wdym","smh","fyi","icymi","nvm","iirc","ikr",
    "for real tho","thats wild bro","oh really","you think so","not wrong",
    "deadass","sheesh","bussin","slay","ate that","its giving","understood",
    "no thoughts","just vibes","im weak","naww","naur","innit","bruv","mate",
    "proper","mad","bare","peak","peng","wagwan","allow it","mandem",
    "ting","swerve","calm","link","gassed","butters","piff","neek",
    "proper mad","well good","bare jokes","that hits different","caught in 4k",
    "not gonna lie tho","genuinely though","lowkey though","ngl fr fr",
    "okay but hear me out","wait actually","hold on though","but like",
]

# Indian internet slang + regional flavors
DESI_SLANG = [
    # Pure desi
    "bc","bsdk","mf","mc","chutiya","harami","bakchod","jhatu","kutte",
    "kamine","teri","teri maa","teri behen","teri aukat","aukat mein reh",
    "apni aukat dekh","chapri","tapori","lafanga","badmaash","rowdy",
    "gunda","mawaali","nalayak","nikamma","berozgaar","bewaqoof","gadha",
    "ullu","tharki","chalu","chamcha","sycophant","chamchi","bootlicker",
    # Soft versions / commonly used
    "abe","oye","sala","saale","yaar sala","bhai sala","abe yaar",
    "abe sala","oye hoye","arey bhai","arre yaar","abe bhai",
    # Praise
    "ekdum mast","full toing","full power","full josh","full scene",
    "dabang","dhamaka","dum hai","paisa vasool","bindaas","jhakaas",
    "fatafat","jhakkas","bambaiya","purana style","classic",
    # Regional flavor words (sanitized for training)
    "machcha","da","pa","anna","akka","bhaiya","didi","dada","didi",
    "macha","kya re","are re","arre bhai","sun re","dekh re",
    "kya re bhai","bhai re bhai","arre re arre","haye haye",
    # Mumbai
    "apun","apna","tension nahi lene ka","ekdum sahi","full on",
    "kya baat bhai","bhai log","solid hai","pukka","pakka bhai",
    # Delhi
    "bhai ek kaam kar","bhai sun na","yaar pakka","solid banda",
    "paisa phekh tamasha dekh","dilli wale","setting ho gayi",
    # Hyderabad
    "kya scene hai bhai","bhai kya hua","bhai solid","scene set",
    # Bangalore
    "da machcha","bro scene","machche","kya machcha","boss solid",
]

# WhatsApp specific patterns
WHATSAPP_PATTERNS = [
    # Typing indicators
    "...","….",",,,,","…",
    # Voice note style
    "sun bhai voice note bhej","bhai call kar","yaar call karte hain",
    # Forwarded message style  
    "forwarded","🔁","📨",
    # Common WA responses
    "seen","read","delivered","blue tick diya","ticked","read kiya",
    "bhai seen kar liya","yaar read nahi kiya abhi","kab se read kiya",
    # Sticker reactions (text form)
    "haha sticker","lol meme","😂 sticker","bhai meme bhej",
    # Time references
    "raat ko baat karte","subah baat karte","kal baat karte",
    "bhai free hai kya","yaar time hai kya","busy hai kya",
]

# Instagram GC specific
INSTA_PATTERNS = [
    # Story reactions
    "bhai story dekhi","yaar reel dekh","bhai post dekh",
    "bhai highlight dekh","yaar bio dekh","bhai close friends mein add kar",
    # DM patterns
    "bhai dm kar","yaar message kar","bhai reply kar reel pe",
    # Engagement
    "bhai like kar","yaar comment kar","bhai share kar",
    "tag kar mujhe","mention kar","bhai collab kar",
    # Aesthetic
    "filter kaunsa tha","preset share kar","edit kaise ki",
    "bhai lightroom preset","yaar vsco filter",
]

# Emoji fillers — extensive Indian GC usage
EMOJI_FILLERS = [
    "😂","💀","🔥","😭","🙄","👀","🤣","😅","🥴","😤","🫡","🤦","💯",
    "😮","🤯","😬","🫠","😑","🤔","🙃","😏","👍","✅","❤️","🫶","🤝",
    "👏","🎉","🥳","😎","💪","🫣","😳","🤌","🙏","⚡","🌚","🌝","🌊",
    "🐐","👑","🚀","💥","🎯","🧠","💡","✨","🌈","🎭","🎪","🏆","🥇",
    "🫰","🤙","✌️","🤞","👊","🫂","💅","🦾","👁️","🗣️","💬","🔊",
    "🎶","🎵","🎤","🎧","📸","🖼️","🎮","🕹️","⚽","🏏","🏀","🎾",
    "🍕","🍔","☕","🧋","🍿","🌮","🎂","🍰","🍜","🍱","🥘","🍛",
    "😈","👿","🤡","🥸","🫥","😶‍🌫️","🤐","🤫","🧐","🥺","🥹",
    "🩷","🧡","💛","💚","💙","💜","🖤","🤍","🤎","❤️‍🔥","💔",
    "🌙","⭐","🌟","💫","☀️","🌈","❄️","🌊","🍃","🌺","🌸",
    # Commonly used in India specifically
    "🙏","🫡","💐","🎊","🪔","🏏","🐄","🦁","🐯","🦊","🐺",
    "🇮🇳","🏳️","🎌","🏴","🚩","⚑",
]

# Hindi romanized starters — very diverse
HI_STARTERS = [
    "bhai","yaar","arre","suno","dekh","bro","sun","hey","oye","abey",
    "guys","bhai sun","yaar sun","arre bhai","ek baat bata","suno zara",
    "bhai ek min","yaar ek cheez","oye bhai","bhai bata na",
    "dekho","suno na","arre yaar","boss sun","bhai dekh","yaar bata",
    "bhai sach","oye sun","arre dekh","yaar please","bhai finally",
    "sun yaar","arre sun na","bhai ek sec","yaar ek baat","oye yaar",
    "bhai honestly","yaar fr","bhai ngl","yaar sach mein","bhai ek cheez",
    "yaar seriously","bhai ekdum","oye seriously","arre seriously",
    "bhai lowkey","yaar highkey","bhai deadass","yaar lagta hai",
    "sun bhai sun","dekh yaar dekh","bhai bhai bhai","yaar yaar yaar",
    "ek second bhai","ek minute yaar","bhai quick","yaar fast",
    "bhai important","yaar urgent","bhai jaldi","yaar abhi",
    "scene ye hai bhai","baat ye hai yaar","update ye hai bhai",
    "news ye hai yaar","situation ye hai bhai",
]

EN_STARTERS = [
    "hey","so","guys","anyone","quick","ok so","wait","listen","btw",
    "honestly","ngl","lowkey","literally","bro","yo","omg","wait so",
    "real talk","genuine question","okay but","not gonna lie",
    "dude","fam","bruh","alright so","hold on","get this","you know what",
    "hear me out","i swear","no joke","dead serious","fun fact",
    "random but","off topic but","idk if this matters but",
    "so basically","here is the thing","update","quick update",
    "heads up","fyi","tbh","ngl though","lowkey though",
    "okay hear me out","wait actually","hold up","pause",
    "scene is","situation is","thing is","update is",
]

# Slang that hits different in Indian context
SLANG_FILLERS = [
    "bc","wtf","ngl","fr","lol","lmao","bro ngl","fr fr","ong","no cap",
    "lowkey","highkey","deadass","bet","slay","bussin","sheesh","yeet",
    "vibe","based","sus","ratio","W","L","cope","seethe","mid",
    "fam","goated","ate","ate that","served","purr","periodt","tea",
    "spill","rent free","main character","delulu","its giving",
    "touch grass","skill issue","ggs","rip","big W","massive L","chad",
    "cringe","vibing","no thoughts","just vibes","im weak","naww","naur",
    # India specific internet slang
    "ded","mawa","paaji","bhai sahab","dost","yarr","bhaiya",
    "solid","jhakaas","bindaas","mast","zabardast","toing","dhamaka",
    "paisa vasool","timepass","jugaad","jugadu","senti","funda",
    "enthu","scene","setting","connection","influence","pehchaan",
    "desi","videshi","firangi","angrezi","madrasi","bhaiya ji",
    "ji haan","ji nahi","bilkul ji","zaroor ji","haan ji theek hai",
]

# Natural conversation continuations (non-signal, after target)
HI_CONTINUATIONS = [
    "haan dekho","accha theek hai","phir kya hua","aur bata","hmm sahi hai",
    "okay bhai","chal theek","samajh gaya","haan haan","lol okay",
    "sahi bola","bilkul","wahi toh","haan okay","chal","phir","aur",
    "accha accha","theek lagta hai","hmm","okay okay","haan sahi",
    "woh toh hai","baat toh sahi hai","accha sun","haan yaar","okay cool",
    "nice bhai","sahi hai yaar","hmm interesting","accha concept hai",
    "okay noted","got it bhai","understood yaar","clear hai",
    "haan bhai samjha","yaar theek bol raha","bhai sahi point",
    "okay fair","haan fair point","accha valid hai","theek hai bhai",
    "chal koi nahi","hota hai yaar","normal hai bhai","sab theek ho jayega",
    "tension mat le","chal jayega","ho jayega","pakka ho jayega",
    "dekho ho jayega","yaar chill kar","bhai relax kar","ek cheez at a time",
    "step by step","dheeraj rakh","patience yaar","sab acha hoga",
    "👍","✅","😌","🙏","💯","👀","🤝","😎",
]

EN_CONTINUATIONS = [
    "yeah makes sense","okay cool","interesting","right right","hmm true",
    "fair enough","got it","yeah i see","oh okay","makes sense now",
    "sure sure","yep","true that","oh right","yeah exactly","same here",
    "honestly yeah","lowkey true","ngl yeah","fr fr","oh wait true",
    "valid point","that tracks","understandable","okay that makes sense",
    "yeah i get it","oh interesting","wait that actually makes sense",
    "noted","understood","got it bro","cool cool","alright alright",
    "yeah fair","that's fair","makes sense honestly","okay yeah",
    "👍","✅","😌","fr","👀","💯","same","mood",
]

# ═══════════════════════════════════════════════════════════════════════════════
# MASSIVE TOPIC BANKS — 3x more diverse, real Indian GC scenarios
# ═══════════════════════════════════════════════════════════════════════════════

TEXT_TOPICS = {
    "coding": [
        # Classic errors
        ("bhai ye python error samajh nahi aa raha kuch bhi kar lo help","high"),
        ("yaar js mein TypeError aa raha hai undefined property pe crash","medium"),
        ("bhai git conflict ho gaya merge karna hai resolve kaise karoon","medium"),
        ("yaar api 404 de rahi hai baaki sab theek hai sirf ye nahi","medium"),
        ("bhai css layout pura toot gaya flexbox se nahi ban raha","low"),
        ("yaar sql query galat data de rahi hai join ka issue lagta","medium"),
        ("bhai react component render nahi ho raha console pe error hai","medium"),
        ("yaar docker container start nahi ho raha port se issue hai","medium"),
        ("bhai npm install fail ho raha dependency conflict aa raha","medium"),
        ("yaar app white screen hai kuch load hi nahi ho raha","medium"),
        ("bro mera code kaam nahi kar raha aaj se tab se try kar raha","medium"),
        ("yaar ye wala error samjhao please kabse dekh raha hun","medium"),
        ("bhai production pe deploy kiya crash ho gaya sab down hai","high"),
        ("yaar senior ne PR reject kar diya reason samajh nahi aaya","medium"),
        ("bhai kal presentation hai aur code chal nahi raha yaar help","high"),
        ("yaar leetcode medium bhi solve nahi ho raha kuch nahi pata","medium"),
        ("bhai system design kaise seekhun kuch samajh nahi aa raha","medium"),
        ("yaar open source mein contribute karna hai kaise shuru karoon","medium"),
        ("bhai github pe portfolio banani hai guidance do please","low"),
        ("yaar code review mein sabne roast kar diya kya karoon ab","medium"),
        # Framework/infra
        ("bhai cors error har request mein block ho rahi hai fix karo","medium"),
        ("yaar firebase auth kaam nahi kar raha signin fail ho rahi","medium"),
        ("bhai nextjs 13 app router samajh nahi aa raha pages se alag","medium"),
        ("yaar prisma migration fail ho rahi hai database lock hua","medium"),
        ("bhai vercel pe build fail ho raha locally chal raha hai","medium"),
        ("yaar redis cache stale data de raha invalidation kaise karoon","medium"),
        ("bhai kafka consumer lag bahut hai messages late aa rahe","medium"),
        ("yaar nginx 502 de raha upstream ko chhod ke sab theek hai","medium"),
        ("bhai kubernetes pod crash loop mein hai logs nahi aa rahe","high"),
        ("yaar jest test flaky hai kabhi pass kabhi fail random","medium"),
        # AI/ML
        ("bhai openai api rate limit bahut jaldi aa jaata hai fix karo","medium"),
        ("yaar langchain tool calling kaam nahi kar raha agent stuck","medium"),
        ("bhai huggingface model download nahi ho raha network timeout","medium"),
        ("yaar ollama model bahut slow hai CPU pe almost unusable","medium"),
        ("bhai stable diffusion OOM error aa raha hai GPU hai mere paas","medium"),
        # Mobile
        ("bhai flutter build fail ho raha android pe ios pe theek hai","medium"),
        ("yaar react native metro bundler crash ho raha hai fresh install","medium"),
        ("bhai android studio emulator start nahi ho raha slow bhi hai","medium"),
        ("yaar ios simulator app install nahi ho raha cert issue lagta","medium"),
        ("bhai unity scene corrupt ho gayi backup nahi tha kya karoon","high"),
        # Extended set
        ("yaar websocket connection drop ho rahi hai random intervals pe","medium"),
        ("bhai graphql resolver infinite loop mein ghoom raha hai","medium"),
        ("yaar typescript strict mode on kiya 200 errors aa gaye bhai","medium"),
        ("bhai memory leak hai node process din bhar mein crash karta","high"),
        ("yaar CI/CD pipeline fail ho rahi hai environment variable missing","medium"),
        ("bhai mongodb atlas connection pool khatam ho jaata hai peak pe","medium"),
        ("yaar tailwind classes kaam nahi kar rahin purge issue lagta","low"),
        ("bhai authentication token expire ho jaata hai random time pe","medium"),
        ("yaar rate limiting implement karni hai kaise karoon express pe","medium"),
        ("bhai socket.io rooms mein broadcast nahi ho raha sirf sender ko","medium"),
        ("yaar cron job ek hi baar chali do baar nahi bhai kya issue","medium"),
        ("bhai pdf generate ho rahi hai lekin images blank aa rahi hain","medium"),
        ("yaar search functionality slow hai 5 second lag raha bhai","medium"),
        ("bhai swagger docs generate nahi ho rahi typescript pe yaar","low"),
        ("yaar microservices mein service discovery kaise karoon consul","medium"),
        ("bhai encryption ka seedha example chahiye AES 256 node pe","medium"),
        ("yaar cloudflare workers pe deploy kiya timeout aa raha hai","medium"),
        ("bhai git rebase vs merge dono use kab karte explain karo","low"),
        ("yaar postgresSQL indexing kab karni chahiye explain karo","medium"),
        ("bhai vim shortcuts sikhne hain basic workflow ke liye please","low"),
        ("yaar regex likhni hai email validate karni hai help karo","low"),
        ("bhai webscraping kaise karoon python mein basic tutorial do","medium"),
        ("yaar browser extension banana chahta hun kahan se shuru karoon","medium"),
        ("bhai chrome dev tools se performance debug kaise karte hain","medium"),
        ("yaar docker compose kaise likhoon 3 services hain mere paas","medium"),
        ("bhai ssh key setup kaise karte hain github ke liye step by step","low"),
        ("yaar vim vs vscode kya better hai beginners ke liye honestly","low"),
        ("bhai python asyncio samajh nahi aata synchronous se kya alag","medium"),
        ("yaar selenium test likhni hai first time guidance do please","medium"),
        ("bhai two sum to hard problems jump karna chahta hun roadmap","medium"),
        ("yaar design patterns kab use karte hain real projects mein","medium"),
    ],

    "study_exam": [
        # Exam panic
        ("bhai kal exam hai kuch nahi pada help karo please raat bhar","high"),
        ("yaar 3 din mein boards hai syllabus 40 percent bhi nahi pada","high"),
        ("bhai aaj last minute hai kya kya zaroori hai batao please","high"),
        ("yaar exam hall mein dimag blank ho jaata hai anxiety hai","high"),
        ("bhai marks kaise laoon subject pasand nahi hai boring hai","medium"),
        # Subject help
        ("yaar organic chemistry mechanisms samajh hi nahi aate help","medium"),
        ("bhai integration by parts kab use karte hain aur kab nahi","medium"),
        ("yaar thermodynamics ke laws sab mix ho jaate hain clear karo","medium"),
        ("bhai english essay structure kaise hoti hai marks ke liye","medium"),
        ("yaar history dates yaad nahi rehte trick batao please","low"),
        # College
        ("bhai attendance shortage ho gayi kya karu detain toh nahi na","high"),
        ("yaar assignment plagiarism pakad gaya sir ko kya bolun help","high"),
        ("bhai group project mein koi kaam nahi kar raha sab pe aa raha","medium"),
        ("yaar backlog subjects ka kya karoon 3 hain abhi tak","medium"),
        ("bhai college dropout karna chahta hun genuine advice do","high"),
        ("yaar JEE advanced clear nahi hua ab kya karoon please help","high"),
        ("bhai NEET mein 500 aaye hain college kaun kaun milenge","medium"),
        ("yaar CAT preparation kaise karoon working professional hun","medium"),
        ("bhai UPSC attempt karna chahta hun realistic hai kya honestly","medium"),
        ("yaar foreign university ke liye SOP kaise likhoon tips do","medium"),
        ("bhai DSA kitna important hai non-CS background hun","medium"),
        ("yaar coding bootcamp worth it hai kya 6 lakh ka hai yaar","medium"),
        ("bhai placement ke liye kya prepare karoon 6 months mein","high"),
        ("yaar resume ATS friendly kaise banaaoon tips do please","medium"),
        ("bhai mock interview practice kahan se karoon best resources","medium"),
        # Extended
        ("yaar physics numericals mein formula confusion hoti hai bhai","medium"),
        ("bhai maths mein coordinate geometry weak hai kaise badhaaun","medium"),
        ("yaar english reading comprehension mein time nahi milta tips","medium"),
        ("bhai accountancy journal entries yaad nahi rehte trick do","low"),
        ("yaar biology diagrams kaise yaad rakhe exam ke liye bhai","low"),
        ("bhai cbse boards mein presentation marks milte hain kya","low"),
        ("yaar open book exam mein kaise approach karoon first time","medium"),
        ("bhai viva mein ghbra jaata hun kaise confident rahun tips","medium"),
        ("yaar internals mein kam marks hain external pe depend hai sab","high"),
        ("bhai reference books zyada hain kaunsa best hai jeemains ke","medium"),
        ("yaar study group banana chahta hun productive kaise karoon","low"),
        ("bhai pomodoro technique try ki but nahi chal rahi mujhe","medium"),
        ("yaar notes banana hai concise wale kaise format karoon","low"),
        ("bhai college mein ek bhi dost nahi hai introvert hun help","high"),
        ("yaar hostel mein padhna mushkil hai shor bahut hai tips do","medium"),
        ("bhai online classes attend nahi kar paata focus nahi hota","medium"),
        ("yaar previous year papers solve karne chahiye kitne honest","medium"),
        ("bhai sem ke 2 din pehle padhna shuru kiya tips dono cover","high"),
        ("yaar professors se doubt poochna dar lagta hai kaise karoon","medium"),
        ("bhai plagiarism checker kaunsa free aur reliable hai bhai","low"),
        ("yaar self study vs coaching dono try kiye confused hun bhai","medium"),
        ("bhai IELTS ke liye kitne mahine chahiye 7 band ke liye","medium"),
        ("yaar GRE preparation kaise karoon 6 mahine mein reasonable","medium"),
        ("bhai college transfer possible hai kya dono deemed hain","medium"),
        ("yaar scholarship ke liye apply karna hai kahan se shuru karoon","medium"),
    ],

    "relationships_family": [
        # Romantic
        ("bhai crush ko confess karna hai kaise karoon help me please","medium"),
        ("yaar breakup hua kuch din pehle bahut bura lag raha hai","high"),
        ("bhai partner se bahut fight hoti hai roz alag baat pe","medium"),
        ("yaar long distance mein trust issues aa rahe hain kya karun","medium"),
        ("bhai propose karna hai special tarika chahiye ideas do","low"),
        ("yaar ex ne text kiya hai kya karoon confused hun bhai","medium"),
        ("bhai one sided love hai 2 saal se kya bataaun kaise","medium"),
        ("yaar arrange marriage ke liye ghar wale bol rahe hain advice","medium"),
        ("bhai partner bahut controlling hai kya normal hai kya","high"),
        ("yaar ghosted ho gaya crush ne kya hua samajh nahi aaya","medium"),
        # Friendship
        ("bhai best friend ne galat baat bol di bahut hurt hua mujhe","medium"),
        ("yaar friend group mein bahut drama ho raha hai kitne dino se","medium"),
        ("bhai yaar ne baad mein distance banana shuru kar diya kyun","medium"),
        ("yaar purani dosti toot rahi hai 5 saal ki kya karoon bhai","high"),
        ("bhai toxic dost hain chhodna chahta hun kaise karoon","medium"),
        # Family
        ("yaar parents career mein bahut interfere kar rahe hain help","medium"),
        ("bhai ghar mein din raat fights hoti hain bohot thak gaya hun","high"),
        ("yaar mummy ko mental health ke baare mein kaise bataaun","high"),
        ("bhai parents divorce ho raha hai adjust karna mushkil hai","high"),
        ("yaar ghar wale shaadi ke liye bohot zyada pressure de rahe","medium"),
        # Mental health
        ("bhai bahut akela feel ho raha hai kisi se baat nahi hoti","high"),
        ("yaar confidence ekdum zero hai kaise banaoon please help","medium"),
        ("bhai social anxiety hai log se milna mushkil lagta hai","high"),
        ("yaar overthinking band nahi hoti raat ko neend nahi aati","medium"),
        ("bhai depression ke signs hain kya sab kuch pointless lagta","high"),
        # Extended
        ("yaar dost ne bina bataye meri baat share kar di kya karoon","medium"),
        ("bhai jealousy feel hoti hai yaar ki success pe kya karoon","medium"),
        ("yaar best friend ki girlfriend mujhe pasand nahi friction hai","medium"),
        ("bhai online relationship hai real mein milna chahta hun tips","medium"),
        ("yaar age difference hai 5 saal ki serious hai kya lagta hai","medium"),
        ("bhai flirting ho rahi thi but ab awkward ho gaya kya karoon","medium"),
        ("yaar girlfriend maa se nahi milna chahti tension ho rahi","high"),
        ("bhai dost ne mujhse paise maange awkward ho gaya relationship","medium"),
        ("yaar shared house mein roommate ke saath issue hai landlord","medium"),
        ("bhai sibling bahut argue karta hai solution kya hai bhai","medium"),
        ("yaar nana nani bohot bimar hain emotionally drain ho raha hun","high"),
        ("bhai shaadi mein parents ka choice pasand nahi kya bolunga","high"),
        ("yaar first relationship mein hun kuch samajh nahi aata","medium"),
        ("bhai breakup ke baad usse block karoon ya nahi dilemma hai","medium"),
        ("yaar apology kaise karoon badi fight ke baad serious wali","medium"),
        ("bhai dost ki problem solve nahi hoti sunta hun but helpless","medium"),
        ("yaar grief chal raha hai relative gaye hain kaise cope karoon","high"),
        ("bhai social media pe ex ki life dekh ke bura lagta hai","medium"),
        ("yaar commitment phobia hai genuinely kya karoon therapy do","high"),
        ("bhai in-laws se adjust nahi ho raha hai shaadi ke baad","high"),
        ("yaar platonic friendship romantic ho rahi hai confusing hai","medium"),
        ("bhai mummy papa ko meri naukri se stress hai samjhao kaise","medium"),
        ("yaar childhood trauma affect kar raha hai relationships ko","high"),
    ],

    "career_work": [
        ("bhai offer aaya hai switch karoon ya current mein rehna chahiye","medium"),
        ("yaar FAANG ka offer hai but location nahi pasand advice do","medium"),
        ("bhai salary negotiate kaise karoon first time hai offer mein","medium"),
        ("yaar toxic workplace hai resign karna chahta hun guide do","medium"),
        ("bhai promotion ke liye manager se kaise baat karoon tips do","medium"),
        ("yaar freelancing full time karna chahta hun jump karoon kya","medium"),
        ("bhai startup join karoon ya MNC advice do genuinely please","medium"),
        ("yaar notice period pe hoon counter offer aaya hai kya karoon","medium"),
        ("bhai side hustle shuru karna chahta hun ideas do please","low"),
        ("yaar LinkedIn profile improve karne ke tips do placement ke","low"),
        ("bhai remote job kaise milegi abroad ke liye guide do yaar","medium"),
        ("yaar appraisal mein low rating mili unfair laga kya karoon","medium"),
        ("bhai 1 saal se ek hi company mein jump karoon kya advice","medium"),
        ("yaar product manager banana chahta hun developer hun currently","medium"),
        ("bhai content creator banana chahta hun kaise shuru karoon","low"),
        ("yaar YouTube channel growth nahi ho rahi 6 mahine ho gaye","medium"),
        ("bhai instagram mein monetize karne ka plan hai advice do","low"),
        ("yaar podcast start karna hai tips do please equipment bhi","low"),
        ("bhai stock market mein seriously invest karna chahta hun guide","medium"),
        ("yaar mutual funds vs FD kya better hai beginner ke liye","low"),
        # Extended
        ("bhai work life balance khatam ho gaya roz 12 ghante kaam","high"),
        ("yaar performance improvement plan pe hun kya karoon bhai","high"),
        ("bhai internship convert karni hai full time kaise impress karoon","medium"),
        ("yaar cold email kaise likhoon job ke liye HR ko bhai","medium"),
        ("bhai job gap explain karna hai 1 saal ka interview mein","medium"),
        ("yaar career change karna chahta hun 30 ke baad realistic kya","medium"),
        ("bhai kaam pe politics bahut hai survive kaise karoon bhai","medium"),
        ("yaar manager mujhe credit nahi deta kya karoon yaar","medium"),
        ("bhai burnout aa gaya hai sab se quit karna chahta hun","high"),
        ("yaar interview mein 5 baar reject hua kya galat kar raha hun","high"),
        ("bhai freelance client payment nahi kar raha kya karoon legal","medium"),
        ("yaar startup ka valuation kaise calculate karte hain early stage","medium"),
        ("bhai MBA karun ya domain expertise badhaaun kya better hai","medium"),
        ("yaar technical interview ke liye 1 mahina hai plan do bhai","high"),
        ("bhai equity vs salary kaise evaluate karoon offer mein","medium"),
        ("yaar government job ke liye prepare karna chahta hun guide","medium"),
        ("bhai data science mein career banana chahta hun roadmap do","medium"),
        ("yaar ek hi client pe depend hun diversify kaise karoon","medium"),
        ("bhai client ke saath disagreement hai project mein kya karoon","medium"),
        ("yaar work from home productivity nahi ho rahi tips do please","medium"),
        ("bhai team mein conflict chal raha hai main bhi involved hun","medium"),
        ("yaar performance review mein kaise present karoon achievements","medium"),
        ("bhai references dene wala koi nahi hai fresher hun help","medium"),
        ("yaar part time job chahiye studies ke saath kahan dhundhun","low"),
        ("bhai employer ko relocation allowance maangna hai kaise bolun","low"),
    ],

    "health_wellness": [
        ("bhai sar mein bahut dard hai din bhar se tablet bhi khayi","medium"),
        ("yaar raat ko 3 baje tak neend nahi aati kya karoon help","medium"),
        ("bhai gym chhodh diya 2 mahine se motivate karo please yaar","low"),
        ("yaar weight loss plateau aa gaya 2 kg se nahi jaa raha","medium"),
        ("bhai healthy khaana shuru karna chahta hun guide do yaar","low"),
        ("yaar back pain bahut hai din bhar chair pe rehta hun office","medium"),
        ("bhai screen se aankhein bahut thak gayi hain kya karoon","low"),
        ("yaar anxiety ke liye kuch natural remedies batao please","medium"),
        ("bhai stress management tips do genuine wale please help","medium"),
        ("yaar meditation shuru karna chahta hun kaise karoon guide","low"),
        ("bhai smoking chhodna chahta hun 3 baar try kar chuka hun","medium"),
        ("yaar hair fall bahut ho raha hai kya khaun kya lagaaun","medium"),
        ("bhai skin dry ho rahi hai winter mein routine kya honi","low"),
        ("yaar pani kam peeta hun kaise badhaaun tips do please","low"),
        ("bhai posture bahut kharab hai laptop pe kaise theek karoon","low"),
        # Extended
        ("yaar thyroid ki problem hai weight control nahi ho raha bhai","medium"),
        ("bhai vitamin D deficiency hai doctor ne bola tips do please","low"),
        ("yaar PCOD hai diet mein kya changes karoon help karo bhai","medium"),
        ("bhai acne bahut ho rahi hai 22 saal ki umar mein yaar","low"),
        ("yaar periods mein bahut dard hota hai normal hai kya bhai","medium"),
        ("bhai gym mein injury ho gayi shoulder ki kya karoon rest ya","medium"),
        ("yaar creatine lena chahta hun safe hai kya beginners ke liye","medium"),
        ("bhai intermittent fasting try karni hai kaise shuru karoon","low"),
        ("yaar protein intake kaise badhaaun vegetarian hun bhai","low"),
        ("bhai cholesterol high hai 24 saal mein lifestyle tips do","medium"),
        ("yaar panic attacks aa rahe hain kaise handle karoon in public","high"),
        ("bhai OCD ke signs hain lag raha hai kya karoon please help","high"),
        ("yaar caffeine addiction ho gayi hai din mein 5 cups chai bhai","low"),
        ("bhai migraine bahut aata hai triggers kya hain tips do","medium"),
        ("yaar periods irregular hain doctor ke alawa kya kar sakta hun","medium"),
        ("bhai 6 pak ke liye realistic timeline kya hai natural mein","low"),
        ("yaar sugar craving control nahi hoti tips do genuine wale","low"),
        ("bhai running shuru karni hai beginner plan do please yaar","low"),
        ("yaar body fat percentage kaise measure karoon ghar pe bhai","low"),
        ("bhai diabetes risk hai family mein diet tips do please","medium"),
        ("yaar teeth sensitivity bahut hai kya karoon dentist ke alawa","low"),
    ],

    "casual_lifestyle": [
        ("bhai bura nahi lagega toh bata phone kab lungu 15k budget","low"),
        ("yaar trip plan karni hai manali ya goa summer mein suggest","low"),
        ("bhai movie recommendation do weekend ke liye mood thriller","low"),
        ("yaar book suggest karo self help wali genuine wali please","low"),
        ("bhai boring ho raha hun ghar pe kya karoon kuch batao","low"),
        ("yaar cooking seekhna chahta hun basic se shuru karoon kaise","low"),
        ("bhai driving seekhni hai kitne dino mein seekh jaata hai","low"),
        ("yaar pet lena chahta hun konsa best rehega flat mein","low"),
        ("bhai gift ideas do girlfriend ke liye birthday aane wala","low"),
        ("yaar fashion sense improve karna chahta hun tips do please","low"),
        ("bhai gaming setup banana chahta hun budget 30k hai guide","low"),
        ("yaar digital detox karna chahta hun kaise karoon properly","low"),
        ("bhai productivity badhani hai procrastinate bahut karta hun","medium"),
        ("yaar time management sikhna chahta hun student hun please","medium"),
        ("bhai morning routine banana chahta hun early bird banna hai","low"),
        # Extended
        ("yaar solo travel karna chahta hun pehli baar tips do bhai","low"),
        ("bhai budget mein ghumna chahta hun 5k mein 3 din suggest","low"),
        ("yaar anime suggest karo beginner hun kahan se shuru karoon","low"),
        ("bhai web series khatam ho gayi naya kuch suggest karo please","low"),
        ("yaar spotify pe playlist banana chahta hun aesthetic wali","low"),
        ("bhai language sikhni hai korean ya japanese kaunsi easier","low"),
        ("yaar photography seekhni hai DSLR ya mirrorless beginner ke","low"),
        ("bhai minimalism try karna chahta hun kahan se shuru karoon","low"),
        ("yaar journaling shuru karna chahta hun format kya hona chahiye","low"),
        ("bhai declutter karna chahta hun room ko tips do konmari wala","low"),
        ("yaar YouTube pe kya dekhun productive wala content suggest","low"),
        ("bhai social media se break lena chahta hun kaise cold turkey","low"),
        ("yaar roommate dhundhna hai Bangalore mein kahan post karoon","low"),
        ("bhai secondhand phone lena chahta hun kya check karoon list","low"),
        ("yaar laptop bag suggest karo office ke liye under 2k please","low"),
        ("bhai earphones suggest karo budget 1500 mein best wale","low"),
        ("yaar plants lagana chahta hun ghar pe indoor wale suggest","low"),
        ("bhai weekend akela boring ho raha hun solo activities batao","low"),
        ("yaar bike trip plan karni hai first time group mein tips","low"),
        ("bhai online courses worth it hain udemy ya coursera suggest","low"),
        ("yaar chai ya coffee health ke liye kya better hai genuinely","low"),
        ("bhai freelance portfolio site banana chahta hun free me","low"),
        ("yaar meditation app suggest karo free wali beginner ke liye","low"),
        ("bhai sustainable living shuru karna chahta hun tips do please","low"),
        ("yaar tinder vs bumble kya better hai Indian market mein","low"),
        ("bhai board games suggest karo 4 logon ke liye fun wale","low"),
    ],

    "life_events_updates": [
        # Good news
        ("bhai finally internship mil gayi Google mein kuch nahi socha tha","high"),
        ("yaar placement ho gayi bhai package 12 LPA first attempt","high"),
        ("bhai startup ka first paying customer aa gaya yaar kya feeling","high"),
        ("yaar app store pe app publish hua finally 6 mahine ka kaam","high"),
        ("bhai 10k followers ho gaye channel pe abhi abhi dekha bhai","medium"),
        # Tough news
        ("yaar exam mein fail ho gaya kya karun ab parents ko kaise bataaun","high"),
        ("bhai layoff ho gaya company se unexpectedly kya karun aage","high"),
        ("yaar phone kho gaya sab data gaya backup nahi tha help karo","medium"),
        ("bhai online scam mein 5k gaye lesson learn kiya advice do","medium"),
        ("yaar pet dog mar gaya 8 saal ka tha bahut miss kar raha hun","high"),
        # Milestones
        ("bhai pehli salary aayi hai 40k celebrate karte hain yaar","medium"),
        ("yaar driving license mil gaya finally 3rd attempt mein yaar","medium"),
        ("bhai apna ghar liya 1BHK rent pe first time alone rehna","medium"),
        ("yaar 100 din streak complete ki gym ki kaise laga feel acha","medium"),
        ("bhai blood pressure normal ho gaya 6 mahine mein lifestyle change se","medium"),
        # Extended
        ("yaar desh chhodh ke ja raha hun 2 hafte mein emotional hun","high"),
        ("bhai pehli baar ghar se bahar rehna hai hostel anxiety hai","medium"),
        ("yaar accident hua tha theek hun but shaken hun bhai help","high"),
        ("bhai badi surgery hone wali hai dar lag raha hai kya karoon","high"),
        ("yaar rishta toot gaya tha lekin patch up ho gaya confused hun","medium"),
        ("bhai pehla car kharida aaj lifelong dream tha feel alag hai","medium"),
        ("yaar scholarship reject ho gayi alternative kya hai bhai","high"),
        ("bhai relative ka nidhan hua grief mein hun kaise cope karoon","high"),
        ("yaar pehli baar vote kiya feeling describe nahi ho sakti yaar","low"),
        ("bhai startup band karni pad rahi hai emotionally mushkil hai","high"),
        ("yaar 30 saal ke ho gaye aaj kuch alag feel ho raha bhai","medium"),
        ("bhai COVID ke baad pehli baar ghar gaya parents ko mila yaar","medium"),
        ("yaar promotion reject ho gayi without reason kya karoon bhai","high"),
        ("bhai long distance relationship finally end hua relief bhi sad bhi","high"),
        ("yaar pehli flight thi akele nervous tha but ho gaya bhai","low"),
        ("bhai naukri chhod ke higher studies jaana chahta hun advice","high"),
        ("yaar bhai ki engagement fix ho gayi ghar mein shor mach gaya","medium"),
        ("bhai flood mein ghar nuksaan hua recover kaise karoon yaar","high"),
        ("yaar 1 crore ka milestone chhua freelance mein feel bata karoon","high"),
        ("bhai therapy shuru ki hai life pehle se better lag rahi hai","medium"),
    ],

    "finance_money": [
        ("bhai credit card bill nahi bhar pa raha minimum bhi mushkil hai","high"),
        ("yaar investment shuru karna chahta hun 22 saal ka hun kahan se","low"),
        ("bhai EMI bahut ho gayi hai cash flow tight hai kya karoon","high"),
        ("yaar term insurance lena chahta hun kaise choose karoon bhai","medium"),
        ("bhai tax filing pehli baar kar raha hun kuch samajh nahi aata","medium"),
        ("yaar UPI fraud hua hai 2k gaye kya recovery possible hai","medium"),
        ("bhai SIP kaisa start karoon 5k per month available hai","low"),
        ("yaar crypto mein 50k lagaye hain ab 20k hai kya karoon","high"),
        ("bhai emergency fund kitna hona chahiye starting mein please","low"),
        ("yaar home loan ke liye credit score improve karna hai kaise","medium"),
        ("bhai freelance income ka tax kaise file karoon first time","medium"),
        ("yaar parents ke liye health insurance lena hai guide do bhai","medium"),
        ("bhai PPF vs NPS kya better hai long term ke liye bhai","low"),
        ("yaar budget banana chahta hun monthly wala template do please","low"),
        ("bhai student loan le liya tha repayment kaise plan karoon","medium"),
        ("yaar gold mein invest karna chahta hun digital ya physical","low"),
        ("bhai share market mein pehla step kaise loon demat account","low"),
        ("yaar 6 mahine ka sabbatical le raha hun financial plan do","medium"),
        ("bhai rent vs buy ghar kaunsa better hai 28 saal mein bhai","medium"),
        ("yaar F.I.R.E movement kya hota hai realistic hai kya India mein","medium"),
    ],

    "tech_gadgets": [
        ("bhai laptop suggest karo college ke liye budget 50k under","low"),
        ("yaar iphone 15 vs samsung s24 honestly kaunsa better hai","low"),
        ("bhai wifi router suggest karo 2bhk ke liye bha lag lag","low"),
        ("yaar mechanical keyboard worth it hai ya normal se kaam chalega","low"),
        ("bhai monitor suggest karo coding ke liye 24 inch under 15k","low"),
        ("yaar smartwatch suggest karo fitness ke liye under 5k please","low"),
        ("bhai NAS setup karna chahta hun home ke liye guide do bhai","medium"),
        ("yaar VPN worth it hai India mein genuinely use case kya hai","low"),
        ("bhai RAM upgrade karni hai laptop ki possible hai kya bhai","low"),
        ("yaar 4G vs 5G kya practical difference hai daily use mein","low"),
        ("bhai portable charger suggest karo best under 1k please","low"),
        ("yaar tablet lena chahta hun drawing ke liye suggest karoon","low"),
        ("bhai SSD upgrade worth it hai 5 saal purane laptop mein","low"),
        ("yaar webcam suggest karo work from home ke liye budget wala","low"),
        ("bhai raspberry pi se kya kya bana sakte hain beginners ke liye","low"),
    ],
}

# MEDIA_TOPICS — implicit situational, NO explicit send/bhej requests
# Structure: {category: [(msg, search_query_max_3_words), ...]}
MEDIA_TOPICS = {
    "funny_moment": [
        ("teacher ne galti se apni dating app screen share kar di class mein 😂", "teacher dating app"),
        ("yaar office mein chair toot gayi meeting mein boss gir gaya sabke saamne", "chair falling fail"),
        ("bhai autocorrect ne itna embarrassing msg bhej diya crush ko ab kya karoon", "autocorrect fail meme"),
        ("cafeteria mein tray le ke ja raha tha pura gir gaya yaar 💀", "cafeteria tray fall"),
        ("teacher ne apna tiktok accidentally projector pe chala diya class mein lmao", "teacher tiktok fail"),
        ("bhai galti se galat group mein personal msg chala gaya sabne padh liya", "wrong group message"),
        ("presentation mein slide galat lag gayi meme wali tha wahan pe 😭", "wrong slide meme"),
        ("bhai zoom pe background nahi lagaya tha room ka pura mess dikha", "zoom fail background"),
        ("yaar interview mein puchha kya weakness hai bola main hi bol deta hun", "interview funny answer"),
        ("bhai ek baar mein 3 log gaye ghar pe galat address diya tha", "wrong address funny"),
        ("mummy ne accidentally mera screenshot stories pe share kar diya yaar 💀", "mom screenshot fail"),
        ("bhai boss ko reply kiya tha yaar tujhe bheja tha galti se wahi gaya", "wrong person message"),
        ("yaar rickshaw wale ne itna weird baat ki ride mein maza aa gaya", "funny rickshaw ride"),
        ("bhai class mein so gaya teacher ne direct naam pucha tha game over", "sleeping class caught"),
        ("yaar delivery boy ne wrong floor pe diya toh 2 log lad gaye neeche", "delivery wrong floor"),
        ("bhai ghar mein ghost movie dekh raha tha light gayi yaar mar gaya", "power cut horror"),
        ("yaar exam mein galat paper fill kiya registration number ka hi nahi tha", "exam wrong paper"),
        ("bhai live stream pe accidentally dusra tab khul gaya sab dikh gaya", "live stream fail"),
        ("yaar office party mein boss ke saamne dance kiya viral ho gaya bhai", "office party dance"),
        ("bhai driving test mein instructor ko hi gate pe rok diya gaya tha", "driving test fail"),
        ("yaar library mein phone ki ringtone baj gayi loud wali exam time pe", "ringtone library fail"),
        ("bhai galat WhatsApp pe voice note bhej diya family wale ko yaar", "wrong WhatsApp voice"),
        ("yaar naya phone liya selfie camera kholke front facing tha bhai 💀", "ugly selfie shock"),
        ("bhai ATM mein PIN bhool gaya queue mein 10 log the bhai yaar", "ATM pin forgot"),
        ("yaar pehli date pe khaana gira diya shirt pe bhai yaar kya tha", "first date spill"),
        ("bhai team call mein mute bhool gaya pura conversation sun gaya sab ne", "unmuted call caught"),
        ("yaar badi line mein tha kuch order karne gaya wahan closed tha bhai", "shop closed waiting"),
        ("bhai gym mein weight drop ho gaya sab ne dekha bhai scene tha yaar", "gym weight drop"),
        ("yaar auto mein ghusa teen log already the ek aur ghus gaya bhai", "auto full funny"),
        ("bhai reply karte karte soye, subah dekha novel likh diya tha bhai", "sleepy texting fail"),
    ],

    "comfort_support": [
        ("bahut down feel ho raha hai aaj kuch accha nahi lag raha kisi se", "comfort hug"),
        ("yaar bahut lonely lag raha hai aaj sab busy hain apni life mein", "feeling lonely sad"),
        ("mood off hai pura din kharab gaya koi cheez sahi nahi hui aaj", "sad comfort meme"),
        ("bhai bahut stressed hai aaj kuch bhi sahi nahi ho raha help karo", "stress overwhelmed"),
        ("feeling really low today everything went wrong ngl kabse aisa hai", "it gets better"),
        ("yaar din bhar bura laga kuch nahi ho raha sahi aisa kyun hota hai", "comfort virtual hug"),
        ("bhai bahut thak gaya hun sab se already weekend pe bhi kaam hai", "exhausted tired"),
        ("yaar kisi baat pe bahut roya aaj achha hua actually feel better ab", "crying feels better"),
        ("bhai presentation kal hai nervous hun bahut kuch galat ho gaya toh", "nervous anxiety comfort"),
        ("yaar kal result hai neend nahi aa rahi anxiety bahut hai bhai", "exam result anxiety"),
        ("bhai exam mein acha nahi gaya lag raha bahut bura feel ho raha", "exam went bad"),
        ("yaar kuch bahut mushkil chal raha hai life mein share nahi kar sakta", "going through it"),
        ("bhai sab kuch sahi nahi ja raha din ki shuruat bhi aisi thi yaar", "bad day comfort"),
        ("yaar pehli baar ghar se itne dur hun homesick ho raha hun bhai", "homesick comfort"),
        ("bhai rejection baar baar aa raha hai give up karne ki feeling hai", "rejection comfort"),
        ("yaar parents se zyada arguments ho rahe hain stress bahut hai bhai", "family stress comfort"),
        ("bhai kuch karna hi nahi chahta aaj sab worthless lag raha hai yaar", "unmotivated comfort"),
        ("yaar purani yaadein aa rahi hain jo nahi chahiye bhai disturb kar rahi", "nostalgic sad comfort"),
        ("bhai best friend se fight ho gayi seriously dukh ho raha bhai yaar", "friendship fight sad"),
        ("yaar 3 baje tak roya hun nahi pata kyun feel nahi better ho raha", "crying night comfort"),
        ("bhai appraisal mein low rating aayi sab mehnat bekar lagi yaar", "work disappointment"),
        ("yaar doctor ne bura news diya hai ghabra raha hun bhai please", "health scare comfort"),
        ("bhai college mein fit nahi ho pa raha 3 mahine ho gaye yaar sad hun", "not fitting in"),
        ("yaar relationship mein kuch theek nahi chal raha bohot hurt hun bhai", "relationship pain"),
        ("bhai self doubt bahut aa raha hai kuch bhi achha nahi kar pa raha", "self doubt comfort"),
    ],

    "celebration_hype": [
        ("bhai aaj mera birthday hai 21 ka turning adult officially ho gaya", "birthday celebration"),
        ("yaar engagement ho gayi ring de diya usne yesterday night bhai", "engagement celebrate"),
        ("exam clear ho gaya bhai sabse mushkil wala paper merit mein aaya", "exam passed celebrate"),
        ("bhai job offer aa gaya finally placement ho gayi package bhi accha", "job offer celebrate"),
        ("yaar pehli salary aai bhai 40k tha kya feeling hai celebrate karte", "first salary celebrate"),
        ("bhai visa approve ho gaya canada ja raha hun next month bhai", "visa approved party"),
        ("yaar hackathon jeet gaye bhai 1st prize tha pehli baar jita kuch", "hackathon win trophy"),
        ("bhai driving license mil gaya 3rd attempt mein finally free hun", "driving license freedom"),
        ("yaar app launch hua play store pe 100 downloads kal mein hi aaye", "app launch success"),
        ("bhai 1 year gym streak complete hua body transformation bhi ayi", "gym transformation"),
        ("yaar promotion ho gayi bhai team lead bana gaya finally mehnat ka", "promotion celebrate"),
        ("bhai 10 kilo weight lose kar liya 6 mahine mein doctor bhi shocked", "weight loss win"),
        ("yaar state level cricket selection ho gayi bhai finally dream aa raha", "sports selection"),
        ("bhai guitar mein pehla song baja sakta hun ek mahine ke baad finally", "learned guitar"),
        ("yaar startup ko pehla paying customer mila aaj milestone hai bhai", "startup first customer"),
        ("bhai college admission ho gayi dream college mein bhai mehnat rang laayi", "college admission"),
        ("yaar 10 saal baad family reunion hua sab mil gaye bhai emotional tha", "family reunion"),
        ("bhai pehla solo travel kiya successfully bhai confidence alag level ho gaya", "solo travel win"),
        ("yaar research paper accept ho gaya international journal mein bhai wow", "research published"),
        ("bhai debt clear ho gaya aaj finally 3 saal mein bhai relief alag hai", "debt free celebrate"),
        ("yaar novel ka pehla draft complete hua 6 mahine ki mehnat thi bhai", "writing milestone"),
        ("bhai pehli baar swim kiya without float aaj 30 saal mein bhai yaar", "adult learning swim"),
        ("yaar business register ho gaya officially proprietor hun ab bhai", "business registered"),
        ("bhai YouTube pe pehla video 100k views gaya viral ho gaya bhai yaar", "video viral celebrate"),
        ("yaar language exam pass kiya B2 German bhai 2 saal ki practice thi", "language exam passed"),
    ],

    "food_craving": [
        ("yaar pizza ki craving ho rahi hai bahut badly raat ko 1 baj gaye hain", "pizza craving"),
        ("bhai biryani ki yaad aa rahi hai mummy wali ghar ki bahut miss kar raha", "biryani craving"),
        ("burger dekhke muh mein paani aa gaya bhai acidity ho jaayegi phir bhi", "burger drooling"),
        ("yaar momos khaane ka mann kar raha hai bahut zyada aaj ki craving", "momos craving"),
        ("chai ki craving hai bhai barish mein pakode ke saath sochke hi maza aa gaya", "chai pakoda rain"),
        ("bhai maggi bana raha hun 2 baje raat ko student life mein ye hota hai", "late night maggi"),
        ("yaar chhole bhature ki craving hai Sunday morning is liye banana padhega", "chole bhature craving"),
        ("bhai gol gappa khaane ka mann hai pani puri wali galli yaad aa rahi", "pani puri craving"),
        ("yaar dosa khaya tha aaj se pehli baar south indian aur life change ho gayi", "dosa south indian"),
        ("bhai rasgulla aur gulab jamun dono ek saath khaaunga aaj koi nahi rokne wala", "sweet craving dessert"),
        ("yaar late night ice cream cravings are hitting different tonight bhai", "ice cream craving"),
        ("bhai haleem ki craving ho rahi hai winters mein kuch aur nahi chahiye", "haleem winter craving"),
        ("yaar shawarma ki craving ho rahi hai raat ko 11 baje band ho gayi dukan", "shawarma craving"),
        ("bhai jalebi aur rabdi combo ho jaye aaj sunday hai bhai", "jalebi rabdi sweet"),
        ("yaar butter chicken naan ke saath bhai ghar ka khana miss ho raha hai", "butter chicken craving"),
        ("bhai mango season shuru hua dasheri aayi hai ghar mein bhai khao", "mango season"),
        ("yaar pani puri wali aunt wahi jagah nahi thi aaj bhai khaali haath aaya", "street food missing"),
        ("bhai samosa chai combo in winters unbeatable hai yaar craving ho rahi", "samosa chai winter"),
        ("yaar kachori dal ke saath bhai Rajasthani vibes yaad aa rahi yaar", "kachori dal craving"),
        ("bhai raat ko 2 baje instant ramen bana raha hun bhai hostel life yaar", "instant ramen night"),
        ("yaar kulfi leke khana chahta hun ghar ke bahar leke jaata hun aaj", "kulfi summer craving"),
        ("bhai achari murg ki craving ho rahi hai pichhle hafte se bhai yaar", "spicy chicken craving"),
        ("yaar bread pakora wali didi kal nahi aayi canteen mein bhai bohot missa", "bread pakora canteen"),
        ("bhai chocolate pastry dekhke kuch control nahi hua bhai ate aaj aur", "chocolate pastry craving"),
        ("yaar aloo paratha makhan wala bhai subah subah ghar yaad aa rahi hai", "aloo paratha craving"),
    ],

    "gaming_sports": [
        ("bhai valorant mein ace maar diya just now 1v5 clutch insane tha yaar", "valorant ace clutch"),
        ("yaar minecraft mein 3 din mein itna insane build banaya castle bana diya", "minecraft build"),
        ("bhai ranked mein diamond hit kar liya grind finally pay off hua yaar", "ranked diamond gaming"),
        ("yaar free fire mein booyah mara bhai pehli baar solo rank match mein", "free fire win"),
        ("bhai BGMI chicken dinner mila last circle mein kaise hua pata nahi", "BGMI chicken dinner"),
        ("yaar last ball pe 6 maar ke jeet liya gully cricket bhai legend ban gaya", "cricket last ball"),
        ("bhai kabaddi mein 5 ko akele touch kiya match turn ho gaya yaar scene tha", "kabaddi solo tackle"),
        ("yaar football mein penalty miss hua bhai 3rd bar aata hun iska yaar", "penalty miss football"),
        ("bhai FIFA mein 97th minute goal daala bhai opponent ne controller pheenka", "FIFA clutch goal"),
        ("yaar badminton mein 21-0 se jeet gaya bhai opponent chidh gaya bilkul", "badminton crushing win"),
        ("bhai chess mein brilliant move se jeet gaya opponent 10 min tak socha", "chess brilliant move"),
        ("yaar kite tournament mein pehla naamber aaya kal bhai maanja bhi accha tha", "kite flying win"),
        ("bhai pubg final zone mein last 2 mein tha heartbeat sun raha tha yaar", "PUBG final zone"),
        ("yaar among us mein impostor tha sab ko convince kar liya bhai legend", "among us win"),
        ("bhai GTA online heist complete kiya first attempt mein smooth tha yaar", "GTA heist success"),
        ("yaar carrom mein board clear kar diya bhai 3 alak ek hi stroke mein", "carrom board clear"),
        ("bhai table tennis mein comeback kiya 0-10 se 11-10 jeet gaya bhai yaar", "table tennis comeback"),
        ("yaar khokho mein sab ko tag kiya akele bhai legend ho gaya school pe", "kho kho win"),
        ("bhai call of duty warzone victory screamed bhai neighbours ne sun liya", "warzone victory scream"),
        ("yaar road cycling pe 50km complete kiya first time legs gone bhai yaar", "cycling milestone"),
        ("bhai pool mein pehli baar bina ruk ke 10 laps kiye bhai proud hun", "swimming laps"),
        ("yaar volleyball smash maar diya point winning wala bhai team khush thi", "volleyball smash win"),
        ("bhai marathon ka 21km half complete kiya bhai time bhi accha tha yaar", "half marathon done"),
        ("yaar esports tournament mein quarterfinal pahuncha bhai first time ever", "esports tournament"),
        ("bhai skateboarding mein pehli ollie clear ki bhai 3 hafte ki practice", "skateboard ollie"),
    ],

    "aesthetic_vibe": [
        ("yaar bahar itni amazing barish ho rahi hai cozy vibes window pe baitha hun", "rain cozy vibes"),
        ("bhai sunset dekha aaj se terrace pe tha itna beautiful tha seriously", "beautiful sunset"),
        ("fog itni thick hai aaj morning mein puri gali silent hill ban gayi bhai", "foggy morning"),
        ("yaar raat ko taarein dikh rahe hain rooftop se koi pollution nahi aaj", "night sky stars"),
        ("bhai snowfall ho rahi hai yahan pehli baar dekh raha hun live feel alag", "snowfall first time"),
        ("yaar monsoon ki pehli barish aayi bhai petrichor smell aaj tha bahut accha", "first monsoon rain"),
        ("bhai aaj itni moonlight thi full moon tha puri raat chamak rahi thi", "full moon night"),
        ("yaar sunrise 5am pe jaag ke dekha tha pehli baar life changed bhai seriously", "sunrise morning"),
        ("bhai rainbow dikha aaj barish ke baad double tha yaar rare cheez hai", "double rainbow"),
        ("yaar samundar pe gaya tha first time bhai waves ki awaz mindblowing thi", "ocean waves beach"),
        ("bhai terrace pe raat ko chai pi raha hun stars aur breeze beautiful bhai", "terrace night breeze"),
        ("yaar himachal trip pe tha mountains mein cloud below tha bhai unreal tha", "mountains cloud below"),
        ("bhai pehli baar desert dekha Rajasthan mein silence aur sand bhai alag", "desert silence vast"),
        ("yaar first snowfall in life dekhi northern trip pe bhai speechless tha", "first snowfall life"),
        ("bhai backwaters of Kerala dekhe shaam ko golden light thi yaar wow", "Kerala backwaters golden"),
        ("yaar old city ki galiyan dekhin raat ko lit up thi bhai aesthetic tha", "old city lights"),
        ("bhai paddy fields mein tha green se green monsoon mein bhai unreal bhai", "green paddy fields"),
        ("yaar cherry blossoms dekhe Shillong mein spring mein bhai India mein yaar", "cherry blossom India"),
        ("bhai jungle mein trek pe tha early morning mist bhai surreal feel tha", "jungle morning mist"),
        ("yaar rooftop pe Diwali dekh raha hun city ki lights bhai overwhelming hai", "Diwali city lights"),
    ],

    "nostalgic_feels": [
        ("yaar purani photo mili school time ki bilkul alag the hum bhai memories", "nostalgic old photos"),
        ("bhai cartoon wali memories aa gayi aaj dragon ball z dekha 10 min trailer", "childhood cartoons"),
        ("yaar wo dino ki yaad aa rahi hai jab sab ek jagah the bhai miss karta hun", "missing old times"),
        ("bhai purana ghar yaad aa gaya aaj kisi ne photo share ki wahan ke yaar", "old home memories"),
        ("yaar first love ki baat sunke hi emotions aa gaye bhai kyun hota hai aisa", "first love memories"),
        ("bhai school ke yaar yaad aa gaye aaj na koi tension na koi stress tha", "school days nostalgia"),
        ("yaar pehli cycle yaad aa gayi bhai kab aayi thi bachpan mein raat tak bhaaga", "childhood cycle"),
        ("bhai wo doordarshan wala cartoon yaad aa gaya chhota bheem se pehle ka", "old cartoon DD"),
        ("yaar handwritten letters milte the bhai ab sirf WhatsApp hai feel nahi", "handwritten letters"),
        ("bhai pados wali didi ki yaad aa gayi bachpan mein khelte the roz yaar", "childhood neighbour"),
        ("yaar Nokia 3310 wale dino ki yaad aayi snake game bhai wo level tha", "Nokia snake game"),
        ("bhai dial up internet ka sound yaad aaya bhai kitna patient tha hum log", "dial up nostalgia"),
        ("yaar cassette player se gaane sunne ki baat alag thi bhai seek nahi hota", "cassette player"),
        ("bhai Doordarshan pe Mahabharata dekhne ka time yaad aa raha bhai family", "Mahabharata DD memory"),
        ("yaar 5 rupaye ka gola milta tha bhai ab 50 ka bhi aisa nahi lagta yaar", "5 rupee gola"),
        ("bhai wo summer vacation wali train journey yaad aa gayi sab saath the bhai", "summer train vacation"),
        ("yaar Ludo board game pe bhai kab khele the real wala plastic wala bhai", "Ludo board game"),
        ("bhai wo wali dukaan band ho gayi jahan chips milti thi school ke baad yaar", "tuck shop closed"),
        ("yaar pehli baar school trip gaye the bhai kitna excited tha wo din bhai", "first school trip"),
        ("bhai orkut pe status likhte the bhai 2008 ki baat hai yaar nostalgia hit", "Orkut nostalgia"),
    ],
}

# REACT_TOPICS — natural moments + emoji
# Structure: {emoji: [(msg, difficulty_hint), ...]}
REACT_TOPICS = {
    "🥳": [
        ("bhai placement ho gayi campus se finally done package bhi 12 LPA mila", "hard"),
        ("yaar result aa gaya first division mein aaya bhai mehnat rang layi", "hard"),
        ("bhai shaadi ki date fix ho gayi december mein bhai finally", "hard"),
        ("yaar abroad masters mein acceptance letter aaya bhai dream university", "hard"),
        ("bhai startup ko seed funding mili investors impressed hue bhai", "hard"),
        ("yaar UPSC CSE prelims clear hua bhai 3rd attempt mein ho gayi", "hard"),
        ("bhai pehli salary 40k tha bhai kya feeling hai celebrate karte yaar", "medium"),
        ("yaar YouTube pe 100k subscribers ho gaye bhai milestone cross hua", "medium"),
        ("bhai visa lagaya tha US ka aaj approve ho gaya bhai yaar kya baat", "hard"),
        ("yaar marathon complete kiya pehli baar 42km bhai legs gone but proud", "hard"),
        ("bhai pehla client mila freelance mein bhai 6 mahine ki koshish thi", "medium"),
        ("yaar college mein gold medal mila convocation pe bhai family roya", "hard"),
        ("bhai driving test 3rd attempt mein clear hua finally bhai free hun", "medium"),
        ("yaar bhai ki job pakki ho gayi abroad mein bhai sab khush hain ghar", "hard"),
        ("bhai research paper accepted hua CVPR mein bhai dream tha bhai yaar", "hard"),
        ("yaar 1 saal ki gym ne result diya transformation dekh ke khud shock hun", "medium"),
        ("bhai debut match mein hat trick ki bhai coach bhi shock tha yaar", "hard"),
        ("yaar sister ne IIT crack kiya bhai ghar mein celebration ho raha", "hard"),
    ],
    "🔥": [
        ("bhai valorant 1v5 clutch maar diya last round mein lobby ka scene tha", "medium"),
        ("yaar deadlift PR tod diya aaj 120kg bhai 6 mahine ki grind pay off hui", "medium"),
        ("bhai presentation khadi hai karo standing ovation mila bhai senior se", "medium"),
        ("yaar code review mein senior bole elegant solution hai bhai kya feeling", "medium"),
        ("bhai cricket mein hat trick li gully mein sab log dang reh gaye", "medium"),
        ("yaar debate competition jeet gaya bhai first time participate kiya tha", "medium"),
        ("bhai poetry sunai public mein pehli baar sab log maan gaye bhai", "easy"),
        ("yaar cooking first time kiya biryani bani bhai sab ne taarif ki", "easy"),
        ("bhai freestyle battle mein sab ko haraaya bhai impromptu tha yaar", "medium"),
        ("yaar project demo boss ne zyada features maange bhai matlab pasand aaya", "medium"),
        ("bhai hackathon solution aisa tha judges poochte rahe bhai proud hun", "medium"),
        ("yaar instagram reel 1M views gaya bhai overnight viral ho gaya yaar", "hard"),
        ("bhai anchoring ki pehli baar stage pe sab haste rahe bhai maza aaya", "easy"),
        ("yaar 5km run time PR kiya 23 min mein bhai training pay off hui", "medium"),
        ("bhai startup pitch mein VC ne directly fund offer kiya bhai surreal tha", "hard"),
        ("yaar art exhibition mein painting bikni bhai pehli sale ever bhai", "hard"),
        ("bhai song release kiya Spotify pe 10k streams aayi bhai yaar", "hard"),
        ("yaar pehle board game tournament mein top 3 aaya bhai surprised hun", "medium"),
    ],
    "😂": [
        ("bhai teacher ne galti se apna phone projector se connect kar diya tha", "easy"),
        ("yaar auto wale ne itna philosophical bola aaj ride mein maza aa gaya", "easy"),
        ("landlord ka private msg galti se GC mein aa gaya sab dang reh gaye", "easy"),
        ("bhai dost ne wrong person ko i love you bhej diya bhai woh number", "easy"),
        ("yaar cafeteria mein sabke saamne tray ke saath gir gaya bhai legendary", "easy"),
        ("bhai interview mein nervous tha puch baith gaya interviewer ki kursi pe", "easy"),
        ("yaar mummy ne meri story repost kar di unhone socha private thi bhai", "easy"),
        ("bhai rickshaw wala philosophy dene laga 2 baje raat ko fare pe yaar", "easy"),
        ("yaar auto mein 4 log the ek aur ghusa conductor bhi nahi bola bhai", "easy"),
        ("bhai office mein sabka tiffin khol ke dekha tha bhookh lagti hai yaar", "easy"),
        ("yaar crush ko 'didi' bol diya galti se bhai literally ground ban gaya", "easy"),
        ("bhai ghar walo ko explain kiya crypto kya hai bhai conversation bhai", "easy"),
        ("yaar pooja mein papa ne selfie stick lagaya pandit ji bhi ruk gaye bhai", "easy"),
        ("bhai relative ne puchha boyfriend hai kya table pe sannata aa gaya yaar", "easy"),
        ("yaar online class mein pet ka naam pucha teacher ne bhai awkward tha", "easy"),
        ("bhai doctor ke clinic mein funny YouTube video dekh raha tha hans diya", "easy"),
        ("yaar galat metro mein baith gaya 4 station aage jake pata chala bhai", "easy"),
        ("bhai password bhool gaya khud ka toh naam daal diya bhai kaam kiya", "easy"),
    ],
    "😱": [
        ("yaar road pe almost accident hote hote bacha 2 second ka fark tha bhai", "hard"),
        ("bhai earthquake aa gayi dar gaya tha sab kuch hilne laga achanak bhai", "hard"),
        ("yaar phone gira 5th floor se concrete pe screen nahi tooti bhai kya", "medium"),
        ("bhai wallet chori ho gaya metro mein pata hi nahi chala kab hua yaar", "medium"),
        ("yaar snake mil gaya room mein bhai puri raat neend nahi aayi phir", "hard"),
        ("bhai bijli ka taar tap gaya haath se shock laga minor tha theek hun", "medium"),
        ("yaar car ke brakes kaam nahi kar rahe the slope pe bhai bacha barely", "hard"),
        ("bhai raat ko akela tha ghar mein darwaza khud khul gaya bhai mara yaar", "hard"),
        ("yaar 20th floor pe tha aur lift band ho gayi andhera bhi tha bhai yaar", "hard"),
        ("bhai galat train mein baitha tha 2 station baad pata chala bhai yaar", "medium"),
        ("yaar balcony pe tha railing hilne lagi bhai neecha dekha bhai dar gaya", "hard"),
        ("bhai deadline kal thi bhool gaya tha raat 11 baje yaad aaya bhai help", "hard"),
        ("yaar passport expire ho gaya tha airport pe pata chala bhai nightmare", "hard"),
        ("bhai AC chala diya kuch smell aaya bijli ka bhai off kar diya call kiya", "medium"),
        ("yaar pet ki collar nikli ghar se bhaag gaya toh tha bhai dar gaya yaar", "hard"),
        ("bhai presentation mein galat slides thi company ki confidential wali bhai", "hard"),
        ("yaar atm se paisa nahi aaya balance kat gaya bhai bank band tha", "medium"),
        ("bhai bina ticket ke train pe tha TC aa gaya bhai dono scared the", "medium"),
    ],
    "💪": [
        ("bhai gym mein pehla pull up maara finally 3 mahine ki koshish thi yaar", "medium"),
        ("yaar 10km run complete kiya pehli baar bhai legs dard kar rahi hain ab", "medium"),
        ("bhai roz subah uthne ki aadat lag gayi 30 din ho gaye bhai discipline", "medium"),
        ("yaar smoking chhodh di bhai 100 din ho gaye clean streak chal raha", "medium"),
        ("bhai cold shower le raha hun roz ab 60 din ka streak hai bhai maza", "easy"),
        ("yaar 20 pages roz likhna shuru kiya novel bane ka sapna hai bhai", "easy"),
        ("bhai 30 din mein 10 books padhi bhai reading habit ban gayi finally", "easy"),
        ("yaar sugar chhod di completely bhai 2 mahine ho gaye skin better lag rahi", "medium"),
        ("bhai 5km se shuru kiya tha ab 15km daud leta hun 6 mahine mein bhai", "medium"),
        ("yaar roz 1 coding problem solve karna shuru kiya 90 din streak bhai", "medium"),
        ("bhai procrastination ki jagah pomodoro try kiya bhai kaam kitna hua yaar", "easy"),
        ("yaar junk food band kiya 3 mahine ho gaye cravings bhi kam ho gayi", "medium"),
        ("bhai 6am uthke joggle laga raha hun 2 mahine se bhai body feel kya hai", "easy"),
        ("yaar 10k steps roz chalte hain bhai streak 45 din ho gayi bhai", "easy"),
        ("bhai daru chhodh di bhai 6 mahine ho gaye clarity kuch aur hi hai yaar", "medium"),
        ("yaar social media daily usage 30 min kar diya bhai focus alag hai", "easy"),
        ("bhai therapy shuru ki bhai life mein bohot fark pada genuinely yaar", "medium"),
        ("yaar first 5K race complete kiya bhai hospital se nahi aaya bhai", "medium"),
    ],
    "😭": [
        ("yaar pet billi mar gayi bhai 8 saal se thi hamare paas bahut yaad aayegi", "hard"),
        ("bhai best dost dusre sheher shift ho gaya permanently kal hi gaya yaar", "medium"),
        ("yaar favorite series khatam ho gayi ab kya dekhu void feel ho raha hai", "easy"),
        ("bhai college ka last day tha emotional ho gaya yaar sab yaad aa raha", "medium"),
        ("yaar dada ji ki purani photo mili bahut miss karta hun unhe bhai", "hard"),
        ("bhai favorite teacher retire ho gaye kal aakhri class tha emotional tha", "medium"),
        ("yaar wo dukan band ho gayi jahan roz chai peete the memories bhai", "easy"),
        ("bhai 10 saal purana phone finally replace kiya emotions aa gaye yaar", "easy"),
        ("yaar batch ka aakhri din tha sab alag ho gaye bohot roya bhai yaar", "medium"),
        ("bhai ghar mein pala hua ped kata dena pada construction ke liye bhai", "hard"),
        ("yaar pehla internship khatam hua mentor se emotional farewell tha bhai", "medium"),
        ("bhai jo dost 10 saal se tha silence pe hai unka samajh nahi aa raha", "hard"),
        ("yaar graduation ceremony mein parents ko dekha emotional ho gaya bhai", "hard"),
        ("bhai pasandida singer ka concert miss ho gaya last moment ticket gaya", "easy"),
        ("yaar boarding school se aaya tha first time 3 saal baad mummy rona lagi", "hard"),
        ("bhai jo dukan school se aate the wahan chips khaate the band ho gayi", "easy"),
        ("yaar pehli baar apna ghar chhoda hostel ke liye sab miss karta hun", "medium"),
        ("bhai gaon se shift ho gaye city mein sab kuch miss ho raha hai yaar", "hard"),
    ],
    "🎉": [
        ("bhai baby aa gaya ghar mein naya member family complete ho gayi yaar", "hard"),
        ("yaar bhai ki shaadi fix ho gayi december mein bhai finally", "hard"),
        ("bhai internship confirm ho gayi Google mein bhai pehli big company", "hard"),
        ("yaar naye ghar mein shift ho gaye griha pravesh kal hai exciting hai", "hard"),
        ("bhai YouTube pe 10K subscribers ho gaye aaj dekha surprise tha bhai", "medium"),
        ("yaar sister ka result aaya topper bani class mein bhai proud hun", "medium"),
        ("bhai dost ki engagement ho gayi finally hum sab ko pata tha aayega", "medium"),
        ("yaar 1 saal company mein hua promotion aa gayi letter bhi diya bhai", "medium"),
        ("bhai naana naani ka 50th anniversary hai puri family aa rahi bhai", "hard"),
        ("yaar school reunion hai 10 saal baad bhai sab mil rahe hain finally", "medium"),
        ("bhai cousins aayi hain ghar pe poora hafte enjoy karenge bhai yaar", "easy"),
        ("yaar dost ki PhD defense pass ho gayi bhai doctor ban gaya officially", "hard"),
        ("bhai colony mein festival hai bhai aaj din kaafi fun wala hoga yaar", "easy"),
        ("yaar pehli baar Diwali akele manayi hostel mein GC pe sabne wish kiya", "medium"),
        ("bhai friend group ka reunion hai bhai 2 saal baad sab mil rahe hain", "medium"),
        ("yaar scholarship announcement aayi bhai college ka naam aaya list mein", "hard"),
        ("bhai IPL team jeet gayi bhai ghar mein pooja-sa mahol chal raha hai", "easy"),
        ("yaar holi pe pura GC rang gaya tha literally ek hi jagah the sab log", "easy"),
    ],
    "❤️": [
        ("yaar best dost ne surprise birthday party plan ki thi emotional ho gaya", "medium"),
        ("bhai mummy ne bina bataye favorite khana banaya tha aaj love karta hun", "easy"),
        ("yaar dost ne 3 baje uthke help kiya exam ke liye true friendship hai bhai", "medium"),
        ("bhai purana school dost mila aaj 7 saal baad same wala bhai emotional", "medium"),
        ("yaar mera dog gate pe intezaar kar raha tha school se aaya toh bhai", "easy"),
        ("bhai nani ne personally sweater banaya mere liye itna precious hai yaar", "medium"),
        ("yaar roommate ne surprise kiya tha akele celebrate karna tha bhai", "easy"),
        ("bhai behen ne apni savings se gift diya bhai emotions aa gaye sachchi", "medium"),
        ("yaar stranger ne umbrella share kiya rain mein aaj bhai goodness hai", "easy"),
        ("bhai mummy ne bina kuch bolte samjh liya kya chal raha hai bhai yaar", "medium"),
        ("yaar aane ke baad dadi ne haath pakad ke rakhya puri shaam bhai yaar", "hard"),
        ("bhai ghar wapas aaya tha pehli baar ghar ki smell se hi rona aaya bhai", "medium"),
        ("yaar dost ne heartbreak mein ghar aake baitha raha kuch nahi bola bas tha", "hard"),
        ("bhai mentor ne apni contact list share ki job ke liye bhai beyond tha", "medium"),
        ("yaar concert ticket sold out tha dost ne apna diya bhai sacrifice yaar", "medium"),
        ("bhai bimar tha akele tha delivery karke gaya dost kuch bola nahi bhai", "medium"),
        ("yaar 5 saal baad school teacher ne message kiya proud hun tumse bhai", "hard"),
        ("bhai neighbor aunty ne khana bheja thi exam ke din bhai unprompted tha", "easy"),
    ],
    "💀": [
        ("yaar ye meme format perfectly timed hai aaj ka context pe bhai ded", "easy"),
        ("bhai itna cringe tha ki literally mar gaya dekhke wahan se hat gaya", "easy"),
        ("yaar teacher ka joke sunke class mein sannata tha bhai cringe level", "easy"),
        ("bhai autocorrect ne galat bheja bhai wo padh ke literally ded hun yaar", "easy"),
        ("yaar wo reply dekh ke bhai kuch bolne ki himmat nahi hai bhai 💀", "easy"),
        ("bhai ye ad dekh ke bhai kitna budget tha unka wo bhi khatam ho gaya", "easy"),
        ("yaar GC mein yaar ne galti se screenshot bhej diya bhai scene tha", "easy"),
        ("bhai presentation mein slide number galat tha bhai 69 pe ruk gaya sab", "easy"),
        ("yaar dunno kaun tha bhai group mein ye type kiya sab ded ho gaye bhai", "easy"),
        ("bhai sir ka outfit dekh ke sab munh dabake hanse bhai vintage level", "easy"),
        ("yaar koi explanation nahi hai is situation ka bhai literally no words", "easy"),
        ("bhai ye caption dekh ke bhai marketing team ko puchha kya soch ke likha", "easy"),
        ("yaar bhai ne mujhe expose kar diya family GC mein bhai literally finished", "easy"),
        ("bhai interview mein poochha gaya apna ek meme describe karo bhai yaar", "easy"),
        ("yaar relative ne puchha ladki dhundh doon kya bhai table pe sannata tha", "easy"),
    ],
    "😎": [
        ("bhai promotion mil gayi salary 40 percent hike bhai kuch nahi bola tha", "hard"),
        ("yaar interview crack kiya first attempt mein bhai preparation nahi thi", "medium"),
        ("bhai coding contest mein first rank aaya yaar bina practice ke bhi", "hard"),
        ("yaar boss ne bola team lead banega ab bhai unexpectedly aaya offer", "hard"),
        ("bhai freelance client ne extra bonus diya work se impressed tha bhai", "medium"),
        ("yaar side project se pehli earning aayi bhai passive income shuru", "medium"),
        ("bhai bina revision ke exam mein acche marks aaye bhai lucky tha yaar", "medium"),
        ("yaar cold email kiya tha startup ko internship mil gayi bhai bold move", "hard"),
        ("bhai blind audition diya band ne direct liya bhai singer nahi tha pehle", "hard"),
        ("yaar last minute assignment submit kiya bhai A grade aaya bhai yaar", "medium"),
        ("bhai presentation bina preparation ke di bhai sabse acchi thi overall", "medium"),
        ("yaar negotiation ki salary 20% upar gayi bhai value pata thi apni", "hard"),
        ("bhai startup idea pitch kiya class mein professor ne VC ko connect kiya", "hard"),
        ("yaar aaj gym record tora bhai kisi ne nahi dekha but maine toh dekha", "easy"),
        ("bhai mock test mein topper aaya bhai bina padhke accha laga genuinely", "medium"),
    ],
    "🌧️": [
        ("bhai pehli barish aayi monsoon shuru ho gaya season ka wait tha yaar", "easy"),
        ("yaar poori raat barish hui itna peaceful tha sonne mein bhai amazing", "easy"),
        ("bhai thunderstorm aa rahi hai lightning dikh rahi khoobsurat lag rahi", "easy"),
        ("yaar baarish mein bheeg gaya college se aate waqt mood happy ho gaya", "easy"),
        ("bhai terrace pe khade hokar barish dekh rahe hain chai bhi hai bhai", "easy"),
        ("yaar barish mein bheeg ke dost se mili thi aaj nostalgia lag raha bhai", "easy"),
        ("bhai monsoon mein ghar mein hai window khuli hai smell amazing hai yaar", "easy"),
        ("yaar pehli baar himachal mein tha baarish aayi pahadi barish alag hai", "easy"),
    ],
    "🤯": [
        ("bhai ye fact sunke literally dimaag ghoom gaya tha kuch din se soch raha", "medium"),
        ("yaar movie ka plot twist dekh ke 10 min tak kuch bolne ki sthiti nahi thi", "easy"),
        ("bhai science experiment result dekha tha class mein seriously mindblown", "easy"),
        ("yaar news padhi thi aaj se kuch cheezein alag lagti hain ab bhai", "medium"),
        ("bhai documentary dekhi bhai perspective hi badal gaya life ka yaar wow", "medium"),
        ("yaar ye statistics sunke bhai literally kuch minute chup raha hun yaar", "medium"),
        ("bhai AI ne ye generate kiya dekha toh bhai kab hoga ye process samjhao", "easy"),
        ("yaar history mein ye hua tha pata hi nahi tha bhai school ne kuch nahi", "medium"),
        ("bhai ye quantum physics ka concept samjha toh bhai literally head spinning", "medium"),
        ("yaar space ki ye photo dekh ke bhai existence questionable lag rahi hai", "easy"),
        ("bhai optical illusion hai ye bhai dimag nahi maanta aankhon ko bhai yaar", "easy"),
        ("yaar usne itna brilliant jugaad kiya bhai engineers shame ho gaye yaar", "easy"),
    ],
}

# IGNORE TOPICS — spam, noise, repetition, keyboard smash
IGNORE_PATTERNS = [
    # Pure laughter chains
    ["lol","haha","😂","💀","hehehehe"],
    ["😂","😂","😂","💀","😭"],
    ["haha","lol","lmao","hehe","xD"],
    ["💀","💀","💀","😭","lol"],
    ["hahahaha","lolll","😂😂","bruh","ded"],
    ["omg lol","haha","😭😭","💀","dying"],
    ["lmaooo","haha same","😂","true","lol"],
    ["bruh lol","haha fr","😭","💀","okay"],
    ["xD xD","haha","😂😂😂","lol","heheh"],
    ["bhai lol","haha yaar","😂","💀","hehehe"],
    ["ahahaha","lolol","😂","dead","hahaha"],
    ["😂😂😂","💀💀","😭😭","lmao","dead"],
    ["literally dead","omg stop","💀💀💀","😭","not okay"],
    ["LMAOOO","bro 💀","no way","DEAD","😭😭"],
    ["bhai 😂","yaar 💀","bro 😭","ded ded","HAHAHA"],
    # Keyboard smash
    ["shjg","sfdd","vsdgsg","dgsg","gfgfhgf"],
    ["asdfgh","qwerty","zxcvbn","poiuyt","lkjhgf"],
    ["1234567","qweasd","zxcqwe","123qwe","asd123"],
    ["aaaaaaa","bbbbbbb","ccccccc","ddddddd","eeeeeee"],
    ["fjdksla","sldkfj","qpwoei","rutyei","aldskf"],
    ["zzzzz","xxxxx","ccccc","vvvvv","bbbbb"],
    ["ajsdbf","lskdjf","qpwodk","zmxncb","alsdkj"],
    # Repetition noise
    ["ok","ok","ok","ok","ok"],
    [".","..","...","....","😶"],
    ["test","test123","testing","yo","nvm"],
    ["bhai","bhai","bhai","bhai","bhai"],
    ["ha","ha","ha","ha","ha"],
    ["k","k","k","k","k"],
    ["???","???","???","???","???"],
    ["na","na","na","na","na"],
    # Spam/forward
    ["WIN FREE IPHONE NOW","CLICK THIS LINK","LIMITED OFFER","CLAIM TODAY","bit.ly/scam"],
    ["Congratulations you won","Click here to claim","Offer expires today","Dont miss out","Forward to 10"],
    ["FREE RECHARGE ALL USERS","Click link now","100% genuine","Only today","wa.me/fake"],
    ["EARN 10000 PER DAY","No investment needed","100% guaranteed","Join now","WhatsApp 9999"],
    ["Make money online easy","Work from home","No experience needed","Daily payment","Join free"],
    ["Send this to 20 people","Good luck will come","Dont break the chain","Forward now","Must share"],
    ["Virus warning share this","Your phone at risk","Share immediately","Protect yourself","Forward now"],
    ["Free Netflix subscription","Click link","Limited time offer","Enter details","Claim now"],
    ["You are lucky winner","Lottery prize ready","Claim in 24 hours","Contact agent now","Send details"],
    ["Investment opportunity big","500% guaranteed returns","Risk free scheme","Join our group","Limited slots"],
    # Mixed noise
    ["lol","😂","okay","sure","haha"],
    ["🔥🔥🔥","💀💀","😂😂","lmao","ded"],
    ["bruh bruh bruh","lol lol","haha","💀","dead"],
    ["omg omg","bhai bhai","yaar yaar","haha","lol"],
    ["😂","😭","💀","🔥","😎"],
]

# ═══════════════════════════════════════════════════════════════════════════════
# MARKOV CORPUS — extended Indian GC style
# ═══════════════════════════════════════════════════════════════════════════════

_HINGLISH_CORPUS = """
bro kya scene hai aajkal kuch toh bata
arre sun na ek baat bata yaar seriously
bhai ye kya ho raha hai samajh nahi aa raha kya karoon
lol kya baat hai yaar bilkul sahi bol raha
haan bhai samajh gaya main kya karna chahte ho
yaar mujhe bhi batao kya hua achha ya bura
bhai aaj bohot bore ho raha hun kuch naya karo
kya kar raha hai tu aajkal busy hai kya
arre yaar tujhe pata hai kya hua kal raat
bhai sun ek interesting cheez batata hun tere ko
haan yaar wahi toh main bhi soch raha tha seriously
bhai seriously ye bahut crazy hai nahi socha tha aisa
oye sun na ek minute ruk kuch baat karni hai
yaar main toh pagal ho jaunga isse kya matlab hai
bhai aaj ka din bahut hectic tha sab kuch ek din mein
kya yaar tu bhi same cheez bol raha hai fr
arre haan mujhe yaad aa gaya bilkul wahi wali baat
bhai chal kuch karte hain bore ho raha hun aajkal
sun bhai ek kaam kar simple sa hai bas
yaar seriously bohot mushkil hai ye situation kya karoon
bhai update kya hai tere side se kuch hua kya
yaar mood kaisa hai aaj theek hai sab kuch
bhai kuch bhi baat karte hain timepass ke liye yaar
oye scene kya hai bhai kuch naya hua kya aaj
yaar ye wala episode dekha kya bhai insane tha
bhai real talk karte hain koi judgment nahi hai
yaar sab theek hai na ghar pe sab kuch
bhai kal plan kya hai kuch karte hain saath mein
yaar ye sab dekh ke dil dukha tha sachchi
bhai jo hua wo toh hua ab aage kya plan
"""

_ENGLISH_CORPUS = """
bro what is going on lately anything new happening
hey listen i need to tell you something important
lol that is actually so funny i cannot believe it
yeah i totally agree with you on that point
wait what are you serious right now no way
honestly i have no idea what happened yesterday bro
so basically the whole thing is like this right
literally cannot believe this happened today feels unreal
ngl that was actually really good not gonna lie
okay but hear me out on this one please
bruh that is absolutely insane how did it even happen
wait so you are telling me that happened for real
no way that actually happened for real i am shocked
honestly i was thinking the same exact thing lowkey
dude that is lowkey hilarious not gonna lie fr
you know what i think about this sometimes actually
that makes so much sense when you explain it
okay cool yeah i get what you mean now
alright fair enough that is a valid point though
hmm interesting never thought about it that way before
same same that happens to me all the time too
wait actually that tracks when you put it like that
not gonna lie that hit different ngl fr fr
bro real talk what do you think about all this
okay hear me out i have been thinking about it
"""

_markov_hi = None
_markov_en = None

if _HAS_MARKOV:
    try:
        _markov_hi = markovify.Text(_HINGLISH_CORPUS, state_size=2)
        _markov_en = markovify.Text(_ENGLISH_CORPUS, state_size=2)
    except Exception:
        pass


def markov_filler(lang="hinglish"):
    model = _markov_hi if lang == "hinglish" else _markov_en
    if model:
        try:
            s = model.make_short_sentence(70, tries=15)
            if s:
                return s.strip().lower()
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def keyboard_smash(length=None):
    l = length or random.randint(4, 12)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(l))

def random_emoji_combo():
    n = random.randint(1, 5)
    return "".join(random.choice(EMOJI_FILLERS) for _ in range(n))

def make_history_msg(lang="hinglish"):
    """Generate realistic history context message."""
    r = random.random()
    if r < 0.25 and _HAS_MARKOV:
        m = markov_filler(lang)
        if m:
            return m

    pool = HI_FILLERS if lang == "hinglish" else EN_FILLERS
    slang_pool = SLANG_FILLERS + DESI_SLANG[:20]

    roll = random.random()
    if roll < 0.45:
        return random.choice(pool)
    elif roll < 0.60:
        return random.choice(EMOJI_FILLERS)
    elif roll < 0.72:
        s = random.choice(HI_STARTERS if lang == "hinglish" else EN_STARTERS)
        f = random.choice(pool[:30])
        return f"{s} {f}"
    elif roll < 0.82:
        return random.choice(slang_pool)
    elif roll < 0.90:
        return f"{random.choice(EMOJI_FILLERS)} {random.choice(EMOJI_FILLERS)}"
    else:
        return random.choice(WHATSAPP_PATTERNS + INSTA_PATTERNS)

def make_continuation(lang="hinglish"):
    """Short non-signal continuation after target in history."""
    pool = HI_CONTINUATIONS if lang == "hinglish" else EN_CONTINUATIONS
    r = random.random()
    if r < 0.7:
        return random.choice(pool)
    elif r < 0.85:
        return random.choice(EMOJI_FILLERS)
    else:
        return random.choice(HI_FILLERS[:20] if lang == "hinglish" else EN_FILLERS[:20])


def build_history(lang="hinglish", n=5):
    """Build N history messages — realistic GC chatter."""
    return [make_history_msg(lang) for _ in range(n)]


def build_candidates(key_msgs, n_fillers=0):
    """Build candidate list with optional fillers. key_msgs placed first."""
    candidates = list(key_msgs)
    # pad with fillers if needed
    while len(candidates) < 3:
        if n_fillers > 0 and len(candidates) < 3:
            candidates.append("...")
            n_fillers -= 1
        else:
            break
    return candidates[:3]


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE FORMAT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def format_input(history, candidates):
    """Format H5+C1-3 input."""
    lines = []
    for i, h in enumerate(history):
        lines.append(f"H{i+1}: {h}")
    for i, c in enumerate(candidates):
        lines.append(f"C{i+1}: {c}")
    return "\n".join(lines)

def make_decision(typ, target=None, effort=None, title=None):
    """Build output string per v4 format. No nulls."""
    if typ == "text":
        return f"R: TYPE=text | TARGET={target} | EFFORT={effort}"
    elif typ == "react":
        return f"R: TYPE=react | TARGET={target} | TITLE={title}"
    elif typ == "media":
        return f"R: TYPE=media | TARGET={target} | TITLE={title}"
    elif typ == "ignore":
        return "R: TYPE=ignore"
    raise ValueError(f"Unknown type: {typ}")

def assign_think_tier(typ, context_hint=""):
    """Assign think tier based on type and context."""
    # @BOT and filler always specific
    if context_hint == "bot":
        return "hard"
    if context_hint == "filler":
        return "medium"

    # By type + randomness weighted
    if typ == "text":
        # Text often needs reasoning
        return random.choices(["hard","medium","easy"], weights=[0.50, 0.35, 0.15])[0]
    elif typ == "react":
        return random.choices(["hard","medium","easy"], weights=[0.25, 0.40, 0.35])[0]
    elif typ == "media":
        return random.choices(["hard","medium","easy"], weights=[0.30, 0.40, 0.30])[0]
    elif typ == "ignore":
        return random.choices(["hard","medium","easy"], weights=[0.05, 0.20, 0.75])[0]
    return "medium"


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE BUILDERS per type
# ═══════════════════════════════════════════════════════════════════════════════

def pick_target_pos(n_candidates=3):
    """
    Pick which slot (1-3) holds the real target message.
    For sparse windows the real message can land in ANY slot --
    remaining slots become fillers. This prevents C1 bias from filler samples.
    Distribution targets C1=25%, C2=35%, C3=40%.
    """
    return random.choices([1, 2, 3], weights=[0.25, 0.35, 0.40])[0]

def vary_msg(msg, lang="hinglish"):
    """Apply light surface variation to key message."""
    hi_pre = ["bhai","yaar","arre","oye","sun","bro","guys","suno","dekho"]
    en_pre = ["hey","so","btw","ngl","honestly","quick","wait","bro","guys"]

    r = random.random()
    if r < 0.35:
        pre = random.choice(hi_pre if lang == "hinglish" else en_pre)
        return f"{pre} {msg}"
    elif r < 0.55:
        return f"{msg} yaar" if lang == "hinglish" else f"{msg} bro"
    elif r < 0.65:
        return f"{msg} 😭" if random.random() < 0.5 else f"{msg} help"
    return msg


def build_text_sample(lang="hinglish"):
    cat = random.choice(list(TEXT_TOPICS.keys()))
    key_base, effort = random.choice(TEXT_TOPICS[cat])
    key_msg = vary_msg(key_base, lang)

    n_real = random.choices([1, 2, 3], weights=[0.15, 0.30, 0.55])[0]
    target_pos = pick_target_pos()  # always from full [1,2,3] distribution

    # Build 3 slots: target_pos gets key_msg, others get filler or real noise
    candidates = []
    real_placed = 0
    for slot in range(1, 4):
        if slot == target_pos:
            candidates.append(key_msg)
        elif real_placed < n_real - 1:
            candidates.append(make_history_msg(lang))
            real_placed += 1
        else:
            candidates.append("...")

    history = build_history(lang)
    target_label = f"C{target_pos}"
    dec = make_decision("text", target_label, effort)
    tier = assign_think_tier("text")

    return {
        "history": history,
        "candidates": candidates,
        "decision": dec,
        "think_tier": tier,
        "_type": "text",
        "_target": target_label,
        "_lang": lang,
    }


def build_react_sample(lang="hinglish"):
    emoji_key = random.choice(list(REACT_TOPICS.keys()))
    key_base, difficulty = random.choice(REACT_TOPICS[emoji_key])
    key_msg = vary_msg(key_base, lang)

    n_real = random.choices([1, 2, 3], weights=[0.15, 0.30, 0.55])[0]
    target_pos = pick_target_pos()

    candidates = []
    real_placed = 0
    for slot in range(1, 4):
        if slot == target_pos:
            candidates.append(key_msg)
        elif real_placed < n_real - 1:
            candidates.append(make_history_msg(lang))
            real_placed += 1
        else:
            candidates.append("...")

    history = build_history(lang)
    target_label = f"C{target_pos}"
    dec = make_decision("react", target_label, title=emoji_key)

    tier_map = {"hard": "hard", "medium": "medium", "easy": "easy"}
    tier = tier_map.get(difficulty, "medium")

    return {
        "history": history,
        "candidates": candidates,
        "decision": dec,
        "think_tier": tier,
        "_type": "react",
        "_target": target_label,
        "_lang": lang,
    }


def build_media_sample(lang="hinglish"):
    cat = random.choice(list(MEDIA_TOPICS.keys()))
    key_base, search_query = random.choice(MEDIA_TOPICS[cat])
    key_msg = vary_msg(key_base, lang)

    n_real = random.choices([1, 2, 3], weights=[0.15, 0.30, 0.55])[0]
    target_pos = pick_target_pos()

    candidates = []
    real_placed = 0
    for slot in range(1, 4):
        if slot == target_pos:
            candidates.append(key_msg)
        elif real_placed < n_real - 1:
            candidates.append(make_history_msg(lang))
            real_placed += 1
        else:
            candidates.append("...")

    history = build_history(lang)
    target_label = f"C{target_pos}"
    dec = make_decision("media", target_label, title=search_query)
    tier = assign_think_tier("media")

    return {
        "history": history,
        "candidates": candidates,
        "decision": dec,
        "think_tier": tier,
        "_type": "media",
        "_target": target_label,
        "_lang": lang,
    }


def build_ignore_sample():
    pattern = random.choice(IGNORE_PATTERNS)

    # Build 5-message history from pattern
    history = []
    for i in range(5):
        history.append(pattern[i % len(pattern)])

    # Candidates are also noise
    candidates = []
    for _ in range(3):
        r = random.random()
        if r < 0.35:
            candidates.append(random.choice(EMOJI_FILLERS) * random.randint(1,4))
        elif r < 0.55:
            candidates.append(keyboard_smash())
        elif r < 0.70:
            candidates.append(random.choice(HI_FILLERS[:15]))
        elif r < 0.80:
            candidates.append(random.choice(pattern))
        else:
            candidates.append("...")

    dec = make_decision("ignore")
    tier = assign_think_tier("ignore")

    return {
        "history": history,
        "candidates": candidates,
        "decision": dec,
        "think_tier": tier,
        "_type": "ignore",
        "_target": "null",
        "_lang": "mixed",
    }


def build_bot_mention_sample(lang="hinglish"):
    """@BOT explicit mention sample. Always text/react/media, never ignore."""
    # Pick type
    typ = random.choices(["text","react","media"], weights=[0.65, 0.20, 0.15])[0]

    # Build the @BOT mention message
    if typ == "text":
        cat = random.choice(list(TEXT_TOPICS.keys()))
        key_base, effort = random.choice(TEXT_TOPICS[cat])
        # Various @BOT mention styles
        styles = [
            f"@BOT {key_base}",
            f"@BOT bhai {key_base}",
            f"hey @BOT {key_base}",
            f"@BOT yaar {key_base}",
            f"{key_base} @BOT help",
            f"@BOT please {key_base}",
        ]
        key_msg = random.choice(styles)
        title_val = None
    elif typ == "react":
        emoji_key = random.choice(list(REACT_TOPICS.keys()))
        key_base, _ = random.choice(REACT_TOPICS[emoji_key])
        styles = [
            f"@BOT {key_base}",
            f"hey @BOT {key_base}",
            f"@BOT dekh {key_base}",
        ]
        key_msg = random.choice(styles)
        title_val = emoji_key
    else:  # media
        cat = random.choice(list(MEDIA_TOPICS.keys()))
        key_base, search_query = random.choice(MEDIA_TOPICS[cat])
        styles = [
            f"@BOT {key_base}",
            f"hey @BOT {key_base}",
            f"@BOT bhai {key_base}",
        ]
        key_msg = random.choice(styles)
        title_val = search_query
        effort = None

    # Sparse window — real message can land in any of the 3 slots
    n_real = random.choices([1, 2], weights=[0.6, 0.4])[0]
    target_pos = pick_target_pos()

    candidates = []
    real_placed = 0
    for slot in range(1, 4):
        if slot == target_pos:
            candidates.append(key_msg)
        elif real_placed < n_real - 1:
            candidates.append(make_history_msg(lang))
            real_placed += 1
        else:
            candidates.append("...")

    history = build_history(lang)
    target_label = f"C{target_pos}"

    if typ == "text":
        dec = make_decision("text", target_label, effort)
    elif typ == "react":
        dec = make_decision("react", target_label, title=title_val)
    else:
        dec = make_decision("media", target_label, title=title_val)

    return {
        "history": history,
        "candidates": candidates,
        "decision": dec,
        "think_tier": "hard",  # @BOT always hard
        "_type": typ,
        "_target": target_label,
        "_lang": lang,
        "_is_bot": True,
    }


def build_filler_sample(lang="hinglish"):
    """Sparse window sample: 1-2 real candidates, rest fillers."""
    typ = random.choices(["text","react","media","ignore"], weights=[0.35, 0.20, 0.20, 0.25])[0]

    n_real = random.choices([1, 2], weights=[0.55, 0.45])[0]

    if typ == "text":
        cat = random.choice(list(TEXT_TOPICS.keys()))
        key_base, effort = random.choice(TEXT_TOPICS[cat])
        key_msg = vary_msg(key_base, lang)
        title_val = None
    elif typ == "react":
        emoji_key = random.choice(list(REACT_TOPICS.keys()))
        key_base, _ = random.choice(REACT_TOPICS[emoji_key])
        key_msg = vary_msg(key_base, lang)
        title_val = emoji_key
    elif typ == "media":
        cat = random.choice(list(MEDIA_TOPICS.keys()))
        key_base, search_query = random.choice(MEDIA_TOPICS[cat])
        key_msg = vary_msg(key_base, lang)
        title_val = search_query
    else:
        key_msg = random.choice(IGNORE_PATTERNS[0])

    n_real = random.choices([1, 2], weights=[0.55, 0.45])[0]
    target_pos = pick_target_pos()

    candidates = []
    real_placed = 0
    for slot in range(1, 4):
        if slot == target_pos:
            candidates.append(key_msg if typ != "ignore" else random.choice(HI_FILLERS))
        elif real_placed < n_real - 1:
            candidates.append(make_history_msg(lang))
            real_placed += 1
        else:
            candidates.append("...")

    history = build_history(lang)
    target_label = f"C{target_pos}" if typ != "ignore" else None

    if typ == "text":
        dec = make_decision("text", target_label, effort if typ == "text" else "low")
    elif typ == "react":
        dec = make_decision("react", target_label, title=title_val)
    elif typ == "media":
        dec = make_decision("media", target_label, title=title_val)
    else:
        dec = make_decision("ignore")

    return {
        "history": history,
        "candidates": candidates,
        "decision": dec,
        "think_tier": "medium",  # filler always medium
        "_type": typ,
        "_target": target_label or "null",
        "_lang": lang,
        "_is_filler": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_text(s):
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def sample_signature(sample):
    key = " | ".join(
        normalize_text(c) for c in sample["candidates"]
    )
    return hashlib.md5(key.encode()).hexdigest()

def jaccard_sim(a, b):
    wa = set(normalize_text(a).split())
    wb = set(normalize_text(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class Deduper:
    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.seen_sigs = set()
        self.seen_key_texts = []       # normal + filler samples
        self.seen_bot_texts = []       # @BOT samples only
        self.dup_count = 0

    def is_duplicate(self, sample):
        sig = sample_signature(sample)
        if sig in self.seen_sigs:
            self.dup_count += 1
            return True

        is_bot = sample.get("_is_bot", False)

        # Strip @BOT prefix before comparing so bot samples compare
        # their actual message content against other bot messages only
        cand_text = normalize_text(" ".join(
            re.sub(r'@BOT\s*', '', c) for c in sample["candidates"] if c != "..."
        ))

        # Bot samples only dedup against other bot samples -- not against
        # normal samples, because @BOT prefix makes them semantically distinct
        # even if the base message is the same
        check_list = self.seen_bot_texts[-600:] if is_bot else self.seen_key_texts[-600:]

        for seen in check_list:
            if jaccard_sim(cand_text, seen) > self.threshold:
                self.dup_count += 1
                return True

        self.seen_sigs.add(sig)
        if is_bot:
            self.seen_bot_texts.append(cand_text)
        else:
            self.seen_key_texts.append(cand_text)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def pick_lang():
    return random.choices(
        ["hinglish", "english"],
        weights=[CFG["lang_dist"]["hinglish"], CFG["lang_dist"]["english"]]
    )[0]

def generate_normal_sample(type_hint=None):
    lang = pick_lang()
    if type_hint is None:
        type_hint = random.choices(
            list(CFG["type_dist"].keys()),
            weights=list(CFG["type_dist"].values())
        )[0]

    if type_hint == "text":
        return build_text_sample(lang)
    elif type_hint == "react":
        return build_react_sample(lang)
    elif type_hint == "media":
        return build_media_sample(lang)
    else:
        return build_ignore_sample()

def finalize_sample(s):
    """Convert internal dict to training format."""
    inp = format_input(s["history"], s["candidates"])
    return {
        "type": "chatml",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": inp},
            {"role": "assistant", "content": s["decision"]},
        ],
        "_meta": {
            "type": s["_type"],
            "target": s["_target"],
            "lang": s["_lang"],
            "think_tier": s.get("think_tier", "medium"),
            "is_bot": s.get("_is_bot", False),
            "is_filler": s.get("_is_filler", False),
        }
    }

def main():
    normal_n = CFG["normal_samples"]
    bot_n = CFG["bot_samples"]
    filler_n = CFG["filler_samples"]
    total_n = normal_n + bot_n + filler_n

    deduper = Deduper(threshold=CFG["dedup_threshold"])
    all_samples = []
    raw_samples = []  # keep internal dicts for CPA

    print(f"🚀 Auroic Router Dataset Generator v4")
    print(f"   Target: {total_n} samples ({normal_n} normal + {bot_n} @BOT + {filler_n} filler)")
    print()

    # ── Phase 1: Normal samples ──────────────────────────────────────────────
    print(f"[Phase 1] Generating {normal_n} normal samples...")
    type_counts = Counter()
    attempts = 0
    max_attempts = normal_n * 12

    while len([s for s in raw_samples if not s.get("_is_bot") and not s.get("_is_filler")]) < normal_n and attempts < max_attempts:
        attempts += 1
        n_normal = len([s for s in raw_samples if not s.get("_is_bot") and not s.get("_is_filler")])
        n_so_far = max(n_normal, 1)

        # Distribution enforcement
        deficits = {}
        for t, frac in CFG["type_dist"].items():
            actual = type_counts.get(t, 0) / n_so_far
            deficits[t] = frac - actual
        most_needed = max(deficits, key=deficits.get)
        type_hint = most_needed if deficits[most_needed] > 0.03 else None

        s = generate_normal_sample(type_hint)
        if deduper.is_duplicate(s):
            continue

        raw_samples.append(s)
        type_counts[s["_type"]] += 1

        if len(type_counts) > 0 and sum(type_counts.values()) % 500 == 0:
            n = sum(type_counts.values())
            print(f"  {n}/{normal_n} | dups: {deduper.dup_count} | dist: " + 
                  " ".join(f"{t}={c}" for t,c in sorted(type_counts.items())))

    print(f"  ✅ Normal: {sum(type_counts.values())} samples")

    # ── Phase 2: @BOT samples ────────────────────────────────────────────────
    print(f"\n[Phase 2] Generating {bot_n} @BOT samples...")
    bot_count = 0
    bot_attempts = 0
    while bot_count < bot_n and bot_attempts < bot_n * 8:
        bot_attempts += 1
        lang = pick_lang()
        s = build_bot_mention_sample(lang)
        if deduper.is_duplicate(s):
            continue
        raw_samples.append(s)
        bot_count += 1

    print(f"  ✅ @BOT: {bot_count} samples")

    # ── Phase 3: Filler samples ──────────────────────────────────────────────
    print(f"\n[Phase 3] Generating {filler_n} filler/sparse samples...")
    filler_count = 0
    filler_attempts = 0
    while filler_count < filler_n and filler_attempts < filler_n * 8:
        filler_attempts += 1
        lang = pick_lang()
        s = build_filler_sample(lang)
        if deduper.is_duplicate(s):
            continue
        raw_samples.append(s)
        filler_count += 1

    print(f"  ✅ Filler: {filler_count} samples")

    # ── Finalize ─────────────────────────────────────────────────────────────
    random.shuffle(raw_samples)
    final_samples = [finalize_sample(s) for s in raw_samples]

    total = len(final_samples)
    print(f"\n{'='*60}")
    print(f"Total samples: {total}")

    # ── Write base JSONL with metadata ───────────────────────────────────────
    base_out = "dataset_v4.jsonl"
    with open(base_out, "w", encoding="utf-8") as f:
        for s in final_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✅ Base dataset → {base_out}")

    # ── Write compact .txt for review ────────────────────────────────────────
    txt_out = "dataset_v4.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        for s in final_samples:
            f.write(s["messages"][1]["content"] + "\n")
            f.write(s["messages"][2]["content"] + "\n\n")
    print(f"✅ Compact txt → {txt_out}")

    # ── Write annotation manifest ─────────────────────────────────────────────
    # Separate by think tier for annotation pipeline
    hard_samples = [s for s in final_samples if s["_meta"]["think_tier"] == "hard"]
    medium_samples = [s for s in final_samples if s["_meta"]["think_tier"] == "medium"]
    easy_samples = [s for s in final_samples if s["_meta"]["think_tier"] == "easy"]

    manifest_out = "annotation_manifest_v4.jsonl"
    with open(manifest_out, "w", encoding="utf-8") as f:
        for s in final_samples:
            record = {
                "id": hashlib.md5(s["messages"][1]["content"].encode()).hexdigest()[:12],
                "think_tier": s["_meta"]["think_tier"],
                "type": s["_meta"]["type"],
                "is_bot": s["_meta"]["is_bot"],
                "is_filler": s["_meta"]["is_filler"],
                "input": s["messages"][1]["content"],
                "output": s["messages"][2]["content"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Annotation manifest → {manifest_out}")
    print(f"\nThink tier breakdown:")
    print(f"  hard   (full think):  {len(hard_samples):5d} ({len(hard_samples)/total*100:.1f}%)")
    print(f"  medium (short think): {len(medium_samples):5d} ({len(medium_samples)/total*100:.1f}%)")
    print(f"  easy   (no think):    {len(easy_samples):5d} ({len(easy_samples)/total*100:.1f}%)")

    # ── Stats ─────────────────────────────────────────────────────────────────
    type_c = Counter(s["_meta"]["type"] for s in final_samples)
    lang_c = Counter(s["_meta"]["lang"] for s in final_samples)
    bot_c = sum(1 for s in final_samples if s["_meta"]["is_bot"])
    filler_c = sum(1 for s in final_samples if s["_meta"]["is_filler"])

    print(f"\n{'='*60}")
    print(f"=== TYPE DISTRIBUTION ===")
    for t, c in sorted(type_c.items(), key=lambda x: -x[1]):
        pct = c/total*100
        bar = "█" * int(pct/2)
        print(f"  {t:8s}: {c:5d} ({pct:5.1f}%) {bar}")

    print(f"\n=== LANGUAGE ===")
    for l, c in lang_c.most_common():
        print(f"  {l:12s}: {c:5d} ({c/total*100:.1f}%)")

    print(f"\n=== SPECIAL BUCKETS ===")
    print(f"  @BOT windows:    {bot_c:5d}")
    print(f"  Filler windows:  {filler_c:5d}")

    # ── Validation ───────────────────────────────────────────────────────────
    print(f"\n=== VALIDATION ===")
    errors = []
    for i, s in enumerate(final_samples):
        dec = s["messages"][2]["content"]
        if not dec.startswith("R: TYPE="):
            errors.append(f"#{i} bad format: {dec[:40]}")
            continue

        parts = {}
        for p in dec.replace("R: ", "").split(" | "):
            if "=" in p:
                k, v = p.split("=", 1)
                parts[k] = v

        t = parts.get("TYPE", "")
        target = parts.get("TARGET", "")
        effort = parts.get("EFFORT", "")
        title = parts.get("TITLE", "")

        if t == "text":
            if not target.startswith("C"):
                errors.append(f"#{i} text bad target: {target}")
            if effort not in ["low","medium","high"]:
                errors.append(f"#{i} text bad effort: {effort}")
            if "TITLE" in parts:
                errors.append(f"#{i} text has TITLE (should not)")
        elif t == "react":
            if not target.startswith("C"):
                errors.append(f"#{i} react bad target: {target}")
            if not title:
                errors.append(f"#{i} react missing TITLE")
            if "EFFORT" in parts:
                errors.append(f"#{i} react has EFFORT (should not)")
        elif t == "media":
            if not target.startswith("C"):
                errors.append(f"#{i} media bad target: {target}")
            if not title:
                errors.append(f"#{i} media missing TITLE")
            if "EFFORT" in parts:
                errors.append(f"#{i} media has EFFORT (should not)")
        elif t == "ignore":
            if len(parts) > 1:
                errors.append(f"#{i} ignore has extra fields: {parts}")
        else:
            errors.append(f"#{i} invalid type: {t}")

    if errors:
        print(f"  ❌ {len(errors)} validation errors:")
        for e in errors[:15]:
            print(f"    {e}")
    else:
        print(f"  ✅ All {total} samples pass validation")

    # ── Sample Previews ───────────────────────────────────────────────────────
    print(f"\n=== SAMPLE PREVIEWS ===")
    preview_indices = [0, 100, 500, 1000, 2000, 4000, total-1]
    for idx in preview_indices:
        if idx < len(final_samples):
            s = final_samples[idx]
            m = s["_meta"]
            print(f"\n[{idx}] type={m['type']} target={m['target']} lang={m['lang']} tier={m['think_tier']} bot={m['is_bot']} filler={m['is_filler']}")
            print(s["messages"][1]["content"])
            print("→", s["messages"][2]["content"])

    print(f"\n{'='*60}")
    print(f"Done! {total} samples ready.")
    print(f"Next step: run annotation pipeline on annotation_manifest_v4.jsonl")
    print(f"  - hard tier:   {len(hard_samples)} samples → full think blocks")
    print(f"  - medium tier: {len(medium_samples)} samples → short think blocks (1-2 sentences)")
    print(f"  - easy tier:   {len(easy_samples)} samples → no think block, direct output")


if __name__ == "__main__":
    main()
"""
Auroic Router Dataset Generator v5
====================================
5000 samples | Semantic deduper (sentence-transformers) | Distribution enforcer
Canonical titles | Adversarial early-targets | Multi-signal context
No thinking blocks — non-thinking SFT only for 0.6B router
"""

import json, random, re, hashlib, string, math
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# ── Optional library imports (graceful fallback) ─────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False
    print("⚠️  sentence-transformers not found, falling back to Jaccard dedup")

try:
    import nltk
    from nltk.corpus import wordnet
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False
    print("⚠️  nltk not found, paraphrase expansion disabled")

try:
    import markovify
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False
    print("⚠️  markovify not found, using static fillers only")

try:
    import emoji as emoji_lib
    _HAS_EMOJI = True
except ImportError:
    _HAS_EMOJI = False

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    "total_samples": 5000,

    "type_dist": {
        "text":        0.33,
        "media":       0.16,
        "react":       0.14,
        "acknowledge": 0.15,
        "translate":   0.13,
        "ignore":      0.09,
    },

    # Updated TARGET distribution — more early targets
    "target_dist": {
        "M5": 0.50,
        "M4": 0.28,
        "M3": 0.12,
        "M2": 0.06,
        "M1": 0.04,
    },

    "lang_dist": {
        "hinglish": 0.70,
        "english":  0.30,
    },

    "content_dist": {
        "real_chat":  0.70,
        "spam_noise": 0.20,
        "emoji_only": 0.10,
    },

    "dedup_threshold": 0.72,
    "semantic_dedup_threshold": 0.80,
    "max_loops": 8,
    "tolerance": 0.035,
}

# Adversarial & multi-signal rates
TARGET_ADVERSARIAL_RATE = 0.15   # 15% force M1/M2
MULTI_SIGNAL_RATE = 0.20          # 20% multi-intent samples

SYSTEM = "You are the Auroic Router. Given 5 chat messages, output exactly one routing decision."

# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════════

HI_FILLERS = [
    "hm","ok","lol","?","bro","haan","nahi","kya","yaar","ugh","ohhh","nice",
    "accha","theek hai","haha","hmm","okay","sure","chal","dekh","sun","bhai",
    "arre","seriously","no way","fr?","sach mein","kya bol raha","bata",
    "acha go on","haan haan","bas","chalo","sun na","dekh na","bol","kya yaar",
    "pagal hai kya","sahi bola","bilkul","nope","yup","nahi re","wahi toh",
    "mujhe pata tha","sahi hai","theek lagta hai","hmm okay","chal theek",
    "aur bata","kya scene hai","bata na","sun bhai","acha acha","bol na",
    "kya hua phir","seriously yaar","arre haan","matlab?","wait kya","huh",
    "abey","oye","arrey","achi baat","noted","got it","samajh gaya","pata hai",
]

EN_FILLERS = [
    "lol","okay","haha","nice","cool","gotcha","makes sense","yeah","sure",
    "no way","interesting","and then","wait what","okay and","right","really",
    "fair enough","i see","hmm","got it","exactly","true","same","fair",
    "i mean","well","anyway","so","oh","ah","wait","nope","yep","mhm",
    "alright","honestly","literally","actually","basically","lowkey","fr",
    "ngl","imo","tbh","rn","omg","wtf","bruh","oof","lmao","lmfao","xd",
]

EMOJI_FILLERS = [
    "😂","💀","🔥","😭","🙄","👀","🤣","😅","🥴","😤","🫡","🤦","💯",
    "😮","🤯","😬","🫠","😑","🤔","🙃","😏","👍","✅","❤️","🫶","🤝",
    "👏","🎉","🥳","😎","💪","🫣","😳","🤌","🙏","⚡","🌚","🌝","🌊",
    "🐐","👑","🚀","💥","🎯","🧠","💡","✨","🌈","🎭","🎪","🏆","🥇",
]

SLANG_FILLERS = [
    "bc","wtf","ngl","fr","lol","lmao","bro ngl","fr fr","ong","no cap",
    "lowkey","highkey","deadass","bet","slay","bussin","sheesh","yeet",
    "vibe","based","sus","ratio","W","L","cope","seethe","mid",
]

HI_STARTERS = [
    "bhai","yaar","arre","suno","dekh","bro","sun","hey","oye","abey",
    "guys","bhai sun","yaar sun","arre bhai","ek baat bata","suno zara",
    "bhai ek min","yaar ek cheez","oye bhai","bhai bata na",
]

EN_STARTERS = [
    "hey","so","guys","anyone","quick","ok so","wait","listen","btw",
    "honestly","ngl","lowkey","literally","bro","yo","omg","wait so",
    "real talk","genuine question","okay but","not gonna lie",
]

# Extended emoji pool from emoji library if available
def get_extended_emojis():
    if _HAS_EMOJI:
        common = list(emoji_lib.EMOJI_DATA.keys())
        # Weight toward common chat emojis - pick subset
        chat_emojis = [e for e in common if len(e) <= 2][:200]
        return chat_emojis if chat_emojis else EMOJI_FILLERS
    return EMOJI_FILLERS

EXTENDED_EMOJIS = get_extended_emojis()

# ═══════════════════════════════════════════════════════════════════════════════
# MARKOV CHAT FILLER GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

_HINGLISH_CORPUS = """
bro kya scene hai aajkal
arre sun na ek baat bata
bhai ye kya ho raha hai samajh nahi aa raha
lol kya baat hai yaar
haan bhai samajh gaya main
yaar mujhe bhi batao kya hua
bhai aaj bohot bore ho raha hun
kya kar raha hai tu aajkal
arre yaar tujhe pata hai kya hua
bhai sun ek interesting cheez batata hun
haan yaar wahi toh main bhi soch raha tha
bhai seriously ye bahut crazy hai
oye sun na ek minute ruk
yaar main toh pagal ho jaunga
bhai aaj ka din bahut hectic tha
kya yaar tu bhi same cheez bol raha hai
arre haan mujhe yaad aa gaya
bhai chal kuch karte hain bore ho raha
sun bhai ek kaam kar
yaar seriously bohot mushkil hai ye
"""

_ENGLISH_CORPUS = """
bro what is going on lately
hey listen i need to tell you something
lol that is actually so funny
yeah i totally agree with you on that
wait what are you serious right now
honestly i have no idea what happened
so basically the thing is this
literally cannot believe this happened today
ngl that was actually really good
okay but hear me out on this one
bruh that is absolutely insane
wait so you are telling me that
no way that actually happened for real
honestly i was thinking the same thing
dude that is lowkey hilarious ngl
"""

_markov_hi = None
_markov_en = None

if _HAS_MARKOV:
    try:
        _markov_hi = markovify.Text(_HINGLISH_CORPUS, state_size=1)
        _markov_en = markovify.Text(_ENGLISH_CORPUS, state_size=1)
    except Exception:
        pass


def markov_filler(lang="hinglish"):
    """Generate a novel filler using Markov chains, fallback to static."""
    model = _markov_hi if lang == "hinglish" else _markov_en
    if model:
        try:
            s = model.make_short_sentence(60, tries=10)
            if s:
                return s.lower()
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# WORDNET PARAPHRASE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_PARAPHRASE_SKIP = {
    "bhai","yaar","arre","oye","sun","bro","hey","guys","kya","hai",
    "ho","raha","kar","karo","de","do","na","mein","se","ke","ka","ki",
    "the","a","an","is","are","was","were","my","your","i","you","me",
    "it","in","on","at","to","for","with","not","no","and","or","but",
}

def synonym_swap(text, swap_prob=0.25):
    """Replace random English words with WordNet synonyms."""
    if not _HAS_NLTK:
        return text
    words = text.split()
    result = []
    for w in words:
        if w.lower() in _PARAPHRASE_SKIP or len(w) < 4 or random.random() > swap_prob:
            result.append(w)
            continue
        syns = wordnet.synsets(w.lower())
        lemmas = list(set(
            l.name().replace("_", " ")
            for s in syns for l in s.lemmas()
            if l.name().lower() != w.lower()
        ))
        if lemmas:
            result.append(random.choice(lemmas[:5]))
        else:
            result.append(w)
    return " ".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# KEYBOARD SPAM GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def keyboard_smash():
    length = random.randint(4, 10)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

def random_emoji_combo():
    pool = EXTENDED_EMOJIS if len(EXTENDED_EMOJIS) > 10 else EMOJI_FILLERS
    n = random.randint(1, 4)
    return "".join(random.choice(pool) for _ in range(n))


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC BANKS — canonical titles (category:intent)
# ═══════════════════════════════════════════════════════════════════════════════

TEXT_TOPICS = {
    "coding": [
        ("bhai ye python error samajh nahi aa raha kuch bhi kar lo","text:python_error"),
        ("js console is full of red errors nothing works yaar","text:javascript_error"),
        ("bhai git conflict aa gaya resolve kaise karu help karo","text:git_conflict"),
        ("my api is returning 404 nothing works at all bro","text:api_error"),
        ("bhai css kaam nahi kar raha layout toot gaya pura","text:css_layout"),
        ("bhai sql query galat aa rahi hai data nahi aa raha","text:database_error"),
        ("my react component is not rendering anymore crashed","text:react_error"),
        ("docker container wont start keeps crashing every time","text:docker_error"),
        ("bhai npm install fail ho raha error aa raha hai","text:npm_error"),
        ("bhai app load nahi ho raha screen blank hai yaar","text:app_loading"),
        ("typescript type error is confusing me badly help karo","text:typescript_error"),
        ("bhai flutter build fail ho raha yaar help please","text:flutter_error"),
        ("bhai python import error module not found kya karu","text:python_import"),
        ("cors policy is blocking my api request yaar fix karo","text:cors_error"),
        ("bhai firebase login kaam nahi kar raha help please","text:firebase_auth"),
        ("bhai localhost nahi chal raha port busy hai yaar","text:localhost_error"),
        ("java nullpointer exception is crashing my app badly","text:java_error"),
        ("bhai vscode extension kaam nahi kar raha yaar help","text:vscode_issue"),
        ("my regex pattern is completely wrong help me fix it","text:regex_help"),
        ("heroku deploy is failing build error again and again","text:heroku_deploy"),
        ("nextjs routing is not working pages are missing bro","text:nextjs_routing"),
        ("mongodb query is returning empty array always help","text:mongodb_error"),
        ("redis cache is not storing data properly fix karo","text:redis_error"),
        ("websocket keeps disconnecting after few seconds yaar","text:websocket_error"),
        ("jwt token expiry is too fast causing logout issues","text:jwt_auth"),
        ("aws s3 upload is failing permissions error aa raha","text:aws_s3"),
        ("nginx giving 502 bad gateway on every request help","text:nginx_error"),
        ("kubernetes pod keeps crashing in restart loop yaar","text:kubernetes_error"),
        ("graphql query is throwing resolver error fix karo","text:graphql_error"),
        ("electron desktop app crashes on startup every time","text:electron_error"),
        ("pandas dataframe operation throwing error help bhai","text:pandas_error"),
        ("tensorflow import failing gpu not detected at all","text:tensorflow_setup"),
        ("django orm query is not joining tables properly","text:django_error"),
        ("fastapi server not starting up properly kya karu","text:fastapi_error"),
        ("tailwind classes not applying in my project at all","text:tailwind_error"),
        ("webpack bundling is failing with weird error message","text:webpack_error"),
        ("jest unit test is failing unexpectedly help please","text:jest_error"),
        ("github actions workflow is failing on every push","text:ci_cd_error"),
        ("stripe payment integration is throwing error bhai","text:stripe_error"),
        ("socket io events not emitting properly fix karo","text:socketio_error"),
        ("vue component lifecycle hook is broken not working","text:vue_error"),
        ("angular module is not loading properly at all yaar","text:angular_error"),
        ("svelte reactive statement not updating the ui help","text:svelte_error"),
        ("prisma database migration is failing today again","text:prisma_error"),
        ("supabase authentication is not working today help","text:supabase_error"),
        ("vercel deployment is failing with build error bro","text:vercel_error"),
        ("cloudflare worker script throwing exception always","text:cloudflare_error"),
        ("express middleware is not executing properly at all","text:express_error"),
        ("apollo client cache is not updating on mutation bro","text:apollo_error"),
        ("redux state is not updating after dispatch action","text:redux_error"),
        ("zustand store reset is not working properly help","text:zustand_error"),
        ("chrome extension content script not loading at all","text:extension_error"),
        ("pwa service worker is not caching properly fix it","text:pwa_error"),
        ("threejs scene is not rendering on canvas element","text:threejs_error"),
        ("d3 chart is not updating with new data coming in","text:d3_error"),
        ("html canvas drawing api is behaving oddly help me","text:canvas_error"),
        ("webassembly module is not loading properly at all","text:wasm_error"),
        ("rust borrow checker is throwing errors everywhere","text:rust_error"),
        ("golang goroutine is causing race condition fix it","text:golang_error"),
        ("swift ios app crashes on specific device model yaar","text:swift_error"),
        ("kotlin android activity is not launching at all","text:android_error"),
        ("unity game object is disappearing randomly in scene","text:unity_error"),
        ("unreal engine blueprint not compiling today at all","text:unreal_error"),
        ("openai api is returning rate limit error every time","text:openai_error"),
        ("langchain chain is not calling tools properly help","text:langchain_error"),
        ("huggingface model download is failing today again","text:huggingface_error"),
        ("stable diffusion out of memory error again and again","text:sd_error"),
        ("ollama model is not responding to any prompts help","text:ollama_error"),
        ("linux command giving permission denied error bhai","text:linux_error"),
        ("bash script is throwing syntax error help me fix","text:bash_error"),
        ("cron job is not executing at scheduled time at all","text:cron_error"),
        ("nginx ssl certificate is expired need help renewing","text:ssl_error"),
        ("docker compose containers not communicating at all","text:compose_error"),
        ("kafka consumer group not receiving messages help","text:kafka_error"),
        ("rabbitmq queue is not processing messages properly","text:rabbitmq_error"),
        ("elasticsearch query not returning any results bro","text:elasticsearch_error"),
        ("redis pubsub not delivering messages properly yaar","text:redis_pubsub"),
        ("prometheus not scraping metrics from app fix karo","text:prometheus_error"),
        ("grafana dashboard showing no data at all help bhai","text:grafana_error"),
    ],

    "study": [
        ("bhai kal exam hai kuch nahi pada help karo please","text:exam_panic"),
        ("bhai ye physics formula samajh nahi aa raha help","text:physics_doubt"),
        ("calculus integration problem i cant solve at all","text:calculus_help"),
        ("bhai organic chemistry reaction samajh nahi aa rahi","text:chemistry_doubt"),
        ("bhai mitosis and meiosis difference kya hai batao","text:biology_doubt"),
        ("i keep forgetting these history dates help me please","text:history_help"),
        ("bhai english grammar rules confuse karte hain mujhe","text:english_grammar"),
        ("how do i structure a good essay properly guide me","text:essay_writing"),
        ("bhai notes bhej do please class miss kar di thi","text:notes_request"),
        ("bhai assignment aaj submit karni hai urgent help","text:assignment_help"),
        ("bhai college admission process kya hota hai batao","text:college_admission"),
        ("bhai jee neet entrance exam prep kaise karein guide","text:entrance_exam"),
        ("how to apply for a scholarship abroad help please","text:scholarship_help"),
        ("i have no idea how to start my thesis writing help","text:thesis_help"),
        ("bhai presentation banani hai tips do please yaar","text:presentation_help"),
        ("group project mein koi kaam nahi kar raha help bhai","text:group_project"),
        ("my essay has accidental plagiarism help me fix it","text:plagiarism_fix"),
        ("how do i cite sources in apa format properly guide","text:citation_help"),
        ("bhai time table banana hai padhai ka help karo na","text:study_schedule"),
        ("i cant focus while studying at all kya karu bhai","text:focus_tips"),
        ("how to memorize things faster for exam tips please","text:memory_tips"),
        ("bhai mock test solve karna hai help karo please","text:mock_test"),
        ("bhai percentage calculate karna hai marks ki help","text:marks_calc"),
        ("bhai internship ke liye kaise apply karein batao na","text:internship_help"),
        ("mera resume bahut weak hai help karo improve karna","text:resume_help"),
        ("bhai interview ke liye kaise prepare karein guide do","text:interview_prep"),
        ("gre exam prep strategy kya honi chahiye batao bhai","text:gre_prep"),
        ("bhai ielts writing task improve karna hai help karo","text:ielts_prep"),
        ("which online course is best for learning python bro","text:course_recommend"),
        ("bhai competitive exam ke liye best books suggest karo","text:book_recommend"),
        ("bhai class skip kar li ab kya karoon notes chahiye","text:class_skip"),
        ("exam mein cheating sahi hai kya honestly batao yaar","text:exam_ethics"),
        ("bhai teacher se problem ho gayi class mein help karo","text:teacher_conflict"),
        ("bhai hostel mein adjust karna mushkil hai tips do","text:hostel_life"),
        ("bhai study group banana chahta hun kaise karoon","text:study_group"),
        ("bhai english improve karna hai kaise karoon guide do","text:language_learning"),
        ("is a coding bootcamp worth the money honestly tell","text:bootcamp_advice"),
        ("bhai backlog clear karna hai kaise karoon help me","text:backlog_help"),
        ("bhai marks kaise improve karoon please batao yaar","text:marks_improve"),
        ("bhai college chhod dene ka mann kar raha hai advice","text:dropout_advice"),
    ],

    "relationships": [
        ("bhai crush ko kya bolun kaise bataaun help karo na","text:crush_advice"),
        ("just broke up and i feel completely terrible help me","text:breakup_support"),
        ("bhai best friend se jhagda ho gaya yaar kya karun","text:friendship_conflict"),
        ("bhai parents career pe pressure de rahe hain help","text:family_pressure"),
        ("my partner is being really toxic help me please bro","text:toxic_relationship"),
        ("bhai long distance relationship kaise chalaye guide","text:long_distance"),
        ("she rejected me i dont know what to do help please","text:rejection_support"),
        ("bhai partner se jealous ho raha hun kyon help karo","text:jealousy_advice"),
        ("i am scared of committing to this relationship help","text:commitment_fear"),
        ("bhai partner pe trust nahi raha kya karoon advice do","text:trust_issue"),
        ("my ex suddenly started texting me again kya karun","text:ex_advice"),
        ("bhai yaar ne friend zone kar diya kya karoon help","text:friendzone_advice"),
        ("bhai friend group mein bahut drama ho raha hai help","text:group_drama"),
        ("bhai behen se bahut fight hoti hai ghar mein advice","text:sibling_conflict"),
        ("bhai ghar wale shaadi ke liye bol rahe hain help","text:marriage_pressure"),
        ("bhai online wale ko trust kar sakte hain kya batao","text:online_relationship"),
        ("bhai ek taraf se pyaar hai kya karoon advice do na","text:one_sided_love"),
        ("i need help moving on after this breakup please bro","text:moving_on"),
        ("bhai bahut purani dosti khatam ho rahi hai help yaar","text:friendship_ending"),
        ("bhai love ya career mein se kya choose karoon advice","text:love_vs_career"),
        ("i think my partner is cheating on me help please","text:cheating_suspicion"),
        ("bhai propose karna hai ideas chahiye help karo na","text:proposal_help"),
        ("first date ke liye kya karoon tips do please bhai","text:first_date"),
        ("bhai kisi ko sorry bolna hai kaise karoon properly","text:apology_advice"),
        ("how do i set boundaries with my partner help me bro","text:boundary_setting"),
        ("bhai partner se baat hi nahi hoti properly help karo","text:communication_fix"),
        ("parents bahut overprotective hain help karo please","text:overprotective_parents"),
        ("different culture wale se love hai yaar advice do","text:cultural_relationship"),
        ("bhai age difference waali relationship kaisi advice","text:age_gap"),
        ("i am completely heartbroken please help me recover","text:heartbreak_recovery"),
        ("bhai log se milne mein darr lagta hai help karo na","text:social_anxiety"),
        ("bhai khud pe confidence nahi aata kaise banaoon do","text:confidence_advice"),
        ("i have been feeling very lonely lately bro help me","text:loneliness_support"),
        ("bhai nayi jagah aaya hun dost kaise banaaoon guide","text:making_friends"),
        ("bhai office mein colleague pe crush hai kya karun","text:workplace_crush"),
        ("bhai parents ka divorce ho raha hai help me please","text:divorce_support"),
        ("i think i am in an abusive situation help me please","text:abuse_support"),
        ("my partner controls everything i do help me escape","text:controlling_partner"),
        ("bhai crush ne ghost kar diya kya karoon advice do","text:ghosted_advice"),
        ("bhai ghar se chhupaake relationship hai kya karun","text:secret_relationship"),
    ],

    "casual": [
        ("bhai bohot bore ho raha hun kuch batao kya karun","text:boredom_help"),
        ("bhai trip plan karni hai suggestions do please yaar","text:trip_planning"),
        ("bhai koi acchi movie suggest karo please bro dekni hai","text:movie_recommend"),
        ("bhai konsa phone lungu budget 15k hai suggest karo","text:phone_buying"),
        ("i cant stop overthinking please help me calm down","text:overthinking_help"),
        ("bhai konsa career choose karoon confused hun advice do","text:career_advice"),
        ("bhai mera startup idea kaisa lagta hai feedback do","text:startup_feedback"),
        ("bhai crypto mein invest karoon kya advice do please","text:crypto_advice"),
        ("bhai gym jaana chodh diya motivate karo please yaar","text:gym_motivation"),
        ("bhai weight loss ke liye kya khaoon suggest karo na","text:diet_plan"),
        ("bhai raat ko neend nahi aati kya karoon help karo","text:sleep_problem"),
        ("i keep procrastinating how to be productive help me","text:productivity_help"),
        ("bhai paise kaise bachaaoon student hun tips do please","text:money_saving"),
        ("bhai simple recipe batao khana banana hai ghar pe","text:cooking_help"),
        ("bhai kya pehnu interview ke liye suggest karo please","text:fashion_advice"),
        ("bhai konsi haircut acchi lagegi suggest karo yaar","text:haircut_advice"),
        ("bhai konsa laptop lungu college ke liye suggest karo","text:laptop_advice"),
        ("bhai 150cc mein kaunsi bike sahi rahegi suggest karo","text:bike_advice"),
        ("bhai pet lena chahta hun konsa accha hai advice do","text:pet_advice"),
        ("bhai girlfriend ko kya gift doon birthday pe ideas","text:gift_ideas"),
        ("i have no time management skills help me organize","text:time_management"),
        ("bhai anxiety bahut hoti hai kaise rokoon help karo","text:anxiety_help"),
        ("i get angry very quickly how to control my temper","text:anger_management"),
        ("bhai phone band karna chahta hun digital detox tips","text:digital_detox"),
        ("bhai new year resolution kaise achieve karoon guide","text:goal_setting"),
        ("bhai freelancing kaise shuru karoon guide do please","text:freelancing_guide"),
        ("bhai stock market mein kaise invest karoon guide do","text:stock_market"),
        ("bhai youtube channel start karna hai tips do please","text:youtube_tips"),
        ("bhai instagram followers kaise badhaaoon tips do na","text:instagram_growth"),
        ("bhai public speaking se dar lagta hai help karo yaar","text:public_speaking"),
    ],

    "health": [
        ("bhai sar dard ho raha hai kya karoon remedy batao","text:headache_remedy"),
        ("bhai cold aa gayi hai kya karoon gharelu upay batao","text:cold_flu"),
        ("my back pain is unbearable please suggest something","text:back_pain"),
        ("bhai aankhein thak gayi hain screen se relief kaise","text:eye_strain"),
        ("bhai bukhar aa gaya hai kya karoon help karo please","text:fever_help"),
        ("i am very stressed how do i calm down help me bro","text:stress_relief"),
        ("bhai kya khaoon healthy rehne ke liye tips do please","text:healthy_eating"),
        ("bhai exercise routine banana hai guide do please yaar","text:exercise_routine"),
        ("bhai neend ka routine kaise improve karu help karo","text:sleep_hygiene"),
        ("bhai bahut burden feel ho raha hai help me please","text:mental_health"),
        ("bhai kuch galat khaa liya stomach upset hai help","text:food_poisoning"),
        ("i keep having allergic reactions help me manage them","text:allergy_help"),
        ("bhai pani kam peeta hun kaise badhaaoon tips do yaar","text:hydration_tips"),
        ("my posture is really bad how to fix it help please","text:posture_fix"),
        ("bhai immunity weak hai kaise badhaaoon tips do bhai","text:immunity_boost"),
        ("period cramps are really bad help please any remedy","text:period_pain"),
        ("bhai face pe pimples bahut hain help karo skin care","text:skin_care"),
        ("bhai bahut hair fall ho raha hai kya karoon help me","text:hair_fall"),
        ("bhai weight gain karna hai healthy tarike se guide do","text:weight_gain"),
        ("bhai smoking chodni hai tips do please help karo yaar","text:quit_smoking"),
    ],

    "life_events": [
        ("bhai internship mil gayi mujhe yaar finally selected","text:internship_news"),
        ("yaar naukri ka offer aaya hai kya karoon advice do","text:job_offer"),
        ("bhai promotion ho gayi office mein aaj celebration","text:promotion_news"),
        ("bhai apna startup launch karne wala hun advice do","text:startup_launch"),
        ("yaar pehli salary aayi hai kya karoon bhai batao","text:first_salary"),
        ("bhai finally bike le li apni dream wali feeling good","text:new_bike"),
        ("yaar driving license mil gaya mujhe finally done","text:driving_license"),
        ("bhai passport ban gaya ab travel kar sakta hun yaar","text:passport_got"),
        ("yaar visa approve ho gaya abroad jaana hai planning","text:visa_approved"),
        ("bhai naye sheher mein aa gaya hun adjust karna hai","text:new_city"),
        ("yaar pehla apna ghar le liya feeling amazing bhai","text:first_apartment"),
        ("bhai hackathon jeet liya yaar kuch nahi socha tha","text:hackathon_win"),
        ("yaar mera app app store pe aa gaya finally published","text:app_published"),
        ("bhai 10k subscribers ho gaye yaar kaise hua amazing","text:youtube_milestone"),
        ("bhai exam mein fail ho gaya kya karoon ab help me","text:exam_failure"),
        ("yaar kaam se nikaala gaya kuch samajh nahi aa raha","text:job_loss"),
        ("bhai business band karna pad raha hai help advice","text:business_failed"),
        ("yaar hafka chhota accident ho gaya theek hun dw","text:minor_accident"),
        ("bhai phone kho gaya sab data chala gaya help karo","text:phone_lost"),
        ("yaar wallet chheen liya kisi ne kya karoon help me","text:wallet_stolen"),
        ("bhai online scam ho gaya paise gaye kya karun ab","text:online_scam"),
        ("yaar meri billi mar gayi bahut dukh ho raha hai bro","text:pet_loss"),
        ("bhai dada ji hospital mein hain please pray for them","text:family_sick"),
        ("yaar ghar shift karna hai next week tips do please","text:house_shifting"),
        ("bhai 10 kilo weight lose kar liya yaar finally done","text:weight_loss_win"),
        ("yaar pehli baar 5k run kiya feeling amazing bhai","text:first_run"),
        ("bhai guitar seekh liya ek song baja sakta hun now","text:guitar_learned"),
        ("yaar khana banana seekh gaya biryani bhi bana sakta","text:cooking_skill"),
        ("bhai 10 books padh li is saal reading streak going","text:reading_milestone"),
        ("yaar 30 din meditation complete kar li feeling great","text:meditation_streak"),
        ("bhai stocks mein invest kiya tha return aaya finally","text:investment_returns"),
        ("yaar pehli emi ka time aa gaya tension hai bhai","text:emi_advice"),
        ("bhai health insurance lena chahta hun guide do yaar","text:insurance_help"),
        ("yaar pehli baar ITR file karni hai help karo please","text:tax_filing"),
        ("bhai bahut mehnatkari raha loan pay kar diya finally","text:loan_cleared"),
        ("yaar sarkari naukri ka form bhara hai advice do bhai","text:govt_job"),
        ("bhai UPSC ki taiyari shuru karni hai guide do please","text:upsc_prep"),
        ("yaar FAANG se offer aaya hai kya karun advice do","text:big_tech_offer"),
        ("bhai side income start karna chahta hun guide do na","text:side_hustle"),
        ("yaar content creator banna chahta hun tips do please","text:content_creator"),
        ("bhai podcast shuru karna hai guide do please yaar","text:podcast_advice"),
        ("yaar blog likhna shuru karna chahta hun tips do bhai","text:blog_advice"),
        ("bhai photography seekhna chahta hun guide do please","text:photography_advice"),
        ("yaar gaming setup banana chahta hun budget hai limited","text:gaming_setup"),
        ("bhai stocks mein loss ho gaya kya karoon help me","text:stock_loss"),
        ("yaar crypto se thoda profit hua kya karoon advice do","text:crypto_profit"),
        ("bhai car lene ka plan kar raha hun guide do please","text:car_buying"),
        ("yaar shaadi plan karna shuru kiya tips do please","text:wedding_planning"),
        ("bhai baby aa rahi hai ghar mein advice do parenting","text:parenting_advice"),
        ("yaar pehla full body checkup karaya sab theek hai","text:health_checkup"),
        ("bhai therapy shuru ki hai bahut helpful lag rahi","text:therapy_update"),
        ("yaar toxic job chhod di bahut better feel ho raha","text:quit_job"),
        ("bhai gap year lena chahta hun advice do please yaar","text:gap_year"),
        ("yaar masters abroad karna chahta hun guide do please","text:study_abroad"),
        ("bhai japanese B1 level clear kar liya feeling great","text:language_achievement"),
        ("yaar state level selection ho gaya cricket mein bhai","text:sports_achievement"),
        ("bhai NGO join kiya volunteer karna tha feeling good","text:volunteering"),
        ("yaar pehli baar blood donate kiya feeling amazing","text:blood_donation"),
        ("bhai ghar mein plants lagaye hain tips do plant care","text:plant_care"),
        ("yaar cooking channel shuru karna chahta hun guide do","text:cooking_channel"),
    ],

    "work": [
        ("bhai office mein politics ho rahi hai kya karoon help","text:office_politics"),
        ("yaar boss se salary raise maangni hai tips do please","text:salary_raise"),
        ("bhai kaam aur ghar ka balance nahi ho raha help me","text:work_life_balance"),
        ("yaar ghar se kaam karte thak gaya hun tips do please","text:remote_work"),
        ("bhai boss bahut toxic hai kya karoon advice do yaar","text:toxic_boss"),
        ("yaar colleague se problem ho gayi office mein help","text:colleague_conflict"),
        ("bhai resign karna chahta hun guidance do please yaar","text:resignation_advice"),
        ("yaar job switch karna chahta hun timeline kya honi","text:job_switch"),
        ("bhai freelancing mein kitna charge karoon rate batao","text:freelance_pricing"),
        ("yaar client payment nahi de raha kya karoon help me","text:client_issue"),
        ("bhai appraisal ke liye kaise prepare karoon guide do","text:appraisal_prep"),
        ("yaar Google se interview call aaya hai help karo bhai","text:interview_call"),
        ("bhai startup join karoon ya badi company advice do","text:startup_vs_corp"),
        ("yaar ghar pe productive nahi hun tips do please bhai","text:wfh_productivity"),
        ("bhai career change karna chahta hun advice do please","text:career_pivot"),
        ("yaar linkedin pe networking kaise karoon tips do bhai","text:networking_tips"),
        ("bhai office mein presentation deni hai help karo yaar","text:presentation_prep"),
        ("yaar professional email likhna hai templates do bhai","text:email_writing"),
        ("bhai bade meeting se darr lagta hai help karo please","text:meeting_anxiety"),
        ("yaar batch mein sab cut ho gaye main bacha layoff","text:layoff_survivor"),
    ],
}

MEDIA_TOPICS = [
    ("bhai ek funny meme bhej yaar please hasna hai","media:funny_meme","high"),
    ("send a cute dog gif please bro want to see","media:dog_gif","medium"),
    ("bhai sad sticker bhej yaar feeling low hai aaj","media:sad_sticker","low"),
    ("yaar gaming rage gif chahiye bhai bhej de please","media:gaming_gif","medium"),
    ("bhai birthday gif bhej celebration wala please yaar","media:birthday_gif","medium"),
    ("send shocked reaction gif please bro need it now","media:shocked_gif","low"),
    ("bhai food craving ho rahi hai pizza gif bhej yaar","media:food_gif","medium"),
    ("yaar workout motivation gif bhej please need push","media:workout_gif","medium"),
    ("send a wholesome hug sticker bro feeling down","media:hug_sticker","low"),
    ("bhai diwali wala gif bhej festival mood hai yaar","media:diwali_gif","low"),
    ("send cat doing something funny gif please bro","media:cat_gif","medium"),
    ("bhai cricket six moment gif chahiye match wala","media:cricket_gif","high"),
    ("yaar anime reaction gif bhej dramatic wala please","media:anime_gif","medium"),
    ("bhai dance gif bhej mood accha hai party time","media:dance_gif","high"),
    ("send a facepalm gif please bruh need it badly","media:facepalm_gif","low"),
    ("bhai rain aesthetic gif bhej vibes wali cozy mood","media:rain_gif","low"),
    ("yaar sleeping gif bhej neend aa rahi hai bhai","media:sleep_gif","low"),
    ("send cooking gif something looks delicious bro","media:cooking_gif","medium"),
    ("bhai superhero landing gif bhej cool wala epic","media:superhero_gif","high"),
    ("yaar baby laughing gif bhej cute wala adorable","media:baby_gif","medium"),
    ("send a mind blown gif bro seriously was shocked","media:mindblown_gif","low"),
    ("bhai nature scenery gif bhej calming wala peaceful","media:nature_gif","low"),
    ("yaar car drifting gif bhej speed wala drift clip","media:car_drift","high"),
    ("bhai sunset beach gif bhej aesthetic wala vibes","media:sunset_gif","low"),
    ("send graduation celebration gif please congrats","media:graduation_gif","medium"),
]

REACT_TOPICS = [
    ("bhai result aa gaya marks acche aaye yaar finally","🥳"),
    ("did you hear that celebrity just got arrested omg","😲"),
    ("bhai rank up kar liya finally diamond mila gaming","🔥"),
    ("aaj mera birthday hai 21 saal ka ho gaya bhai","🎂"),
    ("this baby animal is absolutely adorable omg so cute","🥰"),
    ("bhai jeet gaye hum tournament final jeet liya yaar","🏆"),
    ("this fail compilation is pure comedy gold fr dying","😂"),
    ("that show plot twist completely destroyed me bruh","🤯"),
    ("bhai promotion mil gayi salary hike bhi aai office","😎"),
    ("bhai pehli barish aayi monsoon shuru ho gaya yaar","🌧️"),
    ("just got into my dream college oh my god so happy","🎉"),
    ("bhai aaj ka match insane tha last ball six wow","⚡"),
    ("this song is hitting different at 3am ngl feeling","🎵"),
    ("bhai gym mein personal record tod diya aaj finally","💪"),
    ("this movie ending was absolutely not expected bruh","😱"),
    ("bhai boss ne publicly appreciate kiya aaj office me","🙌"),
    ("just finished my dissertation finally done at last","😮‍💨"),
    ("bhai kal se chutti shuru ho gayi yaar finally free","🌴"),
    ("that meme format is absolutely perfect timing dead","💀"),
    ("bhai arranged marriage fixed ho gayi meri yaar omg","😭"),
    ("just hit 1000 subscribers on youtube channel bro","📈"),
    ("bhai aaj itni garmi hai 42 degrees outside melting","🥵"),
    ("first snow of the year outside my window so pretty","❄️"),
    ("bhai ghar pe gol gappa party hai aaj yaar come over","😋"),
    ("just saw a shooting star make a wish now everyone","⭐"),
]

ACKNOWLEDGE_TOPICS = [
    ("bhai kaam ho gaya assignment submit kar diya finally",["noted done","well done bhai","great job","nicely done","proud of you"]),
    ("i reached home safely just got here now all good",["glad youre safe","safe travels","take rest now","good to know","relieved"]),
    ("bhai payment kar diya upi se transfer hua confirm",["payment received","confirmed got it","noted thanks","received","all good"]),
    ("good morning bhai subah subah gm everyone rise up",["good morning","gm bhai","rise and shine","morning vibes","gm to you too"]),
    ("bhai bohot shukriya mera kaam kar diya tune thanks",["youre welcome","anytime yaar","happy to help","no problem bhai","always here"]),
    ("just sent you the document check your email inbox",["got it thanks","received noted","got the file","thank you","acknowledged"]),
    ("bhai so jaata hun good night everyone gn sweet dreams",["good night","sleep well bhai","gn sweet dreams","rest well","gn take care"]),
    ("i am really sorry about what happened yaar my bad",["its okay bhai","apology accepted","all good now","no worries","we are good"]),
    ("bhai update sun progress update de raha hun project",["got the update","noted thanks","understood bhai","got it","thanks for update"]),
    ("bhai tabiyat theek nahi hai fever hai mujhe aaj sick",["get well soon","rest up bhai","take care yaar","feel better soon","rest karo"]),
    ("reached the hospital safely waiting outside for doc",["okay stay strong","good to know","keep us updated","take care","with you always"]),
    ("bhai file share kar di check karo please google drive",["got the file","received thanks","checking now","noted","got it bhai"]),
    ("just finished the presentation nailed it hopefully",["fingers crossed","you did great","all the best","proud of you","hope it went well"]),
    ("bhai ghar aa gaya safely trip mast rahi enjoyed alot",["glad youre safe","rest kar ab","welcome back","happy to hear","take rest"]),
    ("thanks for being there always appreciate it bro really",["always here for you","anytime bro","that is what friends are for","always","no problem"]),
    ("bhai interview de ke aaya accha laga mujhe went well",["fingers crossed bhai","inshallah ho jayega","all the best","hope it went well","proud of you"]),
    ("submitted the project before deadline finally done yaar",["great job bhai","well done","proud of you","nicely done","you did it"]),
    ("bhai neend aa rahi hai kal milte hain gn sweet dreams",["gn bhai","kal milte hain","sweet dreams","good night","rest well"]),
    ("food is ready come eat everyone at table khana ready",["coming now","on my way","be right there","just a min","coming"]),
    ("bhai movie download ho gayi link bhej raha hun check",["got it thanks","received the link","downloading now","thanks bhai","got it"]),
]

TRANSLATE_TOPICS = [
    ("Muchas gracias por tu ayuda amigo de verdad","translate:spanish_text"),
    ("No entiendo nada de lo que dijiste ahora","translate:spanish_text"),
    ("Por favor ayudame con esto es urgente hoy","translate:spanish_text"),
    ("Estoy muy emocionado por lo que paso hoy","translate:spanish_text"),
    ("No se que hacer en esta situacion ahora mismo","translate:spanish_text"),
    ("Me alegra mucho verte despues de tanto tiempo","translate:spanish_text"),
    ("Esto es absolutamente increible no puedo creer","translate:spanish_text"),
    ("Te quiero mucho amigo siempre estaras en mi corazon","translate:spanish_text"),
    ("Feliz cumpleanos espero que tengas un dia maravilloso","translate:spanish_text"),
    ("El examen fue muy dificil pero creo que lo pase","translate:spanish_text"),
    ("ありがとうございました本当に助かりました","translate:japanese_text"),
    ("お疲れ様でした今日もよく頑張りましたね","translate:japanese_text"),
    ("よろしくお願いしますこれからもよろしく","translate:japanese_text"),
    ("すみません少し助けていただけますか","translate:japanese_text"),
    ("わかりました了解です問題ありません","translate:japanese_text"),
    ("頑張ってください応援していますよ","translate:japanese_text"),
    ("おはようございます今日もよい一日を","translate:japanese_text"),
    ("おやすみなさいゆっくり休んでください","translate:japanese_text"),
    ("本当に嬉しいです信じられないくらいです","translate:japanese_text"),
    ("また会いましょうそれまでお元気で","translate:japanese_text"),
    ("감사합니다 정말 도움이 많이 됐어요","translate:korean_text"),
    ("안녕하세요 처음 뵙겠습니다 잘 부탁드려요","translate:korean_text"),
    ("사랑해요 항상 곁에 있어줘서 고마워요","translate:korean_text"),
    ("미안해요 제가 잘못했어요 용서해 주세요","translate:korean_text"),
    ("정말 대단해요 어떻게 그렇게 잘 하세요","translate:korean_text"),
    ("화이팅 할 수 있어요 믿어요","translate:korean_text"),
    ("배고파요 뭔가 맛있는 거 먹고 싶어요","translate:korean_text"),
    ("오늘 정말 힘든 하루였어요 지쳐버렸어요","translate:korean_text"),
    ("Merci beaucoup pour tout ce que tu as fait","translate:french_text"),
    ("Je ne comprends pas du tout ce que tu veux dire","translate:french_text"),
    ("Sil vous plait aidez moi cest vraiment urgent","translate:french_text"),
    ("Cest tres interessant je naurais pas pense a ca","translate:french_text"),
    ("Je suis desole pour ce qui sest passe hier","translate:french_text"),
    ("Bonne chance pour demain je suis avec toi","translate:french_text"),
    ("Comment ca va depuis la derniere fois quon sest vus","translate:french_text"),
    ("Au revoir et a bientot prends soin de toi","translate:french_text"),
    ("شكراً جزيلاً على مساعدتك لي اليوم","translate:arabic_text"),
    ("لا أفهم هذا الكلام أبداً وضح لي من فضلك","translate:arabic_text"),
    ("من فضلك ساعدني في هذا الأمر المهم","translate:arabic_text"),
    ("أنا سعيد جداً بما حدث اليوم الحمد لله","translate:arabic_text"),
    ("هذا رائع جداً لم أكن أتوقع هذا أبداً","translate:arabic_text"),
    ("كيف حالك اليوم أتمنى أن تكون بخير دائماً","translate:arabic_text"),
    ("مع السلامة وإلى اللقاء قريباً إن شاء الله","translate:arabic_text"),
    ("Vielen Dank fuer deine Hilfe das war sehr nett","translate:german_text"),
    ("Ich verstehe das wirklich nicht erklaer mir das bitte","translate:german_text"),
    ("Bitte hilf mir dabei ich schaffe es alleine nicht","translate:german_text"),
    ("Das ist wirklich sehr interessant hab ich nicht gewusst","translate:german_text"),
    ("Es tut mir sehr leid fuer das was passiert ist","translate:german_text"),
    ("Wie geht es dir ich hoffe du bist gesund heute","translate:german_text"),
    ("Muito obrigado pela sua ajuda foi fundamental hoje","translate:portuguese_text"),
    ("Nao entendo nada do que esta acontecendo aqui","translate:portuguese_text"),
    ("Por favor me ajuda com isso eh muito urgente","translate:portuguese_text"),
    ("Estou muito feliz com tudo que aconteceu hoje","translate:portuguese_text"),
    ("آپ کا بہت بہت شکریہ آپ نے بہت مدد کی","translate:urdu_text"),
    ("مجھے سمجھ نہیں آیا ذرا سمجھائیں مہربانی","translate:urdu_text"),
    ("برائے مہربانی میری مدد کریں بہت ضروری ہے","translate:urdu_text"),
    ("یہ بہت اچھا ہے واقعی بہت خوشی ہوئی آج","translate:urdu_text"),
    ("மிக்க நன்றி உங்கள் உதவிக்கு மிகவும் நன்றி","translate:tamil_text"),
    ("எனக்கு புரியவில்லை கொஞ்சம் விளக்கி சொல்லுங்கள்","translate:tamil_text"),
    ("தயவுசெய்து உதவுங்கள் மிகவும் முக்கியமான விஷயம்","translate:tamil_text"),
    ("यार कल परीक्षा है कुछ नहीं पढ़ा अभी तक","translate:hindi_text"),
    ("भाई बहुत थक गया हूं आज का दिन बुरा था","translate:hindi_text"),
    ("मुझे समझ नहीं आ रहा क्या करूं इस बारे में","translate:hindi_text"),
]

IGNORE_TOPICS = [
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
    ["shjg","sfdd","vsdgsg","dgsg","gfgfhgf"],
    ["asdfgh","qwerty","zxcvbn","poiuyt","lkjhgf"],
    ["1234567","qweasd","zxcqwe","123qwe","asd123"],
    ["aaaaaaa","bbbbbbb","ccccccc","ddddddd","eeeeeee"],
    ["fjdksla","sldkfj","qpwoei","rutyei","aldskf"],
    ["mnbvcx","lkjhgf","poiuyt","qweasd","zxcvbn"],
    ["WIN FREE IPHONE NOW","CLICK THIS LINK","LIMITED OFFER","CLAIM TODAY","bit.ly/scam"],
    ["Congratulations you won","Click here to claim","Offer expires today","Dont miss out","Forward to 10"],
    ["FREE RECHARGE ALL USERS","Click link now","100% genuine","Only today","wa.me/fake"],
    ["You have been selected","God bless you","Send bank details","Claim prize now","Forward for luck"],
    ["EARN 10000 PER DAY","No investment needed","100% guaranteed","Join now","WhatsApp 9999"],
    ["Beta testing invite","Click to join","Free premium access","Limited spots","Refer and earn"],
    ["Your account will be closed","Verify now","Click this link","Urgent action needed","bit.ly/verify"],
    ["Free Amazon gift card","Survey takes 1 min","Click here","Guaranteed reward","Claim now"],
    ["Investment opportunity","500% returns","Risk free","Join our group","Limited slots"],
    ["Make money online easy","Work from home","No experience needed","Daily payment","Join free"],
    ["Send this to 20 people","Good luck will come","Dont break the chain","Forward now","Must share"],
    ["Breaking news share this","Forward to all groups","Must read urgent","Share before deleted","Copy paste now"],
    ["Free Netflix subscription","Click link","Limited time offer","Enter details","Claim now"],
    ["Virus warning forward this","Your phone at risk","Share immediately","Protect yourself","Forward now"],
    ["You are our lucky winner","Lottery prize","Claim in 24 hours","Contact agent","Send details"],
    ["ok","ok","ok","ok","ok"],
    [".","..","...","....","😶"],
    ["test","test123","testing","yo","nvm"],
    ["","hello","hi","hey",""],
    ["bhai","bhai","bhai","bhai","bhai"],
]

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SIGNAL DISTRACTOR TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

DISTRACTOR_MSGS = {
    "tech": [
        "anyone know nginx config fix yaar","bhai docker crash ho raha help",
        "python error aa rahi hai console mein","react component render nahi ho raha",
        "bhai git conflict aa gaya resolve karo","api 404 de rahi hai fix karo",
    ],
    "media": [
        "send cat gif please bro","bhai ek funny meme bhej yaar",
        "yaar gaming gif chahiye send karo","bhai sad sticker bhej na",
        "send a cute dog gif please","bhai birthday gif bhej de",
    ],
    "translate": [
        "Muchas gracias amigo","ありがとうございました","감사합니다","Merci beaucoup",
    ],
    "ack": [
        "bhai kaam ho gaya done","reached home safely all good",
        "payment kar diya upi se","good morning everyone gm",
    ],
    "react_trigger": [
        "bhai result aa gaya marks acche aaye","just got into dream college omg",
        "bhai promotion mil gayi salary hike aai","tournament jeet liya yaar",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDERS — varied realistic filler messages
# ═══════════════════════════════════════════════════════════════════════════════

def hi_context():
    """Generate a realistic hinglish filler message."""
    # Try markov first for novelty
    if random.random() < 0.3:
        m = markov_filler("hinglish")
        if m:
            return m
    patterns = [
        lambda: random.choice(HI_FILLERS),
        lambda: f"{random.choice(HI_STARTERS)} {random.choice(['kya chal raha','kuch batao','sun na','dekh na','bol na'])}",
        lambda: random.choice(EMOJI_FILLERS),
        lambda: f"{random.choice(['haan','okay','accha'])} {random.choice(['bhai','yaar','bro'])}",
        lambda: random.choice(["kya hua","kya scene","bata na","phir kya","sach mein"]),
        lambda: f"{random.choice(EMOJI_FILLERS)} {random.choice(EMOJI_FILLERS)}",
        lambda: random.choice(["samajh gaya","noted","got it","theek hai yaar","chal dekho"]),
        lambda: f"{random.choice(SLANG_FILLERS)} {random.choice(['bhai','yaar','bro'])}",
    ]
    return random.choice(patterns)()

def en_context():
    """Generate a realistic english filler message."""
    if random.random() < 0.3:
        m = markov_filler("english")
        if m:
            return m
    patterns = [
        lambda: random.choice(EN_FILLERS),
        lambda: f"{random.choice(EN_STARTERS)} {random.choice(['whats up','tell me more','go on','okay','really'])}",
        lambda: random.choice(EMOJI_FILLERS),
        lambda: f"{random.choice(['yeah','okay','sure','ngl','lowkey'])} {random.choice(['makes sense','fair enough','true','got it','interesting'])}",
        lambda: random.choice(["wait what","no way","oh really","thats wild","seriously tho"]),
        lambda: random.choice(SLANG_FILLERS),
    ]
    return random.choice(patterns)()

def make_filler(lang="hinglish"):
    return hi_context() if lang == "hinglish" else en_context()

def natural_continuation(lang="hinglish"):
    """Generate messages that look like conversation continuing AFTER a request.
    Used for adversarial samples where target is early (M1/M2)."""
    hi_pool = [
        "haan dekho","accha theek hai","phir kya hua","aur bata","hmm sahi hai",
        "okay bhai","chal theek","samajh gaya","haan haan","lol okay",
        "sahi bola","bilkul","wahi toh","haan okay","chal","phir","aur",
        "accha accha","theek lagta hai","hmm","okay okay","haan sahi",
        "woh toh hai","baat toh sahi hai","accha sun","haan yaar","okay cool",
        "nice bhai","sahi hai yaar","hmm interesting","accha concept hai",
    ]
    en_pool = [
        "yeah makes sense","okay cool","interesting","right right","hmm true",
        "fair enough","got it","yeah i see","oh okay","makes sense now",
        "sure sure","yep","true that","oh right","yeah exactly","same here",
        "honestly yeah","lowkey true","ngl yeah","fr fr","oh wait true",
    ]
    pool = hi_pool if lang == "hinglish" else en_pool
    r = random.random()
    if r < 0.7:
        return random.choice(pool)
    elif r < 0.85:
        return random.choice(EMOJI_FILLERS)
    else:
        return random.choice(HI_FILLERS[:15] if lang == "hinglish" else EN_FILLERS[:15])


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_msgs_with_key(key_msg, target_pos, lang="hinglish", noise_after=True):
    """Build 5 messages placing key_msg at target_pos (1-indexed)."""
    msgs = []
    idx = target_pos - 1

    for i in range(5):
        if i < idx:
            msgs.append(make_filler(lang))
        elif i == idx:
            msgs.append(key_msg)
        else:
            if noise_after:
                short_pool = HI_FILLERS[:20] if lang == "hinglish" else EN_FILLERS[:20]
                msgs.append(random.choice(short_pool + list(EMOJI_FILLERS[:10])))
            else:
                msgs.append(make_filler(lang))
    return msgs

def build_adversarial_msgs(key_msg, target_pos, lang="hinglish"):
    """Build 5 messages for adversarial early-target (M1/M2).
    Messages after target are natural continuations with NO keywords from the request."""
    msgs = []
    idx = target_pos - 1  # 0 or 1

    for i in range(5):
        if i < idx:
            msgs.append(make_filler(lang))
        elif i == idx:
            msgs.append(key_msg)
        else:
            # After target: natural continuation, topic drift, NO request keywords
            msgs.append(natural_continuation(lang))
    return msgs

def pick_target_pos():
    """Pick target position using updated distribution with more early targets."""
    r = random.random()
    if r < 0.50:   return 5    # M5
    elif r < 0.78: return 4    # M4
    elif r < 0.90: return 3    # M3
    elif r < 0.96: return 2    # M2
    else:           return 1   # M1

def pick_adversarial_target():
    """Force M1 or M2 for adversarial samples."""
    return random.choice([1, 2])

def make_decision(typ, target, effort, title):
    t = f"M{target}" if target else "null"
    return f"R: TYPE={typ} | TARGET={t} | EFFORT={effort} | TITLE={title}"


def apply_paraphrase(key_msg):
    """Optionally apply WordNet synonym swap for variety."""
    if _HAS_NLTK and random.random() < 0.30:
        return synonym_swap(key_msg, swap_prob=0.20)
    return key_msg


# ── Suffixes for natural phrasing ──
HI_SUFFIXES = [
    "help karo","batao please","koi bata sakta hai","samjhao zara",
    "guide do","tips do","koi idea hai","kya sochte ho","suggest karo",
    "kaise karoon","kya karoon","please help","urgent hai","pls bata",
    "yaar bahut confuse hun","koi toh bata","kya karun",
]

EN_SUFFIXES = [
    "please help","any ideas","need advice","what should i do",
    "how do i handle this","can someone guide","tips please",
    "help needed","any suggestions","what do you think","urgent",
]


def build_text_sample(lang="hinglish", adversarial=False):
    cat = random.choice(list(TEXT_TOPICS.keys()))
    topic = random.choice(TEXT_TOPICS[cat])
    key_msg_base, canonical_title = topic

    # Apply paraphrase expansion for variety
    key_msg = apply_paraphrase(key_msg_base)

    # Add variation prefix/suffix
    hi_prefixes = ["bhai","yaar","arre","oye","sun","suno","guys"]
    en_prefixes = ["hey","so","btw","quick question","anyone","ngl","honestly"]

    if random.random() < 0.5:
        if lang == "hinglish":
            key_msg = f"{random.choice(hi_prefixes)} {key_msg}"
        else:
            key_msg = f"{random.choice(en_prefixes)} {key_msg}"

    # Enforce minimum length (no label-like messages)
    words = key_msg.strip().split()
    if len(words) < 4:
        if lang == "hinglish":
            key_msg = f"bhai {key_msg_base} {random.choice(HI_SUFFIXES)}"
        else:
            key_msg = f"hey {key_msg_base} {random.choice(EN_SUFFIXES)}"

    if adversarial:
        target_pos = pick_adversarial_target()
        msgs = build_adversarial_msgs(key_msg, target_pos, lang)
    else:
        target_pos = pick_target_pos()
        msgs = build_msgs_with_key(key_msg, target_pos, lang)

    effort = random.choice(["low","medium","high"])
    dec = make_decision("text", target_pos, effort, canonical_title)
    return msgs, dec

def build_media_sample(lang="hinglish", adversarial=False):
    topic = random.choice(MEDIA_TOPICS)
    key_msg_base, canonical_title, effort = topic

    key_msg = apply_paraphrase(key_msg_base)

    hi_variations = [
        key_msg, f"bhai {key_msg}", f"yaar {key_msg}",
        f"{key_msg} please", f"arre {key_msg} na yaar", f"oye {key_msg}",
    ]
    en_variations = [
        key_msg, f"hey {key_msg}", f"can you {key_msg}",
        f"{key_msg} please bro", f"someone {key_msg}",
    ]
    key_msg = random.choice(hi_variations if lang == "hinglish" else en_variations)

    if adversarial:
        target_pos = pick_adversarial_target()
        msgs = build_adversarial_msgs(key_msg, target_pos, lang)
    else:
        target_pos = pick_target_pos()
        msgs = build_msgs_with_key(key_msg, target_pos, lang)

    dec = make_decision("media", target_pos, effort, canonical_title)
    return msgs, dec

def build_react_sample(lang="hinglish", adversarial=False):
    topic = random.choice(REACT_TOPICS)
    key_msg_base, emoji_reaction = topic

    key_msg = apply_paraphrase(key_msg_base)

    hi_variations = [
        key_msg, f"bhai {key_msg}", f"yaar sun {key_msg}",
        f"{key_msg} yaar", f"arre {key_msg}",
    ]
    en_variations = [
        key_msg, f"guys {key_msg}", f"omg {key_msg}",
        f"bro {key_msg}", f"wait {key_msg}",
    ]
    key_msg = random.choice(hi_variations if lang == "hinglish" else en_variations)

    if adversarial:
        target_pos = pick_adversarial_target()
        msgs = build_adversarial_msgs(key_msg, target_pos, lang)
    else:
        target_pos = pick_target_pos()
        msgs = build_msgs_with_key(key_msg, target_pos, lang)

    dec = make_decision("react", target_pos, "null", emoji_reaction)
    return msgs, dec

def build_acknowledge_sample(lang="hinglish", adversarial=False):
    topic = random.choice(ACKNOWLEDGE_TOPICS)
    key_msg_base, reply_pool = topic

    key_msg = apply_paraphrase(key_msg_base)

    hi_variations = [
        key_msg, f"bhai {key_msg}", f"yaar {key_msg}",
        f"{key_msg} bhai", f"btw {key_msg}",
    ]
    en_variations = [
        key_msg, f"hey {key_msg}", f"just fyi {key_msg}",
        f"{key_msg} everyone", f"quick update {key_msg}",
    ]
    key_msg = random.choice(hi_variations if lang == "hinglish" else en_variations)

    if adversarial:
        target_pos = pick_adversarial_target()
        msgs = build_adversarial_msgs(key_msg, target_pos, lang)
    else:
        target_pos = pick_target_pos()
        msgs = build_msgs_with_key(key_msg, target_pos, lang)

    reply = random.choice(reply_pool)
    dec = make_decision("acknowledge", target_pos, "null", reply)
    return msgs, dec

def build_translate_sample(lang="hinglish", adversarial=False):
    topic = random.choice(TRANSLATE_TOPICS)
    foreign_text, canonical_title = topic

    hi_contexts_before = [
        ["bhai sun","haan bol","kuch mila tha message mein","ye dekh kya bol raha hai"],
        ["yaar dekh","kya hai ye","kahin se aaya","samajh nahi aaya"],
        ["bhai","ek message aaya hai","translate kar do","please help"],
        ["yaar sun na","foreign language hai","kya likha hai","bata na"],
        ["oye","ye kya script hai","anime ka tha kya","samjha koi"],
        ["bhai caption mein tha","kaunsi language","pata nahi mujhe","tu bata"],
        ["yaar email mein tha ye","boss ka message tha","kya matlab hai","urgent hai"],
    ]
    en_contexts_before = [
        ["hey guys","got this message","have no idea what it says","can someone translate"],
        ["so","my friend sent this","i think its spanish","what does it mean"],
        ["bro","this was in the caption","what language even","help please"],
        ["wait","got this email","from a client","need a translation asap"],
        ["guys","saw this on instagram","looks japanese","anyone know"],
    ]

    if adversarial:
        target_pos = pick_adversarial_target()
    else:
        target_pos = pick_target_pos()
    idx = target_pos - 1

    contexts = hi_contexts_before if lang == "hinglish" else en_contexts_before
    ctx = random.choice(contexts)

    msgs = []
    for i in range(5):
        if i < idx:
            if i < len(ctx):
                msgs.append(ctx[i])
            else:
                msgs.append(make_filler(lang))
        elif i == idx:
            msgs.append(foreign_text)
        else:
            after_pool = (
                ["kya matlab hai","samjha koi","translate karo","help please","batao na"]
                if lang == "hinglish" else
                ["what does this say","anyone know","translate please","help","what language is this"]
            )
            msgs.append(random.choice(after_pool))

    dec = make_decision("translate", target_pos, "null", canonical_title)
    return msgs, dec

def build_ignore_sample():
    topic = random.choice(IGNORE_TOPICS)
    msgs = [topic[i % len(topic)] for i in range(5)]
    if random.random() < 0.3:
        pos = random.randint(0,4)
        r = random.random()
        if r < 0.4:
            msgs[pos] = random.choice(list(EMOJI_FILLERS) + HI_FILLERS[:15])
        elif r < 0.6:
            msgs[pos] = keyboard_smash()
        else:
            msgs[pos] = random_emoji_combo()
    dec = "R: TYPE=ignore | TARGET=null | EFFORT=null | TITLE=null"
    return msgs, dec


def build_multi_signal_sample(lang="hinglish"):
    """Build a sample with 2-3 competing signals. Only one is correct target."""
    # Pick actual type and generate its key message
    actual_type = random.choices(
        ["text","media","react","acknowledge"],
        weights=[0.4, 0.25, 0.20, 0.15]
    )[0]

    # Build the real sample first
    if actual_type == "text":
        msgs, dec = build_text_sample(lang, adversarial=False)
    elif actual_type == "media":
        msgs, dec = build_media_sample(lang, adversarial=False)
    elif actual_type == "react":
        msgs, dec = build_react_sample(lang, adversarial=False)
    else:
        msgs, dec = build_acknowledge_sample(lang, adversarial=False)

    # Parse target position
    parts = dict(p.split("=",1) for p in dec.replace("R: ","").split(" | "))
    target_str = parts.get("TARGET","null")
    if target_str == "null":
        return msgs, dec
    target_pos = int(target_str[1])

    # Insert 1-2 distractors in non-target positions
    distractor_types = [k for k in DISTRACTOR_MSGS.keys() if k != {
        "text":"tech","media":"media","react":"react_trigger","acknowledge":"ack"
    }.get(actual_type, "")]
    if not distractor_types:
        distractor_types = list(DISTRACTOR_MSGS.keys())

    available_positions = [i for i in range(5) if i != target_pos - 1]
    n_distractors = random.randint(1, min(2, len(available_positions)))
    distractor_positions = random.sample(available_positions, n_distractors)

    for pos in distractor_positions:
        dtype = random.choice(distractor_types)
        msgs[pos] = random.choice(DISTRACTOR_MSGS[dtype])

    return msgs, dec


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC DEDUPLICATION — sentence-transformers + Jaccard fallback
# ═══════════════════════════════════════════════════════════════════════════════

_sbert_model = None

def get_sbert_model():
    global _sbert_model
    if _sbert_model is None and _HAS_SBERT:
        print("📦 Loading sentence-transformers model...")
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model

def normalize_text(s):
    """Normalize for similarity comparison."""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def msg_signature(msgs):
    """Create a hashable signature for a message set."""
    key = " | ".join(normalize_text(m) for m in msgs)
    return hashlib.md5(key.encode()).hexdigest()

def jaccard_similarity(a, b):
    """Jaccard similarity on word tokens."""
    wa = set(normalize_text(a).split())
    wb = set(normalize_text(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class SemanticDeduper:
    """Combined Jaccard + sentence-embedding deduplication."""

    def __init__(self, jaccard_threshold=0.72, semantic_threshold=0.80):
        self.jaccard_threshold = jaccard_threshold
        self.semantic_threshold = semantic_threshold
        self.seen_sigs = set()
        self.seen_key_texts = []   # normalized key message texts
        self.embeddings = []       # numpy arrays
        self.model = get_sbert_model()
        self.dup_count = 0

    def is_duplicate(self, msgs):
        """Check if sample is too similar to existing ones."""
        # 1. Exact hash check
        sig = msg_signature(msgs)
        if sig in self.seen_sigs:
            self.dup_count += 1
            return True

        # 2. Key message Jaccard check (fast pre-filter)
        key_text = normalize_text(" ".join(msgs[2:]))  # M3+M4+M5
        check_pool = self.seen_key_texts[-500:]
        for seen in check_pool:
            if jaccard_similarity(key_text, seen) > self.jaccard_threshold:
                self.dup_count += 1
                return True

        # 3. Semantic embedding check (catches paraphrase duplicates)
        if self.model is not None and len(self.embeddings) > 0:
            full_text = " ".join(msgs)
            emb = self.model.encode([full_text])[0]

            # Check against last 300 embeddings for speed
            check_embs = self.embeddings[-300:]
            if check_embs:
                sims = np.dot(check_embs, emb) / (
                    np.linalg.norm(check_embs, axis=1) * np.linalg.norm(emb) + 1e-8
                )
                if np.max(sims) > self.semantic_threshold:
                    self.dup_count += 1
                    return True
            self.embeddings.append(emb)
        elif self.model is not None:
            # First entry — just encode and store
            full_text = " ".join(msgs)
            emb = self.model.encode([full_text])[0]
            self.embeddings.append(emb)

        # Register as seen
        self.seen_sigs.add(sig)
        self.seen_key_texts.append(key_text)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION ENFORCER
# ═══════════════════════════════════════════════════════════════════════════════

def check_distribution(samples, cfg):
    """Check if current distribution matches config within tolerance."""
    n = len(samples)
    if n == 0:
        return False, {}
    type_counts = Counter(s["_type"] for s in samples)
    issues = {}
    for t, expected_frac in cfg["type_dist"].items():
        actual = type_counts.get(t, 0) / n
        if abs(actual - expected_frac) > cfg["tolerance"]:
            issues[f"type_{t}"] = (actual, expected_frac)
    non_ignore = [s for s in samples if s["_type"] != "ignore"]
    if non_ignore:
        ni = len(non_ignore)
        for tgt, expected_frac in cfg["target_dist"].items():
            actual = sum(1 for s in non_ignore if s["_target"] == tgt) / ni
            if abs(actual - expected_frac) > cfg["tolerance"] + 0.05:
                issues[f"target_{tgt}"] = (actual, expected_frac)
    return len(issues) == 0, issues


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sample(type_hint=None, lang_hint=None, force_adversarial=False, force_multi_signal=False):
    """Generate one sample with optional type/lang/mode hints."""
    lang = lang_hint or (random.choices(
        ["hinglish", "english"],
        weights=[CFG["lang_dist"]["hinglish"], CFG["lang_dist"]["english"]]
    )[0])

    if type_hint is None:
        content_roll = random.random()
        if content_roll < CFG["content_dist"]["real_chat"]:
            type_hint = random.choices(
                list(CFG["type_dist"].keys()),
                weights=list(CFG["type_dist"].values())
            )[0]
        elif content_roll < CFG["content_dist"]["real_chat"] + CFG["content_dist"]["spam_noise"]:
            type_hint = "ignore"
        else:
            type_hint = "ignore"

    # Multi-signal overrides normal building
    if force_multi_signal and type_hint != "ignore":
        msgs, dec = build_multi_signal_sample(lang)
    elif type_hint == "text":
        msgs, dec = build_text_sample(lang, adversarial=force_adversarial)
    elif type_hint == "media":
        msgs, dec = build_media_sample(lang, adversarial=force_adversarial)
    elif type_hint == "react":
        msgs, dec = build_react_sample(lang, adversarial=force_adversarial)
    elif type_hint == "acknowledge":
        msgs, dec = build_acknowledge_sample(lang, adversarial=force_adversarial)
    elif type_hint == "translate":
        msgs, dec = build_translate_sample(lang, adversarial=force_adversarial)
    else:
        msgs, dec = build_ignore_sample()
        lang = "mixed"

    # Parse decision fields
    parts = dict(p.split("=",1) for p in dec.replace("R: ","").split(" | "))
    target = parts.get("TARGET","null").replace("M","") if parts.get("TARGET","null") != "null" else None

    user_content = "\n".join(f"M{i+1}: {msgs[i]}" for i in range(5))

    return {
        "type": "chatml",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": dec},
        ],
        "_type": type_hint,
        "_target": f"M{target}" if target else "null",
        "_lang": lang,
        "_msgs": msgs,
    }


def main():
    target_n = CFG["total_samples"]
    samples = []
    deduper = SemanticDeduper(
        jaccard_threshold=CFG["dedup_threshold"],
        semantic_threshold=CFG["semantic_dedup_threshold"],
    )
    attempts = 0
    max_attempts = target_n * 15

    print(f"🚀 Generating {target_n} samples...")
    print(f"   Jaccard dedup threshold: {CFG['dedup_threshold']}")
    print(f"   Semantic dedup threshold: {CFG['semantic_dedup_threshold']}")
    print(f"   Adversarial rate: {TARGET_ADVERSARIAL_RATE}")
    print(f"   Multi-signal rate: {MULTI_SIGNAL_RATE}")
    print(f"   Max attempts: {max_attempts}")
    print()

    while len(samples) < target_n and attempts < max_attempts:
        attempts += 1

        # Distribution enforcement
        n_so_far = max(len(samples), 1)
        type_counts_now = Counter(s["_type"] for s in samples)

        over_cap_types = set()
        for t, frac in CFG["type_dist"].items():
            actual = type_counts_now.get(t, 0) / n_so_far
            if actual > frac + 0.02:
                over_cap_types.add(t)

        deficits = {}
        for t, frac in CFG["type_dist"].items():
            actual = type_counts_now.get(t, 0) / n_so_far
            deficits[t] = frac - actual
        most_needed = max(deficits, key=deficits.get)

        if deficits[most_needed] > 0.04:
            type_hint = most_needed
        elif over_cap_types:
            allowed = [t for t in CFG["type_dist"] if t not in over_cap_types]
            type_hint = random.choices(
                allowed,
                weights=[CFG["type_dist"][t] for t in allowed]
            )[0]
        else:
            type_hint = None

        # Decide if adversarial or multi-signal
        force_adversarial = random.random() < TARGET_ADVERSARIAL_RATE
        force_multi_signal = random.random() < MULTI_SIGNAL_RATE and not force_adversarial

        sample = generate_sample(
            type_hint=type_hint,
            force_adversarial=force_adversarial,
            force_multi_signal=force_multi_signal,
        )
        msgs = sample["_msgs"]

        # Skip if type over cap
        if sample["_type"] in over_cap_types and deficits.get(sample["_type"], 0) < -0.03:
            continue

        # Dedup check
        if deduper.is_duplicate(msgs):
            continue

        samples.append(sample)

        if len(samples) % 500 == 0:
            ok, issues = check_distribution(samples, CFG)
            status = "✅" if ok else "⚠️"
            print(f"  {status} {len(samples)}/{target_n} | dups rejected: {deduper.dup_count} | issues: {len(issues)}")
            if issues:
                for k,(a,e) in list(issues.items())[:4]:
                    print(f"      {k}: actual={a:.3f} expected={e:.3f}")

    # Final shuffle
    random.shuffle(samples)

    # Strip internal fields
    final = []
    for s in samples:
        final.append({
            "type": s["type"],
            "messages": s["messages"],
        })

    # Write output
    out = "dataset_v3.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Final stats
    print(f"\n{'='*60}")
    print(f"✅ Generated {len(final)} samples → {out}")
    print(f"   Total attempts: {attempts} | Dups rejected: {deduper.dup_count}")
    print(f"   Dedup rate: {deduper.dup_count/max(attempts,1)*100:.1f}%")

    type_c = Counter(s["_type"] for s in samples)
    target_c = Counter(s["_target"] for s in samples)
    lang_c = Counter(s["_lang"] for s in samples)

    print(f"\n=== TYPE (target vs actual) ===")
    for t, frac in sorted(CFG["type_dist"].items(), key=lambda x:-x[1]):
        actual = type_c.get(t,0)
        expected = int(frac * len(samples))
        diff = actual - expected
        mark = "✅" if abs(diff) < len(samples)*CFG["tolerance"] else "❌"
        print(f"  {mark} {t:12s}: {actual:5d} ({actual/len(samples)*100:.1f}%) | expected {expected} ({frac*100:.0f}%)")

    print(f"\n=== TARGET (non-ignore) ===")
    non_ig = [s for s in samples if s["_type"] != "ignore"]
    ni = len(non_ig)
    tgt_c = Counter(s["_target"] for s in non_ig)
    for tgt, frac in sorted(CFG["target_dist"].items(), key=lambda x:-x[1]):
        actual = tgt_c.get(tgt,0)
        mark = "✅" if abs(actual/ni - frac) < CFG["tolerance"]+0.05 else "❌"
        print(f"  {mark} {tgt}: {actual:5d} ({actual/ni*100:.1f}%) | expected ({frac*100:.0f}%)")

    print(f"\n=== LANGUAGE ===")
    for l, c in lang_c.most_common():
        print(f"  {l:12s}: {c:5d} ({c/len(samples)*100:.1f}%)")

    # Validation
    print(f"\n=== VALIDATION ===")
    errors = []
    for i, s in enumerate(samples):
        dec = s["messages"][2]["content"]
        parts = dict(p.split("=",1) for p in dec.replace("R: ","").split(" | "))
        t = parts.get("TYPE","")
        tgt = parts.get("TARGET","")
        eff = parts.get("EFFORT","")
        title = parts.get("TITLE","")
        if t in ["text","media"] and eff not in ["low","medium","high"]:
            errors.append(f"#{i} {t} bad effort: {eff}")
        if t == "ignore" and tgt != "null":
            errors.append(f"#{i} ignore target not null: {tgt}")
        if t not in ["text","media","react","acknowledge","translate","ignore"]:
            errors.append(f"#{i} invalid type: {t}")

    if errors:
        print(f"  ❌ {len(errors)} errors:")
        for e in errors[:10]: print(f"    {e}")
    else:
        print(f"  ✅ All {len(final)} samples valid")

    # Canonical title stats
    print(f"\n=== CANONICAL TITLE STATS ===")
    titles = []
    for s in samples:
        dec = s["messages"][2]["content"]
        title = dec.split("TITLE=")[1] if "TITLE=" in dec else "null"
        titles.append(title)
    tc = Counter(titles)
    canonical = [t for t in tc if ":" in t]
    print(f"  Canonical titles: {len(canonical)} unique")
    print(f"  Top 15:")
    for t,c in tc.most_common(15):
        print(f"    {t}: {c}")

    # Sample previews
    print(f"\n=== SAMPLE PREVIEWS ===")
    preview_indices = [0, 250, 500, 1000, 2000, 3000, 4000, len(samples)-1]
    for idx in preview_indices:
        if idx < len(samples):
            s = samples[idx]
            print(f"\n[{idx}] TYPE={s['_type']} TARGET={s['_target']} LANG={s['_lang']}")
            print(s["messages"][1]["content"])
            print(s["messages"][2]["content"])

    # Generate eval benchmark
    build_eval_benchmark()


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL BENCHMARK — 300 handcrafted test cases, 6 weakness categories × 50
# ═══════════════════════════════════════════════════════════════════════════════

def make_eval(msgs, expected_type, expected_target, expected_effort, title_contains):
    """Build one eval entry with expected fields."""
    user_content = "\n".join(f"M{i+1}: {msgs[i]}" for i in range(5))
    return {
        "input": {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": user_content},
            ]
        },
        "expected": {
            "TYPE":   expected_type,
            "TARGET": expected_target,
            "EFFORT": expected_effort,
            "TITLE_contains": title_contains,
        },
        "_category": title_contains,
    }


def build_eval_benchmark():
    random.seed(99)
    cases = []

    # ── CAT 1: Title drift — key msg in M3/M4, distractor in M1 (50 cases) ──
    title_drift_cases = [
        (["bhai langchain ke baare mein baat kar rahe the","haan","bhai python error samajh nahi aa raha","theek hai","hmm"], "text","M3","text:python"),
        (["yaar college admission tha topic","haan haan","bhai breakup ho gayi yaar help","okay","sure"], "text","M3","text:breakup"),
        (["crypto advice de raha tha","sahi tha","bhai flutter build fail ho raha","accha","lol"], "text","M3","text:flutter"),
        (["yaar trip plan kar rahe the","haan","bhai sql query error aa rahi hai yaar","hmm","ok"], "text","M3","text:database"),
        (["bhai gym ki baat kar rahe the","okay bhai","yaar crush se kaise baat karoon","sahi hai","lol"], "text","M3","text:crush"),
        (["yaar movie dekh rahe the","haan","bhai docker container nahi chal raha","theek","ok"], "text","M3","text:docker"),
        (["bhai startup idea pe baat kar rahe the","haan suno","bhai behen se bahut fight hoti hai","hmm","okay"], "text","M3","text:sibling"),
        (["yaar stock market discuss kar rahe the","haan","bhai typescript error aa raha hai help","lol","sure"], "text","M3","text:typescript"),
        (["bhai instagram ke baare mein tha","sahi hai","yaar anxiety bahut hoti hai help","accha","hmm"], "text","M3","text:anxiety"),
        (["yaar phone buying discuss kar rahe the","haan","bhai cors error block kar raha hai","okay","theek"], "text","M3","text:cors"),
        (["bhai pehle crypto baat ki","haan","accha","yaar redis cache kaam nahi kar raha","lol"], "text","M4","text:redis"),
        (["yaar trip pe gaye the","sahi","hmm","bhai react hooks mein issue hai help","okay"], "text","M4","text:react"),
        (["bhai mummy ki baat thi","accha","theek hai","yaar ex ne text kiya kya karoon","hmm"], "text","M4","text:ex"),
        (["yaar coding bootcamp discuss kar rahe the","haan","okay","bhai mongodb query empty aa rahi","okay lol"], "text","M4","text:mongodb"),
        (["bhai health tips the","sahi","hm","yaar best friend se fight ho gayi","accha"], "text","M4","text:friend"),
        (["yaar langchain model ke baare mein tha","hmm","okay","bhai github actions fail ho rahi hai","lol"], "text","M4","text:ci_cd"),
        (["bhai movie recommend kar raha tha","haan","sure","yaar meri back pain bahut hai help","hmm"], "text","M4","text:back_pain"),
        (["yaar exam ke baare mein baat ki","sahi","theek","bhai websocket disconnect ho raha hai","okay"], "text","M4","text:websocket"),
        (["bhai phone suggest kar raha tha","hmm","okay","yaar proposal karna hai ideas do","lol"], "text","M4","text:proposal"),
        (["yaar freelancing ki baat ki","haan","accha","bhai kubernetes pod crash ho raha hai","okay"], "text","M4","text:kubernetes"),
        (["bhai langchain pe tha convo","haan","aur bata","yaar nginx 502 error aa raha hai","lol bhai"], "text","M4","text:nginx"),
        (["yaar kal match tha","haan wow","sahi tha","bhai jwt token expire ho raha hai","okay"], "text","M4","text:jwt"),
        (["bhai bored ho raha tha","haan yaar","accha","yaar divorce ho rahi hai parents ki help","hmm"], "text","M4","text:divorce"),
        (["yaar langchain ke baare mein tha","haan","theek hai","bhai vim editor nahi samajh aa raha","okay lol"], "text","M4","text:vscode"),
        (["bhai fashion advice de raha tha","haan","okay","yaar stress bahut hai kaise kam karoon","sure"], "text","M4","text:stress"),
        (["bhai langchain mein expert hun","haan wahi toh","okay","hmm","yaar python import error aa rahi hai"], "text","M5","text:python"),
        (["yaar kal trip plan kar rahe the","haan","accha","okay","bhai firebase auth kaam nahi kar raha help"], "text","M5","text:firebase"),
        (["bhai stock market ki baat kar rahe the","haan","sahi","theek","yaar period pain bahut hai help"], "text","M5","text:period"),
        (["yaar langchain use karna tha","haan","hmm","okay","bhai graphql resolver error aa raha hai"], "text","M5","text:graphql"),
        (["bhai mummy ne bola tha kuch","haan","sure","lol","yaar toxic relationship mein hun help karo"], "text","M5","text:toxic"),
        (["yaar pizza kha rahe the","haha","lol","okay","bhai supabase auth nahi chal raha yaar"], "text","M5","text:supabase"),
        (["bhai coding bootcamp discuss kiya","haan","theek hai","accha","yaar one sided love hai kya karoon"], "text","M5","text:one_sided"),
        (["yaar langchain ki baat ki thi","okay","haan","sure","bhai elasticsearch query nahi chal rahi"], "text","M5","text:elasticsearch"),
        (["bhai ghar pe kuch hua tha","haan","okay","hmm","yaar marriage pressure de rahe hain ghar wale"], "text","M5","text:marriage"),
        (["yaar kal ka plan tha","haan","accha","theek","bhai docker compose network issue hai yaar"], "text","M5","text:compose"),
        (["bhai langchain","haan","bhai stripe payment error aa raha hai","ok","hmm"], "text","M3","text:stripe"),
        (["yaar travel","accha","bhai quit smoking karna hai tips do","haan","lol"], "text","M3","text:quit_smoking"),
        (["bhai cricket","haan","yaar hair fall bahut hai kya karoon","hmm","okay"], "text","M3","text:hair_fall"),
        (["yaar coding bootcamp","sahi","bhai prometheus metrics nahi aa rahi","theek","lol"], "text","M3","text:prometheus"),
        (["bhai diwali tha topic","haan","yaar weight gain karna hai advice do","okay","hmm"], "text","M3","text:weight_gain"),
        (["yaar langchain model","haan","bhai kafka consumer nahi chal raha","accha","lol"], "text","M4","text:kafka"),
        (["bhai mummy ki baat","okay","theek","yaar skin care routine kya karoon","hmm"], "text","M4","text:skin"),
        (["yaar food discuss kiya","haan","accha","bhai heroku deploy fail ho rahi hai","lol"], "text","M4","text:heroku"),
        (["bhai movie recommend kiya","okay","sahi","yaar loneliness feel ho rahi hai help","hmm"], "text","M4","text:loneliness"),
        (["yaar plan tha ghumne ka","haan","theek","bhai openai api rate limit aa rahi","lol okay"], "text","M4","text:openai"),
        (["bhai langchain","haan","hmm","lol","bhai fastapi startup nahi ho rahi"], "text","M5","text:fastapi"),
        (["yaar kal exam tha","haan","okay","hmm","bhai freelancing kaise shuru karoon guide do"], "text","M5","text:freelancing"),
        (["bhai game discuss kiya","haan","lol","okay","yaar allergy bahut aa rahi hai help karo"], "text","M5","text:allergy"),
        (["yaar trip ki baat","haan","accha","theek","bhai cloudflare worker error aa raha hai"], "text","M5","text:cloudflare"),
        (["bhai startup topic tha","haan","okay","hmm","yaar overprotective parents hain help karo"], "text","M5","text:overprotective"),
    ]
    for entry in title_drift_cases:
        msgs, typ, target, title_kw = entry
        cases.append(make_eval(msgs, typ, target, None, title_kw))

    # ── CAT 2: Emoji/laughter → IGNORE (50 cases) ──
    emoji_ignore_cases = [
        ["lol","haha","😂","💀","hehehehe"],
        ["😂","😂","😂","💀","😭"],
        ["haha","lol","lmao","hehe","xD"],
        ["💀","💀","💀","😭","lol"],
        ["hahahaha","lolll","😂😂","bruh","ded"],
        ["omg lol","haha","😭😭","💀","dying"],
        ["lmaooo","haha same","😂","true","lol"],
        ["bruh lol","haha fr","😭","💀","okay"],
        ["😂😂😂","💀💀","😭😭","lmao","dead"],
        ["xD","lmao","hahaha","😂","💀"],
        ["shjg","sfdd","vsdgsg","dgsg","gfgfhgf"],
        ["asdfgh","qwerty","zxcvbn","poiuyt","lkjhgf"],
        ["1234567","qweasd","zxcqwe","123qwe","asd123"],
        ["aaaa","bbbb","cccc","dddd","eeee"],
        ["fjdksla","sldkfj","qpwoei","rutyei","aldskf"],
        ["WIN FREE IPHONE NOW","CLICK THIS LINK","LIMITED OFFER","CLAIM TODAY","bit.ly/scam"],
        ["Congratulations you won","Click here to claim","Offer expires today","Dont miss out","Forward to 10"],
        ["FREE RECHARGE ALL USERS","Click link now","100% genuine","Only today","wa.me/fake"],
        ["EARN 10000 PER DAY","No investment needed","100% guaranteed","Join now","WhatsApp 9999"],
        ["Make money online easy","Work from home","No experience needed","Daily payment","Join free"],
        ["ok","ok","ok","ok","ok"],
        [".","..","...","....","😶"],
        ["test","test123","testing","yo","nvm"],
        ["bhai","bhai","bhai","bhai","bhai"],
        ["😂","haha","lol","bruh","💀"],
        ["😭😭😭","💀","omg","lmao","dead"],
        ["hehehe","hahahaha","xdddd","lolol","lmaoo"],
        ["oof","bruh","💀","😭","okay lol"],
        ["bhai lol","yaar haha","😂","💀","hehe okay"],
        ["ha","ha","ha","ha","ha"],
        ["???","???","???","???","???"],
        ["....","....","....","....","...."],
        ["k","k","k","k","k"],
        ["lol lol","haha haha","😂😂","💀💀","😭😭"],
        ["😂 😂 😂","💀 💀","lmao","haha","xd"],
        ["yaar lol","bhai haha","😂","💀","lmaooo"],
        ["Send this to 20 people","Good luck will come","Dont break the chain","Forward now","Must share"],
        ["Virus warning forward this","Your phone at risk","Share immediately","Protect yourself","Forward now"],
        ["Free Netflix subscription","Click link","Limited time offer","Enter details","Claim now"],
        ["You are our lucky winner","Lottery prize","Claim in 24 hours","Contact agent","Send details"],
        ["Investment opportunity","500% returns","Risk free","Join our group","Limited slots"],
        ["😂","😂","😂","😂","😂"],
        ["��","💀","💀","💀","💀"],
        ["haha","haha","haha","haha","haha"],
        ["lol","lol","lol","lol","lol"],
        ["🔥🔥🔥","💀💀","😂😂","lmao","ded"],
        ["bruh bruh bruh","lol lol","haha","💀","dead"],
        ["xD xD xD","lmaooo","hahaha","😂","okay lol"],
        ["asdfjkl","qwiopzx","mnbvcx","lkjhgf","poiuyt"],
        ["FREE PRIZE","CLAIM NOW","URGENT","LIMITED","bit.ly/abc"],
    ]
    for msgs in emoji_ignore_cases:
        if len(msgs) < 5:
            msgs = (msgs * 2)[:5]
        cases.append(make_eval(msgs[:5], "ignore", "null", "null", "null"))

    # ── CAT 3: Translate detection (50 cases) ──
    translate_cases = [
        (["bhai sun","haan","kuch mila","ye dekh","Muchas gracias por tu ayuda"], "translate","M5","translate:spanish"),
        (["yaar dekh","kya hai ye","kahin se aaya","samajh nahi","No entiendo nada de esto"], "translate","M5","translate:spanish"),
        (["bhai","ek message","translate kar do","please","ありがとうございました"], "translate","M5","translate:japanese"),
        (["yaar","foreign hai","kya likha hai","bata na","お疲れ様でした"], "translate","M5","translate:japanese"),
        (["bhai dekh","kaunsi lang","pata nahi","tu bata","감사합니다"], "translate","M5","translate:korean"),
        (["hey guys","got this","no idea","can someone","사랑해요"], "translate","M5","translate:korean"),
        (["bhai caption mein tha","kaunsi lang","pata nahi","tu bata","Merci beaucoup pour tout"], "translate","M5","translate:french"),
        (["so","my friend sent","think its french","what does it mean","Je ne comprends pas"], "translate","M5","translate:french"),
        (["bhai email mein tha","boss ka message","kya matlab","urgent hai","شكراً جزيلاً على مساعدتك"], "translate","M5","translate:arabic"),
        (["wait","got this email","from client","need translation","لا أفهم هذا الكلام"], "translate","M5","translate:arabic"),
        (["bro","caption pe tha","what language","help please","Vielen Dank fuer deine Hilfe"], "translate","M5","translate:german"),
        (["guys","saw on instagram","looks german","anyone know","Ich verstehe das wirklich nicht"], "translate","M5","translate:german"),
        (["bhai sun","haan","foreign script","pata nahi","آپ کا بہت شکریہ"], "translate","M5","translate:urdu"),
        (["yaar dekh","kya hai","script alag hai","bata","مجھے سمجھ نہیں آیا"], "translate","M5","translate:urdu"),
        (["bhai caption mein tha","kaunsi hai","south lang lagti","bata","மிக்க நன்றி உங்களுக்கு"], "translate","M5","translate:tamil"),
        (["hey","this was in bio","what script","anyone","எனக்கு புரியவில்லை"], "translate","M5","translate:tamil"),
        (["bhai sun","haan bol","kuch mila tha","Muchas gracias amigo","kya matlab hai"], "translate","M4","translate:spanish"),
        (["yaar dekh","haan","message aaya","ありがとうございました","samjha do"], "translate","M4","translate:japanese"),
        (["bhai","suno","forward hua","감사합니다","ye kya hai"], "translate","M4","translate:korean"),
        (["yaar","haan","email mein tha","Merci beaucoup pour tout","translate karo"], "translate","M4","translate:french"),
        (["sun","bol","caption tha","شكراً جزيلاً","matlab batao"], "translate","M4","translate:arabic"),
        (["bhai ye dekh","haan","kya hai","Vielen Dank fuer deine Hilfe","matlab bata"], "translate","M4","translate:german"),
        (["yaar","haan","script alag","آپ کا بہت شکریہ","kya likha"], "translate","M4","translate:urdu"),
        (["bhai dekh","haan","south lang","மிக்க நன்றி","translate karo"], "translate","M4","translate:tamil"),
        (["bhai sun","haan","Muchas gracias amigo","kya tha ye","samjhao"], "translate","M3","translate:spanish"),
        (["yaar dekh","haan","ありがとうございました","kya likha hai","translate karo"], "translate","M3","translate:japanese"),
        (["bhai","bol","감사합니다","ye kya hai","batao"], "translate","M3","translate:korean"),
        (["sun bhai","haan","Merci beaucoup","ye kaunsi language","bata"], "translate","M3","translate:french"),
        (["yaar","bol na","شكراً جزيلاً","ye kya hai","translate kar"], "translate","M3","translate:arabic"),
        (["bhai sun","haan","Vielen Dank","kya tha ye","batao"], "translate","M3","translate:german"),
        (["bhai sun","haan","kuch mila","ye dekh","Muito obrigado pela sua ajuda"], "translate","M5","translate:portuguese"),
        (["yaar","kya hai","nahi samjha","bata","Nao entendo nada do que esta acontecendo"], "translate","M5","translate:portuguese"),
        (["bhai sun","haan","kuch mila","ye dekh","यार कल परीक्षा है कुछ नहीं पढ़ा"], "translate","M5","translate:hindi"),
        (["yaar dekh","kya hai","pure hindi","bata","भाई बहुत थक गया हूं आज का दिन"], "translate","M5","translate:hindi"),
        (["ok","ok","ok","ok","감사합니다"], "translate","M5","translate:korean"),
        (["hm","hmm","hmm","hmm","ありがとうございました"], "translate","M5","translate:japanese"),
        (["bhai","yaar","okay","accha","Merci beaucoup"], "translate","M5","translate:french"),
        (["lol","haha","okay","sure","Muchas gracias"], "translate","M5","translate:spanish"),
        (["hey","so","btw","wait","Vielen Dank fuer"], "translate","M5","translate:german"),
        (["bhai sun na","haan bol yaar","theek hai","accha okay","شكراً جزيلاً على مساعدتك"], "translate","M5","translate:arabic"),
        (["yaar sun","okay","haan","hmm","آپ کا بہت شکریہ"], "translate","M5","translate:urdu"),
        (["hey guys","okay","sure","yeah","மிக்க நன்றி உங்களுக்கு"], "translate","M5","translate:tamil"),
        (["bhai","haan","theek","lol","यार कल परीक्षा है कुछ नहीं पढ़ा अभी तक"], "translate","M5","translate:hindi"),
        (["yaar","okay","hmm","accha","Por favor ayudame con esto"], "translate","M5","translate:spanish"),
        (["bhai sun","okay","haan","sure","정말 대단해요"], "translate","M5","translate:korean"),
        (["hey","so","wait","yeah","Je suis desole pour ca"], "translate","M5","translate:french"),
        (["bhai","yaar","lol","hmm","頑張ってください"], "translate","M5","translate:japanese"),
        (["okay","sure","accha","theek","Bitte hilf mir damit"], "translate","M5","translate:german"),
        (["bhai sun","haan","kya tha ye","kaunsi lang","Estoy muy emocionado"], "translate","M5","translate:spanish"),
        (["yaar","haan","kuch mila","caption tha","사랑해요 항상 곁에"], "translate","M5","translate:korean"),
    ]
    for entry in translate_cases:
        msgs, typ, target, title_kw = entry
        cases.append(make_eval(msgs, typ, target, "null", title_kw))

    # ── CAT 4: Acknowledge (50 cases) ──
    ack_cases = [
        (["bhai sun","haan","theek hai","bhai kaam ho gaya yaar","hmm"], "acknowledge","M4","done"),
        (["yaar dekh","okay","accha","reached home safely just now","lol"], "acknowledge","M4","safe"),
        (["bhai","haan","sure","bhai payment kar diya upi se","okay"], "acknowledge","M4","payment"),
        (["okay","hmm","lol","good morning bhai everyone gm","sure"], "acknowledge","M4","morning"),
        (["yaar","haan","theek","bhai bohot shukriya help ke liye","lol"], "acknowledge","M4","welcome"),
        (["bhai","okay","accha","just sent you the document yaar","hmm"], "acknowledge","M4","received"),
        (["haan","okay","sure","bhai so jaata hun good night gn","lol"], "acknowledge","M4","night"),
        (["yaar","theek","hmm","i am really sorry about what happened","okay"], "acknowledge","M4","okay"),
        (["bhai","haan","lol","bhai update sun progress ho gaya","sure"], "acknowledge","M4","update"),
        (["okay","hmm","accha","bhai tabiyat theek nahi hai fever hai","theek"], "acknowledge","M4","well"),
        (["bhai sun","haan","theek hai","accha","bhai kaam ho gaya assignment submit"], "acknowledge","M5","done"),
        (["yaar dekh","okay","haan","hmm","i reached home safely just got here"], "acknowledge","M5","safe"),
        (["bhai","haan","sure","lol","bhai payment kar diya upi transfer hua"], "acknowledge","M5","payment"),
        (["okay","hmm","lol","accha","good morning bhai subah subah gm everyone"], "acknowledge","M5","morning"),
        (["yaar","haan","theek","hmm","bhai bohot shukriya mera kaam kar diya"], "acknowledge","M5","welcome"),
        (["bhai","okay","accha","sure","just sent you the document check inbox"], "acknowledge","M5","received"),
        (["haan","okay","sure","lol","bhai so jaata hun good night everyone gn"], "acknowledge","M5","night"),
        (["yaar","theek","hmm","okay","i am really sorry for what happened yaar"], "acknowledge","M5","okay"),
        (["bhai","haan","lol","accha","bhai update sun project complete ho gaya"], "acknowledge","M5","done"),
        (["okay","hmm","accha","theek","bhai tabiyat theek nahi hai fever aa gaya"], "acknowledge","M5","well"),
        (["bhai sun","haan","bhai kaam ho gaya project done","theek hai","hmm"], "acknowledge","M3","done"),
        (["yaar","okay","reached home safely everyone","lol","sure"], "acknowledge","M3","safe"),
        (["bhai","haan","bhai payment kar diya upi se","accha","lol"], "acknowledge","M3","payment"),
        (["okay","hmm","good morning bhai gm subah","sure","haan"], "acknowledge","M3","morning"),
        (["yaar","theek","bhai bohot shukriya yaar tere bina nahi hota","lol","okay"], "acknowledge","M3","welcome"),
        (["bhai sun","haan","theek","bhai file share kar di check karo","lol"], "acknowledge","M4","file"),
        (["yaar","okay","accha","just finished the presentation nailed it","hmm"], "acknowledge","M4","done"),
        (["bhai","haan","sure","bhai ghar aa gaya safely trip mast rahi","lol"], "acknowledge","M4","safe"),
        (["okay","hmm","lol","thanks for being there always appreciate yaar","accha"], "acknowledge","M4","welcome"),
        (["yaar","theek","okay","bhai interview de ke aaya accha laga mujhe","hmm"], "acknowledge","M4","luck"),
        (["bhai sun","haan","theek","lol","submitted the project before deadline done"], "acknowledge","M5","done"),
        (["yaar","okay","haan","sure","bhai neend aa rahi hai kal milte hain gn"], "acknowledge","M5","night"),
        (["bhai","hmm","lol","accha","food is ready come eat everyone at table"], "acknowledge","M5","coming"),
        (["okay","theek","haan","hmm","bhai movie download ho gayi link bhej raha"], "acknowledge","M5","received"),
        (["yaar","okay","lol","sure","reached the hospital safely waiting outside"], "acknowledge","M5","safe"),
        (["bhai","haan","theek","sure","good morning everyone subah ho gayi gm"], "acknowledge","M5","morning"),
        (["yaar","okay","lol","hmm","good night all sleep well kal milte hain gn"], "acknowledge","M5","night"),
        (["bhai sun","haan","theek","lol","gm bhai gm sab subah uthke chai pi lo"], "acknowledge","M5","morning"),
        (["yaar","okay","haan","accha","gn sab sweet dreams kal milte hain bye"], "acknowledge","M5","night"),
        (["bhai","hmm","lol","sure","bhai aaj ghar aaya safely trip mast thi yaar"], "acknowledge","M5","safe"),
        (["yaar","okay","haan","theek","bhai paytm kar diya check karo na yaar"], "acknowledge","M5","payment"),
        (["bhai sun","haan","accha","lol","upi transfer done check karo please"], "acknowledge","M5","payment"),
        (["okay","hmm","sure","theek","bhai fees bhar di college ki confirm karo"], "acknowledge","M5","payment"),
        (["yaar","haan","lol","accha","rent transfer ho gaya bank se bhai"], "acknowledge","M5","payment"),
        (["bhai","okay","hmm","sure","just sent rent money to your account bro"], "acknowledge","M5","received"),
        (["yaar","okay","haan","lol","yaar thanks a lot tere bina ye nahi hota"], "acknowledge","M5","welcome"),
        (["bhai","hmm","sure","accha","bhai help ke liye bahut shukriya seriously"], "acknowledge","M5","welcome"),
        (["okay","haan","lol","theek","thank you so much for everything always there"], "acknowledge","M5","welcome"),
        (["yaar","okay","haan","sure","appreciate you always being there for me"], "acknowledge","M5","welcome"),
        (["bhai sun","haan","theek","hmm","bhai grateful hun tere liye always yaar"], "acknowledge","M5","welcome"),
    ]
    for entry in ack_cases:
        msgs, typ, target, title_kw = entry
        cases.append(make_eval(msgs, typ, target, "null", title_kw))

    # ── CAT 5: Media request (50 cases) ──
    media_cases = [
        (["bhai sun","haan","theek","bhai ek funny meme bhej yaar","hmm"], "media","M4"),
        (["yaar","okay","accha","send a cute dog gif please bro","lol"], "media","M4"),
        (["bhai","haan","sure","bhai sad sticker bhej feeling low hai","okay"], "media","M4"),
        (["okay","hmm","lol","yaar gaming rage gif chahiye bhai","accha"], "media","M4"),
        (["yaar","theek","haan","bhai birthday gif bhej celebration wala","lol"], "media","M4"),
        (["bhai","okay","sure","send shocked reaction gif please bro","hmm"], "media","M4"),
        (["haan","hmm","lol","bhai food craving ho rahi pizza gif bhej","okay"], "media","M4"),
        (["yaar","theek","okay","send a workout motivation gif bro please","accha"], "media","M4"),
        (["bhai","haan","sure","bhai wholesome hug sticker bhej yaar","lol"], "media","M4"),
        (["okay","hmm","accha","bhai diwali gif bhej festival mood hai","theek"], "media","M4"),
        (["bhai sun","haan","theek","accha","bhai ek funny meme bhej yaar please"], "media","M5"),
        (["yaar dekh","okay","haan","hmm","send a cute dog gif please bro now"], "media","M5"),
        (["bhai","haan","sure","lol","bhai sad sticker bhej feeling low hai yaar"], "media","M5"),
        (["okay","hmm","lol","accha","yaar gaming rage gif chahiye bhai bhej de"], "media","M5"),
        (["yaar","theek","haan","lol","bhai birthday gif bhej party mood hai"], "media","M5"),
        (["bhai","okay","sure","hmm","send shocked reaction gif please bro"], "media","M5"),
        (["haan","hmm","lol","okay","bhai pizza gif bhej food craving hai yaar"], "media","M5"),
        (["yaar","theek","okay","lol","yaar workout motivation gif bhej please bro"], "media","M5"),
        (["bhai","haan","sure","accha","send a wholesome hug sticker please bro"], "media","M5"),
        (["okay","hmm","accha","lol","bhai diwali lights gif bhej festival mood"], "media","M5"),
        (["bhai sun","haan","bhai ek funny meme bhej yaar","theek hai","hmm"], "media","M3"),
        (["yaar","okay","send a cute dog gif please","lol","sure"], "media","M3"),
        (["bhai","haan","bhai sad sticker bhej please","accha","lol"], "media","M3"),
        (["okay","hmm","yaar gaming rage gif chahiye bhai","sure","okay"], "media","M3"),
        (["yaar","theek","bhai birthday gif bhej celebration","lol","haan"], "media","M3"),
        (["bhai sun","haan","theek","koi cricket six moment gif bhej do na yaar","lol"], "media","M4"),
        (["yaar","okay","accha","bhai anime reaction gif bhej dramatic wala","hmm"], "media","M4"),
        (["bhai","haan","sure","send a dance gif bro mood accha hai","lol"], "media","M4"),
        (["okay","hmm","lol","bhai rain aesthetic gif bhej vibes wali yaar","okay"], "media","M4"),
        (["yaar","theek","haan","bhai sleepy panda gif bhej neend aa rahi","accha"], "media","M4"),
        (["bhai","okay","sure","send cooking gif something looks delicious bro","hmm"], "media","M4"),
        (["haan","hmm","lol","bhai superhero landing gif bhej cool wala yaar","okay"], "media","M4"),
        (["yaar","theek","okay","bhai baby laughing gif bhej cute wala please","lol"], "media","M4"),
        (["bhai","haan","sure","send a mind blown gif bro seriously yaar","hmm"], "media","M4"),
        (["okay","hmm","accha","bhai nature scenery gif bhej calming wala yaar","theek"], "media","M4"),
        (["bhai langchain ke baare mein tha","haan","theek","bhai ek funny meme bhej","lol"], "media","M4"),
        (["yaar trip plan kar rahe the","okay","accha","send a dog gif please bro","hmm"], "media","M4"),
        (["bhai coding discuss kiya","haan","sure","bhai birthday gif bhej please","lol"], "media","M4"),
        (["yaar exam ke baare mein tha","okay","theek","yaar gaming gif chahiye bhai","accha"], "media","M4"),
        (["bhai movie discuss kiya","haan","sure","bhai diwali gif bhej yaar please","lol"], "media","M4"),
        (["bhai sun","haan","theek","lol","bhai car drifting gif bhej speed wala"], "media","M5"),
        (["yaar","okay","haan","sure","bhai sunset beach gif bhej aesthetic yaar"], "media","M5"),
        (["bhai","hmm","lol","accha","send graduation celebration gif please bro"], "media","M5"),
        (["okay","theek","haan","hmm","bhai facepalm gif bhej frustrated wala yaar"], "media","M5"),
        (["yaar","okay","lol","sure","send a cat doing something funny gif please"], "media","M5"),
        (["bhai sun","haan","theek","lol","bhai cricket six gif bhej match wala yaar"], "media","M5"),
        (["yaar","okay","haan","accha","yaar anime reaction gif bhej dramatic wala"], "media","M5"),
        (["bhai","hmm","lol","sure","send a wholesome hug sticker for my friend"], "media","M5"),
        (["okay","theek","haan","lol","bhai rain vibes gif bhej cozy wala yaar"], "media","M5"),
        (["yaar","okay","sure","accha","bhai mind blown gif bhej serious moment hai"], "media","M5"),
    ]
    for entry in media_cases:
        msgs, typ, target = entry
        cases.append(make_eval(msgs, typ, target, None, ""))

    # ── CAT 6: React (50 cases) ──
    react_cases = [
        (["bhai sun","haan","theek","bhai result aa gaya marks acche aaye yaar","hmm"], "react","M4","🥳"),
        (["yaar","okay","accha","did you hear that celebrity got arrested","lol"], "react","M4","😲"),
        (["bhai","haan","sure","bhai rank up kar liya finally diamond mila","okay"], "react","M4","🔥"),
        (["okay","hmm","lol","aaj mera birthday hai 21 saal ka ho gaya","accha"], "react","M4","🎂"),
        (["yaar","theek","haan","this baby animal is absolutely adorable omg","lol"], "react","M4","🥰"),
        (["bhai","okay","sure","bhai jeet gaye hum tournament final jeet liya","hmm"], "react","M4","🏆"),
        (["haan","hmm","lol","this fail compilation is pure comedy gold fr","okay"], "react","M4","😂"),
        (["yaar","theek","okay","that show plot twist completely destroyed me","accha"], "react","M4","🤯"),
        (["bhai","haan","sure","bhai promotion mil gayi salary hike bhi aai","lol"], "react","M4","😎"),
        (["okay","hmm","accha","bhai pehli barish aayi monsoon shuru ho gaya","theek"], "react","M4","🌧️"),
        (["bhai sun","haan","theek","accha","bhai result aa gaya marks acche aaye yaar"], "react","M5","🥳"),
        (["yaar dekh","okay","haan","hmm","did you hear that celebrity got arrested"], "react","M5","😲"),
        (["bhai","haan","sure","lol","bhai rank up kar liya finally diamond mila"], "react","M5","🔥"),
        (["okay","hmm","lol","accha","aaj mera birthday hai 21 saal ka ho gaya"], "react","M5","🎂"),
        (["yaar","theek","haan","lol","this baby animal is absolutely adorable omg"], "react","M5","🥰"),
        (["bhai","okay","sure","hmm","bhai jeet gaye hum tournament final jeet liya"], "react","M5","🏆"),
        (["haan","hmm","lol","okay","this fail compilation is pure comedy gold fr"], "react","M5","😂"),
        (["yaar","theek","okay","lol","that show plot twist completely destroyed me"], "react","M5","🤯"),
        (["bhai","haan","sure","accha","bhai promotion mil gayi salary hike bhi aai"], "react","M5","😎"),
        (["okay","hmm","accha","lol","bhai pehli barish aayi monsoon shuru ho gaya"], "react","M5","🌧️"),
        (["bhai sun","haan","bhai result aa gaya marks acche aaye","theek hai","hmm"], "react","M3","🥳"),
        (["yaar","okay","did you hear that celebrity got arrested","lol","sure"], "react","M3","😲"),
        (["bhai","haan","bhai rank up kar liya finally diamond","accha","lol"], "react","M3","🔥"),
        (["okay","hmm","aaj mera birthday hai 21 saal ka","sure","okay"], "react","M3","🎂"),
        (["yaar","theek","this baby animal is absolutely adorable omg","lol","haan"], "react","M3","🥰"),
        (["bhai sun","haan","theek","just got into my dream college oh my god","lol"], "react","M4","🎉"),
        (["yaar","okay","accha","bhai aaj ka match insane tha last ball six","hmm"], "react","M4","⚡"),
        (["bhai","haan","sure","this song is hitting different at 3am ngl bro","lol"], "react","M4","🎵"),
        (["okay","hmm","lol","bhai gym mein personal record tod diya aaj yaar","accha"], "react","M4","💪"),
        (["yaar","theek","haan","this movie ending was absolutely not expected yaar","lol"], "react","M4","😱"),
        (["bhai","okay","sure","bhai boss ne publicly appreciate kiya aaj office","hmm"], "react","M4","🙌"),
        (["haan","hmm","lol","just finished my dissertation finally done yaar bro","okay"], "react","M4","😮‍💨"),
        (["yaar","theek","okay","bhai kal se chutti shuru ho gayi yaar finally","accha"], "react","M4","🌴"),
        (["bhai","haan","sure","just hit 1000 subscribers on youtube channel bro","lol"], "react","M4","📈"),
        (["okay","hmm","accha","bhai aaj itni garmi hai 42 degrees outside yaar","theek"], "react","M4","🥵"),
        (["bhai sun","haan","theek","lol","just got into my dream college oh my god"], "react","M5","🎉"),
        (["yaar","okay","haan","sure","bhai aaj ka match insane tha last ball six"], "react","M5","⚡"),
        (["bhai","hmm","lol","accha","this song is hitting different at 3am ngl bro"], "react","M5","🎵"),
        (["okay","theek","haan","hmm","bhai gym mein personal record tod diya aaj"], "react","M5","💪"),
        (["yaar","okay","lol","sure","this movie ending was absolutely not expected"], "react","M5","😱"),
        (["bhai sun","haan","theek","lol","bhai boss ne publicly appreciate kiya today"], "react","M5","🙌"),
        (["yaar","okay","haan","accha","just finished my dissertation finally done yaar"], "react","M5","😮‍💨"),
        (["bhai","hmm","lol","sure","bhai kal se chutti shuru ho gayi yaar finally"], "react","M5","🌴"),
        (["okay","theek","haan","lol","that meme format is absolutely perfect timing"], "react","M5","💀"),
        (["yaar","okay","sure","accha","bhai arranged marriage fixed ho gayi meri yaar"], "react","M5","😭"),
        (["bhai","haan","hmm","lol","just hit 1000 subscribers on youtube channel"], "react","M5","📈"),
        (["okay","sure","accha","theek","bhai aaj itni garmi hai 42 degrees outside"], "react","M5","🥵"),
        (["yaar","okay","haan","sure","first snow of the year outside my window bro"], "react","M5","❄️"),
        (["bhai sun","haan","theek","lol","bhai ghar pe gol gappa party hai aaj yaar"], "react","M5","😋"),
        (["yaar","okay","haan","accha","just saw a shooting star make a wish now"], "react","M5","⭐"),
    ]
    for entry in react_cases:
        msgs, typ, target, emoji_val = entry
        cases.append(make_eval(msgs, typ, target, "null", emoji_val))

    # ── Write benchmark ──
    random.shuffle(cases)
    bench_out = "eval_benchmark_300.jsonl"
    with open(bench_out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    cat_c = Counter(c["_category"] for c in cases)
    type_c = Counter(c["expected"]["TYPE"] for c in cases)

    print(f"\n{'='*60}")
    print(f"📋 Eval benchmark: {len(cases)} cases → {bench_out}")
    print(f"\n   TYPE breakdown:")
    for t, n in sorted(type_c.items(), key=lambda x:-x[1]):
        print(f"     {t:12s}: {n}")
    print(f"\n   Categories: {len(cat_c)} unique")


if __name__ == "__main__":
    main()

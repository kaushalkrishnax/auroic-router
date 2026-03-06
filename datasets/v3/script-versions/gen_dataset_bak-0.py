"""
Auroic Router Dataset Generator v4
====================================
5000 samples | Semantic deduper | Distribution enforcer | Quality loop
No thinking blocks — non-thinking SFT only for 0.6B router
"""

import json, random, re, hashlib
from collections import Counter, defaultdict
from difflib import SequenceMatcher

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION CONFIG — change these to control everything
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    "total_samples": 5000,

    # TYPE distribution — corrected per review
    "type_dist": {
        "text":        0.33,   # was 0.35, reduce to stop text-heavy bias
        "media":       0.16,   # +1%
        "react":       0.14,   # +2%
        "acknowledge": 0.15,
        "translate":   0.13,
        "ignore":      0.09,   # push harder, deduper kills these fast
    },

    # TARGET distribution (for non-ignore)
    "target_dist": {
        "M5": 0.52,
        "M4": 0.30,
        "M3": 0.14,
        "M2": 0.03,
        "M1": 0.01,
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
    "max_loops": 8,
    "tolerance": 0.035,
}

SYSTEM = "You are the Auroic Router. Given 5 chat messages, output exactly one routing decision."

# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY — massive expanded dictionaries
# ═══════════════════════════════════════════════════════════════════════════════

# Hinglish filler tokens — short realistic chat messages
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

# Hinglish conversation starters
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

# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC BANKS — 500+ distinct topics across categories
# ═══════════════════════════════════════════════════════════════════════════════

TEXT_TOPICS = {
    # CODING (80 topics)
    "coding": [
        ("python syntax error","bhai ye python error samajh nahi aa raha","python error fix"),
        ("javascript crash","js console is full of red errors","javascript debug help"),
        ("git merge conflict","bhai git conflict aa gaya resolve karo","git conflict resolve"),
        ("api not responding","my api is returning 404 nothing works","api error debug"),
        ("css layout broken","bhai css kaam nahi kar raha layout toot gaya","css layout fix"),
        ("sql query error","bhai sql query galat aa rahi hai data nahi","database query fix"),
        ("react component crash","my react component is not rendering anymore","react component fix"),
        ("docker wont start","docker container wont start keeps crashing","docker debug help"),
        ("npm install fail","bhai npm install fail ho raha error aa raha","npm error fix"),
        ("app not loading","bhai app load nahi ho raha screen blank hai","app loading fix"),
        ("typescript error","typescript type error is confusing me badly","typescript error fix"),
        ("flutter build fail","bhai flutter build fail ho raha yaar help","flutter build fix"),
        ("python import error","bhai python import error module not found","python import fix"),
        ("cors blocked","cors policy is blocking my api request yaar","cors error fix"),
        ("firebase auth fail","bhai firebase login kaam nahi kar raha help","firebase auth fix"),
        ("localhost down","bhai localhost nahi chal raha port busy hai","localhost server fix"),
        ("java null pointer","java nullpointer exception is crashing app","java error fix"),
        ("vscode broken","bhai vscode extension kaam nahi kar raha yaar","vscode issue fix"),
        ("regex not matching","my regex pattern is completely wrong help","regex pattern help"),
        ("heroku deploy fail","heroku deploy is failing build error again","heroku deployment fix"),
        ("nextjs routing","nextjs routing is not working pages missing","nextjs routing fix"),
        ("mongodb query","mongodb query is returning empty array always","mongodb query fix"),
        ("redis cache issue","redis cache is not storing data properly","redis cache fix"),
        ("websocket disconnect","websocket keeps disconnecting after seconds","websocket fix"),
        ("jwt token expired","jwt token expiry is too fast causing issues","jwt auth fix"),
        ("aws s3 upload","aws s3 upload is failing permissions error","aws s3 fix"),
        ("nginx 502 error","nginx giving 502 bad gateway on every request","nginx config fix"),
        ("kubernetes pod crash","kubernetes pod keeps crashing in loop","kubernetes fix"),
        ("graphql error","graphql query is throwing resolver error","graphql debug"),
        ("electron app crash","electron desktop app crashes on startup","electron fix"),
        ("python pandas error","pandas dataframe operation throwing error","pandas fix"),
        ("tensorflow import","tensorflow import failing gpu not detected","tensorflow setup"),
        ("django orm error","django orm query is not joining properly","django orm fix"),
        ("fastapi startup","fastapi server not starting up properly","fastapi fix"),
        ("tailwind not working","tailwind classes not applying in project","tailwind css fix"),
        ("webpack bundle error","webpack bundling is failing with error","webpack fix"),
        ("jest test failing","jest unit test is failing unexpectedly","jest test fix"),
        ("github actions fail","github actions workflow is failing on push","ci cd fix"),
        ("stripe payment error","stripe payment integration is throwing error","stripe fix"),
        ("socket io issue","socket io events not emitting properly","socketio fix"),
        ("vue component error","vue component lifecycle hook is broken","vue js fix"),
        ("angular module error","angular module is not loading properly","angular fix"),
        ("svelte reactive","svelte reactive statement not updating ui","svelte fix"),
        ("prisma migration","prisma database migration is failing today","prisma fix"),
        ("supabase auth","supabase authentication is not working today","supabase fix"),
        ("vercel deploy","vercel deployment is failing with build error","vercel fix"),
        ("cloudflare worker","cloudflare worker script throwing exception","cloudflare fix"),
        ("express middleware","express middleware is not executing properly","express fix"),
        ("apollo client","apollo client cache is not updating on mutation","apollo fix"),
        ("redux state","redux state is not updating after dispatch","redux fix"),
        ("zustand store","zustand store reset is not working properly","zustand fix"),
        ("chrome extension","chrome extension content script not loading","extension fix"),
        ("pwa service worker","pwa service worker is not caching properly","pwa fix"),
        ("three js render","threejs scene is not rendering on canvas","threejs fix"),
        ("d3 chart error","d3 chart is not updating with new data","d3 chart fix"),
        ("canvas api issue","html canvas drawing api is behaving oddly","canvas fix"),
        ("web assembly","webassembly module is not loading properly","wasm fix"),
        ("rust borrow error","rust borrow checker is throwing errors","rust fix"),
        ("golang goroutine","golang goroutine is causing race condition","golang fix"),
        ("swift ios crash","swift ios app crashes on specific device","swift ios fix"),
        ("kotlin android","kotlin android activity is not launching","android fix"),
        ("unity game object","unity game object is disappearing randomly","unity fix"),
        ("unreal blueprint","unreal engine blueprint not compiling today","unreal fix"),
        ("openai api error","openai api is returning rate limit error","openai api fix"),
        ("langchain chain","langchain chain is not calling tools properly","langchain fix"),
        ("huggingface model","huggingface model download is failing today","huggingface fix"),
        ("stable diffusion","stable diffusion out of memory error again","sd fix"),
        ("ollama model","ollama model is not responding to prompts","ollama fix"),
        ("linux command","linux command giving permission denied error","linux fix"),
        ("bash script error","bash script is throwing syntax error help","bash fix"),
        ("cron job not running","cron job is not executing at scheduled time","cron fix"),
        ("nginx ssl cert","nginx ssl certificate is expired need help","ssl cert fix"),
        ("docker compose network","docker compose containers not communicating","compose fix"),
        ("kafka consumer","kafka consumer group not receiving messages","kafka fix"),
        ("rabbitmq queue","rabbitmq queue is not processing messages","rabbitmq fix"),
        ("elasticsearch query","elasticsearch query not returning results","es fix"),
        ("redis pub sub","redis pubsub not delivering messages properly","redis pubsub fix"),
        ("prometheus metrics","prometheus not scraping metrics from app","prometheus fix"),
        ("grafana dashboard","grafana dashboard showing no data at all","grafana fix"),
    ],

    # STUDY (40 topics)
    "study": [
        ("exam panic","bhai kal exam hai kuch nahi pada help","last minute study help"),
        ("physics doubt","bhai ye physics formula samajh nahi aaya","physics concept help"),
        ("math calculus","calculus integration problem i cant solve","calculus help"),
        ("chemistry doubt","bhai organic chemistry reaction samajh nahi","chemistry help"),
        ("biology concept","bhai mitosis and meiosis difference kya hai","biology help"),
        ("history dates","i keep forgetting these history dates help","history study help"),
        ("english grammar","bhai english grammar rules confuse karte hain","english grammar help"),
        ("essay writing","how do i structure a good essay properly","essay writing help"),
        ("notes sharing","bhai notes bhej do please class miss ki thi","share study notes"),
        ("assignment urgent","bhai assignment aaj submit karni hai urgent","assignment help"),
        ("college admission","bhai college admission process kya hota hai","college admission advice"),
        ("entrance exam","bhai jee neet entrance exam prep kaise karein","entrance exam prep"),
        ("scholarship apply","how to apply for a scholarship abroad","scholarship advice"),
        ("thesis writing","i have no idea how to start my thesis","thesis writing help"),
        ("presentation tips","bhai presentation banani hai tips do","presentation tips"),
        ("group project","group project mein koi kaam nahi kar raha","group project help"),
        ("plagiarism issue","my essay has accidental plagiarism help","plagiarism fix"),
        ("citation format","how do i cite sources in apa format","citation help"),
        ("study schedule","bhai time table banana hai padhai ka","study schedule help"),
        ("concentration tips","i cant focus while studying at all","focus tips"),
        ("memory technique","how to memorize things faster for exam","memory techniques"),
        ("mock test","bhai mock test solve karna hai help karo","mock test guidance"),
        ("result calculation","bhai percentage calculate karna hai marks ki","marks calculation"),
        ("internship apply","bhai internship ke liye kaise apply karein","internship advice"),
        ("resume writing","mera resume bahut weak hai help karo","resume writing help"),
        ("interview prep","bhai interview ke liye kaise prepare karein","interview prep help"),
        ("gre preparation","gre exam prep strategy kya honi chahiye","gre prep advice"),
        ("ielts writing","bhai ielts writing task improve karna hai","ielts prep help"),
        ("online course","which online course is best for python","course recommendation"),
        ("book recommendation","bhai competitive exam ke liye best books","book recommendation"),
        ("class skip guilt","bhai class skip kar li ab kya karoon","class guilt help"),
        ("cheating temptation","exam mein cheating sahi hai kya honestly","exam ethics advice"),
        ("teacher conflict","bhai teacher se problem ho gayi class mein","teacher conflict"),
        ("hostel life","bhai hostel mein adjust karna mushkil hai","hostel advice"),
        ("study group","bhai study group banana chahta hun kaise","study group advice"),
        ("language learning","bhai english improve karna hai kaise karoon","language learning"),
        ("coding bootcamp","is a coding bootcamp worth the money","bootcamp advice"),
        ("semester backlog","bhai backlog clear karna hai kaise karoon","backlog help"),
        ("marks improvement","bhai marks kaise improve karoon please","marks improvement"),
        ("dropout thoughts","bhai college chhod dene ka mann kar raha","dropout advice"),
    ],

    # RELATIONSHIPS (40 topics)
    "relationships": [
        ("crush confession","bhai crush ko kya bolun kaise bataaun","crush advice"),
        ("breakup pain","just broke up and i feel completely terrible","breakup support"),
        ("friend fight","bhai best friend se jhagda ho gaya yaar","friendship conflict"),
        ("family pressure","bhai parents career pe pressure de rahe hain","family pressure advice"),
        ("toxic partner","my partner is being really toxic help me","toxic relationship advice"),
        ("long distance","bhai long distance relationship kaise chalaye","long distance advice"),
        ("rejection handling","she rejected me i dont know what to do","rejection support"),
        ("jealousy issue","bhai partner se jealous ho raha hun kyon","jealousy advice"),
        ("commitment fear","i am scared of committing to this relationship","commitment advice"),
        ("trust issue","bhai partner pe trust nahi raha kya karoon","trust issue advice"),
        ("ex texting","my ex suddenly started texting me again","ex advice"),
        ("friend zone","bhai yaar ne friend zone kar diya kya karoon","friendzone advice"),
        ("group drama","bhai friend group mein bahut drama ho raha","group drama advice"),
        ("sibling conflict","bhai behen se bahut fight hoti hai ghar mein","sibling conflict"),
        ("marriage pressure","bhai ghar wale shaadi ke liye bol rahe hain","marriage pressure"),
        ("online relationship","bhai online wale ko trust kar sakte hain","online relationship"),
        ("one sided love","bhai ek taraf se pyaar hai kya karoon","one sided love advice"),
        ("moving on","i need help moving on after this breakup","moving on advice"),
        ("friendship ending","bhai bahut purani dosti khatam ho rahi hai","friendship advice"),
        ("love vs career","bhai love ya career mein se kya choose karoon","love career choice"),
        ("cheating partner","i think my partner is cheating on me","cheating suspicion"),
        ("proposal planning","bhai propose karna hai ideas chahiye","proposal help"),
        ("first date tips","first date ke liye kya karoon tips do","first date tips"),
        ("apology needed","bhai kisi ko sorry bolna hai kaise karoon","apology advice"),
        ("boundary setting","how do i set boundaries with my partner","boundary setting"),
        ("communication issue","bhai partner se baat hi nahi hoti properly","communication advice"),
        ("overprotective parents","parents bahut overprotective hain help","parent advice"),
        ("cultural difference","different culture wale se love hai yaar","cultural relationship"),
        ("age gap relationship","bhai age difference waali relationship kaisi","age gap advice"),
        ("heartbreak recovery","i am completely heartbroken please help","heartbreak recovery"),
        ("social anxiety","bhai log se milne mein darr lagta hai","social anxiety help"),
        ("self confidence","bhai khud pe confidence nahi aata kaise banaoon","confidence advice"),
        ("loneliness","i have been feeling very lonely lately bro","loneliness support"),
        ("new city friends","bhai nayi jagah aaya hun dost kaise banaaoon","making friends"),
        ("workplace crush","bhai office mein colleague pe crush hai","workplace crush"),
        ("divorce parents","bhai parents ka divorce ho raha hai help","divorce support"),
        ("abusive situation","i think i am in an abusive situation help","abuse support"),
        ("controlling partner","my partner controls everything i do help","controlling partner"),
        ("ghosted by crush","bhai crush ne ghost kar diya kya karoon","ghosted advice"),
        ("secret relationship","bhai ghar se chhupaake relationship hai","secret relationship"),
    ],

    # CASUAL ADVICE (30 topics)
    "casual": [
        ("bored suggestion","bhai bohot bore ho raha hun kuch batao","activity suggestion"),
        ("trip planning","bhai trip plan karni hai suggestions do","trip planning help"),
        ("movie recommendation","bhai koi acchi movie suggest karo please","movie recommendation"),
        ("phone buying","bhai konsa phone lungu budget 15k hai","phone buying advice"),
        ("overthinking","i cant stop overthinking please help me","overthinking help"),
        ("career choice","bhai konsa career choose karoon confused hun","career advice"),
        ("startup idea","bhai mera startup idea kaisa lagta hai","startup feedback"),
        ("crypto investment","bhai crypto mein invest karoon kya advice","crypto advice"),
        ("gym motivation","bhai gym jaana chodh diya motivate karo","gym motivation"),
        ("diet plan","bhai weight loss ke liye kya khaoon suggest","diet plan help"),
        ("sleep problem","bhai raat ko neend nahi aati kya karoon","sleep problem help"),
        ("productivity tips","i keep procrastinating how to be productive","productivity advice"),
        ("money saving","bhai paise kaise bachaaoon student hun","money saving tips"),
        ("cooking help","bhai simple recipe batao khana banana hai","cooking recipe help"),
        ("fashion advice","bhai kya pehnu interview ke liye suggest karo","fashion advice"),
        ("haircut advice","bhai konsi haircut acchi lagegi suggest karo","haircut advice"),
        ("laptop buying","bhai konsa laptop lungu college ke liye","laptop advice"),
        ("bike recommendation","bhai konsi bike lungu 1 lakh mein suggest","bike advice"),
        ("pet advice","bhai pet lena chahta hun konsa accha hai","pet advice"),
        ("gift ideas","bhai girlfriend ko kya gift doon birthday pe","gift ideas"),
        ("time management","i have no time management skills help me","time management"),
        ("anxiety management","bhai anxiety bahut hoti hai kaise rokoon","anxiety management"),
        ("anger management","i get angry very quickly how to control","anger management"),
        ("digital detox","bhai phone band karna chahta hun tips do","digital detox tips"),
        ("new year goals","bhai new year resolution kaise achieve karoon","goal setting help"),
        ("freelancing start","bhai freelancing kaise shuru karoon guide do","freelancing guide"),
        ("stock market","bhai stock market mein kaise invest karoon","stock market advice"),
        ("youtube channel","bhai youtube channel start karna hai tips","youtube tips"),
        ("instagram growth","bhai instagram followers kaise badhaaoon","instagram growth"),
        ("public speaking","bhai public speaking se dar lagta hai help","public speaking"),
    ],

    # HEALTH (20 topics)
    "health": [
        ("headache relief","bhai sar dard ho raha hai kya karoon","headache remedies"),
        ("cold flu help","bhai cold aa gayi hai kya karoon gharelu upay","cold flu treatment"),
        ("back pain","my back pain is unbearable please suggest","back pain relief"),
        ("eye strain","bhai aankhein thak gayi hain screen se","eye strain relief"),
        ("fever management","bhai bukhar aa gaya hai kya karoon","fever management"),
        ("stress relief","i am very stressed how do i calm down","stress relief tips"),
        ("healthy eating","bhai kya khaoon healthy rehne ke liye","healthy eating tips"),
        ("exercise routine","bhai exercise routine banana hai guide do","exercise routine"),
        ("sleep hygiene","how to improve my sleep quality naturally","sleep hygiene tips"),
        ("mental health","bhai bahut burden feel ho raha hai help","mental health support"),
        ("food poisoning","bhai kuch galat khaa liya stomach upset hai","food poisoning help"),
        ("allergy issue","i keep having allergic reactions help me","allergy management"),
        ("hydration tips","bhai pani kam peeta hun kaise badhaaoon","hydration tips"),
        ("posture fix","my posture is really bad how to fix","posture correction"),
        ("immunity boost","bhai immunity weak hai kaise badhaaoon","immunity boost tips"),
        ("period cramps","period cramps are really bad help please","period pain relief"),
        ("skin problem","bhai face pe pimples bahut hain help karo","skin care help"),
        ("hair fall","bhai bahut hair fall ho raha hai kya karoon","hair fall remedy"),
        ("weight gain","bhai weight gain karna hai healthy tarike se","weight gain advice"),
        ("quit smoking","bhai smoking chodni hai tips do please","quit smoking tips"),
    ],

    # LIFE EVENTS — wins, milestones, problems (60 topics — entropy booster)
    "life_events": [
        ("internship selected","bhai internship mil gayi mujhe yaar finally","internship celebration"),
        ("job offer","yaar naukri ka offer aaya hai kya karoon","job offer advice"),
        ("promotion news","bhai promotion ho gayi office mein aaj","career milestone"),
        ("startup launch","bhai apna startup launch karne wala hun","startup launch advice"),
        ("first salary","yaar pehli salary aayi hai kya karoon","first salary advice"),
        ("bike bought","bhai finally bike le li apni dream wali","new bike share"),
        ("driving license","yaar driving license mil gaya mujhe finally","license milestone"),
        ("passport got","bhai passport ban gaya ab travel kar sakta","passport travel"),
        ("visa approved","yaar visa approve ho gaya abroad jaana hai","visa travel advice"),
        ("moved to new city","bhai naye sheher mein aa gaya hun adjust karna hai","new city life"),
        ("first apartment","yaar pehla apna ghar le liya feeling amazing","first apartment"),
        ("coding contest win","bhai hackathon jeet liya yaar kuch nahi socha tha","hackathon win"),
        ("app published","yaar mera app app store pe aa gaya finally","app launch"),
        ("youtube milestone","bhai 10k subscribers ho gaye yaar kaise hua","youtube milestone"),
        ("failed exam","bhai exam mein fail ho gaya kya karoon ab","exam failure"),
        ("lost job","yaar kaam se nikaala gaya kuch samajh nahi aa raha","job loss"),
        ("business failed","bhai business band karna pad raha hai","business failure"),
        ("accident happened","yaar hafka chhota accident ho gaya theek hun","minor accident"),
        ("phone lost","bhai phone kho gaya sab data chala gaya","lost phone"),
        ("wallet stolen","yaar wallet chheen liya kisi ne kya karoon","stolen wallet"),
        ("got scammed","bhai online scam ho gaya paise gaye","online scam"),
        ("cat died","yaar meri billi mar gayi bahut dukh ho raha","pet loss grief"),
        ("grandparent sick","bhai dada ji hospital mein hain please pray","family sick"),
        ("house shifting","yaar ghar shift karna hai next week tips do","house shifting"),
        ("weight loss achieved","bhai 10 kilo weight lose kar liya yaar finally","weight loss win"),
        ("marathon ran","yaar pehli baar 5k run kiya kuch nahi socha","first run"),
        ("guitar learned","bhai guitar seekh liya ek song baja sakta hun","skill achievement"),
        ("cooking improved","yaar khana banana seekh gaya biryani bhi","cooking skill"),
        ("reading habit","bhai 10 books padh li is saal","reading milestone"),
        ("meditation streak","yaar 30 din meditation complete kar li","wellness streak"),
        ("investment returns","bhai stocks mein invest kiya tha return aaya","investment news"),
        ("emi started","yaar pehli emi ka time aa gaya tension hai","emi advice"),
        ("insurance query","bhai health insurance lena chahta hun guide","insurance advice"),
        ("tax filing","yaar pehli baar ITR file karni hai help","tax filing help"),
        ("loan repaid","bhai bahut mehnatkari raha loan pay kar diya","loan cleared"),
        ("government job","yaar sarkari naukri ka form bhara hai","govt job advice"),
        ("civil services","bhai UPSC ki taiyari shuru karni hai guide","upsc prep"),
        ("coding job offer","yaar FAANG se offer aaya hai kya karun","big tech offer"),
        ("side hustle","bhai side income start karna chahta hun","side hustle advice"),
        ("content creator","yaar content creator banna chahta hun tips","content creator"),
        ("podcast start","bhai podcast shuru karna hai guide do","podcast advice"),
        ("blog writing","yaar blog likhna shuru karna chahta hun","blog advice"),
        ("photography hobby","bhai photography seekhna chahta hun","photography advice"),
        ("gaming setup","yaar gaming setup banana chahta hun budget hai","gaming setup"),
        ("stock loss","bhai stocks mein loss ho gaya kya karoon","stock loss advice"),
        ("crypto profit","yaar crypto se thoda profit hua kya karoon","crypto advice"),
        ("car plan","bhai car lene ka plan kar raha hun guide","car buying advice"),
        ("marriage planning","yaar shaadi plan karna shuru kiya tips","wedding planning"),
        ("baby announcement","bhai baby aa rahi hai ghar mein advice do","parenting advice"),
        ("health checkup","yaar pehla full body checkup karaya sab theek","health update"),
        ("therapy started","bhai therapy shuru ki hai bahut helpful","therapy update"),
        ("quit toxic job","yaar toxic job chhod di bahut better feel","quit job update"),
        ("gap year plan","bhai gap year lena chahta hun","gap year advice"),
        ("abroad study","yaar masters abroad karna chahta hun guide","study abroad"),
        ("language achieved","bhai japanese B1 level clear kar liya","language achievement"),
        ("sports selection","yaar state level selection ho gaya cricket mein","sports achievement"),
        ("ngo work","bhai NGO join kiya volunteer karna tha","volunteering"),
        ("blood donation","yaar pehli baar blood donate kiya feeling good","blood donation"),
        ("plant parent","bhai ghar mein plants lagaye hain tips do","plant care"),
        ("cooking channel","yaar cooking channel shuru karna chahta hun","cooking channel"),
    ],

    # WORK & CAREER (20 topics)
    "work": [
        ("office politics","bhai office mein politics ho rahi hai kya karoon","office politics"),
        ("salary hike ask","yaar boss se salary raise maangni hai tips","salary raise advice"),
        ("work life balance","bhai kaam aur ghar ka balance nahi ho raha","work life balance"),
        ("remote work tips","yaar ghar se kaam karte thak gaya hun tips","remote work advice"),
        ("toxic boss","bhai boss bahut toxic hai kya karoon","toxic boss advice"),
        ("colleague conflict","yaar colleague se problem ho gayi office mein","colleague conflict"),
        ("resignation plan","bhai resign karna chahta hun guidance do","resignation advice"),
        ("job switch","yaar job switch karna chahta hun timeline kya","job switch advice"),
        ("freelance rate","bhai freelancing mein kitna charge karoon","freelance pricing"),
        ("client issue","yaar client payment nahi de raha kya karoon","client problem"),
        ("appraisal tips","bhai appraisal ke liye kaise prepare karoon","appraisal advice"),
        ("interview call","yaar Google se interview call aaya hai help","interview help"),
        ("startup join","bhai startup join karoon ya badi company","startup vs big co"),
        ("wfh productivity","yaar ghar pe productive nahi hun tips do","wfh productivity"),
        ("career pivot","bhai career change karna chahta hun advice","career change"),
        ("networking tips","yaar linkedin pe networking kaise karoon","networking advice"),
        ("presentation skills","bhai office mein presentation deni hai help","presentation help"),
        ("email writing","yaar professional email likhna hai templates","email writing"),
        ("meeting anxiety","bhai bade meeting se darr lagta hai help","meeting anxiety"),
        ("layoff survivor","yaar batch mein sab cut ho gaye main bacha","layoff survivor"),
    ],
}

MEDIA_TOPICS = [
    # (key_msg, title, effort)
    ("bhai ek funny meme bhej yaar", "hilarious surprised face reaction", "high"),
    ("send a cute dog gif please bro", "excited puppy jumping around happy", "medium"),
    ("bhai sad sticker bhej yaar feeling low", "sad crying comfort hug warm", "low"),
    ("yaar gaming rage gif chahiye bhai", "intense gaming clutch moment", "medium"),
    ("bhai birthday gif bhej celebration wala", "confetti balloons birthday party", "medium"),
    ("send shocked reaction gif please", "jaw drop disbelief expression face", "low"),
    ("bhai food craving ho rahi hai pizza gif bhej", "cheesy pizza melting cheese pull", "medium"),
    ("yaar workout motivation gif bhej", "athlete training hard gym grind", "medium"),
    ("send a wholesome hug sticker bro", "warm comforting hug friends together", "low"),
    ("bhai diwali wala gif bhej festival mood", "colorful diwali lights fireworks celebration", "low"),
    ("send cat doing something funny gif", "mischievous cat knocking things over", "medium"),
    ("bhai cricket six moment gif chahiye", "cricket six massive hit stadium crowd", "high"),
    ("yaar anime reaction gif bhej dramatic", "anime character dramatic reaction face", "medium"),
    ("bhai dance gif bhej mood accha hai", "person dancing happily celebration move", "high"),
    ("send a facepalm gif please bruh", "person facepalm frustrated embarrassed moment", "low"),
    ("bhai rain aesthetic gif bhej vibes wali", "rain window cozy evening aesthetic", "low"),
    ("yaar sleeping gif bhej neend aa rahi", "sleepy panda dozing off tired adorable", "low"),
    ("send cooking gif something looks delicious", "chef cooking tossing pan flame", "medium"),
    ("bhai superhero landing gif bhej cool wala", "superhero dramatic landing epic pose", "high"),
    ("yaar baby laughing gif bhej cute wala", "baby giggling laughing cute adorable", "medium"),
    ("send a mind blown gif bro seriously", "mind explosion shock realization moment", "low"),
    ("bhai nature scenery gif bhej calming", "mountain sunrise peaceful nature calm", "low"),
    ("yaar car drifting gif bhej speed wala", "car drifting sharp turn smoke trail", "high"),
    ("bhai sunset beach gif bhej aesthetic", "sunset beach waves orange sky vibe", "low"),
    ("send graduation celebration gif please", "graduation cap toss confetti proud moment", "medium"),
]

REACT_TOPICS = [
    # (key_msg, emoji)
    ("bhai result aa gaya marks acche aaye yaar", "🥳"),
    ("did you hear that celebrity just got arrested", "😲"),
    ("bhai rank up kar liya finally diamond mila", "🔥"),
    ("aaj mera birthday hai 21 saal ka ho gaya", "🎂"),
    ("this baby animal is absolutely adorable omg", "🥰"),
    ("bhai jeet gaye hum tournament final jeet liya", "🏆"),
    ("this fail compilation is pure comedy gold fr", "😂"),
    ("that show plot twist completely destroyed me", "🤯"),
    ("bhai promotion mil gayi salary hike bhi aai", "😎"),
    ("bhai pehli barish aayi monsoon shuru ho gaya", "🌧️"),
    ("just got into my dream college oh my god", "🎉"),
    ("bhai aaj ka match insane tha last ball six", "⚡"),
    ("this song is hitting different at 3am ngl", "🎵"),
    ("bhai gym mein personal record tod diya aaj", "💪"),
    ("this movie ending was absolutely not expected", "😱"),
    ("bhai boss ne publicly appreciate kiya aaj", "🙌"),
    ("just finished my dissertation finally done", "😮‍💨"),
    ("bhai kal se chutti shuru ho gayi yaar finally", "🌴"),
    ("that meme format is absolutely perfect timing", "💀"),
    ("bhai arranged marriage fixed ho gayi meri yaar", "😭"),
    ("just hit 1000 subscribers on youtube channel", "📈"),
    ("bhai aaj itni garmi hai 42 degrees outside", "🥵"),
    ("first snow of the year outside my window", "❄️"),
    ("bhai ghar pe gol gappa party hai aaj yaar", "😋"),
    ("just saw a shooting star make a wish now", "⭐"),
]

ACKNOWLEDGE_TOPICS = [
    # (key_msg, ack_reply_pool)
    ("bhai kaam ho gaya assignment submit kar diya", ["noted done", "well done bhai", "great job", "nicely done", "proud of you"]),
    ("i reached home safely just got here now", ["glad youre safe", "safe travels", "take rest now", "good to know", "relieved"]),
    ("bhai payment kar diya upi se transfer hua", ["payment received", "confirmed got it", "noted thanks", "received", "all good"]),
    ("good morning bhai subah subah gm everyone", ["good morning", "gm bhai", "rise and shine", "morning vibes", "gm to you too"]),
    ("bhai bohot shukriya mera kaam kar diya tune", ["youre welcome", "anytime yaar", "happy to help", "no problem bhai", "always here"]),
    ("just sent you the document check your email", ["got it thanks", "received noted", "got the file", "thank you", "acknowledged"]),
    ("bhai so jaata hun good night everyone gn", ["good night", "sleep well bhai", "gn sweet dreams", "rest well", "gn take care"]),
    ("i am really sorry about what happened yaar", ["its okay bhai", "apology accepted", "all good now", "no worries", "we are good"]),
    ("bhai update sun progress update de raha hun", ["got the update", "noted thanks", "understood bhai", "got it", "thanks for update"]),
    ("bhai tabiyat theek nahi hai fever hai mujhe", ["get well soon", "rest up bhai", "take care yaar", "feel better soon", "rest karo"]),
    ("reached the hospital safely waiting outside", ["okay stay strong", "good to know", "keep us updated", "take care", "with you always"]),
    ("bhai file share kar di check karo please", ["got the file", "received thanks", "checking now", "noted", "got it bhai"]),
    ("just finished the presentation nailed it hopefully", ["fingers crossed", "you did great", "all the best", "proud of you", "hope it went well"]),
    ("bhai ghar aa gaya safely trip mast rahi", ["glad youre safe", "rest kar ab", "welcome back", "happy to hear", "take rest"]),
    ("thanks for being there always appreciate it bro", ["always here for you", "anytime bro", "that is what friends are for", "always", "no problem"]),
    ("bhai interview de ke aaya accha laga mujhe", ["fingers crossed bhai", "inshallah ho jayega", "all the best", "hope it went well", "proud of you"]),
    ("submitted the project before deadline finally done", ["great job bhai", "well done", "proud of you", "nicely done", "you did it"]),
    ("bhai neend aa rahi hai kal milte hain gn", ["gn bhai", "kal milte hain", "sweet dreams", "good night", "rest well"]),
    ("food is ready come eat everyone at table", ["coming now", "on my way", "be right there", "just a min", "coming"]),
    ("bhai movie download ho gayi link bhej raha", ["got it thanks", "received the link", "downloading now", "thanks bhai", "got it"]),
]

TRANSLATE_TOPICS = [
    # (foreign_text, title)
    # Spanish
    ("Muchas gracias por tu ayuda amigo de verdad", "translate spanish message"),
    ("No entiendo nada de lo que dijiste ahora", "translate spanish message"),
    ("Por favor ayudame con esto es urgente hoy", "translate spanish message"),
    ("Estoy muy emocionado por lo que paso hoy", "translate spanish message"),
    ("No se que hacer en esta situacion ahora mismo", "translate spanish message"),
    ("Me alegra mucho verte despues de tanto tiempo", "translate spanish message"),
    ("Esto es absolutamente increible no puedo creer", "translate spanish message"),
    ("Te quiero mucho amigo siempre estaras en mi corazon", "translate spanish message"),
    ("Feliz cumpleanos espero que tengas un dia maravilloso", "translate spanish message"),
    ("El examen fue muy dificil pero creo que lo pase", "translate spanish message"),
    # Japanese
    ("ありがとうございました本当に助かりました", "translate japanese text"),
    ("お疲れ様でした今日もよく頑張りましたね", "translate japanese text"),
    ("よろしくお願いしますこれからもよろしく", "translate japanese text"),
    ("すみません少し助けていただけますか", "translate japanese text"),
    ("わかりました了解です問題ありません", "translate japanese text"),
    ("頑張ってください応援していますよ", "translate japanese text"),
    ("おはようございます今日もよい一日を", "translate japanese text"),
    ("おやすみなさいゆっくり休んでください", "translate japanese text"),
    ("本当に嬉しいです信じられないくらいです", "translate japanese text"),
    ("また会いましょうそれまでお元気で", "translate japanese text"),
    # Korean
    ("감사합니다 정말 도움이 많이 됐어요", "translate korean message"),
    ("안녕하세요 처음 뵙겠습니다 잘 부탁드려요", "translate korean message"),
    ("사랑해요 항상 곁에 있어줘서 고마워요", "translate korean message"),
    ("미안해요 제가 잘못했어요 용서해 주세요", "translate korean message"),
    ("정말 대단해요 어떻게 그렇게 잘 하세요", "translate korean message"),
    ("화이팅 할 수 있어요 믿어요", "translate korean message"),
    ("배고파요 뭔가 맛있는 거 먹고 싶어요", "translate korean message"),
    ("오늘 정말 힘든 하루였어요 지쳐버렸어요", "translate korean message"),
    # French
    ("Merci beaucoup pour tout ce que tu as fait", "translate french message"),
    ("Je ne comprends pas du tout ce que tu veux dire", "translate french message"),
    ("Sil vous plait aidez moi cest vraiment urgent", "translate french message"),
    ("Cest tres interessant je naurais pas pense a ca", "translate french message"),
    ("Je suis desole pour ce qui sest passe hier", "translate french message"),
    ("Bonne chance pour demain je suis avec toi", "translate french message"),
    ("Comment ca va depuis la derniere fois quon sest vus", "translate french message"),
    ("Au revoir et a bientot prends soin de toi", "translate french message"),
    # Arabic
    ("شكراً جزيلاً على مساعدتك لي اليوم", "translate arabic message"),
    ("لا أفهم هذا الكلام أبداً وضح لي من فضلك", "translate arabic message"),
    ("من فضلك ساعدني في هذا الأمر المهم", "translate arabic message"),
    ("أنا سعيد جداً بما حدث اليوم الحمد لله", "translate arabic message"),
    ("هذا رائع جداً لم أكن أتوقع هذا أبداً", "translate arabic message"),
    ("كيف حالك اليوم أتمنى أن تكون بخير دائماً", "translate arabic message"),
    ("مع السلامة وإلى اللقاء قريباً إن شاء الله", "translate arabic message"),
    # German
    ("Vielen Dank fuer deine Hilfe das war sehr nett", "translate german message"),
    ("Ich verstehe das wirklich nicht erklaer mir das bitte", "translate german message"),
    ("Bitte hilf mir dabei ich schaffe es alleine nicht", "translate german message"),
    ("Das ist wirklich sehr interessant hab ich nicht gewusst", "translate german message"),
    ("Es tut mir sehr leid fuer das was passiert ist", "translate german message"),
    ("Wie geht es dir ich hoffe du bist gesund heute", "translate german message"),
    # Portuguese
    ("Muito obrigado pela sua ajuda foi fundamental hoje", "translate portuguese message"),
    ("Nao entendo nada do que esta acontecendo aqui", "translate portuguese message"),
    ("Por favor me ajuda com isso eh muito urgente", "translate portuguese message"),
    ("Estou muito feliz com tudo que aconteceu hoje", "translate portuguese message"),
    # Urdu
    ("آپ کا بہت بہت شکریہ آپ نے بہت مدد کی", "translate urdu message"),
    ("مجھے سمجھ نہیں آیا ذرا سمجھائیں مہربانی", "translate urdu message"),
    ("برائے مہربانی میری مدد کریں بہت ضروری ہے", "translate urdu message"),
    ("یہ بہت اچھا ہے واقعی بہت خوشی ہوئی آج", "translate urdu message"),
    # Tamil
    ("மிக்க நன்றி உங்கள் உதவிக்கு மிகவும் நன்றி", "translate tamil message"),
    ("எனக்கு புரியவில்லை கொஞ்சம் விளக்கி சொல்லுங்கள்", "translate tamil message"),
    ("தயவுசெய்து உதவுங்கள் மிகவும் முக்கியமான விஷயம்", "translate tamil message"),
    # Hindi (hinglish contexts where someone pastes pure hindi)
    ("यार कल परीक्षा है कुछ नहीं पढ़ा अभी तक", "translate hindi text"),
    ("भाई बहुत थक गया हूं आज का दिन बुरा था", "translate hindi text"),
    ("मुझे समझ नहीं आ रहा क्या करूं इस बारे में", "translate hindi text"),
]

IGNORE_TOPICS = [
    # Pure noise chains
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
    # Random keyboard spam
    ["shjg","sfdd","vsdgsg","dgsg","gfgfhgf"],
    ["asdfgh","qwerty","zxcvbn","poiuyt","lkjhgf"],
    ["1234567","qweasd","zxcqwe","123qwe","asd123"],
    ["aaaaaaa","bbbbbbb","ccccccc","ddddddd","eeeeeee"],
    ["fjdksla","sldkfj","qpwoei","rutyei","aldskf"],
    ["mnbvcx","lkjhgf","poiuyt","qweasd","zxcvbn"],
    # Spam messages
    ["WIN FREE IPHONE NOW", "CLICK THIS LINK", "LIMITED OFFER", "CLAIM TODAY", "bit.ly/scam"],
    ["Congratulations you won", "Click here to claim", "Offer expires today", "Dont miss out", "Forward to 10"],
    ["FREE RECHARGE ALL USERS", "Click link now", "100% genuine", "Only today", "wa.me/fake"],
    ["You have been selected", "God bless you", "Send bank details", "Claim prize now", "Forward for luck"],
    ["EARN 10000 PER DAY", "No investment needed", "100% guaranteed", "Join now", "WhatsApp 9999"],
    ["Beta testing invite", "Click to join", "Free premium access", "Limited spots", "Refer and earn"],
    ["Your account will be closed", "Verify now", "Click this link", "Urgent action needed", "bit.ly/verify"],
    ["Free Amazon gift card", "Survey takes 1 min", "Click here", "Guaranteed reward", "Claim now"],
    ["Investment opportunity", "500% returns", "Risk free", "Join our group", "Limited slots"],
    ["Make money online easy", "Work from home", "No experience needed", "Daily payment", "Join free"],
    ["Send this to 20 people", "Good luck will come", "Dont break the chain", "Forward now", "Must share"],
    ["Breaking news share this", "Forward to all groups", "Must read urgent", "Share before deleted", "Copy paste now"],
    ["Free Netflix subscription", "Click link", "Limited time offer", "Enter details", "Claim now"],
    ["Virus warning forward this", "Your phone at risk", "Share immediately", "Protect yourself", "Forward now"],
    ["You are our lucky winner", "Lottery prize", "Claim in 24 hours", "Contact agent", "Send details"],
    # Unrelated/off-topic
    ["ok","ok","ok","ok","ok"],
    [".","..","...","....","😶"],
    ["test","test123","testing","yo","nvm"],
    ["","hello","hi","hey",""],
    ["bhai","bhai","bhai","bhai","bhai"],
]

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDERS — varied realistic filler messages
# ═══════════════════════════════════════════════════════════════════════════════

def hi_context():
    """Generate a realistic hinglish filler message."""
    patterns = [
        lambda: random.choice(HI_FILLERS),
        lambda: f"{random.choice(HI_STARTERS)} {random.choice(['kya chal raha','kuch batao','sun na','dekh na','bol na'])}",
        lambda: random.choice(EMOJI_FILLERS),
        lambda: f"{random.choice(['haan','okay','accha'])} {random.choice(['bhai','yaar','bro'])}",
        lambda: random.choice(["kya hua","kya scene","bata na","phir kya","sach mein"]),
        lambda: f"{random.choice(EMOJI_FILLERS)} {random.choice(EMOJI_FILLERS)}",
        lambda: random.choice(["samajh gaya","noted","got it","theek hai yaar","chal dekho"]),
    ]
    return random.choice(patterns)()

def en_context():
    """Generate a realistic english filler message."""
    patterns = [
        lambda: random.choice(EN_FILLERS),
        lambda: f"{random.choice(EN_STARTERS)} {random.choice(['whats up','tell me more','go on','okay','really'])}",
        lambda: random.choice(EMOJI_FILLERS),
        lambda: f"{random.choice(['yeah','okay','sure','ngl','lowkey'])} {random.choice(['makes sense','fair enough','true','got it','interesting'])}",
        lambda: random.choice(["wait what","no way","oh really","thats wild","seriously tho"]),
    ]
    return random.choice(patterns)()

def make_filler(lang="hinglish"):
    return hi_context() if lang == "hinglish" else en_context()

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_msgs_with_key(key_msg, target_pos, lang="hinglish", noise_after=True):
    """
    Build 5 messages placing key_msg at target_pos (1-indexed).
    Context before: filler (realistic conversation lead-up)
    Context after: filler (reactions, short replies)
    noise_after: if True, messages after key are short/dismissive (more realistic)
    """
    msgs = []
    idx = target_pos - 1

    for i in range(5):
        if i < idx:
            # Before key: lead-up conversation
            msgs.append(make_filler(lang))
        elif i == idx:
            msgs.append(key_msg)
        else:
            # After key: short reaction (makes model learn key is not at end)
            if noise_after:
                short_pool = HI_FILLERS[:20] if lang == "hinglish" else EN_FILLERS[:20]
                msgs.append(random.choice(short_pool + list(EMOJI_FILLERS[:10])))
            else:
                msgs.append(make_filler(lang))

    return msgs

def pick_target_pos():
    r = random.random()
    if r < 0.52: return 5    # M5
    elif r < 0.82: return 4  # M4
    elif r < 0.96: return 3  # M3
    elif r < 0.99: return 2  # M2
    else: return 1            # M1

def make_decision(typ, target, effort, title):
    t = f"M{target}" if target else "null"
    return f"R: TYPE={typ} | TARGET={t} | EFFORT={effort} | TITLE={title}"


# Title variant pools — multiple phrasings per intent to prevent memorization
TITLE_VARIANTS = {
    "python error fix":         ["python error fix","fix python bug","python crash help","debug python code","python issue resolve"],
    "javascript debug help":    ["javascript debug help","js crash fix","debug js error","javascript issue","fix js bug"],
    "git conflict resolve":     ["git conflict resolve","fix git merge","git branch issue","resolve git conflict","git merge fix"],
    "api error debug":          ["api error debug","fix api call","api not working","debug api issue","api response fix"],
    "css layout fix":           ["css layout fix","fix css styling","css not working","layout repair","css responsive fix"],
    "database query fix":       ["database query fix","fix sql query","db query error","sql issue resolve","database error fix"],
    "react component fix":      ["react component fix","fix react bug","react crash resolve","component render fix","react issue"],
    "docker debug help":        ["docker debug help","fix docker container","docker not starting","container issue fix","docker error"],
    "npm error fix":            ["npm error fix","fix node modules","npm install issue","package error fix","npm crash fix"],
    "app loading fix":          ["app loading fix","app not opening","fix app crash","app startup fix","app blank screen fix"],
    "typescript error fix":     ["typescript error fix","ts type error","fix typescript","typescript crash","ts compile fix"],
    "flutter build fix":        ["flutter build fix","flutter error help","dart build issue","flutter crash fix","flutter debug"],
    "python import fix":        ["python import fix","module not found fix","pip install help","python dependency fix","import error"],
    "cors error fix":           ["cors error fix","fix cors policy","cors blocked fix","cors header issue","cors origin fix"],
    "firebase auth fix":        ["firebase auth fix","firebase login fix","firebase error help","firebase crash","auth issue fix"],
    "localhost server fix":     ["localhost server fix","local server down","port busy fix","dev server issue","localhost fix"],
    "java error fix":           ["java error fix","fix java exception","java crash help","java build fix","java debug"],
    "vscode issue fix":         ["vscode issue fix","fix vscode extension","editor not working","vscode crash","ide fix"],
    "regex pattern help":       ["regex pattern help","fix regex match","regex not working","write regex help","pattern fix"],
    "heroku deployment fix":    ["heroku deployment fix","fix heroku deploy","heroku build fail","heroku error","deploy fix"],
    "last minute study help":   ["last minute study help","exam prep urgent","quick revision tips","panic study guide","exam night help"],
    "physics concept help":     ["physics concept help","physics formula help","physics doubt clear","physics explain","physics doubt"],
    "calculus help":            ["calculus help","integration help","differentiation doubt","maths calculus","calc problem"],
    "chemistry help":           ["chemistry help","organic chemistry doubt","chem reaction help","chemistry concept","chem doubt"],
    "biology help":             ["biology help","bio concept doubt","biology explain","bio chapter help","life science doubt"],
    "history study help":       ["history study help","history dates help","memorize history","history revision","history facts"],
    "english grammar help":     ["english grammar help","grammar doubt clear","english writing help","grammar rules","english help"],
    "essay writing help":       ["essay writing help","how to write essay","essay structure help","essay tips","writing guide"],
    "share study notes":        ["share study notes","notes request","send class notes","study material share","notes please"],
    "assignment help":          ["assignment help","deadline assignment","urgent homework","assignment submit","project help"],
    "college admission advice": ["college admission advice","college apply help","admission process","college cutoff help","admission guide"],
    "entrance exam prep":       ["entrance exam prep","jee neet prep","competitive exam help","entrance strategy","exam prep guide"],
    "scholarship advice":       ["scholarship advice","financial aid help","scholarship apply","college funding","scholarship guide"],
    "thesis writing help":      ["thesis writing help","dissertation help","research paper guide","thesis structure","academic writing"],
    "presentation tips":        ["presentation tips","how to present","slides making help","presentation guide","speak present"],
    "group project help":       ["group project help","team assignment issue","group work problem","project team help","collab issue"],
    "plagiarism fix":           ["plagiarism fix","accidental copy fix","paraphrase help","originality check","citation rewrite"],
    "citation help":            ["citation help","apa format guide","reference format","how to cite","bibliography help"],
    "study schedule help":      ["study schedule help","timetable making","study plan create","schedule organize","study routine"],
    "focus tips":               ["focus tips","concentration help","study focus","distraction free study","attention improve"],
    "memory techniques":        ["memory techniques","memorize faster","recall techniques","memory tips","learning strategy"],
    "mock test guidance":       ["mock test guidance","practice test help","mock exam tips","test strategy","practice guidance"],
    "marks calculation":        ["marks calculation","percentage calc","result calculate","grade help","marks tally"],
    "internship advice":        ["internship advice","apply internship","internship hunt","find internship","intern application"],
    "resume writing help":      ["resume writing help","cv building help","improve resume","resume tips","cv guide"],
    "interview prep help":      ["interview prep help","prepare interview","interview tips","job interview guide","crack interview"],
    "gre prep advice":          ["gre prep advice","gre strategy","gre study plan","gre tips","graduate exam prep"],
    "ielts prep help":          ["ielts prep help","ielts writing tips","band score improve","ielts strategy","english test prep"],
    "course recommendation":    ["course recommendation","best online course","learn coding course","course suggest","platform suggest"],
    "book recommendation":      ["book recommendation","suggest books","study books","reading list","best books suggest"],
    "crush advice":             ["crush advice","how to talk crush","impress crush","crush help","confess feelings"],
    "breakup support":          ["breakup support","heartbreak help","move on advice","breakup recovery","relationship ended"],
    "friendship conflict":      ["friendship conflict","friend fight resolve","dost se issue","friend problem","friendship repair"],
    "family pressure advice":   ["family pressure advice","parent pressure help","family conflict","handle parents","family issue"],
    "toxic relationship advice":["toxic relationship advice","leave toxic partner","unhealthy relation","toxic help","control partner"],
    "long distance advice":     ["long distance advice","ldr tips","far relationship help","distance relationship","ldr guide"],
    "rejection support":        ["rejection support","rejected help","handle rejection","cope rejection","after rejection"],
    "jealousy advice":          ["jealousy advice","handle jealousy","partner jealous","jealous issue","trust jealousy"],
    "commitment advice":        ["commitment advice","relationship fear","commit phobia","relationship step","commitment issue"],
    "trust issue advice":       ["trust issue advice","partner trust","rebuild trust","trust broken","relationship trust"],
    "ex advice":                ["ex advice","ex texting back","old flame contact","ex comeback","ex message help"],
    "friendzone advice":        ["friendzone advice","escape friendzone","friend zone stuck","how to leave friendzone","just friends"],
    "group drama advice":       ["group drama advice","friend group fight","group tension","gang drama","group issue"],
    "sibling conflict":         ["sibling conflict","bhai behen fight","sibling rivalry","brother sister issue","sibling help"],
    "marriage pressure":        ["marriage pressure","shaadi pressure help","forced marriage help","family marriage push","wedding pressure"],
    "online relationship":      ["online relationship","trust online person","internet romance","online dating advice","digital love"],
    "one sided love advice":    ["one sided love advice","unrequited love","ek taraf pyaar","one way feelings","love rejection"],
    "moving on advice":         ["moving on advice","get over someone","heal heartbreak","recovery tips","detach help"],
    "friendship advice":        ["friendship advice","dosti save karo","friend repair","fix friendship","save friend"],
    "love career choice":       ["love career choice","love vs job","relationship vs career","prioritize advice","life choice"],
    "cheating suspicion":       ["cheating suspicion","partner cheating","infidelity signs","trust broken","affair suspect"],
    "proposal help":            ["proposal help","how to propose","propose ideas","proposal plan","confess love"],
    "first date tips":          ["first date tips","date ideas","first date guide","dating tips","date advice"],
    "apology advice":           ["apology advice","how to say sorry","apologize properly","forgiveness request","sorry help"],
    "boundary setting":         ["boundary setting","set limits","personal space","healthy limits","boundary help"],
    "communication advice":     ["communication advice","talk better","express feelings","relationship talk","communication fix"],
    "parent advice":            ["parent advice","overprotective parents","strict parents","handle parents","parenting conflict"],
    "cultural relationship":    ["cultural relationship","different culture love","intercultural dating","culture gap","mixed culture"],
    "age gap advice":           ["age gap advice","age difference relation","older younger couple","age gap issue","maturity gap"],
    "heartbreak recovery":      ["heartbreak recovery","heal from love","recover breakup","emotional healing","heart mend"],
    "social anxiety help":      ["social anxiety help","meeting people fear","social fear","crowd anxiety","shyness help"],
    "confidence advice":        ["confidence advice","self confidence build","believe in self","boost confidence","self esteem"],
    "loneliness support":       ["loneliness support","feeling alone help","make friends","overcome lonely","social connect"],
    "making friends":           ["making friends","new city friends","how to socialize","find friends","social circle"],
    "workplace crush":          ["workplace crush","office romance","colleague feelings","work crush","coworker like"],
    "divorce support":          ["divorce support","parents divorce help","family breakup","cope divorce","family split"],
    "abuse support":            ["abuse support","abusive situation help","leave abuse","safety help","dangerous relation"],
    "controlling partner":      ["controlling partner","partner control","possessive partner","control issue","freedom relationship"],
    "ghosted advice":           ["ghosted advice","crush ghost me","no reply help","silent treatment","being ghosted"],
    "secret relationship":      ["secret relationship","hide relation","family hide love","secret love","covert dating"],
    "activity suggestion":      ["activity suggestion","boredom cure","what to do","fun activity","free time ideas"],
    "trip planning help":       ["trip planning help","travel ideas","weekend trip","plan outing","road trip plan"],
    "movie recommendation":     ["movie recommendation","good movie suggest","film recommend","watch what","movie pick"],
    "phone buying advice":      ["phone buying advice","best phone pick","new phone guide","smartphone suggest","phone compare"],
    "overthinking help":        ["overthinking help","stop overthinking","anxious thoughts","calm mind","mental peace"],
    "career advice":            ["career advice","career path help","job choice","career direction","profession guide"],
    "startup feedback":         ["startup feedback","business idea review","startup advice","idea validate","entrepreneur help"],
    "crypto advice":            ["crypto advice","invest crypto","digital currency","btc invest","crypto risk"],
    "gym motivation":           ["gym motivation","workout again","fitness motivation","gym comeback","exercise push"],
    "diet plan help":           ["diet plan help","weight loss food","healthy diet","eat plan","food guide"],
    "sleep problem help":       ["sleep problem help","cant sleep","insomnia fix","sleep routine","night sleep"],
    "productivity advice":      ["productivity advice","stop procrastinate","time use well","work efficient","focus output"],
    "money saving tips":        ["money saving tips","student budget","save paise","frugal tips","expense manage"],
    "cooking recipe help":      ["cooking recipe help","easy recipe","quick cook","simple dish","cook guide"],
    "fashion advice":           ["fashion advice","what to wear","outfit suggest","dress code help","style tip"],
    "haircut advice":           ["haircut advice","new hairstyle","hair suggest","cut hair guide","style hair"],
    "laptop advice":            ["laptop advice","best laptop buy","student laptop","pc suggest","computer pick"],
    "bike advice":              ["bike advice","best bike buy","two wheeler pick","motorcycle suggest","bike compare"],
    "pet advice":               ["pet advice","get a pet","animal care","pet choose","pet guide"],
    "clipt ideas":               ["clipt ideas","present suggest","birthday gift","what to gift","unique present"],
    "time management":          ["time management","manage time","organize day","schedule tips","time control"],
    "anxiety management":       ["anxiety management","handle anxiety","calm anxiety","peace mind","worry control"],
    "anger management":         ["anger management","control anger","temper manage","cool down","anger handle"],
    "digital detox tips":       ["digital detox tips","phone break","screen limit","offline tips","tech detox"],
    "goal setting help":        ["goal setting help","achieve goals","resolution keep","target set","plan goals"],
    "freelancing guide":        ["freelancing guide","start freelance","gig work begin","freelance tips","work remote"],
    "stock market advice":      ["stock market advice","invest shares","equity invest","market tips","trading guide"],
    "youtube tips":             ["youtube tips","channel grow","video content","youtube strategy","creator guide"],
    "instagram growth":         ["instagram growth","followers increase","ig tips","reel strategy","social grow"],
    "public speaking":          ["public speaking","speak confidence","stage fear","presentation speak","talk public"],
    "headache remedies":        ["headache remedies","head pain fix","migraine help","sar dard cure","pain relief"],
    "cold flu treatment":       ["cold flu treatment","cold remedy","flu cure","sick help","fever cold"],
    "back pain relief":         ["back pain relief","backache fix","spine pain","posture pain","back remedy"],
    "eye strain relief":        ["eye strain relief","screen eye pain","tired eyes","eye rest","vision strain"],
    "fever management":         ["fever management","fever remedy","bukhar treatment","temperature drop","fever cure"],
    "stress relief tips":       ["stress relief tips","calm stress","reduce tension","relax tips","anxiety calm"],
    "healthy eating tips":      ["healthy eating tips","eat healthy","nutrition guide","food advice","diet tips"],
    "exercise routine":         ["exercise routine","workout plan","fitness schedule","daily exercise","gym routine"],
    "sleep hygiene tips":       ["sleep hygiene tips","good sleep","sleep routine","rest quality","sleep guide"],
    "mental health support":    ["mental health support","mind health","emotional support","mental burden","wellbeing help"],
    "food poisoning help":      ["food poisoning help","stomach upset","food problem","digestion issue","sick food"],
    "allergy management":       ["allergy management","allergic reaction","allergy control","immune trigger","allergy help"],
    "hydration tips":           ["hydration tips","drink more water","pani peena","water intake","stay hydrated"],
    "posture correction":       ["posture correction","fix posture","straight back","spine posture","posture improve"],
    "immunity boost tips":      ["immunity boost tips","strong immunity","immune system","health defense","body immunity"],
    "period pain relief":       ["period pain relief","menstrual cramps","period remedy","cycle pain","period help"],
    "skin care help":           ["skin care help","pimple fix","face care","acne treatment","skin routine"],
    "hair fall remedy":         ["hair fall remedy","stop hair loss","baal girna","hair health","hair care"],
    "weight gain advice":       ["weight gain advice","gain weight healthy","bulk tips","mass build","weight increase"],
    "quit smoking tips":        ["quit smoking tips","stop smoking","cigarette leave","smoke free","nicotine quit"],
    # translate variants
    "translate spanish message":["translate spanish message","spanish text meaning","what does this say spanish","spanish translate","decode spanish"],
    "translate japanese text":  ["translate japanese text","japanese meaning","what is this japanese","nihongo translate","japanese decode"],
    "translate korean message": ["translate korean message","korean text help","what does korean say","hangul translate","korean meaning"],
    "translate french message": ["translate french message","french text meaning","french translation","what is this french","french decode"],
    "translate arabic message": ["translate arabic message","arabic text help","arabic meaning","what says arabic","arabic decode"],
    "translate german message":  ["translate german message","german text meaning","german translate","what is this german","deutsch decode"],
    "translate portuguese message":["translate portuguese message","portuguese meaning","what says portuguese","portuguese translate","pt decode"],
    "translate urdu message":   ["translate urdu message","urdu text meaning","what is urdu","urdu decode","urdu help"],
    "translate tamil message":  ["translate tamil message","tamil text meaning","what says tamil","tamil decode","south language help"],
    "translate hindi text":     ["translate hindi text","hindi meaning","pure hindi decode","hindi to english","hindi translate"],
    # media variants
    "hilarious surprised face reaction":    ["hilarious surprised face reaction","funny shocked face","surprised humor reaction","surprised humor"],
    "excited puppy jumping around happy":   ["excited puppy jumping around happy","happy dog jumping excited","puppy energy","dog reaction"],
    "sad crying comfort hug warm":          ["sad crying comfort hug warm","comfort sad hug","crying comfort","warm hug sad"],
    "intense gaming clutch moment":         ["intense gaming clutch moment","gaming win moment","clutch gaming win","gamer reaction"],
    "confetti balloons birthday party":     ["confetti balloons birthday party","birthday celebration","party confetti","birthday balloon confetti"],
    "jaw drop disbelief expression face":   ["jaw drop disbelief expression face","shocked jaw drop face","disbelief reaction","jaw drop"],
    "cheesy pizza melting cheese pull":     ["cheesy pizza melting cheese pull","cheesy pizza pull melting","melting cheese","delicious food craving"],
    "athlete training hard gym grind":      ["athlete training hard gym grind","gym workout motivation","workout hard","fitness training hard"],
    "warm comforting hug friends together": ["warm comforting hug friends together","friend warm hug","comforting embrace","wholesome hug"],
    "colorful diwali lights fireworks celebration":["colorful diwali lights fireworks celebration","diwali lights sparkle","festival lights","diwali celebrate"],
    "mischievous cat knocking things over": ["mischievous cat knocking things over","cat mischief chaos","naughty cat","cat chaos"],
    "cricket six massive hit stadium crowd":["cricket six massive hit stadium crowd","cricket six massive hit","match moment","stadium crowd"],
    "anime character dramatic reaction face":["anime character dramatic reaction face","anime dramatic expression","dramatic anime","anime face"],
    "person dancing happily celebration move":["person dancing happily celebration move","person dancing joyful","celebration dance","happy move"],
    "person facepalm frustrated embarrassed moment":["person facepalm frustrated embarrassed moment","facepalm embarrassed moment","embarrassed reaction","frustrated face"],
    "rain window cozy evening aesthetic":   ["rain window cozy evening aesthetic","rain window cozy vibe","cozy rain","window rain"],
    "sleepy panda dozing off tired adorable":["sleepy panda dozing off tired adorable","sleepy panda clip","tired animal","dozing cute"],
    "chef cooking tossing pan flame":       ["chef cooking tossing pan flame","cooking clip","chef pan flip","food prep clip"],
    "superhero dramatic landing epic pose": ["superhero dramatic landing epic pose","superhero clip","epic landing","hero pose"],
    "baby giggling laughing cute adorable": ["baby giggling laughing cute adorable","baby laugh clip","cute baby","giggling baby"],
    "mind explosion shock realization moment":["mind explosion shock realization moment","mind blown clip","shocking realization","explosion head"],
    "mountain sunrise peaceful nature calm":["mountain sunrise peaceful nature calm","nature clip","mountain sunrise","peaceful nature"],
    "car drifting sharp turn smoke trail":  ["car drifting sharp turn smoke trail","car drift clip","racing turn","smoke drift"],
    "sunset beach waves orange sky vibe":   ["sunset beach waves orange sky vibe","sunset beach clip","ocean sunset","beach vibe"],
    "graduation cap toss confetti proud moment":["graduation cap toss confetti proud moment","graduation clip","grad celebration","degree toss"],
}

def get_title(base_title):
    """Return a varied title from the variant pool or the base itself."""
    if base_title in TITLE_VARIANTS:
        return random.choice(TITLE_VARIANTS[base_title])
    return base_title


def build_text_sample(lang="hinglish"):
    cat = random.choice(list(TEXT_TOPICS.keys()))
    topic = random.choice(TEXT_TOPICS[cat])
    key_msg_base, _, title_base = topic

    hi_prefixes = ["bhai","yaar","arre","oye","sun","suno","guys"]
    en_prefixes = ["hey","so","btw","quick question","anyone","ngl","honestly"]

    # Suffix pools — add semantic richness, prevent bare noun key messages
    hi_suffixes = [
        "help karo","batao please","koi bata sakta hai","samjhao zara",
        "guide do","tips do","koi idea hai","kya sochte ho","suggest karo",
        "kaise karoon","kya karoon","please help","urgent hai","pls bata",
        "yaar bahut confuse hun","sochte raho kya","koi toh bata",
    ]
    en_suffixes = [
        "please help","any ideas","need advice","what should i do",
        "how do i handle this","can someone guide","tips please",
        "help needed","any suggestions","what do you think","urgent",
    ]

    if lang == "hinglish":
        variations = [
            f"{random.choice(hi_prefixes)} {key_msg_base}",
            f"{key_msg_base} {random.choice(hi_suffixes)}",
            key_msg_base.replace("bhai","yaar").replace("yaar","bhai"),
            f"yaar {key_msg_base} {random.choice(hi_suffixes)}",
            f"{random.choice(hi_prefixes)} {key_msg_base} {random.choice(hi_suffixes)}",
            f"bhai sun {key_msg_base}",
            f"ek cheez bata {key_msg_base}",
        ]
    else:
        variations = [
            f"{random.choice(en_prefixes)} {key_msg_base}",
            f"{key_msg_base} {random.choice(en_suffixes)}",
            f"{key_msg_base} please help me out",
            f"guys {key_msg_base} {random.choice(en_suffixes)}",
            f"quick question {key_msg_base}",
        ]

    key_msg = random.choice(variations)

    # Enforce minimum meaningful length — bare nouns get suffix added
    words = key_msg.strip().split()
    if len(words) < 4:
        if lang == "hinglish":
            key_msg = f"bhai {key_msg_base} {random.choice(hi_suffixes)}"
        else:
            key_msg = f"hey {key_msg_base} {random.choice(en_suffixes)}"

    target_pos = pick_target_pos()
    msgs = build_msgs_with_key(key_msg, target_pos, lang)
    effort = random.choice(["low","medium","high"])
    title = get_title(title_base)
    dec = make_decision("text", target_pos, effort, title)
    return msgs, dec

def build_media_sample(lang="hinglish"):
    topic = random.choice(MEDIA_TOPICS)
    key_msg, title_base, effort = topic

    hi_variations = [
        key_msg,
        f"bhai {key_msg}",
        f"yaar {key_msg}",
        f"{key_msg} please",
        f"arre {key_msg} na yaar",
        f"oye {key_msg}",
    ]
    en_variations = [
        key_msg,
        f"hey {key_msg}",
        f"can you {key_msg}",
        f"{key_msg} please bro",
        f"someone {key_msg}",
    ]
    variations = hi_variations if lang == "hinglish" else en_variations
    key_msg = random.choice(variations)

    target_pos = pick_target_pos()
    msgs = build_msgs_with_key(key_msg, target_pos, lang)
    title = get_title(title_base)
    dec = make_decision("media", target_pos, effort, title)
    return msgs, dec

def build_react_sample(lang="hinglish"):
    topic = random.choice(REACT_TOPICS)
    key_msg, emoji = topic

    hi_variations = [
        key_msg,
        f"bhai {key_msg}",
        f"yaar sun {key_msg}",
        f"{key_msg} yaar",
        f"arre {key_msg}",
    ]
    en_variations = [
        key_msg,
        f"guys {key_msg}",
        f"omg {key_msg}",
        f"bro {key_msg}",
        f"wait {key_msg}",
    ]
    variations = hi_variations if lang == "hinglish" else en_variations
    key_msg = random.choice(variations)

    target_pos = pick_target_pos()
    msgs = build_msgs_with_key(key_msg, target_pos, lang)
    dec = make_decision("react", target_pos, "null", emoji)
    return msgs, dec

def build_acknowledge_sample(lang="hinglish"):
    topic = random.choice(ACKNOWLEDGE_TOPICS)
    key_msg, reply_pool = topic

    hi_variations = [
        key_msg,
        f"bhai {key_msg}",
        f"yaar {key_msg}",
        f"{key_msg} bhai",
        f"btw {key_msg}",
    ]
    en_variations = [
        key_msg,
        f"hey {key_msg}",
        f"just fyi {key_msg}",
        f"{key_msg} everyone",
        f"quick update {key_msg}",
    ]
    variations = hi_variations if lang == "hinglish" else en_variations
    key_msg = random.choice(variations)

    target_pos = pick_target_pos()
    msgs = build_msgs_with_key(key_msg, target_pos, lang)
    reply = random.choice(reply_pool)
    dec = make_decision("acknowledge", target_pos, "null", reply)
    return msgs, dec

def build_translate_sample(lang="hinglish"):
    topic = random.choice(TRANSLATE_TOPICS)
    foreign_text, title = topic

    # Context messages discussing receiving this foreign text
    hi_contexts_before = [
        ["bhai sun", "haan bol", "kuch mila tha message mein", "ye dekh kya bol raha hai"],
        ["yaar dekh", "kya hai ye", "kahin se aaya", "samajh nahi aaya"],
        ["bhai", "ek message aaya hai", "translate kar do", "please help"],
        ["yaar sun na", "foreign language hai", "kya likha hai", "bata na"],
        ["oye", "ye kya script hai", "anime ka tha kya", "samjha koi"],
        ["bhai caption mein tha", "kaunsi language", "pata nahi mujhe", "tu bata"],
        ["yaar email mein tha ye", "boss ka message tha", "kya matlab hai", "urgent hai"],
    ]
    en_contexts_before = [
        ["hey guys", "got this message", "have no idea what it says", "can someone translate"],
        ["so", "my friend sent this", "i think its spanish", "what does it mean"],
        ["bro", "this was in the caption", "what language even", "help please"],
        ["wait", "got this email", "from a client", "need a translation asap"],
        ["guys", "saw this on instagram", "looks japanese", "anyone know"],
    ]

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
            after_pool = ["kya matlab hai","samjha koi","translate karo","help please","batao na"] if lang == "hinglish" else ["what does this say","anyone know","translate please","help","what language is this"]
            msgs.append(random.choice(after_pool))

    dec = make_decision("translate", target_pos, "null", get_title(title))
    return msgs, dec

def build_ignore_sample():
    topic = random.choice(IGNORE_TOPICS)
    # Use exactly 5 messages from the template, cycling if needed
    msgs = [topic[i % len(topic)] for i in range(5)]
    # Add slight variation
    if random.random() < 0.3:
        pos = random.randint(0,4)
        msgs[pos] = random.choice(list(EMOJI_FILLERS) + HI_FILLERS[:15])
    dec = "R: TYPE=ignore | TARGET=null | EFFORT=null | TITLE=null"
    return msgs, dec

# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

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

def similarity(a, b):
    """Jaccard similarity on word tokens."""
    wa = set(normalize_text(a).split())
    wb = set(normalize_text(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def is_duplicate(new_msgs, seen_sigs, seen_key_msgs, threshold):
    """Check if sample is semantically too similar to existing ones."""
    # Exact hash check
    sig = msg_signature(new_msgs)
    if sig in seen_sigs:
        return True

    # Check key message (M3-M5) similarity against recent seen key msgs
    key_msg = " ".join(new_msgs[2:])  # M3+M4+M5
    key_norm = normalize_text(key_msg)

    # Only check last 500 for speed
    check_pool = list(seen_key_msgs)[-500:]
    for seen in check_pool:
        if similarity(key_norm, seen) > threshold:
            return True

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
    target_counts = Counter(s["_target"] for s in samples)
    lang_counts = Counter(s["_lang"] for s in samples)

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

def what_type_needed(samples, cfg):
    """Return what type is most needed to fix distribution."""
    n = max(len(samples), 1)
    type_counts = Counter(s["_type"] for s in samples)
    deficits = {}
    for t, frac in cfg["type_dist"].items():
        actual = type_counts.get(t, 0) / n
        deficits[t] = frac - actual
    return max(deficits, key=deficits.get)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sample(type_hint=None, lang_hint=None):
    """Generate one sample with optional type/lang hints."""
    lang = lang_hint or (random.choices(
        ["hinglish", "english"],
        weights=[CFG["lang_dist"]["hinglish"], CFG["lang_dist"]["english"]]
    )[0])

    if type_hint is None:
        content_roll = random.random()
        if content_roll < CFG["content_dist"]["real_chat"]:
            # Real chat: pick type from dist
            type_hint = random.choices(
                list(CFG["type_dist"].keys()),
                weights=list(CFG["type_dist"].values())
            )[0]
        elif content_roll < CFG["content_dist"]["real_chat"] + CFG["content_dist"]["spam_noise"]:
            type_hint = "ignore"
        else:
            type_hint = "ignore"  # emoji-only also goes to ignore

    if type_hint == "text":
        msgs, dec = build_text_sample(lang)
    elif type_hint == "media":
        msgs, dec = build_media_sample(lang)
    elif type_hint == "react":
        msgs, dec = build_react_sample(lang)
    elif type_hint == "acknowledge":
        msgs, dec = build_acknowledge_sample(lang)
    elif type_hint == "translate":
        msgs, dec = build_translate_sample(lang)
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





def build_eval_benchmark():
    random.seed(99)
    cases = []

    # ── CAT 1: Title drift — key msg in M3/M4, distractor in M1 (50 cases) ──
    # Router must NOT put title based on M1/M2 context
    title_drift_cases = [
        # msgs=[M1,M2,M3,M4,M5], key at M3 or M4
        (["bhai langchain ke baare mein baat kar rahe the","haan","bhai python error samajh nahi aa raha","theek hai","hmm"], "text","M3","python error"),
        (["yaar college admission tha topic","haan haan","bhai breakup ho gayi yaar help","okay","sure"], "text","M3","breakup"),
        (["crypto advice de raha tha","sahi tha","bhai flutter build fail ho raha","accha","lol"], "text","M3","flutter"),
        (["yaar trip plan kar rahe the","haan","bhai sql query error aa rahi hai yaar","hmm","ok"], "text","M3","sql"),
        (["bhai gym ki baat kar rahe the","okay bhai","yaar crush se kaise baat karoon","sahi hai","lol"], "text","M3","crush"),
        (["yaar movie dekh rahe the","haan","bhai docker container nahi chal raha","theek","ok"], "text","M3","docker"),
        (["bhai startup idea pe baat kar rahe the","haan suno","bhai behen se bahut fight hoti hai","hmm","okay"], "text","M3","sibling"),
        (["yaar stock market discuss kar rahe the","haan","bhai typescript error aa raha hai help","lol","sure"], "text","M3","typescript"),
        (["bhai instagram ke baare mein tha","sahi hai","yaar anxiety bahut hoti hai help","accha","hmm"], "text","M3","anxiety"),
        (["yaar phone buying discuss kar rahe the","haan","bhai cors error block kar raha hai","okay","theek"], "text","M3","cors"),
        # key at M4
        (["bhai pehle crypto baat ki","haan","accha","yaar redis cache kaam nahi kar raha","lol"], "text","M4","redis"),
        (["yaar trip pe gaye the","sahi","hmm","bhai react hooks mein issue hai help","okay"], "text","M4","react"),
        (["bhai mummy ki baat thi","accha","theek hai","yaar ex ne text kiya kya karoon","hmm"], "text","M4","ex"),
        (["yaar coding bootcamp discuss kar rahe the","haan","okay","bhai mongodb query empty aa rahi","okay lol"], "text","M4","mongodb"),
        (["bhai health tips the","sahi","hm","yaar best friend se fight ho gayi","accha"], "text","M4","friend"),
        (["yaar langchain model ke baare mein tha","hmm","okay","bhai github actions fail ho rahi hai","lol"], "text","M4","github"),
        (["bhai movie recommend kar raha tha","haan","sure","yaar meri back pain bahut hai help","hmm"], "text","M4","back pain"),
        (["yaar exam ke baare mein baat ki","sahi","theek","bhai websocket disconnect ho raha hai","okay"], "text","M4","websocket"),
        (["bhai phone suggest kar raha tha","hmm","okay","yaar proposal karna hai ideas do","lol"], "text","M4","proposal"),
        (["yaar freelancing ki baat ki","haan","accha","bhai kubernetes pod crash ho raha hai","okay"], "text","M4","kubernetes"),
        # more varied
        (["bhai langchain pe tha convo","haan","aur bata","yaar nginx 502 error aa raha hai","lol bhai"], "text","M4","nginx"),
        (["yaar kal match tha","haan wow","sahi tha","bhai jwt token expire ho raha hai","okay"], "text","M4","jwt"),
        (["bhai bored ho raha tha","haan yaar","accha","yaar divorce ho rahi hai parents ki help","hmm"], "text","M4","divorce"),
        (["yaar langchain ke baare mein tha","haan","theek hai","bhai vim editor nahi samajh aa raha","okay lol"], "text","M4","vscode"),
        (["bhai fashion advice de raha tha","haan","okay","yaar stress bahut hai kaise kam karoon","sure"], "text","M4","stress"),
        # M5 but with noisy M1/M2 context about something else
        (["bhai langchain mein expert hun","haan wahi toh","okay","hmm","yaar python import error aa rahi hai"], "text","M5","python"),
        (["yaar kal trip plan kar rahe the","haan","accha","okay","bhai firebase auth kaam nahi kar raha help"], "text","M5","firebase"),
        (["bhai stock market ki baat kar rahe the","haan","sahi","theek","yaar period pain bahut hai help"], "text","M5","period"),
        (["yaar langchain use karna tha","haan","hmm","okay","bhai graphql resolver error aa raha hai"], "text","M5","graphql"),
        (["bhai mummy ne bola tha kuch","haan","sure","lol","yaar toxic relationship mein hun help karo"], "text","M5","toxic"),
        (["yaar pizza kha rahe the","haha","lol","okay","bhai supabase auth nahi chal raha yaar"], "text","M5","supabase"),
        (["bhai coding bootcamp discuss kiya","haan","theek hai","accha","yaar one sided love hai kya karoon"], "text","M5","one sided"),
        (["yaar langchain ki baat ki thi","okay","haan","sure","bhai elasticserch query nahi chal rahi"], "text","M5","elasticsearch"),
        (["bhai ghar pe kuch hua tha","haan","okay","hmm","yaar marriage pressure de rahe hain ghar wale"], "text","M5","marriage"),
        (["yaar kal ka plan tha","haan","accha","theek","bhai docker compose network issue hai yaar"], "text","M5","docker"),
        # remaining 15 — mix of M3/M4/M5 with unrelated M1/M2
        (["bhai langchain","haan","bhai stripe payment error aa raha hai","ok","hmm"], "text","M3","stripe"),
        (["yaar travel","accha","bhai quit smoking karna hai tips do","haan","lol"], "text","M3","quit smoking"),
        (["bhai cricket","haan","yaar hair fall bahut hai kya karoon","hmm","okay"], "text","M3","hair fall"),
        (["yaar coding bootcamp","sahi","bhai prometheus metrics nahi aa rahi","theek","lol"], "text","M3","prometheus"),
        (["bhai diwali tha topic","haan","yaar weight gain karna hai advice do","okay","hmm"], "text","M3","weight gain"),
        (["yaar langchain model","haan","bhai kafka consumer nahi chal raha","accha","lol"], "text","M4","kafka"),
        (["bhai mummy ki baat","okay","theek","yaar skin care routine kya karoon","hmm"], "text","M4","skin"),
        (["yaar food discuss kiya","haan","accha","bhai heroku deploy fail ho rahi hai","lol"], "text","M4","heroku"),
        (["bhai movie recommend kiya","okay","sahi","yaar loneliness feel ho rahi hai help","hmm"], "text","M4","loneliness"),
        (["yaar plan tha ghumne ka","haan","theek","bhai openai api rate limit aa rahi","lol okay"], "text","M4","openai"),
        (["bhai langchain","haan","hmm","lol","bhai fastapi startup nahi ho rahi"], "text","M5","fastapi"),
        (["yaar kal exam tha","haan","okay","hmm","bhai freelancing kaise shuru karoon guide do"], "text","M5","freelancing"),
        (["bhai game discuss kiya","haan","lol","okay","yaar allergy bahut aa rahi hai help karo"], "text","M5","allergy"),
        (["yaar trip ki baat","haan","accha","theek","bhai cloudflare worker error aa raha hai"], "text","M5","cloudflare"),
        (["bhai startup topic tha","haan","okay","hmm","yaar overprotective parents hain help karo"], "text","M5","parent"),
    ]

    for entry in title_drift_cases:
        if len(entry) == 3:
            msgs, typ, target = entry
            title_kw = ""
        else:
            msgs, typ, target, title_kw = entry
        cases.append(make_eval(msgs, typ, target, None, title_kw))

    # ── CAT 2: Emoji/laughter → IGNORE not react/text (50 cases) ──
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
        ["💀","💀","💀","💀","💀"],
        ["haha","haha","haha","haha","haha"],
        ["lol","lol","lol","lol","lol"],
        ["🔥🔥🔥","💀💀","😂😂","lmao","ded"],
        ["bruh bruh bruh","lol lol","haha","💀","dead"],
        ["xD xD xD","lmaooo","hahaha","😂","okay lol"],
        ["asdfjkl","qwiopzx","mnbvcx","lkjhgf","poiuyt"],
        ["FREE PRIZE","CLAIM NOW","URGENT","LIMITED","bit.ly/abc"],
        ["bhai haha yaar lol bro 😂 💀","hehe","xd","lmao","dead"],
    ]
    for msgs in emoji_ignore_cases:
        if len(msgs) < 5:
            msgs = (msgs * 2)[:5]
        cases.append(make_eval(msgs[:5], "ignore", "null", "null", "null"))

    # ── CAT 3: Translate detection — foreign text at M3/M4/M5 (50 cases) ──
    translate_cases = [
        # Spanish at M5
        (["bhai sun","haan","kuch mila","ye dekh","Muchas gracias por tu ayuda"], "translate","M5","spanish"),
        (["yaar dekh","kya hai ye","kahin se aaya","samajh nahi","No entiendo nada de esto"], "translate","M5","spanish"),
        # Japanese at M5
        (["bhai","ek message","translate kar do","please","ありがとうございました"], "translate","M5","japanese"),
        (["yaar","foreign hai","kya likha hai","bata na","お疲れ様でした"], "translate","M5","japanese"),
        # Korean at M5
        (["bhai dekh","kaunsi lang","pata nahi","tu bata","감사합니다"], "translate","M5","korean"),
        (["hey guys","got this","no idea","can someone","사랑해요"], "translate","M5","korean"),
        # French at M5
        (["bhai caption mein tha","kaunsi lang","pata nahi","tu bata","Merci beaucoup pour tout"], "translate","M5","french"),
        (["so","my friend sent","think its french","what does it mean","Je ne comprends pas"], "translate","M5","french"),
        # Arabic at M5
        (["bhai email mein tha","boss ka message","kya matlab","urgent hai","شكراً جزيلاً على مساعدتك"], "translate","M5","arabic"),
        (["wait","got this email","from client","need translation","لا أفهم هذا الكلام"], "translate","M5","arabic"),
        # German at M5
        (["bro","caption pe tha","what language","help please","Vielen Dank fuer deine Hilfe"], "translate","M5","german"),
        (["guys","saw on instagram","looks german","anyone know","Ich verstehe das wirklich nicht"], "translate","M5","german"),
        # Urdu at M5
        (["bhai sun","haan","foreign script","pata nahi","آپ کا بہت شکریہ"], "translate","M5","urdu"),
        (["yaar dekh","kya hai","script alag hai","bata","مجھے سمجھ نہیں آیا"], "translate","M5","urdu"),
        # Tamil at M5
        (["bhai caption mein tha","kaunsi hai","south lang lagti","bata","மிக்க நன்றி உங்களுக்கு"], "translate","M5","tamil"),
        (["hey","this was in bio","what script","anyone","எனக்கு புரியவில்லை"], "translate","M5","tamil"),
        # Foreign at M4
        (["bhai sun","haan bol","kuch mila tha","Muchas gracias amigo","kya matlab hai"], "translate","M4","spanish"),
        (["yaar dekh","haan","message aaya","ありがとうございました","samjha do"], "translate","M4","japanese"),
        (["bhai","suno","forward hua","감사합니다","ye kya hai"], "translate","M4","korean"),
        (["yaar","haan","email mein tha","Merci beaucoup pour tout","translate karo"], "translate","M4","french"),
        (["sun","bol","caption tha","شكراً جزيلاً","matlab batao"], "translate","M4","arabic"),
        (["bhai ye dekh","haan","kya hai","Vielen Dank fuer deine Hilfe","matlab bata"], "translate","M4","german"),
        (["yaar","haan","script alag","آپ کا بہت شکریہ","kya likha"], "translate","M4","urdu"),
        (["bhai dekh","haan","south lang","மிக்க நன்றி","translate karo"], "translate","M4","tamil"),
        # Foreign at M3
        (["bhai sun","haan","Muchas gracias amigo","kya tha ye","samjhao"], "translate","M3","spanish"),
        (["yaar dekh","haan","ありがとうございました","kya likha hai","translate karo"], "translate","M3","japanese"),
        (["bhai","bol","감사합니다","ye kya hai","batao"], "translate","M3","korean"),
        (["sun bhai","haan","Merci beaucoup","ye kaunsi language","bata"], "translate","M3","french"),
        (["yaar","bol na","شكراً جزيلاً","ye kya hai","translate kar"], "translate","M3","arabic"),
        (["bhai sun","haan","Vielen Dank","kya tha ye","batao"], "translate","M3","german"),
        # Portuguese
        (["bhai sun","haan","kuch mila","ye dekh","Muito obrigado pela sua ajuda"], "translate","M5","portuguese"),
        (["yaar","kya hai","nahi samjha","bata","Nao entendo nada do que esta acontecendo"], "translate","M5","portuguese"),
        # Hindi pure text
        (["bhai sun","haan","kuch mila","ye dekh","यार कल परीक्षा है कुछ नहीं पढ़ा"], "translate","M5","hindi"),
        (["yaar dekh","kya hai","pure hindi","bata","भाई बहुत थक गया हूं आज का दिन"], "translate","M5","hindi"),
        # Mixed contexts
        (["ok","ok","ok","ok","감사합니다"], "translate","M5","korean"),
        (["hm","hmm","hmm","hmm","ありがとうございました"], "translate","M5","japanese"),
        (["bhai","yaar","okay","accha","Merci beaucoup"], "translate","M5","french"),
        (["lol","haha","okay","sure","Muchas gracias"], "translate","M5","spanish"),
        (["hey","so","btw","wait","Vielen Dank fuer"], "translate","M5","german"),
        # After filler chain
        (["bhai sun na","haan bol yaar","theek hai","accha okay","شكراً جزيلاً على مساعدتك"], "translate","M5","arabic"),
        (["yaar sun","okay","haan","hmm","آپ کا بہت شکریہ"], "translate","M5","urdu"),
        (["hey guys","okay","sure","yeah","மிக்க நன்றி உங்களுக்கு"], "translate","M5","tamil"),
        (["bhai","haan","theek","lol","यार कल परीक्षा है कुछ नहीं पढ़ा अभी तक"], "translate","M5","hindi"),
        (["yaar","okay","hmm","accha","Por favor ayudame con esto"], "translate","M5","spanish"),
        (["bhai sun","okay","haan","sure","정말 대단해요"], "translate","M5","korean"),
        (["hey","so","wait","yeah","Je suis desole pour ca"], "translate","M5","french"),
        (["bhai","yaar","lol","hmm","頑張ってください"], "translate","M5","japanese"),
        (["okay","sure","accha","theek","Bitte hilf mir damit"], "translate","M5","german"),
        (["bhai sun","haan","kya tha ye","kaunsi lang","Estoy muy emocionado"], "translate","M5","spanish"),
        (["yaar","haan","kuch mila","caption tha","사랑해요 항상 곁에"], "translate","M5","korean"),
    ]
    for entry in translate_cases:
        msgs, typ, target, title_kw = entry
        cases.append(make_eval(msgs, typ, target, "null", title_kw))

    # ── CAT 4: Acknowledge vs Text/React confusion (50 cases) ──
    ack_cases = [
        # These should be ACKNOWLEDGE not text
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
        # M5 acknowledge
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
        # M3 acknowledge
        (["bhai sun","haan","bhai kaam ho gaya project done","theek hai","hmm"], "acknowledge","M3","done"),
        (["yaar","okay","reached home safely everyone","lol","sure"], "acknowledge","M3","safe"),
        (["bhai","haan","bhai payment kar diya upi se","accha","lol"], "acknowledge","M3","payment"),
        (["okay","hmm","good morning bhai gm subah","sure","haan"], "acknowledge","M3","morning"),
        (["yaar","theek","bhai bohot shukriya yaar tere bina nahi hota","lol","okay"], "acknowledge","M3","welcome"),
        # Task-done variants that look like text
        (["bhai sun","haan","theek","bhai file share kar di check karo","lol"], "acknowledge","M4","file"),
        (["yaar","okay","accha","just finished the presentation nailed it","hmm"], "acknowledge","M4","done"),
        (["bhai","haan","sure","bhai ghar aa gaya safely trip mast rahi","lol"], "acknowledge","M4","safe"),
        (["okay","hmm","lol","thanks for being there always appreciate yaar","accha"], "acknowledge","M4","welcome"),
        (["yaar","theek","okay","bhai interview de ke aaya accha laga mujhe","hmm"], "acknowledge","M4","luck"),
        # submitted/wrapped up
        (["bhai sun","haan","theek","lol","submitted the project before deadline done"], "acknowledge","M5","done"),
        (["yaar","okay","haan","sure","bhai neend aa rahi hai kal milte hain gn"], "acknowledge","M5","night"),
        (["bhai","hmm","lol","accha","food is ready come eat everyone at table"], "acknowledge","M5","coming"),
        (["okay","theek","haan","hmm","bhai movie download ho gayi link bhej raha"], "acknowledge","M5","received"),
        (["yaar","okay","lol","sure","reached the hospital safely waiting outside"], "acknowledge","M5","safe"),
        # Good morning / good night must be acknowledge not react
        (["bhai","haan","theek","sure","good morning everyone subah ho gayi gm"], "acknowledge","M5","morning"),
        (["yaar","okay","lol","hmm","good night all sleep well kal milte hain gn"], "acknowledge","M5","night"),
        (["bhai sun","haan","theek","lol","gm bhai gm sab subah uthke chai pi lo"], "acknowledge","M5","morning"),
        (["yaar","okay","haan","accha","gn sab sweet dreams kal milte hain bye"], "acknowledge","M5","night"),
        (["bhai","hmm","lol","sure","bhai aaj ghar aaya safely trip mast thi yaar"], "acknowledge","M5","safe"),
        # Payment confirmations
        (["yaar","okay","haan","theek","bhai paytm kar diya check karo na yaar"], "acknowledge","M5","payment"),
        (["bhai sun","haan","accha","lol","upi transfer done check karo please"], "acknowledge","M5","payment"),
        (["okay","hmm","sure","theek","bhai fees bhar di college ki confirm karo"], "acknowledge","M5","payment"),
        (["yaar","haan","lol","accha","rent transfer ho gaya bank se bhai"], "acknowledge","M5","payment"),
        (["bhai","okay","hmm","sure","just sent rent money to your account bro"], "acknowledge","M5","received"),
        # Thank you messages
        (["yaar","okay","haan","lol","yaar thanks a lot tere bina ye nahi hota"], "acknowledge","M5","welcome"),
        (["bhai","hmm","sure","accha","bhai help ke liye bahut shukriya seriously"], "acknowledge","M5","welcome"),
        (["okay","haan","lol","theek","thank you so much for everything always there"], "acknowledge","M5","welcome"),
        (["yaar","okay","haan","sure","appreciate you always being there for me"], "acknowledge","M5","welcome"),
        (["bhai sun","haan","theek","hmm","bhai grateful hun tere liye always yaar"], "acknowledge","M5","welcome"),
        (["okay","sure","accha","lol","thanks for the help man really means a lot"], "acknowledge","M5","welcome"),
    ]
    for entry in ack_cases:
        msgs, typ, target, title_kw = entry
        cases.append(make_eval(msgs, typ, target, "null", title_kw))

    # ── CAT 5: Media request clarity — must be TYPE=media not text (50 cases) ──
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
        # M5
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
        # M3
        (["bhai sun","haan","bhai ek funny meme bhej yaar","theek hai","hmm"], "media","M3"),
        (["yaar","okay","send a cute dog gif please","lol","sure"], "media","M3"),
        (["bhai","haan","bhai sad sticker bhej please","accha","lol"], "media","M3"),
        (["okay","hmm","yaar gaming rage gif chahiye bhai","sure","okay"], "media","M3"),
        (["yaar","theek","bhai birthday gif bhej celebration","lol","haan"], "media","M3"),
        # Varied phrasing — should still be media not text
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
        # After noisy M1/M2 — still media
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
        (["bhai","haan","hmm","lol","yaar dance gif bhej party mood chal raha"], "media","M5"),
    ]
    for entry in media_cases:
        msgs, typ, target = entry
        cases.append(make_eval(msgs, typ, target, None, ""))

    # ── CAT 6: React — must be TYPE=react not text/media (50 cases) ──
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
        # M5
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
        # M3
        (["bhai sun","haan","bhai result aa gaya marks acche aaye","theek hai","hmm"], "react","M3","🥳"),
        (["yaar","okay","did you hear that celebrity got arrested","lol","sure"], "react","M3","😲"),
        (["bhai","haan","bhai rank up kar liya finally diamond","accha","lol"], "react","M3","🔥"),
        (["okay","hmm","aaj mera birthday hai 21 saal ka","sure","okay"], "react","M3","🎂"),
        (["yaar","theek","this baby animal is absolutely adorable omg","lol","haan"], "react","M3","🥰"),
        # Diverse react events
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
        # M5 diverse
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
        (["bhai","hmm","lol","sure","bhai result aa gaya 95 percent mila yaar finally"], "react","M5","🥳"),
    ]
    for entry in react_cases:
        msgs, typ, target, emoji = entry
        cases.append(make_eval(msgs, typ, target, "null", emoji))

    # ── Write benchmark ──────────────────────────────────────────────────────
    random.shuffle(cases)
    bench_out = "eval_benchmark_300.jsonl"
    with open(bench_out, "w", encoding="utf-8") as f:
        for c in cases:
            # strip _category before writing (it's for human reading)
            out_c = {k:v for k,v in c.items()}
            f.write(json.dumps(out_c, ensure_ascii=False) + "\n")

    # Category breakdown
    cat_c = Counter(c["_category"] for c in cases)
    type_c = Counter(c["expected"]["TYPE"] for c in cases)

    print(f"\n{'='*60}")
    print(f"📋 Eval benchmark: {len(cases)} cases → {bench_out}")
    print(f"\n   TYPE breakdown:")
    for t, n in sorted(type_c.items(), key=lambda x:-x[1]):
        print(f"     {t:12s}: {n}")
    print(f"\n   How to use:")
    print(f"     for each case: run model on case['input']['messages']")
    print(f"     parse output: TYPE, TARGET, EFFORT, TITLE")
    print(f"     score:")
    print(f"       TYPE match  → 1.0 pts")
    print(f"       TARGET match → 0.5 pts")
    print(f"       TITLE_contains match → 0.3 pts")
    print(f"     perfect score per case: 1.8 pts")
    print(f"     total possible: {len(cases)*1.8:.0f} pts")
    print(f"\n   Weakness categories detected:")
    print(f"     CAT1 title_drift   → 50 cases (M1/M2 noise → title must match target msg)")
    print(f"     CAT2 ignore        → 50 cases (emoji/spam/noise → must ignore)")
    print(f"     CAT3 translate     → 50 cases (foreign text at M3/M4/M5)")
    print(f"     CAT4 acknowledge   → 50 cases (task-done/gm/gn vs text/react)")
    print(f"     CAT5 media         → 50 cases (gif/sticker request vs text)")
    print(f"     CAT6 react         → 50 cases (celebration/shock vs text)")



def main():
    target_n = CFG["total_samples"]
    samples = []
    seen_sigs = set()
    seen_key_msgs = []
    dup_count = 0
    attempts = 0
    max_attempts = target_n * 15

    print(f"🚀 Generating {target_n} samples...")
    print(f"   Dedup threshold: {CFG['dedup_threshold']}")
    print(f"   Max attempts: {max_attempts}")
    print()

    loop = 0
    while len(samples) < target_n and attempts < max_attempts:
        attempts += 1

        # Hard-cap enforcement: every attempt, check if type would exceed cap
        # This prevents text from ballooning past its target
        n_so_far = max(len(samples), 1)
        type_counts_now = Counter(s["_type"] for s in samples)

        # Check if any type is OVER its cap (using tighter 0.02 ceiling)
        over_cap_types = set()
        for t, frac in CFG["type_dist"].items():
            actual = type_counts_now.get(t, 0) / n_so_far
            if actual > frac + 0.02:
                over_cap_types.add(t)

        # Find most deficient type (force-generate it every 5 samples)
        deficits = {}
        for t, frac in CFG["type_dist"].items():
            actual = type_counts_now.get(t, 0) / n_so_far
            deficits[t] = frac - actual
        most_needed = max(deficits, key=deficits.get)

        # Decide what type to generate
        if deficits[most_needed] > 0.04:
            # Strongly deficient — force it
            type_hint = most_needed
        elif over_cap_types:
            # Some type is over cap — pick from non-overcapped types
            allowed = [t for t in CFG["type_dist"] if t not in over_cap_types]
            type_hint = random.choices(
                allowed,
                weights=[CFG["type_dist"][t] for t in allowed]
            )[0]
        else:
            type_hint = None  # free random

        sample = generate_sample(type_hint=type_hint)
        msgs = sample["_msgs"]

        # Skip if this type is over cap
        if sample["_type"] in over_cap_types and deficits.get(sample["_type"], 0) < -0.03:
            continue

        # Dedup check
        if is_duplicate(msgs, seen_sigs, seen_key_msgs, CFG["dedup_threshold"]):
            dup_count += 1
            continue

        # Register
        sig = msg_signature(msgs)
        seen_sigs.add(sig)
        seen_key_msgs.append(normalize_text(" ".join(msgs[2:])))

        samples.append(sample)

        if len(samples) % 500 == 0:
            ok, issues = check_distribution(samples, CFG)
            status = "✅" if ok else "⚠️"
            print(f"  {status} {len(samples)}/{target_n} | dups rejected: {dup_count} | issues: {len(issues)}")
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
    print(f"   Total attempts: {attempts} | Dups rejected: {dup_count}")
    print(f"   Dedup rate: {dup_count/attempts*100:.1f}%")

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
        expected_frac = frac
        mark = "✅" if abs(actual/ni - expected_frac) < CFG["tolerance"]+0.05 else "❌"
        print(f"  {mark} {tgt}: {actual:5d} ({actual/ni*100:.1f}%) | expected ({expected_frac*100:.0f}%)")

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
        if t == "media" and title:
            for bad in ["meme","gif","sticker","emoji"]:
                if bad in title.lower():
                    errors.append(f"#{i} media title has forbidden word: {title}")
        if t not in ["text","media","react","acknowledge","translate","ignore"]:
            errors.append(f"#{i} invalid type: {t}")

    if errors:
        print(f"  ❌ {len(errors)} errors:")
        for e in errors[:10]: print(f"    {e}")
    else:
        print(f"  ✅ All {len(final)} samples valid")

    # Sample previews
    print(f"\n=== SAMPLE PREVIEWS ===")
    preview_indices = [0, 250, 500, 1000, 2000, 3000, 4000, len(samples)-1]
    for idx in preview_indices:
        if idx < len(samples):
            s = samples[idx]
            print(f"\n[{idx}] TYPE={s['_type']} TARGET={s['_target']} LANG={s['_lang']}")
            print(s["messages"][1]["content"])
            print(s["messages"][2]["content"])

    # ── Generate 300-sample eval benchmark ──────────────────────────────────────
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
            "TITLE_contains": title_contains,   # model title must contain this substring
        },
        "_category": title_contains,
    }




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
            "TITLE_contains": title_contains,   # model title must contain this substring
        },
        "_category": title_contains,
    }


if __name__ == "__main__":
    main()


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
            "TITLE_contains": title_contains,   # model title must contain this substring
        },
        "_category": title_contains,
    }
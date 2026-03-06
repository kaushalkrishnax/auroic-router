const fs = require("fs");

const OUTPUT_FILE = "dataset.jsonl";

// ============================================================
// SCHEMA CONTRACT (from tools.json — DO NOT DEVIATE)
// ============================================================
//
// communicate:
//   operation: "text" | "voice" | "reaction" | "question"
//   tone:      "friendly" | "sarcastic" | "professional" | "angry" | "romantic" | "excited"
//   target:    string (optional)
//   content:   string
//
// media_action:
//   operation: "play" | "search" | "generate" | "send"
//   type:      "music" | "image" | "sticker" | "meme" | "video"
//   query:     string
//   tone:      string (free-form — mood/style, NOT an enum)
//
// retrieve_info:
//   operation: "search" | "lookup" | "analyze"
//   type:      "web" | "user" | "music" | "file"   ← "music" replaces "song" (one enum per concept)
//   query:     string
//   format:    "summary" | "full" | "lyrics" | "detailed"
//              ↑ format:"lyrics" + type:"music" = song lyrics lookup
//
// process_text:
//   operation: "translate" | "summarize" | "rewrite" | "compose"
//   content:   string
//   target:    string (target language or output target)
//   format:    "short" | "detailed" | "bullet" | "formal"
//   tone:      "friendly"|"sarcastic"|"professional"|"angry"|"romantic"|"excited"
//              ↑ ONLY valid on compose; omit for translate/summarize/rewrite
//
// manage_memory:
//   operation: "store" | "recall" | "forget"
//   content:   string
//
// ignore:
//   params: {}
//
// ============================================================

// ---------- DISTRIBUTION ----------
const DISTRIBUTION = {
  communicate:   900,
  media_action:  840,
  retrieve_info: 600,
  process_text:  360,
  manage_memory: 150,
  ignore:        150,
};

// ---------- VOCAB BANKS ----------
const names = [
  "rahul","neha","amit","john","sara","virat","rohit","priya","dev","ananya",
  "kabir","zara","arjun","meera","sam","tanya","aditya","pooja","rishi","natasha",
  "vikram","ishaan","sana","kartik","simran","harsh","divya","milan","sneha","akash",
];

const songs = [
  "Despacito","Shape of You","Perfect","Tum Hi Ho","Believer","Kesariya",
  "Blinding Lights","Levitating","Peaches","Stay","Dynamite","Bad Guy","Senorita",
  "Sunflower","Rockstar","Raataan Lambiyan","Apna Bana Le","Dil Chahta Hai",
  "Tera Yaar Hoon Main","Kho Gaye Hum Kahan","Calm Down","As It Was","Heat Waves",
  "Watermelon Sugar","Golden","Anti-Hero","Flowers","Unholy","Die For You",
  "Cupid","Vampire","Cruel Summer","Love Story",
];

const topics = [
  "cats","dogs","space","AI","football","movies","cricket","climate change",
  "bitcoin","machine learning","pizza","travel","meditation","startups","books",
  "gaming","fashion","cooking","fitness","anime","K-pop","history","politics",
  "science","music production","photography","yoga","mental health","stock market",
  "web development","cybersecurity","archaeology","astronomy",
];

// ---- STRICT ENUM: communicate.tone (6 values only) ----
const COMMUNICATE_TONES = [
  "friendly","sarcastic","professional","angry","romantic","excited",
];

// ---- communicate.tone with realistic weighted distribution ----
// neutral/casual users map to "friendly" — no leakage
function pickCommunicateTone() {
  const r = Math.random();
  if (r < 0.25) return "friendly";
  if (r < 0.40) return "excited";
  if (r < 0.53) return "romantic";
  if (r < 0.66) return "professional";
  if (r < 0.79) return "sarcastic";
  return "angry";
}

// ---- media_action.tone: 7-value set matching communicate.tone enum + neutral
// Weighted: neutral fires ~15% → ~120+ samples; others balanced
const MEDIA_TONES = [
  "neutral","neutral","neutral","neutral","neutral",   // ~25% weight → targets ~120+ across 840 * 0.45 tone samples
  "friendly","friendly",            // ~10%
  "romantic","romantic",            // ~10%
  "excited","excited",              // ~10%
  "angry",                           // ~5%
  "sarcastic",                       // ~5%
  "professional",                    // ~5%
];

// ---- process_text.tone: same enum as communicate (ONLY for compose) ----
const PROCESS_COMPOSE_TONES = COMMUNICATE_TONES;

const languages = [
  "Hindi","English","Spanish","French","German","Japanese","Marathi","Tamil",
  "Punjabi","Portuguese","Italian","Korean",
];

const retrieveFormats = ["summary","full","lyrics","detailed"];
const processFormats  = ["short","detailed","bullet","formal"];

// media_action.type enum: music|image|sticker|meme|video
const mediaTypes = ["music","image","sticker","meme","video"];

const fileTypes = [
  "pdf","image","note","contact","spreadsheet","document",
  "voice memo","chat backup","photo","video clip",
];

// ---------- UTILS ----------
function rand(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
function maybe(prob = 0.5) { return Math.random() < prob; }
function confidence() {
  const r = Math.random();
  if (r < 0.70) return +(0.95 + Math.random() * 0.05).toFixed(2);
  if (r < 0.90) return +(0.85 + Math.random() * 0.09).toFixed(2);
  return +(0.70 + Math.random() * 0.14).toFixed(2);
}

// ---------- COMMUNICATE TEMPLATES ----------
const textTemplates = [
  (name, tone) => `drop a ${tone} hi to ${name}`,
  (name, tone) => `text ${name} something ${tone}`,
  (name)       => `ping ${name} real quick`,
  (name)       => `msg ${name} bhai`,
  (name)       => `send a message to ${name}`,
  (name)       => `${name} ko ek msg bhejo`,
  (name)       => `tell ${name} i'll be late`,
  (name)       => `let ${name} know i'm on my way`,
  (name)       => `bro text ${name} for me`,
  (name, tone) => `${name} ko ${tone} wala msg bhejo`,
  (name)       => `quickly ping ${name}`,
  (name)       => `send ${name} a hey`,
  (name)       => `hmu to ${name}`,
  (name)       => `drop ${name} a line`,
  (name)       => `text ${name} rn`,
  (name)       => `${name} ko bol dena coming soon`,
  (name)       => `forward msg to ${name}`,
  (name)       => `message ${name} asap`,
  (name)       => `${name} bhai ko ek baar message kar`,
  (name)       => `slide into ${name}'s dms`,
];

const textContents = [
  "Hello! 👋","Hey, what's up?","Yo yo!","Sup bro","Hey!! Miss you 😊",
  "Heyy, free ho kya?","Kya scene hai?","Bhai call kar","Just checking in",
  "You good?","Oye hoye!","Heyyy how are you doing?","Long time no see!",
  "Thinking of you 💭","Call me when you can","You free tonight?",
  "Bhai kahan ho?","Miss you yaar","Wanna hang?","All good?",
];

const voiceContents = [
  "Hey I am busy right now, will call you back",
  "Bhai abhi baad mein baat karte hain",
  "On my way, be there in 10",
  "Can't talk, in a meeting",
  "Yaar baad mein baat karte hain",
  "Just leaving home, give me 20 mins",
  "In the middle of something, will ping you",
  "Driving rn, call you back",
  "Bro just woke up, give me a sec",
  "Out for a run, text me",
  "Eating dinner, hmu later",
  "Bhai abhi class chal raha hai",
  "Working from home, kinda busy",
  "On a call, ping me after",
];

const voiceTemplates = [
  (name) => `voice note bhejo ${name} ko`,
  (name) => `${name} ko voice message kar`,
  (name) => `send a voice note to ${name}`,
  (name) => `record a message for ${name}`,
  (name) => `${name} ko ek voice bhejo saying i'm busy`,
  (name) => `voice msg ${name}`,
  (name) => `audio message ${name}`,
  ()     => `send voice note`,
  (name) => `${name} ko bolna hai voice mein`,
  (name) => `leave a voice message for ${name}`,
];

const questionTemplates = [
  (name) => `ask ${name} what they're up to`,
  (name) => `${name} ko pucho kya kar rahe hain`,
  (name) => `question ${name} about their plans`,
  (name) => `check with ${name} if they're free`,
  (name) => `${name} se pooch are they coming`,
  ()     => `ask what's the plan for tonight`,
  (name) => `${name} ko pucho kab free hoge`,
  (name) => `ask ${name} for their opinion`,
  (name) => `${name} ke baare mein pooch`,
  ()     => `what are you doing tonight? send this`,
];

const questionContents = [
  "Kya kar rahe ho?","Tum kya kar rahe ho?","Plans kya hain?","Free ho aaj?",
  "Kab miloge?","Kya scene hai aaj?","What are you up to?","You free today?",
  "Coming tonight?","What do you think?","Suggestions dena yaar","Bata kya plan hai",
  "Game kheloge aaj?","Dinner pe milte hain?","Movie dekhne chaloge?",
];

const reactionTemplates = [
  () => `react with 😂 on their message`,
  () => `send a haha`,
  () => `heart react karo`,
  () => `thumbs up bhejo`,
  () => `react 🔥 on this`,
  () => `send a 😍 to that`,
  () => `react with 🫡`,
  () => `give a 👀 reaction`,
  () => `reply with lol`,
  () => `send 😭 as reaction`,
  () => `react 🙏 karo`,
  () => `drop a 💀 reaction`,
];

const reactionContents = [
  "😂","❤️","👍","🔥","😍","🫡","👀","😭","🙏","💀",
  "🤣","😮","😱","🥹","🤌","✨","💯",
];

// ---------- MEDIA TEMPLATES ----------
const playMusicTemplates = [
  (song) => `play ${song}`,
  (song) => `chhado yaar ${song} lgao`,
  (song) => `${song} bajao na`,
  (song) => `put on ${song}`,
  (song) => `play ${song} please`,
  (song) => `${song} sunna hai`,
  (song) => `queue up ${song}`,
  (song) => `start playing ${song}`,
  (song) => `${song} chala do`,
  (song) => `bro play ${song} rn`,
  (song) => `next song should be ${song}`,
  (song) => `add ${song} to queue`,
  (song) => `play me ${song}`,
  (song) => `play ${song} on full volume`,
];

const generateImageTemplates = [
  (topic) => `generate image of ${topic}`,
  (topic) => `make a pic of ${topic}`,
  (topic) => `create an image about ${topic}`,
  (topic) => `generate me a visual for ${topic}`,
  (topic) => `ai image banao ${topic} ka`,
  (topic) => `make a cool image of ${topic}`,
  (topic) => `create a ${topic} image`,
  (topic) => `design an image about ${topic}`,
  (topic) => `make a realistic image of ${topic}`,
  (topic) => `generate a cute image of ${topic}`,
  (topic) => `create a digital art for ${topic}`,
  (topic) => `paint me a picture of ${topic}`,
];

const searchMediaTemplates = [
  (topic) => `find ${topic} meme`,
  (topic) => `search ${topic} video`,
  (topic) => `find a funny ${topic} reel`,
  (topic) => `${topic} ka koi video dhundo`,
  (topic) => `look up ${topic} content`,
  (topic) => `search for ${topic} clips`,
  (topic) => `find ${topic} trending reel`,
  (topic) => `ek ${topic} gif dhundo`,
  (topic) => `find me a ${topic} sticker`,
  (topic) => `look for ${topic} highlight`,
];

const sendMediaTemplates = [
  (topic) => `send ${topic} gif`,
  (topic) => `bhejo ek ${topic} meme`,
  (topic) => `share ${topic} sticker`,
  (topic) => `forward that ${topic} video`,
  (topic) => `${topic} ka sticker bhejo`,
  (topic) => `send a ${topic} reel`,
  (topic) => `${topic} related image share karo`,
  (topic) => `drop a ${topic} meme`,
  (topic) => `send that ${topic} clip`,
];

// ---------- RETRIEVE INFO TEMPLATES ----------
const webSearchTemplates = [
  (topic) => `tell me about ${topic}`,
  (topic) => `${topic} ke baare mein bata`,
  (topic) => `google kar ${topic}`,
  (topic) => `search karo ${topic}`,
  (topic) => `what is ${topic}`,
  (topic) => `explain ${topic} to me`,
  (topic) => `latest news on ${topic}`,
  (topic) => `${topic} kya hota hai`,
  (topic) => `find info on ${topic}`,
  (topic) => `give me details about ${topic}`,
  (topic) => `${topic} pe kuch bata`,
  (topic) => `research ${topic} for me`,
  (topic) => `${topic} news kya hai`,
  (topic) => `kuch interesting ${topic} ke baare mein`,
  (topic) => `facts about ${topic}`,
  (topic) => `${topic} summary de do`,
  (topic) => `quick recap of ${topic}`,
  (topic) => `what's happening with ${topic}`,
  (topic) => `break down ${topic} for me`,
  (topic) => `ek dum basic mein ${topic} explain karo`,
];

const lyricsSearchTemplates = [
  (song) => `lyrics dhundo ${song} ke`,
  (song) => `${song} ke words kya hain`,
  (song) => `what are the lyrics of ${song}`,
  (song) => `find me lyrics for ${song}`,
  (song) => `${song} song ke lyrics chahiye`,
  (song) => `get me the words to ${song}`,
  (song) => `${song} lyrics`,
  (song) => `look up ${song} lyrics`,
];

const userLookupTemplates = [
  (name) => `${name} ke baare mein kya pata hai`,
  (name) => `look up ${name}'s profile`,
  (name) => `find info about ${name}`,
  (name) => `${name} ka number dhundo`,
  (name) => `who is ${name}?`,
  (name) => `search ${name} contact`,
  (name) => `${name} ki detail nikalo`,
];

const fileSearchTemplates = [
  (type) => `find my recent ${type}`,
  (type) => `search for ${type} in my files`,
  (type) => `wo ${type} dhundo jo maine bheja tha`,
  (type) => `locate that ${type} I saved`,
  (type) => `${type} kahan gaya mera`,
  (type) => `find old ${type}`,
  (type) => `retrieve my ${type} from last week`,
];

// ---------- PROCESS TEXT TEMPLATES ----------
const translateTemplates = [
  (lang) => `translate this to ${lang}`,
  (lang) => `${lang} mein translate karo`,
  (lang) => `isko ${lang} mein kar do`,
  (lang) => `convert to ${lang}`,
  (lang) => `${lang} translation chahiye`,
  (lang) => `make this ${lang}`,
  (lang) => `turn this into ${lang}`,
  (lang) => `${lang} version dedo`,
  (lang) => `translate in ${lang} please`,
  ()     => `translate this`,
  ()     => `iska translation karo`,
];

const translateContents = [
  "I am feeling happy today",
  "Can we meet tomorrow?",
  "The weather is really nice outside",
  "I miss you so much",
  "Let's go for a walk",
  "I need help with this project",
  "Thank you so much for everything",
  "I don't understand what you're saying",
  "Please send me the details",
  "She's my best friend",
  "We should talk about this",
  "I'm running a bit late",
  "This is absolutely amazing",
  "Are you coming to the party?",
  "I haven't slept properly in days",
];

const summarizeTemplates = [
  () => `summarize this text`,
  () => `iska summary de do`,
  () => `short mein bata`,
  () => `condense this`,
  () => `TL;DR dedo`,
  () => `give me the gist`,
  () => `what's the main point?`,
  () => `brief summary chahiye`,
  () => `summarize in 2 lines`,
  () => `short kar do ye`,
  () => `summarize it for me`,
  () => `what does this say?`,
  () => `quick summary please`,
];

const summarizeContents = [
  "Climate change is a global emergency requiring immediate action from governments and individuals alike. Rising temperatures, melting ice caps, and extreme weather patterns are just a few consequences of unchecked carbon emissions.",
  "Machine learning is a subset of artificial intelligence where algorithms learn from data to make predictions or decisions without being explicitly programmed.",
  "The Indian economy has been growing steadily at around 6-7% GDP annually. Sectors like IT, manufacturing and services have been major contributors.",
  "Recent studies show that regular exercise improves mental health significantly. Even 30 minutes of moderate activity 5 days a week makes a measurable difference.",
  "The startup ecosystem in India has exploded in recent years with Bengaluru, Mumbai and Delhi NCR emerging as major tech hubs attracting billions in venture capital.",
];

const rewriteTemplates = [
  (fmt) => `rewrite this in ${fmt} style`,
  (fmt) => `make this more ${fmt}`,
  (fmt) => `${fmt} style mein likhdo`,
  ()    => `rewrite this nicely`,
  ()    => `isko better kar do`,
  ()    => `improve this text`,
  ()    => `make this sound professional`,
  ()    => `rephrase this`,
  ()    => `rewrite this to sound friendlier`,
  ()    => `clean up this writing`,
  ()    => `polish this text`,
  ()    => `fix the tone of this`,
  ()    => `make this flow better`,
];

const rewriteContents = [
  "i cant come today sorry",
  "the meeting was not good",
  "i think your idea is bad",
  "please help me now",
  "this is taking too long",
  "the food was okay",
  "i don't like this",
  "can you do this for me please thanks",
  "tell me when you're done",
  "i was late because of traffic",
];

const composeTemplates = [
  (fmt)  => `write a ${fmt} message`,
  (fmt)  => `compose a ${fmt} note`,
  ()     => `write something cool`,
  ()     => `ek accha sa message likh`,
  (fmt)  => `draft a ${fmt} email`,
  ()     => `compose a birthday wish`,
  ()     => `write an apology message`,
  ()     => `likhdo ek motivation quote`,
  ()     => `write a thank you note`,
  ()     => `compose a leave application`,
  (fmt)  => `write a ${fmt} caption for my post`,
  ()     => `write something funny`,
  ()     => `ek romantic message likh do`,
  ()     => `write a professional bio`,
  ()     => `draft an introduction message`,
];

const composeContents = [
  "write a birthday wish","compose apology note","draft thank you message",
  "write a leave request","make a professional introduction",
  "write a motivation message","compose a goodbye note",
  "draft a complaint email","write a congrats message","compose a cover letter",
];

// ---------- MEMORY TEMPLATES ----------
const storeMemoryTemplates = [
  (fact) => `remember that ${fact}`,
  (fact) => `save this: ${fact}`,
  (fact) => `note kar lo ${fact}`,
  (fact) => `don't forget: ${fact}`,
  (fact) => `store this info: ${fact}`,
  (fact) => `keep in mind that ${fact}`,
  (fact) => `add to memory: ${fact}`,
  (fact) => `ek note rakh lo: ${fact}`,
  (fact) => `mark this: ${fact}`,
  (fact) => `jot down: ${fact}`,
];

const userFacts = [
  "I like pizza","my birthday is on 15th August","I prefer tea over coffee",
  "I'm allergic to peanuts","I work from 9 to 5","my favourite colour is blue",
  "I hate mornings","I'm a night owl","I love Bollywood music",
  "I have a dog named Bruno","I'm vegetarian","my mom's name is Sunita",
  "I like action movies","I wake up at 6am daily","I'm learning guitar",
  "I recently moved to Bangalore","I prefer WhatsApp over calls",
  "my favourite food is biryani","I go to the gym 3 times a week",
  "I'm currently reading Atomic Habits","I have a sister named Priya",
  "I prefer cold weather","I dislike crowded places",
  "I save 20% of my income monthly","I binge watch anime on weekends",
];

const recallTemplates = [
  () => `what do you remember about me?`,
  () => `kya pata hai tumhe mere baare mein?`,
  () => `recall my preferences`,
  () => `what have I told you before?`,
  () => `mujhe kya pasand tha bata`,
  () => `remind me what you know`,
  () => `fetch my saved info`,
  () => `what's in your memory about me`,
  () => `what did I tell you last time?`,
  () => `retrieve my preferences`,
  () => `bata kya kya yaad hai`,
  () => `what do you know about my likes?`,
];

const forgetTemplates = [
  (fact) => `forget ${fact}`,
  (fact) => `delete memory of ${fact}`,
  (fact) => `${fact} wala note hatao`,
  (fact) => `remove ${fact} from memory`,
  (fact) => `clear my ${fact} preference`,
  (fact) => `erase ${fact} from notes`,
  (fact) => `unremember ${fact}`,
  (fact) => `${fact} bhool jao`,
];

const forgetFacts = [
  "my pizza preference","my birthday","my coffee order","my work schedule",
  "my food allergy","my favourite colour","all my preferences",
  "my personal info","my daily routine","what I told you",
];

// ---------- IGNORE TEMPLATES ----------
const ignoreInputs = [
  "asdfgh","....","k","hmm","???","wait","lol","ok","ugh","meh",
  "brb","gtg","💀","👀","😶","🙃","ok bye","hm","no","ye","na","wut","huh",
  "hm ok","sure i guess","whatever","...ok","mm","ah","ohh","ugh ok","nvm",
  "nm","lmao k","aight","bro","yolo","bbye","ok fine","nothing",
  "no response needed","just testing","test","hello?","anyone there",
  "kuch nahi","chodo yaar","baat nahi","bye","thik hai","ok done",
  "😂😂😂","💯","👍👍","🔥🔥","xD",":)",":/","XD ok",
];

// ============================================================
// SAMPLE BUILDERS — strict schema enforcement
// ============================================================

function buildCommunicate() {
  const operation = rand(["text","voice","reaction","question"]);
  const name = rand(names);
  const tone = pickCommunicateTone(); // always valid enum value

  // Build params — only include valid schema fields
  const params = { operation, content: "" };
  if (maybe(0.60)) params.target = name;
  if (maybe(0.65)) params.tone = tone;

  let user;

  if (operation === "text") {
    const tpl = rand(textTemplates);
    user = tpl(name, tone);
    params.content = rand(textContents);
  } else if (operation === "voice") {
    const tpl = rand(voiceTemplates);
    user = tpl(name);
    params.content = rand(voiceContents);
  } else if (operation === "reaction") {
    const tpl = rand(reactionTemplates);
    user = tpl();
    params.content = rand(reactionContents);
    // reactions rarely need a tone
    if (maybe(0.3) && params.tone) delete params.tone;
  } else {
    const tpl = rand(questionTemplates);
    user = tpl(name);
    params.content = rand(questionContents);
  }

  return { user, action: "communicate", params };
}

function buildMedia() {
  // play weighted 3x to keep music dominant
  const operation = rand(["play","play","play","search","generate","send"]);

  // SCHEMA: type enum = ["music","image","sticker","meme","video"]
  // When playing, type MUST be "music"
  // For other operations: pick from full mediaTypes enum
  const type = operation === "play" ? "music" : rand(mediaTypes);

  const params = { operation, type, query: "" };

  // tone uses MEDIA_TONES (7-value set aligned with communicate.tone enum)
  if (maybe(0.45)) params.tone = rand(MEDIA_TONES);

  // NOTE: NO "mood" field — not in schema
  // NOTE: type is never "song" in media_action — song lookup uses retrieve_info type:"music" + operation:"lookup"

  let user;

  if (operation === "play") {
    const song = rand(songs);
    params.query = song;
    const tpl = rand(playMusicTemplates);
    user = tpl(song);
  } else if (operation === "generate") {
    const t = rand(topics);
    params.query = t;
    const tpl = rand(generateImageTemplates);
    user = tpl(t);
  } else if (operation === "search") {
    const t = rand(topics);
    params.query = t;
    const tpl = rand(searchMediaTemplates);
    user = tpl(t);
  } else { // send
    const t = rand(topics);
    params.query = t;
    const tpl = rand(sendMediaTemplates);
    user = tpl(t);
  }

  return { user, action: "media_action", params };
}

function buildRetrieve() {
  // SCHEMA: type enum = ["web","user","music","file"]
  // type:"music" + operation:"lookup" + format:"lyrics" = song lyrics lookup
  const retrieveType = rand(["web","web","music","user","file"]); // web weighted 2x
  const params = { operation: "search", type: retrieveType };

  if (maybe(0.55)) params.format = rand(retrieveFormats);

  let user;

  if (retrieveType === "web") {
    const topic = rand(topics);
    params.query = topic;
    if (maybe(0.15)) {
      // ~15% of web queries use operation "analyze"
      params.operation = "analyze";
      params.format = "detailed";
      const analyzeTpls = [
        (t) => `analyze this ${t} situation`,
        (t) => `deep dive into ${t} for me`,
        (t) => `do a full analysis of ${t}`,
        (t) => `${t} ka thorough breakdown do`,
        (t) => `analyze ${t} trends`,
        (t) => `run an analysis on ${t}`,
        (t) => `give me a detailed analysis of ${t}`,
      ];
      user = rand(analyzeTpls)(topic);
    } else {
      const tpl = rand(webSearchTemplates);
      user = tpl(topic);
    }
  } else if (retrieveType === "music") {
    // Music lookup = get lyrics or info about a song
    // type:"music" + operation:"lookup" — format:"lyrics" distinguishes from play
    const song = rand(songs);
    params.query = song;
    params.operation = "lookup";
    if (maybe(0.7)) params.format = "lyrics"; else params.format = "detailed";
    const tpl = rand(lyricsSearchTemplates);
    user = tpl(song);
  } else if (retrieveType === "user") {
    const name = rand(names);
    params.query = name;
    params.operation = "lookup";
    const tpl = rand(userLookupTemplates);
    user = tpl(name);
  } else { // file
    const ftype = rand(fileTypes);
    params.query = ftype;
    params.operation = "analyze";
    const tpl = rand(fileSearchTemplates);
    user = tpl(ftype);
  }

  return { user, action: "retrieve_info", params };
}

function buildProcess() {
  const operation = rand(["translate","summarize","rewrite","compose"]);
  const params = { operation };

  let user;

  if (operation === "translate") {
    const lang = rand(languages);
    params.target = lang;      // target = language name
    params.content = rand(translateContents);
    if (maybe(0.5)) params.format = rand(processFormats);
    // NO tone on translate — not semantically valid
    const tpl = rand(translateTemplates);
    user = tpl(lang);

  } else if (operation === "summarize") {
    params.content = rand(summarizeContents);
    if (maybe(0.5)) params.format = rand(processFormats);
    // NO tone on summarize
    const tpl = rand(summarizeTemplates);
    user = tpl();

  } else if (operation === "rewrite") {
    const fmt = rand(processFormats);
    params.content = rand(rewriteContents);
    params.format = fmt;
    // NO tone on rewrite — format carries the style signal
    const tpl = rand(rewriteTemplates);
    user = tpl(fmt);

  } else { // compose
    const fmt = rand(processFormats);
    params.content = rand(composeContents);
    params.format = fmt;
    // tone IS valid on compose (user may specify style)
    if (maybe(0.45)) params.tone = rand(PROCESS_COMPOSE_TONES);
    const tpl = rand(composeTemplates);
    user = tpl(fmt);
  }

  return { user, action: "process_text", params };
}

function buildMemory() {
  const operation = rand(["store","recall","forget"]);
  const params = { operation };

  let user;

  if (operation === "store") {
    const fact = rand(userFacts);
    params.content = fact;
    const tpl = rand(storeMemoryTemplates);
    user = tpl(fact);
  } else if (operation === "recall") {
    params.content = "user preferences";
    const tpl = rand(recallTemplates);
    user = tpl();
  } else { // forget
    const fact = rand(forgetFacts);
    params.content = fact;
    const tpl = rand(forgetTemplates);
    user = tpl(fact);
  }

  return { user, action: "manage_memory", params };
}

function buildIgnore() {
  return { user: rand(ignoreInputs), action: "ignore", params: {} };
}

// ---------- EDGE CASES (hand-crafted, schema-verified) ----------
const edgeCaseUserInputs = [
  {
    user: "bhai kal kya scene hai? rahul ko bhi bata dena",
    action: "communicate",
    params: { operation: "text", content: "Kal kya scene hai?", target: "rahul", tone: "friendly" },
  },
  {
    user: "Shape of You bajao aur lights dim karo",
    action: "media_action",
    params: { operation: "play", type: "music", query: "Shape of You" },
  },
  {
    user: "isko hindi mein translate karo phir neha ko bhejo",
    action: "process_text",
    params: { operation: "translate", target: "Hindi", content: "Please translate this message" },
  },
  {
    user: "sunflower lyrics dhundo aur mujhe bhejo",
    action: "retrieve_info",
    params: { operation: "lookup", type: "music", query: "Sunflower", format: "lyrics" },
  },
  {
    user: "AI ke baare mein kuch interesting bata",
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: "AI", format: "summary" },
  },
  {
    user: "birthday wish likhke priya ko bhejo",
    action: "process_text",
    params: { operation: "compose", content: "write a birthday wish", format: "short", tone: "friendly" },
  },
  {
    user: "remember i hate mornings",
    action: "manage_memory",
    params: { operation: "store", content: "user hates mornings" },
  },
  {
    user: "yaar despacito bajao please mood hai",
    action: "media_action",
    params: { operation: "play", type: "music", query: "Despacito", tone: "romantic" },
  },
  {
    user: "this is too long, short kar do",
    action: "process_text",
    params: { operation: "summarize", content: "provided text", format: "short" },
  },
  {
    user: "forget my food preferences",
    action: "manage_memory",
    params: { operation: "forget", content: "food preferences" },
  },
  {
    user: "😂😂 yaar ye sun",
    action: "communicate",
    params: { operation: "reaction", content: "😂" },
  },
  {
    user: "amit ko voice note bhejo i'll be late for meeting",
    action: "communicate",
    params: { operation: "voice", content: "I'll be late for meeting", target: "amit" },
  },
  {
    user: "make a cool wallpaper of space",
    action: "media_action",
    params: { operation: "generate", type: "image", query: "space" },
  },
  {
    user: "translate this to French and make it formal",
    action: "process_text",
    params: { operation: "translate", target: "French", content: "provided text", format: "formal" },
  },
  {
    user: "search for cricket highlights from yesterday",
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: "cricket", format: "detailed" },
  },
  {
    user: "john ko romantic message bhejo",
    action: "communicate",
    params: { operation: "text", content: "Thinking of you 💭", target: "john", tone: "romantic" },
  },
  {
    user: "anti hero ki lyrics chahiye",
    action: "retrieve_info",
    params: { operation: "lookup", type: "music", query: "Anti-Hero", format: "lyrics" },
  },
  {
    user: "mujhe kitna kuch yaad hai tumhe mere baare mein?",
    action: "manage_memory",
    params: { operation: "recall", content: "user preferences" },
  },
  {
    user: "kabir ka number dhundo",
    action: "retrieve_info",
    params: { operation: "lookup", type: "user", query: "kabir" },
  },
  {
    user: "send a cricket meme to virat",
    action: "media_action",
    params: { operation: "send", type: "meme", query: "cricket" },
  },
];

// ============================================================
// SCHEMA VALIDATOR — catch any violation before writing
// ============================================================
const VALID_ACTIONS = new Set(["communicate","media_action","retrieve_info","process_text","manage_memory","ignore"]);
const COMMUNICATE_OPS = new Set(["text","voice","reaction","question"]);
const COMMUNICATE_TONE_SET = new Set(["friendly","sarcastic","professional","angry","romantic","excited"]);
const MEDIA_OPS = new Set(["play","search","generate","send"]);
const MEDIA_TYPES = new Set(["music","image","sticker","meme","video"]);
const RETRIEVE_OPS = new Set(["search","lookup","analyze"]);
const RETRIEVE_TYPES = new Set(["web","user","music","file"]);
const RETRIEVE_FORMATS = new Set(["summary","full","lyrics","detailed"]);
const PROCESS_OPS = new Set(["translate","summarize","rewrite","compose"]);
const PROCESS_FORMATS = new Set(["short","detailed","bullet","formal"]);
const MEMORY_OPS = new Set(["store","recall","forget"]);

function validateSample(obj) {
  const inner = JSON.parse(obj.messages[1].content);
  const { action, params } = inner;
  const errors = [];

  if (!VALID_ACTIONS.has(action)) {
    errors.push(`Invalid action: ${action}`);
    return errors;
  }

  if (action === "communicate") {
    if (!COMMUNICATE_OPS.has(params.operation))
      errors.push(`communicate.operation invalid: ${params.operation}`);
    if (params.tone && !COMMUNICATE_TONE_SET.has(params.tone))
      errors.push(`communicate.tone invalid: ${params.tone}`);
    if (params.type !== undefined)
      errors.push(`communicate should not have .type`);
  }

  if (action === "media_action") {
    if (!MEDIA_OPS.has(params.operation))
      errors.push(`media_action.operation invalid: ${params.operation}`);
    if (!MEDIA_TYPES.has(params.type))
      errors.push(`media_action.type invalid: ${params.type}`);
    if (params.format !== undefined)
      errors.push(`media_action should not have .format`);
    if (params.mood !== undefined)
      errors.push(`media_action should not have .mood (use .tone)`);
  }

  if (action === "retrieve_info") {
    if (!RETRIEVE_OPS.has(params.operation))
      errors.push(`retrieve_info.operation invalid: ${params.operation}`);
    if (!RETRIEVE_TYPES.has(params.type))
      errors.push(`retrieve_info.type invalid: ${params.type}`);
    if (params.format && !RETRIEVE_FORMATS.has(params.format))
      errors.push(`retrieve_info.format invalid: ${params.format}`);
    if (params.tone !== undefined)
      errors.push(`retrieve_info should not have .tone`);
  }

  if (action === "process_text") {
    if (!PROCESS_OPS.has(params.operation))
      errors.push(`process_text.operation invalid: ${params.operation}`);
    if (params.format && !PROCESS_FORMATS.has(params.format))
      errors.push(`process_text.format invalid: ${params.format}`);
    if (params.tone && !COMMUNICATE_TONE_SET.has(params.tone))
      errors.push(`process_text.tone invalid: ${params.tone}`);
    // tone only allowed on compose
    if (params.tone && params.operation !== "compose")
      errors.push(`process_text.tone only valid on compose, got: ${params.operation}`);
  }

  if (action === "manage_memory") {
    if (!MEMORY_OPS.has(params.operation))
      errors.push(`manage_memory.operation invalid: ${params.operation}`);
  }

  return errors;
}

// ============================================================
// SAMPLE CREATION
// ============================================================
function createSample(tool) {
  let sample;
  switch (tool) {
    case "communicate":   sample = buildCommunicate(); break;
    case "media_action":  sample = buildMedia();       break;
    case "retrieve_info": sample = buildRetrieve();    break;
    case "process_text":  sample = buildProcess();     break;
    case "manage_memory": sample = buildMemory();      break;
    case "ignore":        sample = buildIgnore();      break;
  }
  return {
    messages: [
      { role: "user", content: sample.user },
      {
        role: "assistant",
        content: JSON.stringify({
          action: sample.action,
          params: sample.params,
          confidence: confidence(),
        }),
      },
    ],
  };
}

// ============================================================
// MAIN
// ============================================================
function main() {
  const lines = [];

  for (const [tool, count] of Object.entries(DISTRIBUTION)) {
    for (let i = 0; i < count; i++) {
      lines.push(JSON.stringify(createSample(tool)));
    }
  }

  // Edge cases — loop to ~60 samples
  for (let i = 0; i < 60; i++) {
    const sample = rand(edgeCaseUserInputs);
    lines.push(JSON.stringify({
      messages: [
        { role: "user", content: sample.user },
        {
          role: "assistant",
          content: JSON.stringify({
            action: sample.action,
            params: sample.params,
            confidence: confidence(),
          }),
        },
      ],
    }));
  }

  // Load conflict samples
  let conflictLines = [];
  try {
    conflictLines = fs.readFileSync("conflict_samples.jsonl","utf8").split("\n").filter(Boolean);
  } catch {
    console.log("⚠️  conflict_samples.jsonl not found, skipping.");
  }
  lines.push(...conflictLines);

  // Shuffle
  for (let i = lines.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [lines[i], lines[j]] = [lines[j], lines[i]];
  }

  // Validate every sample — collect and report violations
  let violations = 0;
  const validatedLines = lines.map((l, idx) => {
    const obj = JSON.parse(l);
    const errors = validateSample(obj);
    if (errors.length > 0) {
      violations++;
      console.error(`[LINE ${idx}] SCHEMA VIOLATION:`);
      errors.forEach((e) => console.error(`  → ${e}`));
      console.error(`  Sample: ${l.slice(0, 120)}`);
    }
    return l;
  });

  fs.writeFileSync(OUTPUT_FILE, validatedLines.join("\n"));

  // ---- DIAGNOSTICS ----
  const stats = {};
  const toneCount = {};
  const typeCount = {};
  let totalTokens = 0;
  let confSum = 0;
  let confCount = 0;
  let targetCount = 0;

  for (const l of validatedLines) {
    const obj = JSON.parse(l);
    const inner = JSON.parse(obj.messages[1].content);
    const a = inner.action;
    const p = inner.params || {};
    const op = p.operation || "-";
    const typ = p.type || "-";
    const key = `${a}|${op}|${typ}`;
    stats[key] = (stats[key] || 0) + 1;

    if (p.tone) toneCount[p.tone] = (toneCount[p.tone] || 0) + 1;
    if (p.type) typeCount[p.type] = (typeCount[p.type] || 0) + 1;
    if (p.target) targetCount++;

    const userLen = obj.messages[0].content.length;
    totalTokens += Math.ceil(userLen / 4);

    if (inner.confidence) { confSum += inner.confidence; confCount++; }
  }

  console.log(`\n✅ Dataset generated: ${OUTPUT_FILE}`);
  console.log(`📊 Total samples  : ${validatedLines.length}`);
  console.log(`🔢 Avg user tokens: ${(totalTokens / validatedLines.length).toFixed(1)}`);
  console.log(`📈 Avg confidence : ${(confSum / confCount).toFixed(3)}`);
  console.log(`🎯 Target present : ${targetCount} (${((targetCount / validatedLines.length) * 100).toFixed(1)}%)`);
  console.log(`🚨 Schema violations: ${violations}`);

  console.log(`\n--- Action × Operation × Type ---`);
  const sorted = Object.entries(stats).sort((a, b) => b[1] - a[1]);
  for (const [k, v] of sorted) console.log(`  ${k.padEnd(45)}: ${v}`);

  console.log(`\n--- Tone distribution ---`);
  for (const [k, v] of Object.entries(toneCount).sort((a, b) => b[1] - a[1]))
    console.log(`  ${k.padEnd(16)}: ${v}`);

  console.log(`\n--- Type distribution ---`);
  for (const [k, v] of Object.entries(typeCount).sort((a, b) => b[1] - a[1]))
    console.log(`  ${k.padEnd(16)}: ${v}`);
}

main();

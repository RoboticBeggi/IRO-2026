"""Talking with ARC: the conversation features (24 Oct), used by iro/talk.py.

 1. Interrupt by talking: speak over ARC and it stops and listens. Its own voice comes back into the
    mic from the HDMI speaker (no echo cancelling there), so an interruption must pass every check:
    speech by the VAD, louder than ARC's own echo should be, and the words heard are NOT the words
    ARC is saying right now (`judge`). Noise, coughs and ARC's echo do not interrupt.
 2. Polite ducking: the moment it might be you, ARC's voice drops to a third; if it was only noise,
    the voice comes back up within a second.
 3. Stop words: "stop", "wait", "hold on", "quiet"... stop ARC at once, and it just listens.
 4. "Go on" / "continue": after an interruption ARC carries on from the sentence it was cut in.
 5. "What?" / "say that again": the last reply again, instantly (from its recording, no GPU).
 6. "Louder" / "softer": the volume changes and is remembered (config/voice.yaml).
 7. "Keep it short" / "more detail": shorter or longer answers, remembered.
 8. "Heard you" tick: a soft click as soon as ARC has understood you (voice.yaml earcon: false = off).
 9. Noise and filler turns are ignored ("uh", "hmm", one stray word) instead of answered.
10. Sleep and wake by name (owner): "ARC, go to sleep" = ARC ignores everything until it hears its
    name ("ARC", "hey ARC", "wake up"). Visitors cannot put it to sleep.

And for the owner: "ARC, take a note: ..." / "ARC, remember that ..." writes the order to
arc/owner_notes.md (dated), so improvements asked for by voice are not lost. `iro arc notes` lists them.
"""

import re
import time

from iro import ROOT

NOTES = ROOT / "arc" / "owner_notes.md"

STOP_PHRASES = re.compile(r"\b(stop|wait|hold on|hang on|quiet|shush|shh+|enough|pause|silence|be quiet|"
                          r"one (moment|second|sec)|okay okay|berhenti)\b", re.I)
FILLERS = set("uh um umm uhm hmm hm mm mhm ah oh er erm huh eh ahem".split())
SHORT_OK = set("yes no yeah yep nope okay ok hello hi hey thanks thank bye goodbye sure please stop wait "
               "what why how who pardon again more continue louder softer".split())

COMMANDS = [   # (kind, pattern, owner only) - matched on the whole (short) utterance
    ("repeat", r"^(what|pardon( me)?|sorry|come again|say (that|it) again|repeat( that| it)?( please)?|again|"
               r"what did you say|i didn'?t (catch|hear) (that|you)|one more time)( please)?\W*$", False),
    # "please continue explaining", "go on with the story", "pick up where you left off", "where were we?"
    ("go_on", r"^((please|okay|ok|yes|yeah|so|sorry|now|arc)[ ,]+)*(can you |could you |would you )?"
              r"((go on|continue|carry on|keep (going|explaining|talking|telling)|resume|proceed)\b.*|"
              r"(go ahead|as you were( saying)?|you were saying|where were we|finish (what you were saying|"
              r"the story|explaining)|pick up where you (left off|stopped)|(where|from where) you (left off|stopped))"
              r"\b.*)$", False),
    ("louder", r"\b(louder|speak up|turn (it )?up|volume up|i can'?t hear you)\b", False),
    ("softer", r"\b(quieter|softer|too loud|turn (it )?down|volume down|lower your voice)\b", False),
    ("shorter", r"\b(keep (it|them) short|shorter( answers)?|be brief|too long|short answers)\b", False),
    ("longer", r"\b(more detail(s)?|in detail|longer answers|elaborate)\b", False),
    ("sleep", r"\b(go to sleep|sleep mode|stand ?by|take a break|go (and )?rest)\b", True),
    ("note", r"^(please )?(take a note|make a note|note (that|this|down)|remember (that|this)|note to self|"
             r"add (this )?to (your|the) (to-?do|list))\b\W*(?P<arg>.*)$", True),
]
NAME = re.compile(r"^\W*(hey|hi|hello|okay|ok|oh)?\W*\b(arc|ark|arch)\b\W*", re.I)
WAKE = re.compile(r"\b(arc|ark|arch|wake up)\b", re.I)
LINES = {   # what ARC says back (pre-recorded at start-up, so they are instant)
    "louder": "Is that better?", "softer": "Is this better?", "shorter": "Short and sweet, then.",
    "longer": "I will go into more detail.", "sleep": "Going quiet. Say my name when you need me.",
    "wake": "I am here.", "wake_owner": "At your service, sir.", "note": "Noted, sir.",
    "no_repeat": "I have not said anything yet.", "no_go_on": "I had finished, actually.",
}


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def strip_name(text):
    """'ARC, tell me about...' -> 'tell me about...' (being addressed is not part of the question)."""
    out = NAME.sub("", text, count=1).strip()
    return out[:1].upper() + out[1:] if out else text


def command(text, owner):
    """(kind, arg) of a conversation command, or (None, None)."""
    t = strip_name(text).strip()
    if len(words(t)) > 12:
        return None, None
    for kind, pat, owner_only in COMMANDS:
        m = re.search(pat, t, re.I)
        if m and (owner or not owner_only):
            return kind, (m.groupdict().get("arg") or "").strip()
    return None, None


def is_stop(text):
    """A short "stop" / "wait" / "hold on": stop talking and listen (feature 3)."""
    return len(words(text)) <= 4 and bool(STOP_PHRASES.search(text))


EMPTY = set("i i'm said say saying mean meant so well just it me my you okay right like and but then "
            "that this was is".split())


def is_noise(text):
    """A turn not worth answering: only fillers, one stray word, or nothing with meaning ("I said.",
    24 Oct: it got a long made-up reply) (feature 9)."""
    w = [x for x in words(text) if x not in FILLERS]
    if not w:
        return True
    if any(x in SHORT_OK for x in w):
        return False
    if all(x in EMPTY for x in w):
        return True
    return len(w) == 1 and len(w[0]) < 4


FUNCTION = set("the and you can could would tell what about for with this that are is was who how why when "
               "where your there have has please just like then than not but from into our its it's his her "
               "she they them their we were had been i'm you're that's there's let's".split())
STOPS = set("stop wait hold hang quiet shush enough pause silence arc ark okay".split())


def _stem(w):
    return w[:5]


def judge(heard, spoken, min_new=2):
    """Something was heard while ARC talks: "stop", "interrupt" (a person said something new) or "echo"
    (ARC's own voice coming back, or noise). heard: its transcript; spoken: what ARC was saying."""
    if not heard.strip():
        return "echo"
    mine = {_stem(x) for x in words(spoken)}
    content = [x for x in words(heard) if len(x) > 2 and x not in FILLERS and x not in FUNCTION]
    new = [x for x in content if _stem(x) not in mine]
    beyond = [x for x in new if x not in STOPS]
    if is_stop(heard) and not beyond:
        return "stop"                              # "stop", "wait, wait": stop and listen
    overlap = 1 - len(new) / max(len(content), 1)
    # a name nobody was saying (Mozart, France) makes 2 new words enough: misheard echo comes out as
    # everyday words ("Our hips, stole")
    toks = re.findall(r"[A-Za-zÀ-ÿ']+", heard)
    named = [t for t in toks[1:] if t[0].isupper() and _stem(t.lower()) not in mine and t.lower() not in FUNCTION]
    need = min(min_new, 2) if named else min_new
    if (len(new) >= need and overlap < 0.6) or (is_stop(heard) and beyond):
        return "interrupt"                         # "wait, what about Japan?": a new question
    if command(heard, owner=False)[0] in ("repeat", "louder", "softer", "shorter", "longer"):
        return "interrupt"
    return "echo"


def add_note(text):
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    new = not NOTES.exists()
    with NOTES.open("a") as f:
        if new:
            f.write("# Orders and notes from the owner (said to ARC)\n\n")
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')}: {text.strip()}\n")


def cmd_notes():
    print(NOTES.read_text() if NOTES.exists() else "No notes yet. Say: \"ARC, take a note: ...\"")
    return 0

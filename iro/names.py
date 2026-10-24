"""Fix names the speech recogniser mishears, by how they SOUND (24 Oct, Moonshine: "Molocha
Sultanate" for Malacca, "Wei Wang Koolin" for wayang kulit).

    fix("Tell me about Wei Wang Koolin")  -> "Tell me about wayang kulit"

Each 1-3 word stretch of the transcript is compared with the known names (the knowledge cards'
names + the list below) by Soundex code (same consonant skeleton) AND spelling similarity; only
close matches are replaced, and common English words are never touched.
"""

import difflib
import re
from functools import lru_cache

EXTRA = ["Malacca", "Melaka", "Parameswara", "Iskandar Shah", "Hang Tuah", "Kelantan", "Terengganu", "Tok Dalang",
         "wayang kulit", "Mak Yong", "silat", "songket", "batik", "tepak sirih", "keris", "Hanuman", "Seri Rama",
         "Siti Dewi", "Maharaja Wana", "Pak Dogol", "Wak Long", "Ramayana", "kelir", "serunai", "gamelan", "pantun",
         "Dondang Sayang", "Mek Mulung", "Sultanate", "Kuala Lumpur", "Malaysia", "Vienna", "Habsburg",
         "Maria Theresa", "Lipizzaner", "Hofburg", "Schonbrunn", "Mozart", "Strauss", "Danube", "ARC"]
COMMON = set("the and that this with what when where which who how about tell talk show your yours there their "
             "they them then than have make made like look looking take time just very much more most some "
             "many come came give gave good great small little long way wing wang well were will would want "
             "know known song sing king kind molar".split())


def soundex(s):
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return ""
    codes = {**{c: "1" for c in "bfpv"}, **{c: "2" for c in "cgjkqsxz"}, **{c: "3" for c in "dt"},
             "l": "4", **{c: "5" for c in "mn"}, "r": "6"}
    out, last = s[0].upper(), codes.get(s[0], "")
    for c in s[1:]:
        d = codes.get(c, "")
        if d and d != last:
            out += d
        if c not in "hw":
            last = d
    return (out + "000")[:4]


@lru_cache(maxsize=1)
def vocabulary():
    names = list(EXTRA)
    try:
        from iro.arc import card, topic
        for c in topic()["cards"]:
            names += [n for n in card(c)[0].get("names", []) if len(n) > 3]
    except Exception:                              # noqa: BLE001 - the list alone still works
        pass
    return {n: soundex(n.replace(" ", "")) for n in dict.fromkeys(names)}


def fix(text):
    words = text.split()
    voc = vocabulary()
    out, i = [], 0
    while i < len(words):
        done = False
        for n in (3, 2, 1):
            chunk = words[i:i + n]
            if len(chunk) < n:
                continue
            raw = " ".join(chunk)
            core = re.sub(r"[^\w\s'-]", "", raw)
            low = core.lower()
            if n == 1 and (low in COMMON or len(low) < 5):
                continue
            joined = low.replace(" ", "")
            code = soundex(joined)
            best, best_r = None, 0.0
            for name, ncode in voc.items():
                if ncode != code:
                    continue
                r = difflib.SequenceMatcher(None, joined, name.lower().replace(" ", "")).ratio()
                if r > best_r:
                    best, best_r = name, r
            if best and low.rstrip("s") == best.lower().rstrip("s"):
                break                              # the same name (or its plural): leave it
            if best and best_r >= 0.55 and best.lower() != low:
                trail = raw[len(raw.rstrip(".,!?;:")):]
                out.append(best + trail)
                i += n
                done = True
                break
            if best and best.lower() == low:
                break
        if not done:
            out.append(words[i])
            i += 1
    return " ".join(out)


if __name__ == "__main__":
    for t in ["Tell me about Wei Wang Koolin.", "Who founded the Molocha Sultanate?", "Tell me about those shadow puppets.",
              "What is a Tok Daling?", "I like the kelear screen", "Who was Paramisura?", "How are you today?"]:
        print(f"{t!r:45} -> {fix(t)!r}")

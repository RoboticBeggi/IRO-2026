"""Text written for a screen -> text a voice can say naturally (TTS text normalisation).

    speakable("Founded c. 1402 CE (by Parameswara), 15th century: 2 km & 40% of it!")
    -> "Founded around fourteen oh two C E, by Parameswara, fifteenth century: two kilometres and
        forty percent of it!"

Years are read the way people say them (1402 "fourteen oh two", 1957 "nineteen fifty-seven",
2005 "two thousand and five", 2026 "twenty twenty-six"); other numbers in words; ordinals; common
abbreviations and symbols; brackets become commas (a voice cannot pronounce a bracket).
"""

import re

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
       "twelve": "twelfth"}
ABBR = [
    (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\betc\.", "and so on"), (r"\bc\.\s*(?=\d)", "around "),
    (r"\bca\.\s*(?=\d)", "around "), (r"\bapprox\.", "about"), (r"\bvs\.?\b", "versus"), (r"\bDr\.", "Doctor"),
    (r"\bMr\.", "Mister"), (r"\bMrs\.", "Missus"), (r"\bSt\.(?= [A-Z])", "Saint"), (r"\bBCE\b", "B C E"),
    (r"\bCE\b", "C E"), (r"\bBC\b", "B C"), (r"\bAD\b", "A D"), (r"\bkm\b", "kilometres"), (r"\bcm\b", "centimetres"),
    (r"\bkg\b", "kilograms"), (r"°C\b", " degrees Celsius"), (r"\bm\b(?= (?:tall|long|high|wide))", "metres"),
    (r"\bRM\s?(\d+(?:[.,]\d+)*)", r"\1 ringgit"), (r"€\s?(\d+(?:[.,]\d+)*)", r"\1 euros"), (r"\$\s?(\d+(?:[.,]\d+)*)", r"\1 dollars"),
]


def words(n):
    """0 <= n < 1e9 in words (British style: "one hundred and five")."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" and " + words(n % 100) if n % 100 else "")
    for size, name in ((10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            rest = n % size
            sep = " and " if 0 < rest < 100 else (" " if rest else "")
            return words(n // size) + " " + name + (sep + words(rest) if rest else "")
    return str(n)


def year(n):
    """How people say a year."""
    if 2000 <= n <= 2009:
        return "two thousand" + (" and " + ONES[n % 10] if n % 10 else "")
    hi, lo = divmod(n, 100)
    if lo == 0:
        return words(hi) + " hundred"
    return words(hi) + (" oh " + ONES[lo] if lo < 10 else " " + words(lo))


def ordinal(n):
    w = words(n)
    last = re.split(r"[ -]", w)[-1]
    if last in ORD:
        new = ORD[last]
    elif last.endswith("y"):
        new = last[:-1] + "ieth"
    else:
        new = last + "th"
    return w[: len(w) - len(last)] + new


def speakable(t):
    t = re.sub(r"[*_#`]+", "", t)
    t = re.sub(r"\s*[\(\[]\s*", ", ", t)
    t = re.sub(r"\s*[\)\]]\s*(?=[.,!?;:]|$)", "", t)
    t = re.sub(r"\s*[\)\]]\s*", ", ", t)
    t = t.replace("—", ", ").replace("–", " to ").replace("&", " and ").replace("%", " percent").replace("/", " or ")
    for pat, rep in ABBR:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\b(\d{1,3})(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)
    t = re.sub(r"\b(\d{1,3}(?:,\d{3})+)\b", lambda m: words(int(m.group(1).replace(",", ""))), t)
    t = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: words(int(m.group(1))) + " point " +
               " ".join(ONES[int(d)] for d in m.group(2)), t)
    t = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})s\b", lambda m: year(int(m.group(1))).replace("ty", "ties") if
               m.group(1).endswith("0") else year(int(m.group(1))) + "s", t)
    t = re.sub(r"\b(1[1-9][0-9]{2}|20[0-9]{2})\b", lambda m: year(int(m.group(1))), t)
    t = re.sub(r"\b\d{1,9}\b", lambda m: words(int(m.group(0))), t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,")
    return t


if __name__ == "__main__":
    for s in ["Founded c. 1402 CE (by Parameswara), 15th century: 2 km & 40% of it!",
              "Malaya became independent in 1957; Silat was listed in 2019, Mak Yong in 2005.",
              "It costs RM 250, about 3.5 hours (approx.) in the 1990s and 2026."]:
        print(s, "\n ->", speakable(s))

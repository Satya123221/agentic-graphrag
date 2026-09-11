"""Shared person-name parsing for athlete catalog and WON_BY extraction."""

import re
import unicodedata

PARTICLES = {
    "van", "von", "de", "den", "der", "di", "da", "del", "della", "dos",
    "das", "du", "la", "le", "bin", "al", "ter", "ten", "af", "st", "st.",
}

NAME_SUFFIXES = {"jr", "jr.", "junior", "sr", "sr.", "ii", "iii", "iv"}

NON_PERSON_VALUES = {
    "vacant", "none", "n/a", "na", "stripped", "disqualified", "dq",
    "cancelled", "canceled", "not awarded", "unawarded",
}

COUNTRIES_AND_TEAMS = {
    "australia", "austria", "belarus", "belgium", "brazil", "bulgaria",
    "canada", "china", "chinese taipei", "croatia", "cuba", "czech republic",
    "czechoslovakia", "denmark", "east germany", "egypt", "estonia",
    "ethiopia", "finland", "france", "germany", "great britain", "greece",
    "hong kong", "hungary", "india", "iran", "ireland", "israel", "italy",
    "jamaica", "japan", "kazakhstan", "kenya", "latvia", "lithuania",
    "mexico", "netherlands", "the netherlands", "new zealand", "nigeria",
    "north korea", "norway", "olympic athletes from russia", "poland",
    "portugal", "puerto rico", "roc", "romania", "russia", "serbia",
    "slovakia", "slovenia", "south africa", "south korea", "soviet union",
    "spain", "sweden", "switzerland", "turkey", "ukraine", "unified team",
    "united team", "united kingdom", "united states", "ussr", "west germany",
    "yugoslavia",
}

NOC_CODES = {
    "ALG", "ARG", "AUS", "AUT", "AZE", "BAH", "BAR", "BEL", "BLR", "BRA",
    "BRN", "BUL", "CAN", "CHI", "CHN", "CIV", "CMR", "COL", "CRO", "CUB",
    "CZE", "DEN", "DOM", "ECU", "EGY", "ESP", "EST", "ETH", "FIJ", "FIN",
    "FRA", "GBR", "GEO", "GER", "GRE", "GRN", "GUA", "HKG", "HUN", "INA",
    "IND", "IRI", "IRL", "ISR", "ISV", "ITA", "JAM", "JPN", "KAZ", "KEN",
    "KGZ", "KOR", "KSA", "LAT", "LTU", "LUX", "MAR", "MAS", "MEX", "MGL",
    "NED", "NGR", "NOR", "NZL", "PAN", "PER", "PHI", "POL", "POR", "PRK",
    "PUR", "QAT", "ROC", "ROU", "RSA", "RUS", "SLO", "SRB", "SUI", "SVK",
    "SWE", "THA", "TJK", "TPE", "TTO", "TUN", "TUR", "UAE", "UKR", "URS",
    "USA", "UZB", "VEN", "VIE", "YUG", "ZIM",
}

SPORTS = {
    "alpine skiing", "archery", "athletics", "badminton", "beach volleyball",
    "biathlon", "bobsleigh", "boxing", "canoeing", "cross-country skiing",
    "curling", "cycling", "diving", "equestrian", "fencing", "field hockey",
    "figure skating", "football", "freestyle skiing", "golf", "gymnastics",
    "handball", "ice hockey", "judo", "luge", "marathon swimming",
    "modern pentathlon", "nordic combined", "rowing", "rugby sevens",
    "sailing", "shooting", "short-track speed skating", "skateboarding",
    "skeleton", "ski jumping", "snowboarding", "speed skating", "swimming",
    "synchronized swimming", "table tennis", "taekwondo", "tennis",
    "triathlon", "water polo", "weightlifting", "wrestling",
}

NON_PERSON_FRAGMENTS = {
    "academy award", "academy awards", "world championship", "world indoor",
    "best actor", "best actress", "best picture", "best director",
    "best action", "best cinematography", "best supporting actor",
    "filmfare award", "olympic games",
}

NATIONALITY_PREFIX_RE = re.compile(
    r"^(?:"
    r"[a-z]+(?:\s+[a-z]+)?(?:'s|'s)|"
    r"dutchman|dutchwoman|american|austrian|hungarian|canadian|"
    r"jamaican|turkish|ukrainian|brazilian|german|french|italian|"
    r"russian|chinese|japanese|korean|australian|british|spanish|"
    r"swedish|norwegian|polish|czech|swiss|danish|finnish|irish|"
    r"mexican|cuban|kenyan|ethiopian|nigerian|indian|iranian|"
    r"seventeen-year-old american"
    r")\s+",
    re.IGNORECASE,
)

INFOBOX_LINE_RE = re.compile(r"^\s+[A-Za-z0-9_]+:\s*")
MEDAL_FIELD_RE = re.compile(
    r"^\s*(gold2?|silver2?|bronze2?):\s*(.*)$",
    re.IGNORECASE,
)
GOLD_FIELD_RE = re.compile(r"^\s*gold2?:\s*(.*)$", re.IGNORECASE)
NOC_GLUE_RE = re.compile(
    r"\b(" + "|".join(sorted(NOC_CODES)) + r")(?=[A-ZÀ-ÖŠŽČĆĐ])"
)
TOKEN_RE = re.compile(r"[^\s,;:/&]+")


def normalize(value):
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def make_id(name):
    slug = normalize(name).replace(" ", "_")
    return slug.strip("_")


def is_particle(token):
    return token.lower().rstrip(".") in PARTICLES


def is_suffix(token):
    return token.lower().rstrip(".") in {s.rstrip(".") for s in NAME_SUFFIXES} or (
        token.lower() in NAME_SUFFIXES
    )


def is_country_or_team(name):
    norm = normalize(name)
    if not norm:
        return False
    if norm in COUNTRIES_AND_TEAMS or norm in SPORTS:
        return True
    if norm.upper() in NOC_CODES and " " not in norm:
        return True
    if norm.endswith(" team") or norm.endswith(" federation"):
        return True
    return False


def is_non_person_value(name):
    norm = normalize(name)
    if not norm:
        return True
    if norm in NON_PERSON_VALUES or norm in NON_PERSON_FRAGMENTS:
        return True
    if is_country_or_team(name):
        return True
    return False


def clean_display_name(name):
    cleaned = name.replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = NATIONALITY_PREFIX_RE.sub("", cleaned)
    cleaned = cleaned.strip(" ,;.-")
    return cleaned


def first_alpha_is_upper(name):
    for ch in name:
        if ch.isalpha():
            return ch.isupper()
    return False


def looks_like_person_name(name):
    if not name or "\n" in name:
        return False
    if is_non_person_value(name):
        return False

    tokens = TOKEN_RE.findall(name)
    real_tokens = [tok for tok in tokens if not is_suffix(tok)]
    if len(real_tokens) < 2 or len(real_tokens) > 5:
        return False
    if is_particle(real_tokens[-1]):
        return False
    if any(tok.isdigit() for tok in tokens):
        return False
    if ". " in name or name.lower().startswith("best "):
        return False
    if not first_alpha_is_upper(name):
        return False

    letter_count = sum(1 for ch in name if ch.isalpha())
    if letter_count < 4:
        return False
    return True


def split_camel_case(value):
    """Split LastnameFirstname concatenations without breaking McIntosh / DeVries."""
    chars = []
    for i, ch in enumerate(value):
        if i > 0 and ch.isupper() and value[i - 1].islower():
            k = i - 1
            while k >= 0 and not value[k].isupper() and value[k].isalpha():
                k -= 1
            prev_word_len = i - k if k >= 0 and value[k].isupper() else i
            if prev_word_len >= 4:
                chars.append(" ")
        chars.append(ch)
    return "".join(chars)


def insert_glued_and(value):
    return re.sub(
        r"(?<=[aeiouyàáâäåæèéêëìíîïòóôöøùúûüýÿ])and(?=\s*[A-ZÀ-ÖŠŽČĆĐ])",
        " and ",
        value,
    )


def strip_glued_noc(value):
    return NOC_GLUE_RE.sub(r"\1 ", value)


def preprocess_medal_value(value):
    value = value.replace("\n", " ").strip()
    value = re.sub(r"\([^)]*\)", " ", value)
    value = insert_glued_and(value)
    value = strip_glued_noc(value)
    value = split_camel_case(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,;")


def extract_infobox_fields(text, gold_only=False):
    values = []
    in_infobox = False
    pattern = GOLD_FIELD_RE if gold_only else MEDAL_FIELD_RE

    for line in text.splitlines()[:80]:
        if line.startswith("[Infobox"):
            in_infobox = True
            continue
        if not in_infobox:
            continue
        if line.strip() == "":
            break
        if not INFOBOX_LINE_RE.match(line) and not line.startswith("["):
            break
        match = pattern.match(line)
        if match:
            value = match.group(match.lastindex).strip()
            if value:
                values.append(value)
    return values


def pack_person_names(chunk):
    tokens = TOKEN_RE.findall(chunk)
    if not tokens:
        return []

    names = []
    i = 0
    n = len(tokens)
    while i < n:
        if is_suffix(tokens[i]) and names:
            names[-1] = names[-1] + " " + tokens[i]
            i += 1
            continue

        if i + 2 < n and is_particle(tokens[i + 1]):
            j = i + 2
            while j < n and is_particle(tokens[j]):
                j += 1
            if j < n:
                end = j + 1
                if n - end == 1:
                    end = n
                names.append(" ".join(tokens[i:end]))
                i = end
                continue

        remaining = n - i
        if remaining == 1:
            names.append(tokens[i])
            i += 1
        elif remaining == 3:
            names.append(" ".join(tokens[i:i + 3]))
            i += 3
        elif remaining >= 5 and remaining % 4 == 3:
            names.append(" ".join(tokens[i:i + 3]))
            i += 3
        elif remaining % 2 == 1 and remaining >= 5:
            names.append(" ".join(tokens[i:i + 2]))
            i += 2
        else:
            names.append(" ".join(tokens[i:i + 2]))
            i += 2

    return names


def split_medal_value(value):
    prepared = preprocess_medal_value(value)
    if not prepared or is_non_person_value(prepared):
        return []

    chunks = re.split(r"\s*(?:,|;|&|/| and )\s*", prepared, flags=re.IGNORECASE)
    names = []
    for chunk in chunks:
        chunk = chunk.strip(" ,;")
        chunk = re.sub(r"^(?:and|&)\s+", "", chunk, flags=re.IGNORECASE)
        if not chunk:
            continue
        if is_suffix(chunk) and names:
            names[-1] = names[-1] + " " + chunk
            continue
        names.extend(pack_person_names(chunk))

    cleaned = []
    for name in names:
        name = clean_display_name(name)
        if not name or is_non_person_value(name):
            continue
        cleaned.append(name)
    return cleaned

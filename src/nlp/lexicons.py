# Lexicons for simple rules-based parsing

# Synonyms indicating user feels hot/warm
HOT_SYNONYMS = {
    "hot", "warm", "sweaty", "stuffy", "boiling", "toasty", "burning",
    "roasting", "heated"
}

# Synonyms indicating user feels cold/chilly
COLD_SYNONYMS = {
    "cold", "freezing", "chilly", "frozen", "icy", "nippy", "shivery",
    "cool"
}

# Intensifiers affecting magnitude
INTENSIFIERS_STRONG = {
    "very", "really", "extremely", "so", "super", "too"
}

INTENSIFIERS_WEAK = {
    "slightly", "a bit", "little", "somewhat", "kinda", "kind of", "a little"
}

# Fan level words to absolute levels (0..4)
FAN_LEVEL_WORDS = {
    "off": 0,
    "low": 1,
    "medium": 2,
    "mid": 2,
    "med": 2,
    "high": 3,
    "max": 4,
    "maximum": 4,
    "full": 4
}

# Verbs that often indicate setting an absolute value
SET_VERBS = {
    "set", "make", "put", "keep", "adjust", "change"
}

# Words that refer to temperature explicitly
TEMP_WORDS = {"temperature", "temp", "ac", "air", "thermostat"}

# Words that refer to fan
FAN_WORDS = {"fan", "blower"}

# Reasonable device bounds (Celsius) and fan bounds
MIN_TEMP_C = 16
MAX_TEMP_C = 30
MIN_FAN = 0
MAX_FAN = 4



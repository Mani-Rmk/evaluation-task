import spacy
from spacy.matcher import Matcher, PhraseMatcher
from word2number import w2n

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Price matcher
price_matcher = Matcher(nlp.vocab)
price_pattern = [
    {"IS_CURRENCY": True, "OP": "?"},
    {"LIKE_NUM": True, "OP": "?"},
    {"LOWER": {"IN": ["usd", "dollars", "rs", "inr", "euro"]}, "OP": "?"}
]
price_matcher.add("PRICE", [price_pattern])

# List of amenities
amenities = [
    "wifi",
    "free wifi",
    "free wi-fi",
    "pool",
    "swimming pool",
    "parking",
    "free parking",
    "gym",
    "fitness center",
    "breakfast",
    "free breakfast",
    "air conditioning",
    "air conditioner",
    "spa",
    "hot tub",
    "jacuzzi",
    "pet friendly",
    "pets allowed",
    "restaurant",
    "bar",
    "24-hour front desk",
    "elevator",
    "wheelchair accessible",
    "laundry service",
    "room service",
    "kitchen",
    "microwave",
    "refrigerator",
    "coffee maker",
    "balcony",
    "terrace",
    "garden",
    "business center",
    "conference room",
    "free internet",
    "internet access",
    "tv",
    "cable tv",
    "flat screen tv",
    "hair dryer",
    "iron",
    "safe",
    "heating",
    "non-smoking rooms",
    "airport shuttle",
    "shuttle service",
    "luggage storage",
    "daily housekeeping",
    "child friendly",
    "children's playground",
    "pets allowed",
    "smoking allowed",
    "wheelchair accessible",
    "dry cleaning",
]

#amenities phrase matcher
amenities_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
amenity_patterns = [nlp.make_doc(text) for text in amenities]
amenities_matcher.add("AMENITY", amenity_patterns)


#price conditions patterns
price_condition_pattern =[
    "less than", "more than", "equal to",
    "under", "below", "above", "greater than", "equals"
]
price_codtion_phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
pattern_docs = [nlp.make_doc(p) for p in price_condition_pattern]
price_codtion_phrase_matcher.add("PRICE_CONDITION", pattern_docs)

# Country list 
countries = [
    "india", "united states", "usa", "canada", "australia",
    "united kingdom", "france", "germany", "japan", "china", "italy","thailand"
]

# Category list
categories = [
    "beach resort", "mountain lodge", "city hotel",
    "desert camp", "lake view resort", "jungle lodge"
]

#country phase matcher
country_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
country_patterns = [nlp.make_doc(text) for text in countries]
country_matcher.add("COUNTRY", country_patterns)

# category phase matcher
category_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
category_patterns = [nlp.make_doc(text) for text in categories]
category_matcher.add("CATEGORY", category_patterns)

def ner_data(USER_INPUT: str):
    doc = nlp(USER_INPUT)
    
    places = []
    amenities = []
    prices=[]
    price_condition=[]
    country=[]
    category=[]
    
    #place matches
    for ent in doc.ents:
        if ent.label_ in ("GPE","LOC", "FAC") and ent.text.lower() not in countries:
            places.append(ent.text)

    
    # Price matches
    price_matches = price_matcher(doc)
    for match_id, start, end in price_matches:
        span = doc[start:end]
        first_token = span[0].text.lower()
        try:
            if span[0].like_num:
                number_value = float(span[0].text)
            else:
                number_value = w2n.word_to_num(first_token)
            prices.append(number_value)
        except ValueError:
            continue


    # Amenity matches
    amenity_matches = amenities_matcher(doc)
    for match_id, start, end in amenity_matches:
        span = doc[start:end]
        amenities.append(span.text)

    
    # Price condition matches
    price_condition_matches = price_codtion_phrase_matcher(doc)
    for match_id, start, end in price_condition_matches:
        span = doc[start:end]
        price_condition.append(span.text)


    country_matches = country_matcher(doc)
    for match_id, start, end in country_matches:
        span = doc[start:end]
        country.append(span.text)


    category_matches = category_matcher(doc)
    for match_id, start, end in category_matches:
        span = doc[start:end]
        category.append(span.text)

    return {
        "place": places if places else {},
        "country": country if country else {},
        "price": prices[0] if prices else None,
        "amenities": amenities,
        "category": category if category else {},
        "price_condition": price_condition[0] if price_condition else None
    }


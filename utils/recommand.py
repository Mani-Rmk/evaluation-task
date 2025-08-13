from sentence_transformers import SentenceTransformer, util
from utils.ner import ner_data

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_popular_destinations(df, top_n=5):
    df['popularity_score'] = df['Rating'] * df['Number of Reviews']
    top_destinations_df = df.sort_values(by='popularity_score', ascending=False).head(top_n)
    return top_destinations_df[['Hotel Name', 'Location', 'Country', 'Price per Night (USD)', 'Amenities']]

def filter_hotels(df, user_input):
    try:
        ner_input = ner_data(user_input) or {}
    except Exception as e:
        print(f"NER extraction failed: {e}")
        ner_input = {}

    place = [p.lower() for p in ner_input.get('place', [])]
    country = [c.lower() for c in ner_input.get('country', [])]
    category = [cat.lower() for cat in ner_input.get('category', [])]
    amenities = [a.lower() for a in ner_input.get('amenities', [])]
    price = ner_input.get('price')
    price_condition = ner_input.get('price_condition')
    
    #print("price:", price)
    filtered_df = df.copy()

    # Price filter
    if price is not None:
        if price_condition in ["less than", "under", "below"]:
            filtered_df = filtered_df[filtered_df['Price per Night (USD)'] <= price]
        elif price_condition in ["more than", "above"]:
            filtered_df = filtered_df[filtered_df['Price per Night (USD)'] >= price]
        else:
            filtered_df = filtered_df[filtered_df['Price per Night (USD)'] == price]

    # Place filter
    if place:
        filtered_df = filtered_df[
            filtered_df['Location'].astype(str).str.lower().apply(
                lambda loc: any(p in loc for p in place)
            )
        ]

    # Country filter
    if country:
        filtered_df = filtered_df[
            filtered_df['Country'].astype(str).str.lower().apply(
                lambda c: any(country_name in c for country_name in country)
            )
        ]

    # Category filter
    if category:
        filtered_df = filtered_df[
            filtered_df['Category'].astype(str).str.lower().apply(
                lambda cat: any(c in cat for c in category)
            )
        ]

    # Amenities filter
    if amenities:
        filtered_df = filtered_df[
            filtered_df['Amenities'].astype(str).str.lower().apply(
                lambda am: all(a in am for a in amenities)
            )
        ]


    if not filtered_df.empty:
            filtered_df['combined_text'] = (
            filtered_df['Hotel Name'].astype(str) + " " +
            filtered_df['Location'].astype(str) + " " +
            filtered_df['Country'].astype(str) + " " +
            filtered_df['Amenities'].astype(str)
        )
    else:
        return None

    # Similarity ranking
    hotel_embeddings = model.encode(filtered_df['combined_text'].tolist(), convert_to_tensor=True)
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, hotel_embeddings)[0]
    top_idx = cos_scores.topk(k=len(cos_scores)).indices.cpu().numpy()
    top_results = filtered_df.iloc[top_idx]

    return top_results[['Hotel Name', 'Location', 'Country', 'Price per Night (USD)', 'Amenities']].reset_index(drop=True)
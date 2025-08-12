# Hotel Search System with NLP-based Filtering and Recommendations

This project is a **hotel search and recommendation system** that uses natural language processing (NLP) to understand user preferences and filters hotels accordingly. It leverages **spaCy** for Named Entity Recognition (NER) to extract entities like locations, amenities, price conditions, and more from user input, and **SentenceTransformers** to rank hotel results based on semantic similarity.

---

## Features

- Extracts location, country, price, price conditions (e.g., under $200), amenities, and category from user queries using spaCy NER and custom pattern matching.
- Filters hotels dataset based on extracted parameters.
- Ranks filtered hotels using semantic similarity of user input and hotel metadata via SentenceTransformers (`all-MiniLM-L6-v2` model).
- Provides popular destination recommendations if no results match the filter.
- Simple interactive UI built with **Streamlit** for user input and displaying results.

---

## How It Works

1. **NER Extraction:**  
   Parses the user input text to detect:
   - Places (cities, locations)  
   - Countries  
   - Price and price conditions (e.g., under $200)  
   - Amenities (e.g., wifi, pool)  
   - Hotel categories (e.g., beach resort)

2. **Filtering:**  
   Uses the extracted entities to filter the hotels dataset:
   - Filters by price according to conditions like "less than" or "above"  
   - Matches place, country, category, and amenities (all specified amenities must be present)

3. **Ranking:**  
   Applies sentence transformer embeddings to rank hotels based on semantic similarity with the original user query.

4. **Results Display:**  
   Shows recommended hotels or popular destinations using Streamlit UI.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Mani-Rmk/evaluation-task.git
   cd hotel-search-nlp
2.Create a Python virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

3.Install required packages:

pip install -r requirements.txt

4.Download the spaCy English model:

python -m spacy download en_core_web_sm

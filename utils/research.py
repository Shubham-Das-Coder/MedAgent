# utils/research.py
import requests

def fetch_research(query):
    """Fetch research articles from PubMed API."""
    url = f"https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/?term={query}"
    response = requests.get(url)
    return response.json()

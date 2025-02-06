import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from deep_translator import GoogleTranslator

def translate_text(text, target_language="fr"):
    """Traduit le texte en français."""
    return GoogleTranslator(source="auto", target=target_language).translate(text)

def fetch_wikipedia():
    urls = [
        "https://fr.wikipedia.org/wiki/National_Basketball_Association",
        "https://fr.wikipedia.org/wiki/Playoffs_NBA",
        "https://fr.wikipedia.org/wiki/Draft_NBA",
        "https://fr.wikipedia.org/wiki/Palmar%C3%A8s_NBA"
    ]
    text_data = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text_data.extend([translate_text(p.get_text().strip()) for p in paragraphs if p.get_text().strip()])
    
    df = pd.DataFrame(text_data, columns=["Texte"])
    df.to_csv("data/nba_wikipedia.csv", index=False, encoding="utf-8")

def fetch_basketball_reference():
    urls = [
        "https://www.basketball-reference.com/leagues/NBA_2024.html",
        "https://www.basketball-reference.com/players/",
        "https://www.basketball-reference.com/teams/"
    ]
    text_data = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table")
        text_data.extend([translate_text(table.get_text().strip()) for table in tables if table.get_text().strip()])
    
    df = pd.DataFrame(text_data, columns=["Texte"])
    df.to_csv("data/nba_stats.csv", index=False, encoding="utf-8")

def fetch_nba_com():
    urls = [
        "https://www.nba.com/stats",
        "https://www.nba.com/standings",
        "https://www.nba.com/draft",
        "https://www.nba.com/news"
    ]
    text_data = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for url in urls:
            page.goto(url)
            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            text_data.extend([translate_text(p.get_text().strip()) for p in soup.find_all("p") if p.get_text().strip()])
        
        browser.close()
    
    df = pd.DataFrame(text_data, columns=["Texte"])
    df.to_csv("data/nba_website.csv", index=False, encoding="utf-8")

def main():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    print("Scraping Wikipedia en français...")
    fetch_wikipedia()
    
    print("Scraping Basketball Reference et traduction...")
    fetch_basketball_reference()
    
    print("Scraping NBA.com et traduction...")
    fetch_nba_com()
    
    print("Scraping terminé et sauvegardé en CSV en français !")

if __name__ == "__main__":
    main()

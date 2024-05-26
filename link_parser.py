import requests
from bs4 import BeautifulSoup

class LinkParser:
    def __init__(self):
        pass
    def extract_text_from_website(self,url):
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all text elements
            text_elements = soup.find_all(text=True)
            
            # Filter out empty and whitespace-only strings
            text_content = [text.strip() for text in text_elements if text.strip()]
            
            # Join the text content into a single string
            extracted_text = ' '.join(text_content)
            
            return extracted_text
        else:
            raise Exception(f"Failed to fetch URL: {url}")
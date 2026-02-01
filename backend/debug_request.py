
import requests

url = "http://127.0.0.1:8000/translate/debug_client"
files = {'file': ('test.pdf', b'%PDF-1.4 empty content', 'application/pdf')}
data = {
    'api_key': '',  # Empty
    'target_lang': 'Arabic'
}

try:
    resp = requests.post(url, files=files, data=data)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text}")
except Exception as e:
    print(f"Request Failed: {e}")

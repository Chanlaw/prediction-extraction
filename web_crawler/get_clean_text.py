import sys
import requests
import json 
import csv
from bs4 import BeautifulSoup


def get_text(url) : 
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  ret = ''
  for p in soup.find_all(class_="story-body-text"): 
    ret += ' '.join(p.text.encode('utf-8').split())
    ret += ' '
  return ret


for url in sys.stdin : 
  url = url.strip()
  sys.stderr.write("Getting text from %s\n"%url)
  try: 
    txt = get_text(url)
  except :
    continue
  if not(txt.strip() == ''):
    print txt
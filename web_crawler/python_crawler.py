#!/usr/bin/python

import sys
import urllib2
from cookielib import CookieJar
import lxml.etree
import lxml.html
import re

#Only get links from Paul Krugman's blog
DOMAIN = 'http://krugman.blogs.nytimes.com/' 

def crawl(root_url, domain=''):
  visited = set()
  stack = [root_url]
  cj = CookieJar()
  url = root_url
  sys.stderr.write("Crawling from page %s\n"%url)

  opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))

  #Make NYT not think we're a bot
  opener.addheaders = [('User-agent', 'Chrome/44.0.2403.107')]
  site = opener.open(url).read() 
  doc = lxml.etree.HTML(site) 

  #Use regex to get links from page
  result = doc.xpath("//div//p") 
  link = re.compile('href="(.*?)"') 

  #Print all links not visited
  for item in result:
      source = lxml.html.tostring(item)
      if link.search(source): 
        link_address = link.search(source).group(1) 
        if link_address.startswith(DOMAIN) and link_address not in visited: 
            print link_address 
            visited.add(link_address)

if __name__ == "__main__":
   for i in range(1,56): 
     crawl('http://krugman.blogs.nytimes.com/page/%s'%i, domain=DOMAIN)


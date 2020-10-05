import requests
import re

x = ""

BoM_link = f'https://www.sacred-texts.com/mor/book00.htm'

# BOM scrape example

# for i in range(9):
#     txt = requests.get(f'https://www.sacred-texts.com/mor/book0{i}.htm')
#     nochaptertxt = re.sub(r"(\d\s)?([a-zA-Z]+\s)+(\d+\:\d+)", "", txt.text)
#     nonumbertxt = re.sub(r"\d+", "", nochaptertxt)
#     nochaptertxtagain = re.sub(r"Chapter", "", nonumbertxt)
#     nonewlinetxt = re.sub(r"\n{2,}", "", nochaptertxtagain)
#     cleantxt = re.sub(r"<.+>", "", nonewlinetxt)
#     x += cleantxt

# for i in range(10, 15, 1):
#     txt = requests.get(f'https://www.sacred-texts.com/mor/book{i}.htm')
#     nochaptertxt = re.sub(r"(\d\s)?([a-zA-Z]+\s)+(\d+\:\d+)", "", txt.text)
#     nonumbertxt = re.sub(r"\d+", "", nochaptertxt)
#     nochaptertxtagain = re.sub(r"Chapter", "", nonumbertxt)
#     nonewlinetxt = re.sub(r"\n{2,}", "", nochaptertxtagain)
#     cleantxt = re.sub(r"<.+>", "", nonewlinetxt)
#     x += cleantxt

# with open("BOMtxt.txt", "w", encoding="utf-8") as f:
#     f.write(x)

debate_link = "https://www.rev.com/blog/transcripts/donald-trump-joe-biden-1st-presidential-debate-transcript-2020"
x = requests.get(debate_link)
nohtmltext = re.sub(r"<.+>", "", x.text)
nosingletagtext = re.sub(r"<.+/>", "", nohtmltext)
with open("debate2020.txt", "w", encoding="utf-8") as f:
    f.write(nosingletagtext)
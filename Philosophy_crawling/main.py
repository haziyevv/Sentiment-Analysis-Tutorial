import requests
from bs4 import BeautifulSoup, Tag
import pdb
import time, re
import util
import sys


BASE_URL='https://en.wikipedia.org/'


def get_page(url):
    res = requests.get(url)
    html = res.text

    soup = BeautifulSoup(html, 'lxml')
    util.decomposer(soup)
    main_content = soup.find('div', id='bodyContent')

    try:
        paragraphs = main_content.find_all('p')
    except AttributeError as e:
        print(e)
        
    for par in paragraphs:
        str_par = str(par)

        # find links with paranthesis and convert them to uid
        links = re.findall(r"href=[\"'][/\w_\-]+\(\w+\)", str(par))
        link_ids = util.link_to_dict(links)
        str_par = util.to_uid(str_par, link_ids) 
        str_par = util.remove_text_between_parens(str_par)
        str_par = util.to_str(str_par, link_ids)

        new_par = BeautifulSoup(str_par, "lxml") 
        links = new_par.findChildren('a')
        for link in links:
            if 'wiki' not in link['href']:
                continue
            url = f"{BASE_URL}{link['href']}"
            return url



url = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Special:Random"


def check(url):
    print(f"Starting url: {url}")
    seen = set()
    i = 50
    while i > 0:
        i -= 1
        url = get_page(url)
        if not url:
            print(f"oops... this page: {url}  does not have an outer link")
            break
        elif url in seen:
            print(f"there is a loop, this will stuck in {url}")
            break
        elif 'Philosophy' in url:
            print(f"Philosophy url is reached: {url}")
            break
        else:
            seen.add(url)
            print(url)
            time.sleep(0.5)

check(url)

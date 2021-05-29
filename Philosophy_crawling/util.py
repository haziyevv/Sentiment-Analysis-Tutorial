from bs4 import BeautifulSoup
import re, uuid



def remove_text_between_parens(text):
    n = 1
    while n:
        text, n = re.subn(r'\([^()]+\)', '', text)
    return text


def link_to_dict(links):
    link_id = {}
    for link in links:
        id_ = uuid.uuid1()
        link_id[link] = id_

    return link_id


def to_uid(str_par, link_toids):
    for link, id_ in link_toids.items():
        str_par = str_par.replace(link, str(id_))
    return str_par

def to_str(str_par, link_toids):
    for link, id_ in link_toids.items():
        str_par = str_par.replace(str(id_), link)
    return str_par


def decomposer(b_soup, uw=["table", "i", "span"]):
    for x in uw:
        dumps = b_soup.find_all(x)
        for dump in dumps:
            dump.decompose()



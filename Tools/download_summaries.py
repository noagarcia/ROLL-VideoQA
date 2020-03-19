from bs4 import BeautifulSoup
import urllib.request
import csv


def read_url(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    fp = urllib.request.urlopen(req)
    mybytes = fp.read()
    urlstr = mybytes.decode("utf8")
    fp.close()
    return urlstr


def text_cleaning(text):
    text = text.replace("\n"," ")
    return text


if __name__ == "__main__":

    # Needed urls
    baseurl = 'https://the-big-bang-theory.com'
    season_baseurl = 'https://the-big-bang-theory.com/episodeguide/season/'
    seasons = list(range(1,10))

    # output
    recap_dict = {}
    outfile = 'Data/tbbt_summaries.csv'

    # Iterate over each season menu to get list of episode links --> episodes_urls, episode_ids
    for s in seasons:
        seasons_url = season_baseurl + str(s)
        seasondata = read_url(seasons_url)
        seasonSoup = BeautifulSoup(seasondata, features="html.parser")
        episodes_urls, episode_ids = [], []
        for link in seasonSoup.findAll('a', {"class": "stitle"}):
            episodes_urls.append(baseurl + link.get('href'))
            episode_ids.append(link.get_text())

        # Iterate over each episode url to get the recap info --> recap_dict
        for eurl, eid in zip(episodes_urls,episode_ids):
            episodedata = read_url(eurl)
            episodeSoup = BeautifulSoup(episodedata, features="html.parser")

            # Get all textBlock class in the code
            textBlocks = episodeSoup.findAll('div', {"class": "textBlock"})

            # Get the textBlock class that is preceeded by the title "Recap"
            for t in textBlocks:
                prevSib = t.find_previous_sibling("h3", string='Recap')
                if prevSib is not None:
                    episode_recap = t.get_text()
                    recap_dict[eid] = episode_recap

    # Save summaries into a file
    with open(outfile, 'w') as f:
        w = csv.writer(f)
        w.writerow(["Episode", "Recap"])
        w.writerows(recap_dict.items())



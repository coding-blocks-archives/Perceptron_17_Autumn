import requests
from bs4 import BeautifulSoup as BS
import pandas as pd

url = 'http://www.imdb.com/chart/top?ref_=nv_mv_250_6'
r = requests.get(url)

soup = BS(r.text, 'html.parser')

divs = soup.find_all('div', {'class':'lister'})
div = divs[0]

rows = div.find_all('tr')

URLS = []
TITLES = []
YEARS = []
RATINGS = []

for row in rows[1:]:
	tds = row.find_all('td')
	data = tds[1]
	anchor = data.find('a')
	url = 'https://www.imdb.com/' + str(anchor['href'])
	URLS.append(str(url.encode('utf-8', 'ignore')))
	
	title = anchor.text
	TITLES.append(str(title.encode('utf-8', 'ignore')))
	
	span = data.find('span')
	year = span.text.replace(')', '').replace('(','')
	YEARS.append(str(year))
	
	ratingdata = tds[2]
	rating = ratingdata.find('strong').text
	RATINGS.append(str(rating))

initial = pd.DataFrame({'Title': TITLES, 'URL': URLS, 'Year': YEARS, 'Rating': RATINGS})
initial.to_csv('./IMDb_Database.csv', header=True, columns = ['Title', 'URL', 'Year', 'Rating'], sep = ',', index = False)
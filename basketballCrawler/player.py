from .soup_utils import getSoupFromURL
import re
import logging
import json
from bs4 import Comment
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from datetime import date
from datetime import datetime

class Player(object):
    # Regex patterns for player info
    POSN_PATTERN = u'(Point Guard|Center|Power Forward|Shooting Guard|Small Forward)'
    HEIGHT_PATTERN = u'([0-9]-[0-9]{1,2})'
    WEIGHT_PATTERN = u'([0-9]{2,3})lb'

    name = None

    positions = []
    height = None
    weight = None
    salaries = []
    age = None

    overview_url = None
    overview_url_content = None
    gamelog_data = None
    gamelog_url_list = []
    current_stats = {}

    def __init__(self,_name,_overview_url,scrape_data=True):
        self.name = _name
        self.overview_url = _overview_url

        # Explicitly declaring all fields in the constructor will ensure that
        # they're included in JSON serialization
        self.positions = []
        self.height = None
        self.weight = None
        self.salaries = []

        self.overview_url_content = None
        self.gamelog_data = None
        self.gamelog_url_list = []
        self.age = None
        self.current_stats = {}

        if scrape_data:
            self.scrape_data()

    def scrape_data(self):
        print(self.name,self.overview_url)
        if self.overview_url_content is not None:
            raise Exception("Can't populate this!")

        overview_soup = getSoupFromURL(self.overview_url)
        self.overview_url_content = overview_soup.text

        try:
            player_position_text = overview_soup.findAll(text=re.compile(u'(Point Guard|Center|Power Forward|Shooting Guard|Small Forward)'))[0]
            player_height_text = overview_soup.findAll(text=re.compile(self.HEIGHT_PATTERN))[0]
            player_weight_text = overview_soup.findAll(text=re.compile(self.WEIGHT_PATTERN))[0]
            self.height = re.findall(self.HEIGHT_PATTERN,player_height_text)[0].strip()
            self.weight = re.findall(self.WEIGHT_PATTERN,player_weight_text)[0].strip()
            tempPositions = re.findall(self.POSN_PATTERN,player_position_text)
            self.positions = [position.strip() for position in tempPositions]
            self.salaries = self.__findSalaries(overview_soup)
            self.age = self.__findAge(overview_soup)

            self.current_stats = self.__findCurrentStats(overview_soup)

        except Exception as ex:
            logging.error(ex)
            self.positions = []
            self.height = None
            self.weight = None

        # the links to each year's game logs are in <li> tags, and the text contains 'Game Logs'
        # so we can use those to pull out our urls.
        for li in overview_soup.find_all('li'):
            game_log_links = []
            if 'Game Logs' in li.getText():
                game_log_links =  li.findAll('a')

            for game_log_link in game_log_links:
                self.gamelog_url_list.append('http://www.basketball-reference.com' + game_log_link.get('href'))

    def __findSalaries(self, soupped):
        total_salaries = []
        all_all_salaries = soupped.find("div", {"id": "all_all_salaries"})
        comments=all_all_salaries.find_all(string=lambda text:isinstance(text,Comment))
        raw_salary_rows = BeautifulSoup(comments[0], "lxml").find("tbody").find_all("tr")
        for each_raw_salary in raw_salary_rows:
            year = each_raw_salary.find("th").text.replace("-","_")
            salary = self.__salaryTextToFloat(each_raw_salary.find_all("td")[2].text)
            total_salaries.append((year, salary))
        return total_salaries

    def __findAge(self, soupped):
        year = soupped.find("span", {"id": "necro-birth"})
        birth_year = year.get("data-birth")
        birth_time = datetime.strptime(birth_year, '%Y-%m-%d')
        rel =  relativedelta(date.today(), birth_time)
        return rel.years

    def __salaryTextToFloat(self, text):
        return float(text[1:].replace(",",""))

    def __findCurrentStats(self, soupped):
        year = str(datetime.now().year)
        current_stats = {}

        per_game = soupped.find("tr", {"id": "per_game." + year })

        all_advanced = soupped.find("div", {"id": "all_advanced" })
        comments = all_advanced.find_all(string=lambda text:isinstance(text,Comment))
        all_advanced_rows = BeautifulSoup(comments[0], "lxml").find("tbody")

        for game_stat in [per_game, all_advanced_rows]:
            for td in game_stat.find_all("td"):
                data_stat = td.get("data-stat")
                if data_stat in ["Xxx", "Yyy"]:
                    continue
                data_val = td.text
                current_stats[data_stat] = data_val

        return current_stats

    def to_json(self):
        return json.dumps(self.__dict__)


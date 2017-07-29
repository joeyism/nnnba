import nba_py
import requests
from nba_py import player
from lxml import html
import re
from .logger import *

class NBA_player:

    def __init__(self, player_id, name_reverse, name):
        self.player_id = player_id
        self.name_reverse = name_reverse
        self.name = name
        self.stats = []
        self.salaries = []
        self.projected_salaries = []
        self.header = [] 
        self.age = None
        self.positions = []
        #TODO: get player metadata, such as team, etc.

    def __trimData__(self, measure_type, array): 
        if measure_type == "Base":
            return array[5:33] #[5:-30] #removed basic rankings
        elif measure_type == "Advanced":
            return array[10:25] #+ array[30:-32] #[10:-32] #removed adv ranking
        elif measure_type == "Scoring":
            return array[10:25] #+ array[30:-2] #[15:-22] # removed ranking
        elif measure_type == "Usage":
            return array[10:25] #+ array[30:-2] #[11:-25] # removed ranking
        elif measure_type == "Misc":
            return array[10:18]
        return array

    def __joinData__(self, list1, list2):
        list_tot = []
        for i, row in enumerate(list1):
            list_tot.append((row[0], row[1] + list2[i][1]))
        return list_tot

    def getPlayerStats(self, measure_type="Base"):
        yoy = player.PlayerYearOverYearSplits(measure_type=measure_type, player_id=self.player_id)
        json = yoy.json
        total = []
        header = self.__trimData__(measure_type, json["resultSets"][0]["headers"])
        for yeardata in json["resultSets"][1]["rowSet"]:
            year = yeardata[-1]
            yeardata = self.__trimData__(measure_type, yeardata)
            total.append((year, yeardata))
        
        self.stats = dict(total)
        self.header = header
        self.projected_salaries = self.getProjectedSalary()

        return total

    def getPlayerAdvStats(self):
        measure_types = ["Advanced", "Scoring", "Usage", "Misc"]
        yoy = self.getPlayerStats()
        yoy_header = self.header
        yoy_tot = yoy
        yoy_tot_header = yoy_header
        for measure_type in measure_types:
            yoy_adv = self.getPlayerStats(measure_type=measure_type)
            yoy_adv_header = self.header
            yoy_tot = self.__joinData__(yoy_tot, yoy_adv)
            yoy_tot_header = yoy_tot_header + yoy_adv_header

        self.stats = dict(yoy_tot)
        self.header = yoy_tot_header
        return yoy_tot

    def setSalaries(self, salaries):
        self.salaries = dict(salaries)

    def setAge(self, age):
        self.age = age

    def setPositions(self, positions):
        self.positions = positions
    
    def __parseSalaryText(self, rawSalary):
        textSalary = re.sub('\s+', '', rawSalary)
        try:
            return int(textSalary[1:].replace(",",""))
        except:
            return 0
    
    def getProjectedSalary(self):
        playerName = "-".join(self.name.replace(".","").split(" "))
        url = "http://hoopshype.com/player/{}/salary/".format(playerName)
        page = requests.get(url)
        tree = html.fromstring(page.content)
        salaryParents = tree.xpath('//*[@id="content"]/div[2]/div[3]/div[1]/div[1]/table/tbody/tr')
        salaries = []
        for salaryParent in salaryParents:
            thisSalary = self.__parseSalaryText(salaryParent.getchildren()[1].text)
            salaries.append(thisSalary)
        return salaries

    def summarize(self):
        return { "name": self.name, "salaries": self.salaries, "stats": self.stats, "header": self.header, "projected_salaries": self.projected_salaries, "age": self.age , "positions": self.positions}


def getAllPlayers():
    playerlist = nba_py.player.PlayerList()
    json = playerlist.json
    rowSet = json["resultSets"][0]["rowSet"]
    total = []
    for player in rowSet:
        total.append(player[0:3])
    return total

def manualFix(nba_player):
    fix_names = { 
        "Wade Baldwin IV": "Wade Baldwin",
        "James Ennis III": "James Ennis",
        "AJ Hammons": "A.J. Hammons",
        "Tim Hardaway Jr.": "Tim Hardaway",
        "Johnny O'Bryant III": "Johnny O'Bryant",
        "Nene": "Nene Hilario",
        "Derrick Jones Jr.": "Derrick Jones",
        "RJ Hunter": "R.J. Hunter",
        "CJ McCollum": "C.J. McCollum",
        "KJ McDaniels": "K.J. McDaniels",
        "CJ Miles": "C.J. Miles",
        "Kelly Oubre Jr.": "Kelly Oubre",
        "Gary Payton II": "Gary Payton",
        "Otto Porter Jr.": "Otto Porter",
        "Taurean Prince": "Taurean Waller-Prince",
        "JJ Redick": "J.J. Redick",
        "Glenn Robinson III": "Glenn Robinson",
        "JR Smith": "J.R. Smith",
        "PJ Tucker": "P.J. Tucker",
        "TJ Warren": "T.J. Warren",
        "CJ Wilcox": "C.J. Wilcox"
    }
    for listed_name in fix_names:
        if nba_player.name == listed_name:
            nba_player.name = fix_names[listed_name]
            return nba_player

    return nba_player

def test_headers(measure_type="Scoring"):
    import pandas as pd
    pd.set_option('display.max_columns', None)
    nba_player = NBA_player("203382", "Baynes, Aron", "Aron Baynes")
    nba_player.getPlayerStats(measure_type=measure_type)
    df = pd.DataFrame(columns = nba_player.header)
    df.loc[0] = nba_player.getPlayerStats(measure_type=measure_type)[0][1]
    print(df)
    return nba_player

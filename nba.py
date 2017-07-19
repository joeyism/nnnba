import nba_py
import requests
from nba_py import player
from lxml import html
import re

class NBA_player:

    def __init__(self, player_id, name_reverse, name):
        self.player_id = player_id
        self.name_reverse = name_reverse
        self.name = name
        self.stats = []
        self.salaries = []
        self.header = []

    def __trimData__(self, measure_type, array):
        if measure_type == "Base":
            return array[5:-2]
        elif measure_type == "Advanced":
            return array[10:-2]

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
        return total

    def getPlayerAdvStats(self):
        yoy = self.getPlayerStats()
        yoy_header = self.header
        yoy_adv = self.getPlayerStats(measure_type="Advanced")
        yoy_adv_header = self.header

        yoy_tot = self.__joinData__(yoy, yoy_adv)
        yoy_tot_header = yoy_header + yoy_adv_header

        self.stats = dict(yoy_tot)
        self.header = yoy_tot_header
        return yoy_tot

    def setSalaries(self, salaries):
        self.salaries = dict(salaries)
    
    def __parseSalaryText(rawSalary):
        textSalary = re.sub('\s+', '', rawSalary)
        return int(textSalary[1:].replace(",",""))
    
    def getProjectedSalary(self):
        playerName = "-".join(playerName.replace(".","").split(" "))
        url = "http://hoopshype.com/player/{}/salary/".format(self.name)
        page = requests.get(url)
        tree = html.fromstring(page.content)
        salaryParents = tree.xpath('//*[@id="content"]/div[2]/div[3]/div[1]/div[1]/table/tbody/tr')
        salaries = []
        for salaryParent in salaryParents:
            thisSalary = self.__parseSalaryText(salaryParent.getchildren()[1].text)
            salaries.append(thisSalary)
        return salaries

    def summarize(self):
        return { "name": self.name, "salaries": self.salaries, "stats": self.stats, "header": self.header}


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

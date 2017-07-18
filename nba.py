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

    def getPlayerStats(self):
        yoy = player.PlayerYearOverYearSplits(measure_type="Base", player_id=self.player_id)
        json = yoy.json
        total = []
        for yeardata in json["resultSets"][0]["rowSet"]:
            yeardata = yeardata[5:]
            year = yeardata.pop()
            total.append((year, yeardata))
        
        self.stats = dict(total)
        return total

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
        return { "name": self.name, "salaries": self.salaries, "stats": self.stats}


def getAllPlayers():
    playerlist = nba_py.player.PlayerList()
    json = playerlist.json
    rowSet = json["resultSets"][0]["rowSet"]
    total = []
    for player in rowSet:
        total.append(player[0:3])
    return total

def manualFix(nba_player):
    if nba_player.name == "Wade Baldwin IV":
        nba_player.name = "Wade Baldwin"
    elif nba_player.name == "James Ennis III":
        nba_player.name = "James Ennis"
    elif nba_player.name == "AJ Hammons":
        nba_player.name = "A.J. Hammons"
    elif nba_player.name == "Tim Hardaway Jr.":
        nba_player.name = "Tim Hardaway"
    elif nba_player.name == "Johnny O'Bryant III":
        nba_player.name = "Johnny O'Bryant"
    elif nba_player.name == "Nene":
        nba_player.name = "Nene Hilario"
    elif nba_player.name == "Derrick Jones Jr.":
        nba_player.name == "Derrick Jones"
    return nba_player

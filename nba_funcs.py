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

    def getPlayerStats(self):
        yoy = player.PlayerYearOverYearSplits(measure_type="Base", player_id=self.player_id)
        json = yoy.json
        total = []
        for yeardata in json["resultSets"][0]["rowSet"]:
            total.append(yeardata[5:])
    
        return total
    
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


def getAllPlayers():
    playerlist = nba_py.player.PlayerList()
    json = playerlist.json
    rowSet = json["resultSets"][0]["rowSet"]
    total = []
    for player in rowSet:
        total.append(player[0:3])
    return total


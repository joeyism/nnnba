import nba_py
import requests
from nba_py import player
from lxml import html
import re


def getPlayerStats(player_id):
    yoy = player.PlayerYearOverYearSplits(measure_type="Base", player_id=player_id)
    json = yoy.json
    total = []
    for yeardata in json["resultSets"][0]["rowSet"]:
        total.append(yeardata[5:])

    return total

def getAllPlayers():
    playerlist = nba_py.player.PlayerList()
    json = playerlist.json
    rowSet = json["resultSets"][0]["rowSet"]
    total = []
    for player in rowSet:
        total.append(player[0:3])
    return total

def parseSalaryText(rawSalary):
    textSalary = re.sub('\s+', '', rawSalary)
    return int(textSalary[1:].replace(",",""))

def getProjectedSalary(playerName):
    playerName = "-".join(playerName.replace(".","").split(" "))
    url = "http://hoopshype.com/player/{}/salary/".format(playerName)
    page = requests.get(url)
    tree = html.fromstring(page.content)
    salaryParents = tree.xpath('//*[@id="content"]/div[2]/div[3]/div[1]/div[1]/table/tbody/tr')
    salaries = []
    for salaryParent in salaryParents:
        thisSalary = parseSalaryText(salaryParent.getchildren()[1].text)
        salaries.append(thisSalary)
    return salaries

import nba_funcs
import basketballCrawler as bc

bballref_players = bc.loadPlayerDictionary('crawled_data/players.json')
nbastats = nba_funcs.getAllPlayers()

X = []

for player in bballref_players:
    

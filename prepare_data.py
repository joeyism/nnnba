import os
from . import nba
from .nba import *
from .  import basketballCrawler as bc
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import json
from .logger import *

def start(parallel=True, measure_type="Advanced"):
    pool = ThreadPool(4)
    
    def process_to_raw_data(bballref_players, player_stat_info): #TODO: Add position

        nba_player = NBA_player(player_stat_info[0], player_stat_info[1], player_stat_info[2])

        try: #ignore ones who aren't playing this upcoming year, i.e. retired
            bballref_players[nba_player.name]
        except:
            return None

        logger.debug(nba_player.name)
        nba_player = nba.manualFix(nba_player)
        if measure_type == "Advanced":
            nba_player.getPlayerAdvStats()
        elif measure_type == "Basic":
            nba_player.getPlayerStats()
        nba_player.setSalaries(bballref_players[nba_player.name].salaries)
        nba_player.setAge(bballref_players[nba_player.name].age)
        nba_player.setPositions(bballref_players[nba_player.name].positions)
        return nba_player.summarize()
    
    print ("Start")
    print ("nbastats = nba.getAllPlayers()")
    nbastats = nba.getAllPlayers()

    fn = os.path.join(os.path.dirname(__file__), 'crawled_data/players.json')
    print ("bballref_players = bc.loadPlayerDictionary('" + fn + "')")
    bballref_players = bc.loadPlayerDictionary(fn)
    
    print ("process to raw data")
    raw_data = []
    if parallel:
        logger.debug("Computing in Parallel")
        func = partial(process_to_raw_data, bballref_players)
        raw_data = [x for x in pool.map(func, nbastats) if x is not None]
    else:
        logger.debug("Computing in Series")
        for player_stat_info in nbastats:
            raw_data.append(process_to_raw_data(bballref_players, player_stat_info))
        raw_data = [x for x in raw_data if x is not None]
            
    fn = os.path.join(os.path.dirname(__file__), "crawled_data/raw_data.json")
    with open(fn, "w") as outfile:
        json.dump(raw_data, outfile)

if __name__ == "__main__":
    start()

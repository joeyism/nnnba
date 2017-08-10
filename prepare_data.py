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

    def to_float(s):
        try:
            return float(s)
        except:
            return s
    
    def process_to_raw_data(bballref_players, player_stat_info): #TODO: Add position

        nba_player = NBA_player(player_stat_info[0], player_stat_info[1], player_stat_info[2])

        nba_player = nba.manualFix(nba_player)
        logger.debug(nba_player.name)



        try: #ignore ones who aren't playing this upcoming year, i.e. retired
            this_bballref_player = bballref_players[nba_player.name]
            this_bballref_player.stats.keys()
        except:
            logger.debug("ERROR 1: " + player_stat_info[2])
            return None

        if measure_type == "Advanced":
            nba_player.getPlayerAdvStats()
        elif measure_type == "Basic":
            nba_player.getPlayerStats()

        successful = False

        for bballref_year in list(this_bballref_player.stats.keys()):
            try:
                nba_player.stats[bballref_year] = nba_player.stats[bballref_year] + list(map(to_float, list(this_bballref_player.stats[bballref_year].values())))
                successful = True
            except:
                logger.debug("ERROR 2: " + player_stat_info[2] + " " + str(bballref_year))

        if successful:
            nba_player.header = nba_player.header + list(this_bballref_player.stats[bballref_year].keys())

        nba_player.setSalaries(this_bballref_player.salaries)
        nba_player.setAge(this_bballref_player.age)
        nba_player.setPositions(this_bballref_player.positions)
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
    logger.debug("done")

if __name__ == "__main__":
    start()

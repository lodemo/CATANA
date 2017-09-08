# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
 
from youtubeAPICrawler.database import *


q = raw_input("Updating Database scheme, potential data loss!\nAre you sure? y/n\n")
if q != 'y':
    exit

db = YTDatabase()

# Updates the database scheme if new tables were added in database.py
db.createDatabase(False, True)

# CAUTION: potential data loss, adding additional tables works without data loss, changing existing tables could!




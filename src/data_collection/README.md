# Data Collection

Contains different crawler for the sites YouTube, Socialblade and Google Image Search (not used).
All crawler are based on the [Scrapy](https://scrapy.org/) Framework.
*link

Needs valid YouTube API key!
Needs Scrapy to be installed!


scrapy_cron.sh:
- file to add to crontab for daily execution of the YouTube crawler

estimate_quota.py:
- simple script for estimating the daily found videos and YouTube API quote needed, outputs csv file


## Directories

### youtubeAPICrawler
- contains scrapy crawler implementing the described crawler of the thesis.
- crawls: populate and update
- usage:
    - navigate to youtubeAPICrawler/youtubeAPICrawler
    - open console, start crawl with scrapy crawl *spider*
        - scrapy crawl populate or crawl scrapy update

- make sure youtubeAPICrawler/youtubeAPICrawler/database.py has correct MySQL credentials
- YouTube API keys are fixed in the spider .py files
- initial channel list for populate crawler is set in the spider PopulateYTSpider.py file
- updatepopulation spider is a extended spider for additional crawling of added information, like thumbnail URLs


### sblade
- contains a crawler for the website Socialblade, used to acquire a sample of channel per network (MCN)
- crawls: broadbandtv, makerstudio, studio71

- every crawler reads file in data directory and writes new found channel into a file per network
- crawler have max iteration and request delay to prevent ip ban
- crawler can be executed multiple times (files are extended)



### images
- contains Google images search crawler
- crawls: images

- crawler gets channel stored in MySQL database, requests images based on channel title and saves them on disk




import pandas as pd
import numpy as np
import requests
import json
import re
from unidecode import unidecode
from cleantext import clean
from datetime import datetime
pd.set_option('display.max_columns', None)


class YTDataConvertion:
    def __init__(self, apiKey, channelId):
        self.apiKey = apiKey
        self.channelID = channelId
        self.fileName = None


    def findFileName(self):
        channelURL = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={self.channelID}&key={self.apiKey}"
        jsonUrl = requests.get(channelURL)
        data = json.loads(jsonUrl.text)

        try:
            title = data["items"][0]["snippet"]["title"]
            title = clean(title, no_emoji=True)
            title = title.replace(" ", "_").lower()
            self.fileName = title

        except:
            title = None
            print("Channel Title = None")

        return title


    def ISO8601ToHMS(self, durationISO8601):
        match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', durationISO8601)

        if not match:
            raise ValueError("Not Valid ISO 8601 Format")

        hours = int(match.group(1)[:-1]) if match.group(1) else 0
        minutes = int(match.group(2)[:-1]) if match.group(2) else 0
        seconds = int(match.group(3)[:-1]) if match.group(3) else 0

        return "{:d}:{:d}:{:d}".format(hours, minutes, seconds)


    def convertToSeconds(self, duration):
        hours, minutes, seconds = map(int, duration.split(':'))
        totalSeconds = hours * 3600 + minutes * 60 + seconds
        return totalSeconds


    def convertJSONToCSV(self):

        assert self.fileName is not None, "File Name is None" #This ensures that before continuing with the script a fileName actually exists, if not it returns and AssertionError explaining "File Name is None"

        #try:
        jsonFile = self.fileName + ".json"

        rawData = pd.read_json(jsonFile)
        rawData = rawData.T #Transposing the DataFrame from horizontal to vertical
        rawData.reset_index(inplace=True)
        rawData.rename(columns={'index': 'videoID'}, inplace=True)

        movingIndex = rawData.pop("videoID")
        rawData.insert(0, "videoID", movingIndex)

        #rawData = rawData.drop(columns=['channelId', 'channelTitle', 'thumbnails', 'categoryId', 'defaultLanguage', 'description', 'liveBroadcastContent', 'localized', 'defaultAudioLanguage', 'licensedContent', 'favoriteCount', 'definition', 'projection', 'caption', 'topicCategories', 'contentRating', 'tags', 'regionRestriction', 'dimension'], errors="ignore")

        rawData = rawData[['videoID', 'publishedAt', 'title', 'viewCount', 'likeCount', 'commentCount', 'duration']] #Keeping only these columns

        rawData = rawData.dropna() #Removing rows with Na

        rawData["publishedAt"] = pd.to_datetime(rawData["publishedAt"], format='%Y-%m-%dT%H:%M:%SZ', errors="coerce")
        rawData["duration"] = rawData["duration"].apply(lambda videoDuration: self.ISO8601ToHMS(videoDuration)) #Adapting ISO8601 video duration to "classic" duration format
        rawData["title"] = rawData["title"].apply(lambda videoTitle: unidecode(videoTitle)) #Approximating every character to the closest ASCII "equivalent"
        rawData["title"] = rawData["title"].apply(lambda videoTitle: clean(videoTitle, no_emoji=True)) #Removing possible emojis from YouTube Channel titles
        rawData["totalSecondsDuration"] = rawData["duration"].apply(lambda duration: self.convertToSeconds(duration))  # Adding video duration in seconds column into the DF

        rawData["commentCount"] = rawData["commentCount"].astype("int64", errors="raise")
        rawData["viewCount"] = rawData["viewCount"].astype("int64", errors="raise")
        rawData["likeCount"] = rawData["likeCount"].astype("int64", errors="raise")

        rawData["commentViewRatio"] = rawData["commentCount"]/rawData["viewCount"] #Number of comments divided by the number of views
        rawData["likeViewRatio"] = rawData["likeCount"]/rawData["viewCount"] #Number of likes divided by number of views, the result can't be 0 since if there aren't views there can't be likes too

        rawData["videoAge"] = rawData["publishedAt"].apply(lambda date: (datetime.now() - datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")).days) #Video age obtained by subtracting video timestamp (converted to string just for this purpose) to the current date and returning just the number of days that are passed since the video timestamp

        #NOTE Deleting all rows where commentViewRatio and likeViewRatio are 0 since it means that there aren't any comments or likes
        rawData = rawData[rawData["commentViewRatio"] != 0]
        rawData = rawData[rawData["likeViewRatio"] != 0]

        print(rawData)
        print()

        print("DataFrame Column Names:")
        print(rawData.columns)
        print()

        print("Nas Sum in Every Column:")
        print(rawData.isna().sum())
        print("Nas Removed")
        print()

        print("Unique Years:")
        print(rawData['publishedAt'].dt.year.unique())
        print()

        print("Unique Years Video Count")
        print(rawData["publishedAt"].groupby(rawData['publishedAt'].dt.year).count().reset_index(name='VideoCount'))
        print()

        print("DataFrame Shape:")
        print(rawData.shape)
        print()

        print("Target Features: ")
        print(rawData[["viewCount", "totalSecondsDuration", "commentViewRatio", "likeViewRatio", "videoAge"]])
        print()

        rawData.to_csv(f"{self.fileName}CSV.csv", index=False, encoding="utf-8") #Not saving the index as a column in the CSV file
        print("CSV File Exported Correctly")

        #except:
            #print("Error Raised, Maybe File Not Existing")

        return None


































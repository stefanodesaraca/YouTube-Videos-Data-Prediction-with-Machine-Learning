import pandas as pd
import numpy as np
import requests
import json
import re
from unidecode import unidecode
from cleantext import clean
from datetime import datetime
pd.set_option('display.max_columns', None)


class YTDataConversion:
    def __init__(self, apiKey, channelId):
        self.apiKey = apiKey
        self.channelID = channelId
        self.fileName = None


    def findChannelTitle(self):
        channelURL = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={self.channelID}&key={self.apiKey}"
        jsonUrl = requests.get(channelURL)
        data = json.loads(jsonUrl.text)

        try:
            title = data["items"][0]["snippet"]["title"]
            title = clean(title, no_emoji=True, to_ascii=True)
            title = title.replace(" ", "_").lower()
            self.fileName = f"./YouTubeML/{title}/{title}" #This is intended to stay without extensions to be used in multiple cases (with different extensions)

        except:
            title = None
            print("\033[91mChannel Title = None\033[0m")

        return title


    @staticmethod
    def ISO8601ToHMS(durationISO8601: str):
        match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', durationISO8601) #This Regular Expression checks respectively: H (Hours), M (Minutes), S (Seconds)

        if match:
            days = 0
            hours = int(match.group(1)[:-1]) if match.group(1) else 0
            minutes = int(match.group(2)[:-1]) if match.group(2) else 0
            seconds = int(match.group(3)[:-1]) if match.group(3) else 0

            return "{:d}:{:d}:{:d}:{:d}".format(days, hours, minutes, seconds)

        elif not match:

            try:
                #This Regular Expression is specifically created for videos longer than 24H (24H = 1D) checks respectively: D (Days), H (Hours), M (Minutes), S (Seconds). In these cases the duration doesn't start with "PT", but with "P"
                #To know: the reason of the starting letter described in the previous comment is because the T indicates the time since it's called "Time Designator", while the P stands for "Period", in fact it is called the "Period Designator"
                match = re.match(r'P(\d+D)T(\d+H)?(\d+M)?(\d+S)?', durationISO8601)

                days = int(match.group(1)[:-1]) if match.group(1) else 0
                hours = int(match.group(2)[:-1]) if match.group(2) else 0
                minutes = int(match.group(3)[:-1]) if match.group(3) else 0
                seconds = int(match.group(4)[:-1]) if match.group(4) else 0

                return "{:d}:{:d}:{:d}:{:d}".format(days, hours, minutes, seconds)

            except ValueError:
                print("\033[91mNot Valid ISO 8601 Format\033[0m")
                exit()

        else:
            raise Exception("Unknown error raised, unable to continue.")




    @staticmethod
    def convertToSeconds(duration: str):
        days, hours, minutes, seconds = map(int, duration.split(':'))
        totalSeconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        return totalSeconds


    @staticmethod
    def dataOverview(data):

        print(data)
        print()

        print("DataFrame Column Names:")
        print(data.columns)
        print()

        print("Nas Sum in Every Column:")
        print(data.isna().sum())
        print("Nas Removed")
        print()

        print("Unique Years:")
        print(data['publishedAt'].dt.year.unique())
        print()

        print("Unique Years Video Count")
        print(data["publishedAt"].groupby(data['publishedAt'].dt.year).count().reset_index(name='VideoCount'))
        print()

        print("DataFrame Shape:")
        print(data.shape)
        print()

        print("Target Features: ")
        print(data[["viewCount", "totalSecondsDuration", "commentViewRatio", "likeViewRatio", "videoAge"]])
        print()

        return None



#To know the download mode check the downloadModeFile.txt in the YouTuber custom folder
class convertManualFileType(YTDataConversion):
    def __init__(self, apiKey, channelId):
        super().__init__(apiKey, channelId)
        self.mode = "Manual"

    def __repr__(self):
        return f"\nMode: {self.mode} | API KEY: {self.apiKey} | YouTube Channel ID: {self.channelID}\n"  # Returns a formatted string with the used API KEY and chosen YouTube Channel ID


    def convertManualJSONToCSV(self):

        assert self.fileName is not None, "File Name is None" #This ensures that before continuing with the script a fileName actually exists, if not it returns and AssertionError explaining "File Name is None"

        try:

            rawData = pd.read_json(self.fileName + ".json")
            rawData = rawData.T #Transposing the DataFrame from horizontal to vertical

            #print(rawData)

            rawData.reset_index(inplace=True)
            rawData.rename(columns={'index': 'videoID'}, inplace=True)

            movingIndex = rawData.pop("videoID")
            rawData.insert(0, "videoID", movingIndex)

            #rawData = rawData.drop(columns=['channelId', 'channelTitle', 'thumbnails', 'categoryId', 'defaultLanguage', 'description', 'liveBroadcastContent', 'localized', 'defaultAudioLanguage', 'licensedContent', 'favoriteCount', 'definition', 'projection', 'caption', 'topicCategories', 'contentRating', 'tags', 'regionRestriction', 'dimension'], errors="ignore")

            rawData = rawData[['videoID', 'publishedAt', 'title', 'viewCount', 'likeCount', 'commentCount', 'duration']] #Keeping only these columns

            rawData = rawData.dropna() #Removing rows with NaNs

            rawData["publishedAt"] = pd.to_datetime(rawData["publishedAt"], format='%Y-%m-%dT%H:%M:%SZ', errors="coerce")
            rawData["duration"] = rawData["duration"].apply(lambda videoDuration: self.ISO8601ToHMS(videoDuration)) #Adapting ISO8601 video duration to "classic" duration format
            rawData["title"] = rawData["title"].apply(lambda videoTitle: unidecode(videoTitle)) #Approximating every character to the closest ASCII "equivalent"
            rawData["title"] = rawData["title"].apply(lambda videoTitle: clean(videoTitle, no_emoji=True, to_ascii=True)) #Removing possible emojis from YouTube Channel titles
            rawData["totalSecondsDuration"] = rawData["duration"].apply(lambda duration: self.convertToSeconds(duration))  # Adding video duration in seconds column into the DF

            rawData["commentCount"] = rawData["commentCount"].astype("int64", errors="raise")
            rawData["viewCount"] = rawData["viewCount"].astype("int64", errors="raise")
            rawData["likeCount"] = rawData["likeCount"].astype("int64", errors="raise")

            rawData["commentViewRatio"] = rawData["commentCount"]/rawData["viewCount"] #Number of comments divided by the number of views
            rawData["likeViewRatio"] = rawData["likeCount"]/rawData["viewCount"] #Number of likes divided by number of views, the result can't be 0 since if there aren't views there can't be likes too

            rawData["videoAge"] = rawData["publishedAt"].apply(lambda date: (datetime.now() - datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")).days) #Video age obtained by subtracting video timestamp (converted to string just for this purpose) to the current date and returning just the number of days that are passed since the video timestamp

            #Deleting all rows where commentViewRatio and likeViewRatio are 0 since it means that there aren't any comments or likes
            rawData = rawData[rawData["commentViewRatio"] != 0]
            rawData = rawData[rawData["likeViewRatio"] != 0]


            self.dataOverview(rawData)


            rawData.to_csv(f"{self.fileName}CSV.csv", index=False, encoding="utf-8") #Not saving the index as a column in the CSV file
            print("CSV File Exported Correctly")

        except:
            print("\033[91mError raised, maybe file not existing or wrong download mode chosen\033[0m")
            exit()

        return None



class convertGoogleClientFileType(YTDataConversion):
    def __init__(self, apiKey, channelId):
        super().__init__(apiKey, channelId)
        self.mode = "Google API Python Client"

    def __repr__(self):
        return f"\nMode: {self.mode} | API KEY: {self.apiKey} | YouTube Channel ID: {self.channelID}\n" #Returns a formatted string with the used API KEY and chosen YouTube Channel ID


    def convertGClientJSONtoCSV(self):

        assert self.fileName is not None, "File Name is None" #This ensures that before continuing with the script a fileName actually exists, if not it returns and AssertionError explaining "File Name is None"

        try:

            videoData = {}
            topics = {} #Topics for every video -> videoID: [topic1, topic2, ...]


            with open(f"{self.fileName}" + ".json", "r") as jFile:

                jsonData = json.load(jFile)

                for vid in jsonData.keys():

                    #print(key, vid)

                    #By using the get() method we avoid KeyErrors if there aren't values for that key
                    vidID = vid #The key is also the videoID
                    publishedAt = jsonData[vid]["snippet"].get("publishedAt", None)
                    title = jsonData[vid]["snippet"].get("title", None)
                    #description = jsonData[vid]["snippet"].get("description", None)
                    viewCount = jsonData[vid]["statistics"].get("viewCount", 0)
                    likeCount = jsonData[vid]["statistics"].get("likeCount", 0)
                    #favoriteCount = jsonData[vid]["statistics"].get("favorite", 0)
                    commentCount = jsonData[vid]["statistics"].get("commentCount", 0)
                    duration = jsonData[vid]["contentDetails"].get("duration", 0)

                    vidTopics = jsonData[vid]["snippet"].get("tags", None)

                    topics.update({vidID: vidTopics})

                    videoData.update({vidID: {"videoID": vid,
                                              "publishedAt": publishedAt,
                                              "title": title,
                                              "viewCount": viewCount,
                                              "likeCount": likeCount,
                                              "commentCount": commentCount,
                                              "duration": duration}})

            rawData = pd.DataFrame(videoData).T


            rawData["publishedAt"] = pd.to_datetime(rawData["publishedAt"], format='%Y-%m-%dT%H:%M:%SZ', errors="coerce")
            rawData = rawData.dropna()  # Removing rows with NaNs

            rawData["publishedAt"] = pd.to_datetime(rawData["publishedAt"], format='%Y-%m-%dT%H:%M:%SZ', errors="coerce")
            rawData["duration"] = rawData["duration"].apply(lambda videoDuration: self.ISO8601ToHMS(videoDuration))  # Adapting ISO8601 video duration to "classic" duration format
            rawData["title"] = rawData["title"].apply(lambda videoTitle: unidecode(videoTitle))  # Approximating every character to the closest ASCII "equivalent"
            rawData["title"] = rawData["title"].apply(lambda videoTitle: clean(videoTitle, no_emoji=True, to_ascii=True))  # Removing possible emojis from YouTube Channel titles
            rawData["totalSecondsDuration"] = rawData["duration"].apply(lambda duration: self.convertToSeconds(duration))  # Adding video duration in seconds column into the DF

            rawData["commentCount"] = rawData["commentCount"].astype("int64", errors="raise")
            rawData["viewCount"] = rawData["viewCount"].astype("int64", errors="raise")
            rawData["likeCount"] = rawData["likeCount"].astype("int64", errors="raise")

            rawData["commentViewRatio"] = rawData["commentCount"]/rawData["viewCount"]  # Number of comments divided by the number of views
            rawData["likeViewRatio"] = rawData["likeCount"]/rawData["viewCount"]  # Number of likes divided by number of views, the result can't be 0 since if there aren't views there can't be likes too

            rawData["videoAge"] = rawData["publishedAt"].apply(lambda date: (datetime.now() - datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")).days)  # Video age obtained by subtracting video timestamp (converted to string just for this purpose) to the current date and returning just the number of days that are passed since the video timestamp

            # Deleting all rows where commentViewRatio and likeViewRatio are 0 since it means that there aren't any comments or likes
            rawData = rawData[rawData["commentViewRatio"] != 0]
            rawData = rawData[rawData["likeViewRatio"] != 0]

            self.dataOverview(rawData)

            rawData.to_csv(f"{self.fileName}CSV.csv", index=False, encoding="utf-8") #Not saving the index as a column in the CSV file
            print("CSV File Exported Correctly")

            self.exportTopics(topics)

            return None

        except Exception as JSONReadingError:
            print(f"\033[91mError:\033[0m {JSONReadingError}", f"\033[91mwhile trying to read:\033[0m {self.fileName}.json")
            exit()


    def exportTopics(self, topics: dict):

        with open(f"YouTubeML/{self.findChannelTitle()}/videoTopics.txt", "w") as topicsFile:
            topicsFile.write("VideoID: Topics\n")
            for vid, top in topics.items():
                topicsFile.write(f"{vid}: {top}\n")

        return None













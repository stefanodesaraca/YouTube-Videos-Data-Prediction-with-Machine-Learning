import requests
import json
from tqdm import tqdm
import os
from pathlib import Path
from cleantext import clean


class YTStats:
    def __init__(self, apiKey, channelId):
        self.apiKey = apiKey
        self.channelID = channelId
        self.channelStatistics = None
        self.videoData = None
        self.nptList = []
        self.lastTimestampFileName = "lastTimestampFile.txt"


    def getChannelStatistics(self):
        channelURL = f"https://www.googleapis.com/youtube/v3/channels?part=statistics&id={self.channelID}&key={self.apiKey}"
        #print(channelURL)
        jsonUrl = requests.get(channelURL)
        data = json.loads(jsonUrl.text)
        #print(data)

        try:
            data = data["items"][0]["statistics"]
        except:
            data = None
            print("Data = None")

        self.channelStatistics = data
        return data

        #ALTERNATIVE METHOD:
        #youtube = build("youtube", "v3", developerKey=self.apiKey)
        #rq = youtube.channels().list(part="statistics", id=self.channelID)
        #print(rq.execute())


    def getChannelVideoData(self):

        channelVideos = self.getChannelVideos(limit=50) #Receiving the dictionary containing all videos gathered (each video is a dictionary itself, so the data structure is a dictionary of dictionaries)

        print(f"Channel Videos: {channelVideos}")
        print(f"Channel Videos Dictionary Length: {len(channelVideos)}")

        try:
            videoParts = ["snippet", "statistics", "contentDetails", "topicDetails"]
            for videoID in tqdm(channelVideos):
                for videoPart in videoParts:
                    data = self.getSingleVideoData(videoID, videoPart)
                    channelVideos[videoID].update(data)

            self.videoData = channelVideos

            return channelVideos

        except:
            print("Empty Channel Videos Dictionary")
            return None


    def getSingleVideoData(self, videoId, part):
        url = f"https://www.googleapis.com/youtube/v3/videos?part={part}&id={videoId}&key={self.apiKey}"
        jsonUrl = requests.get(url)
        videoData = json.loads(jsonUrl.text)
        try:
            videoData = videoData['items'][0][part] #Accessing the first element of the list and then the part
        except KeyError:
            print("Error in Getting Single Video Data")
            videoData = dict()

        return videoData



    def getChannelVideos(self, limit):

        url = f"https://www.googleapis.com/youtube/v3/search?key={self.apiKey}&channelId={self.channelID}&part=id&order=date"

        if limit is not None and isinstance(limit, int): #Checking if limit variable is an integer by using isinstance()
            url += "&maxResults=" + str(limit)

        #vid = None
        #npt = None

        #Vid is a dictionary containing the videos IDs
        if Path(self.lastTimestampFileName).is_file() == False: #If a previous lastTimestamp file doesn't exist this should be the first call of the class in the MainScript
            vid = self.getChannelVideosPerPage(url)
            npt = self.getNPT(url)
            print(url)

        else:

            lastTimestamp = self.loadLastTimestampFile()

            if lastTimestamp is not None:

                url = None #Resetting the url, otherwise during every run it will "append" the next timestamp one next to each other
                url = f"https://www.googleapis.com/youtube/v3/search?key={self.apiKey}&channelId={self.channelID}&part=id&publishedBefore={lastTimestamp}&order=date"
                url += "&maxResults=" + str(limit)

                print(url)

                vid = self.getChannelVideosPerPage(url)
                npt = self.getNPT(url)

            else:
                npt = None
                vid = None

        idx = 0

        while(npt is not None and idx < 9):
            nextUrl = url + "&pageToken=" + npt

            nextVideo = self.getChannelVideosPerPage(nextUrl)
            npt = self.getNPT(nextUrl)

            vid.update(nextVideo) #Merging dictionaries to add videos to the vid dictionary (each video is a dictionary itself) to build a dictionary of dictionaries

            idx += 1

        print(f"NPT List: {self.nptList}")

        return vid


    def getChannelVideosPerPage(self, url):

        jsonUrl = requests.get(url)
        data = json.loads(jsonUrl.text)

        channelVideos = {} #A dictionary of dictionaries containing channel videos (each dictionary is a video)
        if "items" not in data:
            return channelVideos, None

        itemData = data["items"]
        nextPageToken = data.get("nextPageToken", None)

        if nextPageToken is not None:
            self.nptList.append(nextPageToken) #Appening to NPT list only valid tokens and not Nones
        else:
            pass

        #print(self.nptList)

        for item in itemData:
            try:
                kind = item["id"]["kind"]

                if kind == "youtube#video":
                    video_id = item["id"]["videoId"]
                    channelVideos[video_id] = dict()
            except KeyError:
                print("Error in Channel Videos Per Page")

        return channelVideos #Returning the dictionary of dictionaries



    def getNPT(self, url):
        jsonUrl = requests.get(url)
        data = json.loads(jsonUrl.text)

        npt = data.get("nextPageToken", None)

        return npt



    def dump(self):
        if self.channelStatistics is None or self.videoData is None:
            print("No Data")
            return

        lastTS, channelTitle = self.getChannelInfo()

        self.writeLastTimestampFile(lastTS)

        channelTitle = clean(channelTitle, no_emoji=True)
        channelTitle = channelTitle.replace(" ", "_").lower() #Stardizing eventual spaces or uppercase letters
        fileName = channelTitle + ".json"

        if Path(fileName).is_file() == True:
            with open(fileName, "r+") as jFile: #Reading and writing mode
                loadedData = json.load(jFile)
                loadedData.update(self.videoData)

                jFile.truncate() #Erasing JSON file's content to then write old and new data gathered
                jFile.seek(0) #Resetting cursor to the first byte

                json.dump(loadedData, jFile, indent=4)

            jFile.close()

        else:
            jsonData = self.videoData

            with open(fileName, "w") as FirstJFile:
                json.dump(jsonData, FirstJFile, indent=4)

            FirstJFile.close()

            print("JSON File Correctly Created")

        return None #TODO TEST IF THIS WORKS CORRECTLY


    def writeLastTimestampFile(self, lastTimestamp):

        if lastTimestamp is not None:

            print("Last Timestamp Found: " + lastTimestamp)

            with open(self.lastTimestampFileName, "w") as ltsF:
                ltsF.write(lastTimestamp)
        else:
            print("No Last Timestamp Found")
            pass

        return None

    def loadLastTimestampFile(self):

        if Path(self.lastTimestampFileName).is_file() == True:
            with open(self.lastTimestampFileName, "r") as ltsF:
                lastTimestamp = ltsF.readlines()[-1]

            return lastTimestamp

        else:
            return None

    def delLastTimestampFile(self):
        if Path(self.lastTimestampFileName).is_file() == True:
            os.remove(self.lastTimestampFileName)
            return None
        else:
            return None

    def getChannelInfo(self):

        lastVideoDataKey = list(self.videoData.keys())[-1]
        lastVideo = self.videoData[lastVideoDataKey]

        lastTimestamp = lastVideo.get("publishedAt", None)
        channelTitle = lastVideo.get("channelTitle", None)

        return lastTimestamp, channelTitle







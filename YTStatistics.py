from YTDataFormatConversion import YTDataConversion
import requests
import json
from tqdm import tqdm
import os
from pathlib import Path
from cleantext import clean
from googleapiclient.discovery import build


class YTStatsDownloader:
    def __init__(self, apiKey, channelId):
        self.apiKey = apiKey
        self.channelID = channelId
        self.channelStatistics = None
        self.videoData = None
        self.nptList = []
        self.lastTimestampFileName = "lastTimestampFile.txt"
        self.currentWorkingDirectoryPath = os.path.abspath(os.getcwd())


    def createYouTubeMLDir(self, channelTitle):


        # Creating YouTubeML Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML")
        except FileExistsError:
            print("\033[91mYouTubeML folder already exists\033[0m")
            pass

        # Creating YouTubeML YouTuber Custom Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML/{channelTitle}")
        except FileExistsError:
            print(f"\033[91m./YouTubeML/{channelTitle} folder already exists\033[0m")
            pass

        # Creating Models File Folder - Pickle Models Files
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML/{channelTitle}/YTModels")
        except FileExistsError:
            print(f"\033[91m./{channelTitle}/YTModels folder already exists\033[0m")
            pass

        # Models Prediction Plots Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML/{channelTitle}/YTModelsPredictionPlots")
        except FileExistsError:
            print(f"\033[91m./{channelTitle}/YTModelsPredictionPlots folder already exists\033[0m")
            pass


        # Creating YouTube ML EDA Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML/{channelTitle}/YouTubeMLEDA")
        except FileExistsError:
            print(f"\033[91m./{channelTitle}/YouTubeMLEDA folder already exists\033[0m")
            pass


        # Creating YouTube Plots Folder Inside the YouTube ML EDA One
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML/{channelTitle}/YouTubeMLEDA/YTPlots")
        except FileExistsError:
            print(f"\033[91m./{channelTitle}/YouTubeMLEDA/YTPlots folder already exists\033[0m")
            pass


        # Creating Shapiro-Wilk Test Plots Folder Inside the YouTube ML EDA One
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML/{channelTitle}/YouTubeMLEDA/YTShapiroWilkPlots")
        except FileExistsError:
            print(f"\033[91m./{channelTitle}/YouTubeMLEDA/YTShapiroWilkPlots folder already exists\033[0m")
            pass


        # Creating Models Test Plots Folder - Test Scores Comparison Between the Models (Plotly Charts)
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML/{channelTitle}/YTModelsTestScoreSummaryPlots")
        except FileExistsError:
            print(f"\033[91m./{channelTitle}/ModelsTestScoreSummaryPlots folder already exists\033[0m")
            pass


        # Creating Models Report Folder - CSV Files
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YouTubeML/{channelTitle}/YTModelsReport")
        except FileExistsError:
            print(f"\033[91m./{channelTitle}/YTModelsReport folder already exists\033[0m")
            pass


        return None


    @staticmethod
    def getNPT(url):
        jsonUrl = requests.get(url)
        data = json.loads(jsonUrl.text)

        npt = data.get("nextPageToken", None)

        return npt


    def dump(self):
        if self.channelStatistics is None or self.videoData is None:
            print("No Data")
            return

        lastTS, channelTitle = self.getChannelInfo()

        self.createYouTubeMLDir(channelTitle) #Create the custom folders for the YT channel
        self.writeLastTimestampFile(lastTS, channelTitle) #The lastTimestampFile is directly written during the execution of the dump method

        fileName = f"./YouTubeML/{channelTitle}/{channelTitle}" + ".json"

        #print(self.videoData)

        if Path(fileName).is_file() is True:
            with open(fileName, "r+") as jFile: #Reading and writing mode
                loadedData = json.load(jFile)

                loadedData.update(self.videoData)

                jFile.truncate() #Erasing JSON file's content to then write old and new data merged together
                jFile.seek(0) #Resetting cursor to the first byte

                json.dump(loadedData, jFile, indent=4)

            jFile.close()

        else:
            jsonData = self.videoData

            with open(fileName, "w") as FirstJFile:
                json.dump(jsonData, FirstJFile, indent=4)

            FirstJFile.close()

            print("JSON File Correctly Created")

        return None

    def writeLastTimestampFile(self, lastTimestamp, channelTitle):

        if lastTimestamp is not None:

            print("Last Timestamp Found: " + lastTimestamp)

            with open(f"./YouTubeML/{channelTitle}/{self.lastTimestampFileName}", "w") as ltsF:
                ltsF.write(lastTimestamp)
        else:
            print("No Last Timestamp Found")
            pass

        return None


    def loadLastTimestampFile(self):

        yt = YTDataConversion(self.apiKey, self.channelID)
        channelTitle = yt.findChannelTitle()

        if Path(f"./YouTubeML/{channelTitle}/{self.lastTimestampFileName}").is_file() is True:
            with open(f"./YouTubeML/{channelTitle}/{self.lastTimestampFileName}", "r") as ltsF:
                lastTimestamp = ltsF.readlines()[-1]
            print("\033[92mlastTimestampFile.txt loaded\033[0m")

            return lastTimestamp

        else:
            print("\033[91mCouldn't load lastTimestampFile.txt because it wasn't in the YouTuber's folder\033[0m")
            return None



    def delLastTimestampFile(self):

        yt = YTDataConversion(self.apiKey, self.channelID)
        channelTitle = yt.findChannelTitle()

        if Path(f"./YouTubeML/{channelTitle}/{self.lastTimestampFileName}").is_file() is True:
            os.remove(f"./YouTubeML/{self.lastTimestampFileName}")
            print("\033[92mlastTimestampFile.txt deleted\033[0m")
            return None

        else:
            print("\033[91mCouldn't delete lastTimestampFile.txt because it wasn't in the YouTuber's folder\033[0m")
            return None


    def getChannelInfo(self):

        lastVideoDataKey = list(self.videoData.keys())[-1]
        lastVideo = self.videoData[lastVideoDataKey]

        #print(lastVideo) #This can be useful in case of errors raised because of dictionary keys not found

        lastTimestamp = lastVideo.get("publishedAt", None)
        channelTitle = lastVideo.get("channelTitle", None)

        if lastTimestamp is None:

            try:
                lastTimestamp = lastVideo["snippet"].get("publishedAt", None)
            except KeyError:
                print("Couldn't find lastTimestamp")
                exit()

        else:
            pass

        if channelTitle is None:

            try:
                channelTitle = lastVideo["snippet"].get("channelTitle", None)
            except KeyError:
                print("Couldn't find channel title")
                exit()

        else:
            pass

        channelTitle = clean(channelTitle, no_emoji=True, to_ascii=True)
        channelTitle = channelTitle.replace(" ", "_").lower()

        return lastTimestamp, channelTitle


    def writeModeFile(self, mode):

        YTDC = YTDataConversion(apiKey=self.apiKey, channelId=self.channelID)
        filename = YTDC.findChannelTitle()

        with open(f"YouTubeML/{filename}/downloadModeFile.txt", "w") as m:
            m.write(mode)

        return None


    def deleteModeFile(self):

        YTDC = YTDataConversion(apiKey=self.apiKey, channelId=self.channelID)
        filename = YTDC.findChannelTitle()

        os.remove(f"YouTubeML/{filename}/downloadModeFile.txt")

        return None


#Using inheritance to define classes which will use different approach to download data. The approach is defined as "mode"

#This class will collect videos data by changing URLs and using the requests module to retrieve the data
class YTStatsDownloaderManual(YTStatsDownloader):

    def __init__(self, apiKey, channelId):
        super().__init__(apiKey, channelId)
        self.mode = "Manual"

    def __repr__(self):
        return f"\nMode: {self.mode} | API KEY: {self.apiKey} | YouTube Channel ID: {self.channelID}\n" #Returns a formatted string with the used API KEY and chosen YouTube Channel ID


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

            #IMPORTANT TO KNOW: Dictionaries can't have the same key in more than one key-value pair, so if duplicated videos (a key-value pair) are ever returned these will just overwrite the old one

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
        if Path(self.lastTimestampFileName).is_file() is False: #If a previous lastTimestamp file doesn't exist this should be the first call of the class in the MainScript
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

        while npt is not None and idx < 9:
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


class YTStatsDownloaderGClient(YTStatsDownloader):

    def __init__(self, apiKey, channelId):
        super().__init__(apiKey, channelId)
        self.youtube = build("youtube", "v3", developerKey=apiKey)
        self.mode = "Google API Python Client"
        self.lastTimestamp = None #Later assigned in the program

    def __len__(self):
        if self.videoData is not None:
            return len(self.videoData) #Prints the number of key-value pairs (videos) that the dictionary contains
        else:
            return "No Data"

    def __repr__(self):
        return f"\nMode: {self.mode} | API KEY: {self.apiKey} | YouTube Channel ID: {self.channelID}\n" #Returns a formatted string with the used API KEY and chosen YouTube Channel ID


    def getChannelStatistics(self):

        channelStatisticsRequest = self.youtube.channels().list(
            part="statistics",
            id=self.channelID
        )

        response = channelStatisticsRequest.execute()

        #print("Full Response: ", response, "\n")

        videoCount = response["items"][0]["statistics"]["videoCount"]
        subsCount = response["items"][0]["statistics"]["subscriberCount"]
        viewCount = response["items"][0]["statistics"]["viewCount"]

        #print("Video count: ", videoCount)
        #print("Subscriber count: ", subsCount)
        #print("View count: ", viewCount)

        return videoCount, subsCount, viewCount


    def getVideoIDs(self):

        videoIDsRequest = None
        videoIDs = []
        npt = None

        self.lastTimestamp = self.loadLastTimestampFile()
        if self.lastTimestamp is not None: print(f"Last Timestamp Found in lastTimestampFile.txt: {self.lastTimestamp}")


        while True:

            if self.lastTimestamp is not None:
                videoIDsRequest = self.youtube.search().list(
                    part="id",
                    channelId=self.channelID,
                    maxResults=50,
                    pageToken=npt,
                    type="video",
                    publishedBefore=self.lastTimestamp,
                    order="date"
                )

            elif self.lastTimestamp is None:
                videoIDsRequest = self.youtube.search().list(
                    part="id",
                    channelId=self.channelID,
                    maxResults=50,
                    pageToken=npt,
                    type="video",
                    order="date"
                )

            response = videoIDsRequest.execute()

            for item in response['items']:
                try:
                    videoIDs.append(item['id']["videoId"])
                except KeyError as itemError:
                    print("\033[91mError in Getting VideoID of Item With ETag:\033[0m", response["etag"], " \033[91mError:\033[0m", itemError)

            npt = response.get('nextPageToken')
            #print(npt)

            if npt is None: break


        nDuplicates = len(videoIDs) - len(set(videoIDs))

        print("Total Videos Returned by The YouTube Data API V3: ", len(videoIDs))

        videoIDs = list(set(videoIDs)) #Using the set() function since the API can give duplicated results
        videoIDs = {i: {} for i in videoIDs}

        #print(videoIDs)
        #print(len(videoIDs)) #Length of the video IDs list obtained without duplicated filtering

        print("Duplicates Count: ", nDuplicates)
        print("Number of Videos Without Duplicates: ", len(videoIDs))

        return videoIDs


    def getVideosData(self, videos):

        toPop = []
        #print(videos)

        for vidId in videos.keys():

            #It's not unusual to get HTTP errors of some sort, for this reason we'll implement a try-except block to manage that
            try:

                #print(f"Requested Video: ", vidId)
                statsRequest = self.youtube.videos().list(
                    part="snippet, statistics, contentDetails, topicDetails",
                    id=vidId
                )

                statsResponse = statsRequest.execute()
                #print(statsRequest)
                #print(statsResponse) #Useful to check if response is not what we're waiting for

                videos[vidId].update(statsResponse["items"][0])

            except Exception as Error:
                print(f"\033[91mError in Getting Statistics for Video:\033[0m", vidId, "\033[91mWith Error:\033[0m", Error)
                toPop.append(vidId)

        all(videos.pop(vid) for vid in videos.keys() if vid in toPop) #In case we couldn't fetch the data for a specific video we'll then just remove it from the dictionary

        print(f"Total Errors During Statistics Fetching Phase: {len(toPop)}")

        #print(videos)

        return videos #This is a dataframe containing all the videos and relative data


    #nCycles is the number of times the data collection function will run
    def executeDataCollectionProcess(self, nCycles: int):

        self.channelStatistics = list(self.getChannelStatistics())
        print("Video Count: ", self.channelStatistics[0], " | Subscriber Count: ", self.channelStatistics[1], " | View Count: ", self.channelStatistics[2])

        videoIDsCollection = self.getVideoIDs()

        for j in range(nCycles):

            if self.videoData is None:
                self.videoData = self.getVideosData(videoIDsCollection)

            elif isinstance(self.videoData, dict):
                self.videoData.update(self.getVideosData(videoIDsCollection))

                print(f"Updated Videos Dictionary Length at the {j}th Cycle: {len(self.videoData)}")

            else:
                raise Exception("Videos Data Is Not Stored in a Dictionary")


        return None


#Using polymorphism to create a class identical to the one which it obtains methods and attributes, but with redefined methods or attributes
#In this case we're creating a class with the same properties as the YTStatsDownloaderGClient, but with a different getVideoIDs() method which implements a different approach
#Also the mode class attribute will be different
#Important to know that the Uploads Playlist method is significantly cheaper in terms of API quotas used compared to the other ones
#To know: there's an API requests limit of 20.000 by design, so that should be the maximum number of videos we could retrieve in one go
class YTStatsDownloaderGClientUploadsPlaylist(YTStatsDownloaderGClient):
    def __init__(self, apiKey, channelId):
        super().__init__(apiKey, channelId)
        self.youtube = build("youtube", "v3", developerKey=apiKey)
        self.mode = "Google API Python Client - Uploads Playlist Method"


    def getVideoIDs(self):

        #IMPORTANT: Every channel has a so-called "Uploads Playlist" which includes all the non-private videos uploaded from the YT channel itself
        uploadsPlaylistRequest = self.youtube.channels().list(
                                                              fields='items/contentDetails/relatedPlaylists/uploads',
                                                              part='contentDetails',
                                                              id=self.channelID,
                                                              maxResults=1
                                                          )

        uploadsPlaylistResponse = uploadsPlaylistRequest.execute()
        items = uploadsPlaylistResponse.get("items", None)

        if items is not None:
            uploadsPlaylistID = items[0]['contentDetails']['relatedPlaylists'].get('uploads')
            #print(uploadsPlaylistID)

        else:
            raise Exception("Uploads Playlist Not Found")


        videoIDs = []


        self.lastTimestamp = self.loadLastTimestampFile()
        if self.lastTimestamp is not None: print(f"Last Timestamp Found in lastTimestampFile.txt: {self.lastTimestamp}")


        # We don't need to specify again the channelID because we already did it in the previous request for the Uploads Playlist which supposedly should be already associated to the YouTube channel
        videoIDsRequest = self.youtube.playlistItems().list(
            part="id, snippet",
            playlistId=uploadsPlaylistID,
            maxResults=50,
        )


        #The loop runs as long as a response exists
        while videoIDsRequest:

            response = videoIDsRequest.execute()


            #Differently from the other methods, with the Uploads Playlist one you'll receive a youtube#playlistItem which has an id that doesn't correspond to the video one, in fact the video ID is found in the "snippet" dictionary in the "resourceId" key (which is a dictionary too) and in the "videoId" key
            #We can confirm that the item we retrieved is actually a video by checking the "kind" key
            for item in response['items']:
                if item['snippet']['resourceId']['kind'] == 'youtube#video':
                    videoIDs.append(item["snippet"]["resourceId"]["videoId"])
                    #print(item)
                else:
                    print("Not Video Resource Found With ID", item["id"])


            videoIDsRequest = self.youtube.playlistItems().list_next(videoIDsRequest, response) #This will let us obtain the next chunk of video IDs




        nDuplicates = len(videoIDs) - len(set(videoIDs))

        print("Total Videos Returned by The YouTube Data API V3: ", len(videoIDs))

        videoIDs = list(set(videoIDs))  # Using the set() function since the API can give duplicated results
        videoIDs = {i: {} for i in videoIDs}

        # print(videoIDs)
        # print(len(videoIDs)) #Length of the video IDs list obtained without duplicated filtering

        print("Duplicates Count: ", nDuplicates)
        print("Number of Videos Without Duplicates: ", len(videoIDs))

        return videoIDs



























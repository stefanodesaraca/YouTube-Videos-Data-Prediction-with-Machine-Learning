from YTStatistics import YTStatsDownloaderManual, YTStatsDownloaderGClient, YTStatsDownloader, YTStatsDownloaderGClientUploadsPlaylist
from YTDataFormatConversion import convertManualFileType, convertGoogleClientFileType, YTDataConversion
from YTMLAnalysis import YouTubeML
import YTAPIFile
import math


API_KEY = YTAPIFile.KEY
channel_id = YTAPIFile.ID



def GetYTStats(mode: str):

    if mode == "1":

        yt = YTStatsDownloaderManual(apiKey=API_KEY, channelId=channel_id)

        YTData = yt.getChannelStatistics()

        #nVideos = YTData["videoCount"]
        #nRuns = math.ceil(int(nVideos)/500) #Number of videos divided by the max amount of videos obtainable by one query

        nRuns = 1

        for _ in range(nRuns):

            yt.getChannelVideoData()
            yt.dump()

        #yt.delLastTimestampFile() #When the whole channel's videos download is finished then delete lastTimeStampFile.txt
        print(YTData)

        yt.writeModeFile(yt.__class__.__name__)

        return None

    elif mode == "2":

        yt = YTStatsDownloaderGClient(apiKey=API_KEY, channelId=channel_id)
        print(repr(yt))

        nVideos, _, _ = yt.getChannelStatistics()

        #nRuns = math.ceil(int(nVideos)/500) #Number of videos divided by the max amount of videos obtainable by one query
        yt.executeDataCollectionProcess(nCycles=1) #The for cycle in this case is already built-in the function

        print("Number of Videos With Collected Statistics: ", len(yt))

        yt.dump()

        yt.writeModeFile(yt.__class__.__name__)

        #yt.delLastTimestampFile()

        return None


    elif mode == "3":

        yt = YTStatsDownloaderGClientUploadsPlaylist(apiKey=API_KEY, channelId=channel_id)
        print(repr(yt))

        nVideos, _, _ = yt.getChannelStatistics()

        # nRuns = math.ceil(int(nVideos)/500) #Number of videos divided by the max amount of videos obtainable by one query
        yt.executeDataCollectionProcess(nCycles=1)  # The for cycle in this case is already built-in the function

        print("Number of Videos With Collected Statistics: ", len(yt))

        yt.dump()

        yt.writeModeFile(yt.__class__.__name__)

        # yt.delLastTimestampFile()

        return None


    else:
        raise Exception("Wrong Download Mode. Restart the Program and Choose a Valid One")


def ConvertJSONDataIntoCSV(mode: str):

    if mode == "1":

        YTConvertor = convertManualFileType(apiKey=API_KEY, channelId=channel_id)

        filename = YTConvertor.findChannelTitle()
        print("File Name: " + filename + ".json")

        YTConvertor.convertManualJSONToCSV()


    elif mode == "2":

        YTConvertor = convertGoogleClientFileType(apiKey=API_KEY, channelId=channel_id)

        filename = YTConvertor.findChannelTitle()
        print("File Name: " + filename + ".json")

        YTConvertor.convertGClientJSONtoCSV()


    else:
        raise Exception("Wrong Download Mode. Restart the Program and Choose a Valid One")


    return None


def AnalyzeML():

    titleFinder = YTDataConversion(API_KEY, channel_id)
    channelTitle = titleFinder.findChannelTitle()

    YTA = YouTubeML(channelTitle) #YouTubeML Object names "YouTube Analysis"

    print(repr(YTA)) #Representation of the API KEY and Channel ID used in the YouTubeML class

    YTA.setYTDFs()

    YTA.ExploratoryDataAnalysis()

    YTA.VariableSelection()
    YTA.executeMLAnalysis()



    return None


def createFolders():

    titleFinder = YTDataConversion(API_KEY, channel_id)
    channelTitle = titleFinder.findChannelTitle()

    yt = YTStatsDownloader(API_KEY, channel_id)
    yt.createYouTubeMLDir(channelTitle)

    return None



def main():
    while True:
        print("1. Download YouTube Videos Data")
        print("2. Convert JSON Data Into CSV File")
        print("3. Analyze Dataset")
        print("4. Create Custom Folders for The YouTube Channel")
        print("0. Exit")

        option = input("Choice: ")
        print()

        if option == "1":

            print("""Choose Download Method:
            1. YouTube Data API V3 - Manual Mode
            2. YouTube Data API V3 - Google Client Mode (Classic)
            3. YouTube Data API V3 - Google Client Mode (Uploads Playlist Method)
            """)

            downloadOption = input("Input Option: ")

            if downloadOption in ["1", "2", "3"]:
                GetYTStats(downloadOption)

            else:
                print("Insert a valid option")
            print()


        elif option == "2":

            print("""Choose Conversion Method:
            1. YouTube Data API V3 - Manual Mode
            2. YouTube Data API V3 - Google Client Mode (Classic or Uploads Playlist Method)
            Use the Equivalent For the Download Method You Have Chosen Before
            """)

            conversionOption = input("Input Option: ")
            if conversionOption == "1":
                ConvertJSONDataIntoCSV(conversionOption)

            elif conversionOption == "2":
                ConvertJSONDataIntoCSV(conversionOption)

            else:
                print("Insert a valid option")
            print()


        elif option == "3":
            AnalyzeML()
            print()

        elif option == "4":
            createFolders()
            print()

        elif option == "0":
            break

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()








































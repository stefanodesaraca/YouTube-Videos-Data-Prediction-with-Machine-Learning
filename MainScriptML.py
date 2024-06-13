from YTStatistics import YTStats
from YTDataFormatConvertion import YTDataConvertion
from YTMLAnalysis import YouTubeML
import YTAPIFile
import math


API_KEY = YTAPIFile.KEY
channel_id = YTAPIFile.ID


def GetYTStats():

    yt = YTStats(apiKey=API_KEY, channelId=channel_id)

    YTData = yt.getChannelStatistics()

    #nVideos = YTData["videoCount"]
    #nRuns = math.ceil(int(nVideos)/500) #Number of videos divided by the max amount of videos obtainable by one query

    nRuns = 1

    for _ in range(nRuns):

        yt.getChannelVideoData()
        yt.dump()

    #yt.delLastTimestampFile() #When the whole channel's videos download is finished then delete lastTimeStampFile.txt
    print(YTData)

    return None


def ConvertJSONIntoCSV():

    YTConvertor = YTDataConvertion(apiKey=API_KEY, channelId=channel_id)

    filename = YTConvertor.findFileName()
    print("File Name: " + filename + ".json")

    YTConvertor.convertJSONToCSV()

    return None


def AnalyzeML():

    YTA = YouTubeML()

    print(repr(YTA)) #Representation of the API KEY and Channel ID used in the YouTubeML class

    YTA.createYTMLDocumentsDir() #Creating Necessary Folders
    YTA.setYTDFs()

    #YTA.ExploratoryDataAnalysis()

    YTA.VariableSelection()
    YTA.executeMLAnalysis()



    return None


def main():
    while True:
        print("1. Download YouTube Videos Data")
        print("2. Convert JSON Data Into CSV File")
        print("3. Analyze Dataset")
        print("0. Exit")
        option = input("Choice: ")
        print()

        if option == "1":
            GetYTStats()
            print()

        elif option == "2":
            ConvertJSONIntoCSV()
            print()

        elif option == "3":
            AnalyzeML()
            print()

        elif option == "0":
            break

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()








































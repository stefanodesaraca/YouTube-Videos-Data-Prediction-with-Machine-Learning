import YTAPIFile
from YTDataFormatConvertion import YTDataConvertion #To extract the fileName calling the method of the class
import pandas as pd
from scipy import stats #CAVEAT For Shapiro-Wilk Normality Test
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import numpy as np
from tqdm import tqdm
import pickle
import os
from collections import ChainMap
from functools import wraps
from warnings import simplefilter
import inspect
import warnings
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


from sklearn.preprocessing import MinMaxScaler, RobustScaler #MinMax Scaler and (Standard Scaler + Z-Score)
from sklearn.feature_selection import SelectFromModel #Forward Method and L1 Penalty
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error


def savePlots(plotFunction):

    def checkPlots(plotNames, plots):
        if isinstance(plotNames, list) and isinstance(plots, list):
            return True
        else:
            #print("\033[91mCheckPlots: object obtained are not lists\033[0m")
            return False

    def checkPlotsTypeAndSave(plotName, plots, filePath):
        if isinstance(plots, (plt.Figure, plt.Axes, sns.axisgrid.FacetGrid, sns.axisgrid.PairGrid, list)):
            plt.savefig(f"{filePath}{plotName}.png", dpi=300)
            print(f"{plotName} Exported Correctly")

        elif isinstance(plots, plotly.graph_objs._figure.Figure):
            plots.write_html(f"{filePath}{plotName}.html")
            print(f"{plotName} Exported Correctly")

        else:
            try:
                plt.savefig(f"{filePath}{plotName}.png", dpi=300)
                print(f"{plotName} Exported Correctly")
            except:
                print("\033[91mExporting the plots wasn't possible, the returned type is not included in the decorator function\033[0m")

        return None

    @wraps(plotFunction)
    def wrapper(*args, **kwargs):

        plotsNames, generatedPlots, filePath = plotFunction(*args, **kwargs)
        #print("File path: " + filePath)

        if checkPlots(plotsNames, generatedPlots) == True:

            for plotName, plot in zip(plotsNames, generatedPlots):
                checkPlotsTypeAndSave(plotName, plot, filePath)

        elif checkPlots(plotsNames, generatedPlots) == False:
            #print("Saving Single Graph...")
            checkPlotsTypeAndSave(plotsNames, generatedPlots, filePath)

        else:
            print(f"\033[91mExporting the plots wasn't possible, here's the data types obtained by the decorator: PlotNames: {type(plotsNames)}, Generated Plots (could be a list of plots): {type(generatedPlots)}, File Path: {type(filePath)}\033[0m")

        return None

    return wrapper



class YouTubeML:
    def __init__(self):
        self.fileName = None #Assigned later in the program
        self.apiKey = YTAPIFile.KEY
        self.channelID = YTAPIFile.ID
        self.currentWorkingDirectoryPath = os.path.abspath(os.getcwd())
        self.plotsPath = "./YTMLDocuments/YTDataPlots/"
        self.modelsPath = "./YTMLDocuments/YTDataModels/"
        self.TestScorePlotsPath = "./YTMLDocuments/YTDataPlots/TestScorePlots/" #TODO EDIT THESE NAMES REMOVING "TEST", IT'S USELESS
        self.modelsReportPath = "./YTMLDocuments/YTModelsReport/"
        self.testReportPath = "./YTMLDocuments/YTModelsReport/TestReports/"
        self.EDAYTDF = None #YouTube EDA DataFrame available to all class functions
        self.YTDF = None #GLOSS YouTube Data DataFrame for ML related functions
        self.targetVariablesDict = {"likeViewRatio": 3, 'videoAge': 4, 'viewCount': 0, 'totalSecondsDuration': 1, 'commentViewRatio': 2}
        self.FeatureSelectionTTSVariables = None #This will be a list containing just the names of the variables that we'll use later in the feature selection process
        self.bar = "--------------------------------------------------------------"
        self.FeaturesDict = {}
        self.models = [RandomForestRegressor(n_jobs=-1, random_state=100), BaggingRegressor(random_state=100), AdaBoostRegressor(random_state=100), KNeighborsRegressor(n_jobs=-1), DecisionTreeRegressor(random_state=100), GradientBoostingRegressor(random_state=100), XGBRegressor(n_jobs=-1, random_state=100)]
        self.modelNames = ["RandomForestRegressor", "BaggingRegressor", "AdaBoostRegressor", "KNeighborsRegressor", "DecisionTreeRegressor", "GradientBoostingRegressor", "XGBRegressor"]
        self.targetFeaturesTrainMLDict = {}
        self.targetFeaturesTestMLDict = {}
        self.scalingLevelHigh = False
        self.CVShuffle = True
        self.scoring = {"R2": make_scorer(r2_score),
                        "MSE": make_scorer(mean_squared_error),
                        "RMSE": make_scorer(root_mean_squared_error),
                        "MAE": make_scorer(mean_absolute_error)}

    def __repr__(self):
        return f"API KEY: \033[92m{self.apiKey}\033[0m and YouTube Channel ID: \033[91m{self.channelID}\033[0m\n" #Returns a formatted string with the used API KEY and chosen YouTube Channel ID


    def getCSVFileName(self):

        YTDC = YTDataConvertion(self.apiKey, self.channelID)

        try:
            channelTitle = YTDC.findFileName()
            fileName = channelTitle + "CSV.csv"

            self.fileName = fileName

        except FileNotFoundError:
            print("File Not Found")

        return None


    def setYTDFs(self):

        try:
            self.getCSVFileName()
            self.EDAYTDF = pd.read_csv(self.fileName)
            self.EDAYTDF["publishedAt"] = pd.to_datetime(self.EDAYTDF["publishedAt"])

            self.YTDF = self.EDAYTDF[['viewCount', 'totalSecondsDuration', 'commentViewRatio', "likeViewRatio", 'videoAge']]

            self.FeatureSelectionTTSVariables = [x for x in self.YTDF.columns] #Adding column names to self.FeatureSelectionTTSVariables so we can later use them in feature selection

            self.setScaling(scalingLevelH=self.scalingLevelHigh) #Scaling the data, by default self.scalingLeveHigh is False

            print("\nDataFrame Shape After Scaling: ", self.YTDF.shape)

            return None

        except FileNotFoundError:
            print("\033[91mFile Not Found\033[0m")
            raise FileNotFoundError("File not found, have you set the correct channel ID or converted the JSON file into CSV?")


    def setScaling(self, scalingLevelH):

        #I decided to implement a scaling level supposing that not all YouTube channel videos datasets are the same, so if needed you can use the RobustScaler, otherwise you can leave the MinMaxScaler as default
        if scalingLevelH == False:

            scaler = MinMaxScaler()
            self.YTDF["commentViewRatio"] = scaler.fit_transform(self.YTDF[["commentViewRatio"]])
            self.YTDF["likeViewRatio"] = scaler.fit_transform(self.YTDF[["likeViewRatio"]])
            #self.YTDF["videoAge"] = scaler.fit_transform(self.YTDF[["videoAge"]])

        elif scalingLevelH == True:

            scaler = RobustScaler()
            self.YTDF["commentViewRatio"] = scaler.fit_transform(self.YTDF[["commentViewRatio"]])
            self.YTDF["likeViewRatio"] = scaler.fit_transform(self.YTDF[["likeViewRatio"]])
            self.YTDF["videoAge"] = scaler.fit_transform(self.YTDF[["videoAge"]])

        #NOTE In case of future maintenance this is to ensure that the structure of the setScaling function remains the same
        else:
            print("\033[91mScaling Level Not Found\033[0m")
            raise Exception("Scaling Level Not Set. Control Scaling Level and Restart the Program")

        #print(self.YTDF)

        return None


    #CAVEAT Declaring ZScore as a static method means that it still is a method of the class, but it's not bound to the class instance (object), so it doesn't use the self parameter because it can't modify the class state
    @staticmethod
    def ZScore(DF, column):
        DF["ZScore"] = (DF[column] - DF[column].mean()) / DF[column].std()
        filteredDF = DF[(DF["ZScore"] > -3) & (DF["ZScore"] < 3)]  # Filtering all the records that have on the column we'd like to apply a z score greater than 3 (which is the typical range between 99.7% of the data falls in the Gaussian distribution)
        filteredDF = filteredDF.drop(columns="ZScore")
        return filteredDF


    @savePlots
    def ShapiroWilkTest(self, targetFeatureName, data):

        plotName = targetFeatureName + inspect.currentframe().f_code.co_name

        print(f"Shapiro-Wilk Normality Test On \033[92m{targetFeatureName}\033[0m Target Feature")
        _, SWH0PValue = stats.shapiro(data) #Executing the Shapiro-Wilk Normality Test - This method returns a 'scipy.stats._morestats.ShapiroResult' class object with two parameters inside, the second is the H0 P-Value
        #print(type(stats.shapiro(data)))
        print(f"Normality Probability (H0 Hypothesis P-Value): \033[92m{SWH0PValue}\033[0m")

        fig, ax = plt.subplots()
        SWQQPlot = stats.probplot(data, plot=ax)
        ax.set_title(f"Probability Plot for {targetFeatureName}")

        return plotName, SWQQPlot, self.plotsPath


#TODO SAVE PLOTS DIRECTLY INTO THEIR FOLDER WITHOUT THE ADDITIONAL TestScorePlots FOLDER
    def createYTMLDocumentsDir(self):
        #Creating YTMLDocuments Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YTMLDocuments")
        except FileExistsError:
            print("\033[91mYTMLDocuments folder already exists\033[0m")
            pass

        #Creating Models File Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YTMLDocuments/YTDataModels")
        except FileExistsError:
            print("\033[91mYTDataModels folder already exists\033[0m")
            pass

        #Creating Plots Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YTMLDocuments/YTDataPlots")
        except FileExistsError:
            print("\033[91mYTDataPlots folder already exists\033[0m")
            pass

        #Creating Models Test Plots Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YTMLDocuments/YTDataPlots/TestScorePlots")
        except FileExistsError:
            print("\033[91mModels Score Plot folder already exists\033[0m")
            pass

        #Creating Models Report Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YTMLDocuments/YTModelsReport")
        except FileExistsError:
            print("\033[91mYTModelsReport folder already exists\033[0m")
            pass

        #Creating Models Training and Testing Reports Folder
        try:
            os.mkdir(f"{self.currentWorkingDirectoryPath}/YTMLDocuments/YTModelsReport/TestReports")
        except FileExistsError:
            print("\033[91mModel Test Reports folders already exists\033[0m")
            pass

        return None


    def ExploratoryDataAnalysis(self):

        if len(self.EDAYTDF.index) > 450: #Leaving room for a good number of videos to analyze

            print(self.bar)
            print(f"\nNumber of Rows Before Z-Score: {len(self.EDAYTDF.index)}")

            #Calling self in front of ZScore (a static method) is fine, it will just be ignored
            #Executing Z-Score on columns which are more likely to contain outliers
            self.EDAYTDF = self.ZScore(self.EDAYTDF, ["commentCount"])  #Applying a Z-Score to the commentCount column to filter all outliers rows for that parameter

            self.EDAYTDF = self.ZScore(self.EDAYTDF, ["totalSecondsDuration"])
            self.EDAYTDF = self.ZScore(self.EDAYTDF, ["totalSecondsDuration"])  #IDEA Executing it twice for better results, but this is optional

            self.EDAYTDF = self.ZScore(self.EDAYTDF, ["viewCount"])

            print(f"Number of Rows After Z-Score: {len(self.EDAYTDF.index)}")
            print("Z-Score Applied Successfully\n")
            print(self.bar + "\n")

        else:
            pass


        YTNumVariablesCorrelation = self.EDAYTDF.corr(numeric_only=True)

        print("Correlation Between Numeric Variables of the DataFrame")
        print(YTNumVariablesCorrelation)
        print()


        print(self.bar)
        #Executing the Shapiro-Walk Test for every target feature
        for targetFeature in self.targetVariablesDict.keys():
            self.ShapiroWilkTest(targetFeature, self.EDAYTDF[targetFeature])

        print(self.bar + "\n")


        print(self.EDAYTDF[["viewCount", "likeCount", "commentCount", "totalSecondsDuration", "commentViewRatio", "likeViewRatio", "videoAge"]].describe())

        print("\nTop 20 Videos by View Count")
        print(self.EDAYTDF[["title", "viewCount", "publishedAt"]].sort_values(by="viewCount", ascending=False).head(20))

        print("\nTop 20 Videos by Like Count")
        print(self.EDAYTDF[["title", "likeCount", "publishedAt"]].sort_values(by="likeCount", ascending=False).head(20))

        print("\nLast 5 Videos by View Count")
        print(self.EDAYTDF[["title", "viewCount", "publishedAt"]].sort_values(by="viewCount").head(5))

        print("\nLast 5 Videos by Like Count")
        print(self.EDAYTDF[["title", "likeCount", "publishedAt"]].sort_values(by="likeCount").head(5))

        print()

        self.createExploratoryAnalysisPlots()

        return None


    def createExploratoryAnalysisPlots(self):

        @savePlots
        def YTNumDataCorrPlot():
            plotName = inspect.currentframe().f_code.co_name #Getting current function name
            YTNumDataCorrGraph = sns.pairplot(self.EDAYTDF, hue="viewCount", palette="viridis", height=3.5, aspect=1.5)
            #YTNumDataCorrPlot.fig.suptitle("Correlation Between Numeric Variables of the DataFrame", y=1.05)
            #plt.tight_layout()
            #print(type(YTNumDataCorrGraph))
            return plotName, YTNumDataCorrGraph, self.plotsPath

        @savePlots
        def YTDataLinePlot():
            plotName = inspect.currentframe().f_code.co_name
            plt.figure(figsize=(16, 9))
            YTDataLinePlot = sns.lineplot(self.EDAYTDF[["likeCount", "commentCount"]]).set(title='Likes and Comments Line Plot')
            #YTDataLinePlot.set_axis_labels(x_var="All Videos", y_var="Likes / Comments", fontdict=dict(weight="bold"))
            plt.tight_layout()
            #print(type(YTDataLinePlot))
            return plotName, YTDataLinePlot, self.plotsPath

        @savePlots
        def viewsDistributionPlot():
            plotName = inspect.currentframe().f_code.co_name
            plt.figure(figsize=(16, 9))
            viewsDistributionGraph = sns.displot(self.EDAYTDF, x="viewCount", hue="viewCount", multiple="stack", fill=True, common_norm=False, palette="flare", alpha=0.5, linewidth=0, legend=False).set(title='Likes and Comments Line Plot')
            #viewsDistributionGraph.set_titles(template='Views Histogram Plot', fontdict=dict(weight="bold"))
            viewsDistributionGraph.set_xlabels(label="View Count", fontdict=dict(weight="bold"))
            #plt.tight_layout()
            return plotName, viewsDistributionGraph, self.plotsPath

        @savePlots
        def secondsDurationAndLikesPlot():
            plotName = inspect.currentframe().f_code.co_name
            plt.figure(figsize=(16, 9))
            secondsDurationAndLikesGraph = sns.scatterplot(self.EDAYTDF, x="likeCount", y="totalSecondsDuration", hue="likeCount", palette="magma", size="likeCount", sizes=(20, 200))
            secondsDurationAndLikesGraph.set_title("Videos Duration Histogram", fontdict=dict(weight="bold"))
            secondsDurationAndLikesGraph.set_ylabel("Duration (Seconds)", fontdict=dict(weight="bold"))
            secondsDurationAndLikesGraph.set_xlabel("Like Count", fontdict=dict(weight="bold"))
            #plt.tight_layout()
            return plotName, secondsDurationAndLikesGraph, self.plotsPath

        @savePlots
        def likeViewRatioScatterPlot():
            plotName = inspect.currentframe().f_code.co_name
            plt.figure(figsize=(16, 9))
            viewLikeRatePlot = sns.scatterplot(data=self.EDAYTDF, x="viewCount", y="likeViewRatio", hue="likeViewRatio", palette="viridis", size="likeViewRatio", sizes=(20, 200), legend=False)
            viewLikeRatePlot.set_title("Like/View Rate Scatterplot", fontdict=dict(weight="bold"))
            viewLikeRatePlot.set_ylabel("Like/View Rate")
            viewLikeRatePlot.set_xlabel("View Count")
            #plt.tight_layout()

            return plotName, viewLikeRatePlot, self.plotsPath

        @savePlots
        def groupedFeaturesBarPlot():

            plotName = inspect.currentframe().f_code.co_name
            groupedYearsDF = self.EDAYTDF[["totalSecondsDuration", "commentViewRatio"]].groupby(self.EDAYTDF['publishedAt'].dt.year).median()
            groupedYearsDF.reset_index(inplace=True)
            #print(groupedYearsDF)

            barWidth = 0.25
            barPositions = range(len(groupedYearsDF))

            fig, ax0 = plt.subplots(figsize=(16, 9))

            ax0.set_title("Median Value of Video Duration and Comment/View Ratio", fontdict=dict(weight="bold"))

            ax0.bar(x=[pos - barWidth / 2 for pos in barPositions], height=groupedYearsDF["totalSecondsDuration"], label="Video Duration (Seconds)", width=0.25, edgecolor="white", linewidth=0.7, color="#115f9a")

            ax0.set_xlabel("Year", fontdict=dict(weight="bold"))
            ax0.set_ylabel("Median Video Duration (Seconds)", fontdict=dict(weight="bold"))
            ax0.set_xticks(barPositions)
            ax0.set_xticklabels(groupedYearsDF["publishedAt"])
            ax0.tick_params(axis='y')

            ax1 = ax0.twinx()

            ax1.bar(x=[pos + barWidth / 2 for pos in barPositions], height=groupedYearsDF["commentViewRatio"], label="Comment/View Ratio", width=0.25, edgecolor="white", linewidth=0.7, color="#5ad45a")
            ax1.set_ylabel("Median Comment/View Ratio", fontdict=dict(weight="bold"))
            ax1.tick_params(axis='y')

            fig.legend(loc="upper right", prop=dict(size="medium"))

            return plotName, fig, self.plotsPath

        @savePlots
        def likesByYearSubPlots():
            plotName = inspect.currentframe().f_code.co_name
            groupedYearsDF = self.EDAYTDF[["likeCount"]].groupby(self.EDAYTDF['publishedAt'].dt.year).median()
            groupedYearsDF.reset_index(inplace=True)

            colorPalette = sns.color_palette('viridis')

            fig, axes = plt.subplots(1, 2, figsize=(25, 10))

            axes[0].bar(x=groupedYearsDF['publishedAt'], height=groupedYearsDF["likeCount"])
            axes[0].grid(axis="y")
            axes[0].tick_params(axis='x', rotation=-45)
            axes[0].set_xlabel('Year', fontweight='bold', fontsize=10)
            axes[0].set_ylabel('Like Count', fontweight='bold', fontsize=10)
            axes[0].set_title('Yearly Likes Total', fontweight='bold', fontsize=20)

            donutLabels = [f"{i} ({int(j)})" for i, j in zip(groupedYearsDF['publishedAt'], groupedYearsDF['likeCount'])]
            axes[1].pie(groupedYearsDF['likeCount'], labels=donutLabels, autopct="%.2f%%", textprops=dict(fontsize=13), colors=colorPalette)
            axes[1].axis("equal")

            circle = plt.Circle(xy=(0, 0), radius=0.75, facecolor='white')
            axes[1].add_artist(circle)

            axes[1].set_title("Total Likes by Years (Donut Plot)", x=0.5, y=1.05, fontweight='bold', fontsize=20)
            axes[1].annotate(text=f'Total Likes: {int(groupedYearsDF["likeCount"].sum())}', xy=(0, 0), ha='center', weight='bold', size=20)

            #print(type(plt))

            return plotName, fig, self.plotsPath


        plotsFunctionsList = [YTNumDataCorrPlot, YTDataLinePlot, viewsDistributionPlot, secondsDurationAndLikesPlot, likeViewRatioScatterPlot, groupedFeaturesBarPlot, likesByYearSubPlots]
        all((plotsFunctionsList[i](), plt.clf()) for i in tqdm(range(len(plotsFunctionsList)))) #Executing comprehension without generating a list using the all() method

        return None


    def getMoreMetrics(self, featureUsed, VariableSelectionMethod, scoringName, metricsDataFrame):

        avgScoreValueIndex = metricsDataFrame["avg_score"].idxmax()
        avgScoreValueRow = metricsDataFrame.loc[avgScoreValueIndex]

        averageScoreValue = avgScoreValueRow["avg_score"]
        averageScoreFeatures = avgScoreValueRow["feature_names"]

        #print(self.bar)
        #print(f"\033[92m{VariableSelectionMethod} Metrics DataFrame For Feature (\033[93m{featureUsed}\033[0m\033[92m): \033[0m")
        #print(metricsDataFrame)
        #print(self.bar)
        #print("Average Score Value Row: ")
        #print(avgScoreValueRow)
        print(self.bar)
        print(f"Average {scoringName} Score Value For Feature (\033[93m{featureUsed}\033[0m): \033[92m{averageScoreValue}\033[0m")
        #print(f"Chosen Feature Names: \033{averageScoreFeatures}\033[0m")
        print(self.bar)

        return None


    def VariableSelection(self):

        #print(f"DataFrame Shape: {self.YTDF.shape}\n")

        def getForwardMethod():

            ForwardRegMetricsDict = {}
            #Executing Variable Selection for Regression Models
            for feat, i in self.targetVariablesDict.items():
                featuresSelected = self.ForwardMethodFS(feat, i)
                ForwardRegMetricsDict.update({feat: list(featuresSelected)}) #Creating a dictionary of lists where every key is the target variable taken in consideration and the list values are the selected variables

            #selectedFeaturesFWDF = pd.DataFrame(ForwardRegMetricsDict)

            #print("Three Features Selected for Each Target Feature: ")
            #print(selectedFeaturesFWDF)
            #print()

            return ForwardRegMetricsDict

        def getL1Method():

            L1FeaturesDict = {}
            #Executing Variable Selection for Regression Models
            for feat, i in self.targetVariablesDict.items():
                featuresSelected = self.L1PenaltyFS(feat, i)
                L1FeaturesDict.update({feat: list(featuresSelected)}) #Creating a dictionary of lists where every key is the target variable taken in consideration and the list values are the selected variables

            print(self.bar)
            #print("Features Selected for Each Target Feature: ")
            #print(L1FeaturesDict)
            #print()

            return L1FeaturesDict

        print("\n\033[94mForward Method\033[0m")
        print("==============================================================\n")
        FWFeaturesDict = getForwardMethod()

        print("\n\n\033[94mLasso L1 Penalty Method\033[0m")
        print("==============================================================\n")
        L1FeaturesDict = getL1Method()


        FirstFeature = dict([FWFeaturesDict.popitem()]) #Contains the first target feature's chosen variables

        L1FeaturesDict.popitem() #Removing the first key-value pair that we won't use
        OtherFeatures = L1FeaturesDict

        self.FeaturesDict = dict(ChainMap(FirstFeature, OtherFeatures)) #Adding one key-value pair to a dictionary
        self.FeaturesDict["likeViewRatio"] = ['viewCount', 'totalSecondsDuration', 'commentViewRatio', 'videoAge'] #Updating only the likeViewRatio feature predictors as request by the project

        print("\n\033[94mFeatures Selected For Each Target Variable Dictionary: \033[0m")
        print(self.FeaturesDict)
        print()

        return None


    def ForwardMethodFS(self, feature, targetColumn): #Feature Selection

        scoreName = "r2"

        X = self.YTDF.loc[:, self.FeatureSelectionTTSVariables] #All the rows and only specific columns declared in self.FeatureSelectionTTSVariables in the setYTDF() function, be aware that since only here we are using string indexers for columns we're using the .loc method and NOT .iloc
        Y = self.YTDF.iloc[:, targetColumn] #Setting the target variable for each for cycle

        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.3, random_state=100) #Setting test dataset as 25% of the original one and random state (seed) as 100

        #For the Forward Method I decided to use a RandomForestRegressor
        ForwardFS = SequentialFeatureSelector(RandomForestRegressor(n_jobs=-1, random_state=100), k_features=2, forward=True, floating=False, scoring=scoreName, cv=5) #n_jobs=-1 means the usage of all the cores available on the local machine (computer, server, etc.)
                                                                    #Most of the times 2 features were more than enough to get a pretty good score               #Executing 5 times cross-validation, floating is set to False because we don't want a mixed approach to variable selection, instead we only want the "Forward" one
        ForwardFS.fit(XTrain, YTrain) #Fitting the model

        print(f"\nSelected Features Names For Feature (\033[93m{feature}\033[0m):")
        print(f"\033[92m{ForwardFS.k_feature_names_}\033[0m")
        #print("Selected Features Indexes:")
        #print(f"\033[92m{ForwardFS.k_feature_idx_}\033[0m")

        metricsDF = pd.DataFrame(ForwardFS.get_metric_dict()).T
        self.getMoreMetrics(feature, "Forward Method", scoreName, metricsDF)

        return ForwardFS.k_feature_names_


    def L1PenaltyFS(self, feature, targetColumn):

        X = self.YTDF.loc[:, self.FeatureSelectionTTSVariables] #All the rows and only specific columns declared in self.FeatureSelectionTTSVariables in the setYTDF() function, be aware that since only here we are using string indexers for columns we're using the .loc method and NOT .iloc
        Y = self.YTDF.iloc[:, targetColumn] #Setting the target variable for each for cycle

        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.3, random_state=100)

        L1FS = SelectFromModel(Lasso(alpha=0.5, random_state=100)) #Linear regression model with already-executed L1 Penalization (Lasso)

        L1FS.fit(XTrain, YTrain)

        if len(L1FS.get_feature_names_out()) == 0: #If not features were selected by model selection then:

            LassoReg = Lasso(alpha=0.5, random_state=100)
            LassoReg.fit(XTrain, YTrain)

            #LassoRegCoefficients = np.abs(LassoReg.coef_)
            #print("Lasso Regression Coefficients For Feature (\033[93m{feature}\033[0m: ", LassoRegCoefficients)

            minFeatures = 2
            selectedFeatures = list(np.where(LassoReg.coef_ != 0)[0])

            nFeatures = np.partition(selectedFeatures, -minFeatures)[-minFeatures:] #Extracting the n feature with the highest absolute coefficients
            nFeatures = list(self.YTDF.columns[nFeatures]) #Returning a list of the columns names corresponding to the relative index found by the np.partition function above

            print(self.bar)
            print(f"Significant Features for Target Feature (\033[93m{feature}\033[0m): ")
            print(nFeatures)

            #print(type(XTrain))
            #print(type(YTrain))

            XTrainTLasso = XTrain.loc[:, nFeatures]
            XTestTLasso = XTest.loc[:, nFeatures]

            self.finalL1Process(feature, XTrainTLasso, XTestTLasso, YTrain, YTest) #To not repeat myself twice I just incorporated the last part of the L1PenaltyFS Function in a separate one

            return nFeatures

        else:
            print(self.bar)
            print(f"Significant Features for Target Feature (\033[93m{feature}\033[0m): ")
            print(L1FS.get_feature_names_out())

            XTrainTLasso = L1FS.transform(XTrain) #T stands for "transformed"
            XTestTLasso = L1FS.transform(XTest) #T stands for "transformed"

            self.finalL1Process(feature, XTrainTLasso, XTestTLasso, YTrain, YTest)

            return L1FS.get_feature_names_out()


    def finalL1Process(self, feature, XTrainTLasso, XTestTLasso, YTrain, YTest):

        LM = LinearRegression(n_jobs=-1)
        LM.fit(XTrainTLasso, YTrain)

        YPrediction = LM.predict(XTestTLasso)

        print(f"Mean Squared Error (MSE) For Feature (\033[93m{feature}\033[0m):")
        print(mean_squared_error(YTest, YPrediction))

        print(f"R2 Score *Sometimes Approximated* For Feature (\033[93m{feature}\033[0m):")
        print(r2_score(YTest, YPrediction))

        return None


    def getImportantFeaturesTTS(self, targetFeature): #Get Important Features Test-Train-Split

        targetFeatureIndipendentVariablesList = list(self.FeaturesDict[targetFeature]) #This list contains specific predictors (returned by feature selection functions) for each target feature (at each call of this function the target feature, and so also its predictors, will change)

        #print("DataFrame Shape: ", self.YTDF.shape)
        #print("Columns in YTDF:", self.YTDF.columns)

        X = self.YTDF.loc[:, targetFeatureIndipendentVariablesList]  # Subsetting the dataframe with only the important feature to predict the specific feature passed as formal parameter to the function
        Y = self.YTDF.loc[:, targetFeature]  # Subset of the dataframe with only the target feature we want to predict

        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.3, random_state=100)

        return XTrain, XTest, YTrain, YTest


    def getCrossValidation(self, XTrain, YTrain, Model):
        KFoldCV = KFold(n_splits=15, shuffle=self.CVShuffle, random_state=100)
        scores = cross_validate(Model, XTrain, YTrain, scoring=self.scoring, cv=KFoldCV, return_train_score=False) #Training and validating the model on each CV Fold and evaluating it using various scores, return_test_score is set to False as default to indicate that we want the test scores

        #print(scores)

        return scores


    @staticmethod
    def RSE(YTrue, YPredicted):
        YTrue = np.array(YTrue)
        YPredicted = np.array(YPredicted)
        RSS = np.sum(np.square(YTrue - YPredicted)) #Residuals' Square Sum

        rse = np.sqrt(RSS / (len(YTrue) - 2))
        return rse


    def modelSaver(self, targetFeature, model):
        modelFileName = f"{self.modelsPath}{targetFeature}Model{model.__class__.__name__}.pkl" #Every model passed is the instance of the relative class (all the instances are contained in self.models) and mode.__class__.__name__ gets the name of the class
        with open(modelFileName, 'wb') as modelFile:  # Writing in binary mode to preserve data and state of the model
            pickle.dump(model, modelFile)
        modelFile.close()

        print(f"{targetFeature} {model.__class__.__name__} Model Exported Correctly as Pickle File")

        return None


    #Be aware that these are the models predictions BEFORE cross-validation
    @savePlots
    def modelPredictionsPlot(self, targetFeature, model, XTest, YPredictions):
        plotName = inspect.currentframe().f_code.co_name
        plotName = targetFeature + model.__class__.__name__ + plotName

        try:
            plotName.replace("model", "")
        except:
            pass

        fig, ax = plt.subplots()
        ax.scatter(XTest, YPredictions, edgecolors=(0, 0, 0))
        ax.plot([YPredictions.min(), YPredictions.max()], [YPredictions.min(), YPredictions.max()], 'k--', lw=4)
        ax.set_title(targetFeature + " " + model.__class__.__name__ + " Model Plot", fontdict=dict(weight="bold"))
        ax.set_xlabel('Truth')
        ax.set_ylabel('Predicted')

        return plotName, fig, self.plotsPath


    def saveModelReport(self, modelReport, targetFeature):

        modelsReportPath = self.testReportPath

        modelReportName = targetFeature + "ModelsReport.csv"
        modelsReportPath += modelReportName

        modelReport.to_csv(modelsReportPath, index=False) #Exporting Model Report as a CSV file without including the rows index column

        return None


    def YTRegressionModelsLearning(self, targetFeature):

        XTrain, XTest, YTrain, YTest = self.getImportantFeaturesTTS(targetFeature) #Train Test Split results are returned by this function

        #print("XTrain shape:", XTrain.shape)
        #print("YTrain shape:", YTrain.shape)
        #print("XTest shape:", XTest.shape)
        #print("YTest shape:", YTest.shape)
        #print()

        YPrediction = None

        R2 = []
        MSE = []
        RMSE = []
        MAE = []

        for i in range(len(self.models)):

            self.models[i].fit(XTrain, YTrain)

            YPrediction = self.models[i].predict(XTest)

            CrossValidatedModelResults = self.getCrossValidation(XTrain, YTrain, self.models[i]) #Executing Cross Validation

            #print(np.array(YTest)[-1]) #Truth
            #print(YPrediction[-1]) #Model prediction

            #print("Pre: ", mean_absolute_error(YTest, YPrediction))


            print("Aft R2: ", r2_score(YTest, YPrediction))
            print("Aft MSE: ", mean_squared_error(YTest, YPrediction))
            print("Aft RMSE: ", root_mean_squared_error(YTest, YPrediction))
            print("Aft MAE: ", mean_absolute_error(YTest, YPrediction))

            self.modelPredictionsPlot(targetFeature, self.models[i], YTest, YPrediction)
            self.modelSaver(targetFeature, self.models[i]) #Calling "model" to save fitted model

            R2.append(np.median((CrossValidatedModelResults["test_R2"])))
            MSE.append(np.median((CrossValidatedModelResults["test_MSE"])))
            RMSE.append(np.median((CrossValidatedModelResults["test_RMSE"])))
            MAE.append(np.median(CrossValidatedModelResults["test_MAE"]))


        #print("Number of Models: ", len(self.modelNames))
        #print("Model Names: ", self.modelNames)

        # print("Number of R2 Test Scores: ", len(R2))
        # print("R2 Test Scores", R2)

        #print("Number of MSE Scores: ", len(MSE))
        #print("MSE Scores", MSE)

        #print("Number of RMSE Scores: ", len(RMSE))
        #print("RMSE Scores", RMSE)

        #print("Number of MAE Scores: ", len(MAE))
        #print("MAE Scores", MAE)


        modelReport = pd.DataFrame({"ModelName": self.modelNames, "R2": R2, "MSE": MSE, "RMSE": RMSE, "MAE": MAE})

        self.saveModelReport(modelReport, targetFeature) #Saving the report in a CSV file

        print(self.bar)
        print(f"\033[94mModel Report For {targetFeature} Feature: \033[0m")
        print(modelReport)
        print(self.bar, "\n")


        return modelReport


    @savePlots
    def generateMLPlots(self, targetFeaturesMLDict, filePath):

        #print("File Path in generateMLPlots Function" + filePath)

        FeaturesModelNames = []
        ModelPlots = []

        for feat, ModelReport in targetFeaturesMLDict.items():
            ModelPxPlot = px.bar(ModelReport, x='ModelName', y='R2', color='ModelName', title=f"{feat} Models R^2 Plot")

            FeaturesModelNames.append(feat + "ModelsPlot")
            ModelPlots.append(ModelPxPlot)

            #print(type(ModelPxPlot))

        #print(len(FeaturesModelNames), len(ModelPlots))

        return FeaturesModelNames, ModelPlots, filePath


    def executeMLAnalysis(self):

        #Putting the test score results of the various models into their specific dictionary and then saving the plots in a separate folder
        for feat in self.targetVariablesDict.keys():
            self.targetFeaturesTestMLDict.update({feat: self.YTRegressionModelsLearning(feat)})

        self.generateMLPlots(self.targetFeaturesTestMLDict, self.TestScorePlotsPath)

        return None




























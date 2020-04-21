from xml.etree import cElementTree as ElementTree
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

from . import XML
from . import RateInstruments
from . import Timer

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from dateutil.relativedelta import *

class RateCurve():
    def __init__(self, stringURL : str, verbose:bool = False):
        # fetch the zip file from the website
        resp = urlopen(stringURL)
        # unzip and decode the file 
        zipfile = ZipFile(BytesIO(resp.read()))
        OIS_curve_xml = zipfile.open(zipfile.namelist()[1]).readlines()[0].decode('utf-8')
        if verbose:
            print(OIS_curve_xml)

        # iterate over the xml nodes
        root = ET.fromstring(OIS_curve_xml)
        templist = []

        for child in root.findall('./deposits/curvepoint'):
            data_child = XML.XmlListConfig(child)
            data_child.insert(0,'deposit')
            templist.append(data_child)

        for child in root.findall('./swaps/curvepoint'):
            data_child = XML.XmlListConfig(child)
            data_child.insert(0,'swap')
            templist.append(data_child)

        # init of the instruments data
        self.df = pd.DataFrame(templist, columns = ['source','tenor','maturity', 'parrate'])
        self.df['maturity']  = self.df['maturity'].apply(datetime.strptime,args=['%Y-%m-%d'])
        self.df.set_index('maturity',  inplace = True)
        
        # init of other data
        self.effectiveDate =  datetime.strptime(XML.XmlListConfig(root.findall('./effectiveasof'))[0], '%Y-%m-%d')
        self.spotdate =  datetime.strptime(XML.XmlListConfig(root.findall('./deposits/spotdate'))[0], '%Y-%m-%d')
        self.currency =  XML.XmlListConfig(root.findall('./currency'))[0]
        
        # preparation for calibration (maybe not to be done here)
        self.df['parrate'] = pd.to_numeric(self.df['parrate'])
        self.df['YTM'] = self.df['tenor'].apply(XML.tenorToFloat)
        self.df['ZCrate'] = self.df['parrate']
        self.df['df']=0.0
        self.cacheZCVector = None
    
    def prepareForInterpolate(self, discountDate):
        # initialize the cache if not done already
        if self.cacheZCVector is None:
            differential = pd.DataFrame(np.eye(len(self.df), dtype=float), index = self.df.index, columns = self.df.index)
            self.cacheZCVector = pd.concat([self.df['ZCrate'],differential], axis=1)
            
        # add the new data in the cache if the data doesn't exist yet
        if discountDate not in self.cacheZCVector.index:
            # insertion of the new value to interpolate 
            self.cacheZCVector.loc[discountDate] = None
            # linear interpolation of the value and of the differentiate vector (flat before and after first / last value)
            self.cacheZCVector = self.cacheZCVector.interpolate(method = 'time')

    def cleanCache(self):
        # initialize the cache if not done already
        self.cacheZCVector = None
            
    def interpolateRate(self, discountDate):
        # prepare the interpolation if needed
        self.prepareForInterpolate(discountDate)
        # return the interpolated value
        return self.cacheZCVector['ZCrate'].loc[discountDate]

    def getRateDiff(self, discountDate):
        # prepare the interpolation if needed
        self.prepareForInterpolate(discountDate)
        # return the interpolated value
        return self.cacheZCVector.drop('ZCrate', axis=1).loc[[discountDate]]
    
    def getDiscountFactor(self, discountDate):
        discount_factor = 1
        if discountDate > self.spotdate:
            delta_T = (discountDate-self.spotdate).days/365
            discount_factor = np.exp(-delta_T * self.interpolateRate(discountDate)) 
        return discount_factor

    def getDiscountFactorDiff(self, discountDate):
        # return dDF
        delta_T = (discountDate-self.spotdate).days/365
        return (- delta_T * self.getRateDiff(discountDate) * np.exp(-delta_T * self.interpolateRate(discountDate)) )
    
    def plot(self):
        # plot each instrument with a different color
        colors = {'deposit': 'purple', 'swap': 'blue'}
        fig, ax = plt.subplots()
        ax.scatter(self.df['YTM'], self.df['ZCrate'], c=self.df['source'].apply(lambda x: colors[x]))
        plt.show() 
        
        
    def getCurvePL():
        if 'MV' in self.df.columns:
            globalMV = self.df['MV'].sum()*1000000 
        else: 1000000
            
        return globalMV
        
    def calibrateCurve(self):
        for i in[1, 2, 3, 4, 5, 6]:
            
            self.plot()
        
            spot = self.spotdate

            for index in self.df.index:
                line = self.df.loc[index]
                source = line['source']
                parrate = line['parrate']
                maturity = index
                if (source == 'deposit'):
                    self.df.loc[index, "instrumentList"] = RateInstruments.DepositInstrument(spot, maturity, parrate, self)
                elif (source == 'swap'):
                    ytm = line['YTM']
                    self.df.loc[index, "instrumentList"] = RateInstruments.FixFloatSwapInstrument(spot, maturity, ytm, 6, 3, parrate, self, self)
                    jacobienMatrix = pd.DataFrame(0, index = self.df.index, columns = self.df.index)

            for i in self.df.index:
                #really shitty performance here
                jacobienMatrix.loc[i] = self.df.loc[i,'instrumentList'].getZCsensi()    
                self.df.loc[i,'MV']  = self.df.loc[i,'instrumentList'].getMV().iloc[0]
            
            # problem to be seen later
            # print("Global MV : \n\n", self.getCurvePL(), "\n\n")  
            print("detailed MV : \n\n", self.df['MV']*1000000, "\n\n")
            
            # dataframe not required here as it crates a copy in memory
            inv_J = pd.DataFrame(np.linalg.pinv(jacobienMatrix.values), jacobienMatrix.columns, jacobienMatrix.index)
            # inv_J.dot(jacobienMatrix) cette quantité est presque une matrice diagonale
            epsilon = -inv_J.dot(self.df['MV'])
            self.df['ZCrate'] += epsilon
            self.cleanCache()

        
        
 
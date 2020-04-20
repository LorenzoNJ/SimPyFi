from xml.etree import cElementTree as ElementTree
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from dateutil.relativedelta import *
import numpy as np


# QUESTION : is it necessary to have a class for this? 
class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)

                    
                    
                    
def tenorToFloat(string : str):
    result = float("Nan")
    if string.endswith('M') or string.endswith('m'):
        result = float(string[:-1])/12
    elif string.endswith('Y') or string.endswith('y'):
        result = float(string[:-1])
    return result




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
            data_child = XmlListConfig(child)
            data_child.insert(0,'deposit')
            templist.append(data_child)

        for child in root.findall('./swaps/curvepoint'):
            data_child = XmlListConfig(child)
            data_child.insert(0,'swap')
            templist.append(data_child)

        # init of the instruments data
        self.df = pd.DataFrame(templist, columns = ['source','tenor','maturity', 'parrate'])
        self.df['maturity']  = self.df['maturity'].apply(datetime.strptime,args=['%Y-%m-%d'])
        self.df.set_index('maturity',  inplace = True)
        
        # init of other data
        self.effectiveDate =  datetime.strptime(XmlListConfig(root.findall('./effectiveasof'))[0], '%Y-%m-%d')
        self.spotdate =  datetime.strptime(XmlListConfig(root.findall('./deposits/spotdate'))[0], '%Y-%m-%d')
        self.currency =  XmlListConfig(root.findall('./currency'))[0]
        
        # preparation for calibration (maybe not to be done here)
        self.df['parrate'] = pd.to_numeric(self.df['parrate'])
        self.df['YTM'] = self.df['tenor'].apply(tenorToFloat)
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
        ax.scatter(self.df['YTM'], self.df['parrate'], c=self.df['source'].apply(lambda x: colors[x]))
        plt.show()        
        
        
        
        
        
        
        
        
        
class DepositInstrument():
    def __init__(self, spotDate, maturityDate, rate, discountCurve):
        self.spotDate = spotDate
        self.maturityDate = maturityDate
        self.rate = rate
        self.discountCurve = discountCurve

        self.flowKey = ["flowType", "flowDate"]
        self.flows = None 
        self.dNDF = None 
        self.dF = None 

        
    def computeNonDiscountedFlows(self):
        if(self.flows is None):
            # dataframe init 
            self.flows = pd.DataFrame(None, columns = self.flowKey+["flowValue"]).set_index(self.flowKey)
            self.dNDF = pd.DataFrame(None, columns = self.flowKey+self.discountCurve.df.index.tolist()).set_index(self.flowKey)
            # lending flow
            self.flows.loc[("CAP", self.spotDate),"flowValue"] = -1
            self.dNDF.loc[("CAP", self.spotDate),:] = 0
            # reimbursement of the capital flow
            self.flows.loc[("CAP", self.maturityDate),"flowValue"] = 1
            self.dNDF.loc[("CAP", self.maturityDate),:] = 0
            # interest flow 
            self.flows.loc[("INT", self.maturityDate),"flowValue"] = self.rate * (self.maturityDate - self.spotDate).days/360
            self.dNDF.loc[("INT", self.maturityDate),:]= 0   
           
    
    #TODO: factorize in a mother class
    def computeDiscountedFlows(self):
        # compute non-discounted flows
        if self.flows is None:
            self.computeNonDiscountedFlows()
        if self.dF is None:
            self.dF = pd.DataFrame(None, columns = self.flowKey+self.discountCurve.df.index.tolist()).set_index(self.flowKey)
            # discount the flows and record the discounted flows in the swap
            for flowKey in self.flows.index:
                DF = self.discountCurve.getDiscountFactor(flowKey[1])
                dDF = self.discountCurve.getDiscountFactorDiff(flowKey[1])
                self.flows.loc[flowKey,"DF"] = DF
                self.flows.loc[flowKey,"discountedValue"] = self.flows.loc[flowKey,"flowValue"] * DF
                self.dF.loc[flowKey,:] = DF * self.dNDF.loc[flowKey] + self.flows.loc[flowKey,"flowValue"] * dDF.iloc[0]

    def getZCsensi(self):
        self.computeDiscountedFlows()
        return self.dF.sum()
    
    def getMV(self):
        self.computeDiscountedFlows()
        return self.flows[["discountedValue"]].sum()
    
    
    
    
        

class FixFloatSwapInstrument():
    def __init__(self, spotDate, maturityDate, maturityInYears, fixPeriodInMonth, floatPeriodInMonth, swapRate, evalCurve, discountCurve):
        self.spotDate = spotDate
        self.maturityDate = maturityDate
        self.fixPeriodInMonth = fixPeriodInMonth
        self.floatPeriodInMonth = floatPeriodInMonth
        self.swapRate = swapRate

        self.evalCurve = evalCurve
        self.discountCurve = discountCurve

        self.flowKey = ["flowType", "flowDate"]
        self.flows = None
        self.dNDF = None
        self.dF = None        
        
        # remove the field after and replace with difference between spot date and maturity
        self.maturityInYears = maturityInYears
        
    def computeNonDiscountedFlows(self):
        if self.flows is None:
            # date generator for fixed flows 
            dFixedFlows = []
            nbFixedFlows = int(self.maturityInYears * 12 / self.fixPeriodInMonth)
            for i in range(0, nbFixedFlows):
                dFixedFlows.append( self.spotDate+relativedelta(months=(i+1)*self.fixPeriodInMonth) )
            # date generator for floating flows 
            dFloatingFlows = []
            nbFloatingFlows = int(self.maturityInYears * 12 / self.floatPeriodInMonth)
            for i in range(0, nbFloatingFlows):
                dFloatingFlows.append( self.spotDate+relativedelta(months=(i+1)*self.floatPeriodInMonth) )
                
            # dataframe init
            self.flows = pd.DataFrame(None, columns = self.flowKey+["flowValue"]).set_index(self.flowKey)
            self.dNDF = pd.DataFrame(None, columns = self.flowKey+self.discountCurve.df.index.tolist()).set_index(self.flowKey)
            # fixed flows computation
            for i in range(len(dFixedFlows)):
                flowDate = dFixedFlows[i]
                previousFlowDate = self.spotDate if i == 0 else dFixedFlows[i-1]
                delta_t = ( flowDate - previousFlowDate).days/360            
                self.flows.loc[("INT-FIX", flowDate),"flowValue"] = -self.swapRate * delta_t
                self.dNDF.loc[("INT-FIX", flowDate),:] = 0
            # floating flows computation
            for i in range(len(dFloatingFlows)):
                flowDate = dFloatingFlows[i]
                previousFlowDate = self.spotDate if i == 0 else dFloatingFlows[i-1]
                DF1 = self.evalCurve.getDiscountFactor(previousFlowDate)
                DF2 = self.evalCurve.getDiscountFactor(flowDate)
                dDF1 = self.evalCurve.getDiscountFactorDiff(previousFlowDate)
                dDF2 = self.evalCurve.getDiscountFactorDiff(flowDate)
                self.flows.loc[("INT-FLT", flowDate),"flowValue"] = DF1 / DF2 - 1
                self.dNDF.loc[("INT-FLT", flowDate),:] = 1/DF2 * dDF1.iloc[0]  - DF1/(DF2*DF2) * dDF2.iloc[0]

        return self.flows
        
        
    #TODO: factorize in a mother class
    def computeDiscountedFlows(self):
        # compute non-discounted flows 
        if self.flows is None:
            self.computeNonDiscountedFlows()
        if self.dF is None:
            self.dF = pd.DataFrame(None, columns = self.flowKey+self.discountCurve.df.index.tolist()).set_index(self.flowKey)
            # discount the flows and record the discounted flows in the swap
            for flowKey in self.flows.index:
                DF = self.discountCurve.getDiscountFactor(flowKey[1])
                dDF = self.discountCurve.getDiscountFactorDiff(flowKey[1])
                self.flows.loc[flowKey,"DF"] = DF
                self.flows.loc[flowKey,"discountedValue"] = self.flows.loc[flowKey,"flowValue"] * DF
                self.dF.loc[flowKey,:] = DF * self.dNDF.loc[flowKey] + self.flows.loc[flowKey,"flowValue"] * dDF.iloc[0]
    
    def getZCsensi(self):
        self.computeDiscountedFlows()
        return self.dF.sum()

    def getMV(self):
        self.computeDiscountedFlows()
        return self.flows[["discountedValue"]].sum()
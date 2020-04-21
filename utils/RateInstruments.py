import pandas as pd
import numpy as np  
from .Timer import Timer
from dateutil.relativedelta import *



       
class CalibrationInstrument():
    def __init__(self):
        pass

    #TODO: factorize in a mother class
    def computeDiscountedFlows(self):
        # compute non-discounted flows
        if self.flows is None:
            with Timer("computeNonDiscountedFlows"):
                self.computeNonDiscountedFlows()
        if self.dF is None:
            with Timer("computeDiscountedFlows"):
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
    
    
        
class DepositInstrument(CalibrationInstrument):
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
           
    

    
    
    
    
        

class FixFloatSwapInstrument(CalibrationInstrument):
    def __init__(self, spotDate, maturityDate, maturityInYears, fixPeriodInMonth, floatPeriodInMonth, swapRate, evalCurve, discountCurve):
        self.spotDate = spotDate
        self.maturityDate = maturityDate
        self.fixPeriodInMonth = fixPeriodInMonth
        self.floatPeriodInMonth = floatPeriodInMonth
        self.swapRate = swapRate

        self.evalCurve = evalCurve
        self.discountCurve = discountCurve      
        
        # remove the field after and replace with difference between spot date and maturity
        self.maturityInYears = maturityInYears
        
        self.flowKey = ["flowType", "flowDate"]
        self.flows = None 
        self.dNDF = None 
        self.dF = None    
        
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
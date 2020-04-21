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
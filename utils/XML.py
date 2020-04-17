from xml.etree import cElementTree as ElementTree
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pandas as pd


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

def ParseXMLRateCurve(string : str):
    # fetch the zip file from the website
    resp = urlopen(string)
    # unzip and decode the file 
    zipfile = ZipFile(BytesIO(resp.read()))
    OIS_curve_xml = zipfile.open(zipfile.namelist()[1]).readlines()[0].decode('utf-8')
    
    # iterate over the xml nodes
    root = ET.fromstring(OIS_curve_xml)
    templist = []

    for child in root.findall('./deposits/curvepoint'):
        data_child = XmlListConfig(child)
        data_child.insert(0,'deposit')
        templist.append(data_child)
        
    for child in root.findall('./swaps/curvepoint'):
        data_child = XmlListConfig(child)
        data_child.insert(0,'swaps')
        templist.append(data_child)
        
    df = pd.DataFrame(templist, columns = ['source','tenor','maturity', 'parrate'])
    
    df['parrate'] = pd.to_numeric(df['parrate'])
    df['YTM'] = df['tenor'].apply(tenorToFloat)
   
    # return the rate curve parsed under a DataFrame format
    return df




def tenorToFloat(string : str):
    result = float("Nan")
    if string.endswith('M') or string.endswith('m'):
        result = float(string[:-1])/12
    elif string.endswith('Y') or string.endswith('y'):
        result = float(string[:-1])
    return result
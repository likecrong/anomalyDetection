# The data analysis for the anomaly detection
- The anomalies was detected for the datafame using ADTK.
- ADTK 라이브러리를 이용하여 데이터프레임에 대한 이상치 탐지를 진행한다.

## 1. What is the ADTK?
- ADTK is an anomaly detection toolkit for considering with time series by supported with python3.
- there are some module like aggregator, transformer, detector, and pipe.
- I would use algorithm of the PersistAD(detector) because I need to change the algorithm a little.
- There was not 'resample' function originally but I added that.
- https://adtk.readthedocs.io/en/stable/index.html

### The algorithm of the PersistAD

![persistAD02](https://user-images.githubusercontent.com/112467598/200116170-b0f42b18-6f82-468d-92f7-6002a2a5b802.png)

## 2. Why did I add the 'resample' function in the algorithm?

### The example of after applying the DoubleRollingAggregate using the l1

![double01](https://user-images.githubusercontent.com/112467598/200116532-c15848cb-bf29-444a-a847-c7fc8833b219.PNG)

- Look at the section where the value spikes in the first graph.
- 첫번째 그래프에서 값이 급증이 일어난 구간을 보세요.
- In the second graph, you can see that the section was more emphasized than the others.
- 두번째 그래프에서 해당 구간이 다른 구간보다 강조된 것을 확인할 수 있습니다.


```python
import os
import pandas as pd
import pickle
import numpy as np
```


```python
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree

#동영상 파일경로 불러오기
xml_file = './path.xml'
doc = ET.parse(xml_file)
root = doc.getroot()
filepath = root.find("pathOfdata").text
```


```python
# 저장한 피클 파일 리스트 불러오기
fileList = os.listdir(filepath)
file_list_pkl = [file for file in fileList if file.endswith(".pkl") and len(file)==10]
print(file_list_pkl)
num = len(file_list_pkl)
print(num)

# 피클 파일 리스트로부터 데이터프레임 리스트 생성
df = []
for i,name in enumerate(file_list_pkl):
    data = pd.read_pickle(filepath+name)
    #print(data)
    df.append(data)
```

### The result of after applying the DoubleRollingAggregate using the l1


```python
from adtk.data import validate_series
from adtk.transformer import DoubleRollingAggregate
from adtk.visualization import plot
```


```python
s = df[0].result
s_transformed1 = DoubleRollingAggregate(
    agg='mean',
    window=('1h',1),
    center=True,
    diff="l1").transform(s).rename("DoubleRolling01h")
s_transformed2 = DoubleRollingAggregate(
    agg='mean',
    window=('12h',1),
    center=True,
    diff="l1").transform(s).rename("DoubleRolling12h")
#plot(pd.concat([s, s_transformed1, s_transformed2], axis=1));
```

![double02](https://user-images.githubusercontent.com/112467598/200153258-2572eb73-49a9-4478-9e8f-1346229915e8.png)

- Although the DoubleRollingAggregates were applied, there was no section that had been emphasized differently from the example.
- DoubleRollingAggregate를 적용했음에 불구하고 예시와 다르게 강조되는 구간이 없다.
- The performance of the detector will be degraded When the series is applied to the InterQuartileRangeAD.
- 해당 그래프를 InterQuartileRangeAD에 적용할 경우, detector의 성능이 저하된다.
- It will be helped to improve the performance of the detector by adding a resample function.
- resample 함수를 추가하여 detector의 성능 향상을 돕는다.

### The result of applying the DoubleRollingAggregates and the 'resample' method


```python
#s_transformed2.resample('1h').sum().rename('resample1h').plot()
```

![drars01](https://user-images.githubusercontent.com/112467598/200153324-f5215c14-0cd4-4996-ae36-9bda87c60287.png)


```python
#s_transformed2.resample('12h').sum().rename('resample12h').plot()
```

![drars02](https://user-images.githubusercontent.com/112467598/200153340-52fb8056-3582-4b49-9c07-aded58e02dce.png)

- You can see the result which was the highlights areas drastically reduced if the 'resample' method was applied.
- resample을 적용하면 급감한 구역이 강조된다.

## 3. Applying the algorithm of the PersistAD with the 'resample' method


```python
from adtk.data import validate_series
from adtk.transformer import DoubleRollingAggregate
from adtk.transformer import RollingAggregate
from adtk.detector import InterQuartileRangeAD
from adtk.detector import ThresholdAD
from adtk.aggregator import AndAggregator
from adtk.visualization import plot
import matplotlib.pyplot as plt
from datetime import datetime
```


```python
roWin=['01H', '03H' ,'06H', '12H', '24H', '48H']
reWin=['01H', '03H' ,'06H', '12H', '24H', '48H']
```


```python
targetList=['A','B']

for idx in range(len(targetList)):
    s = df[idx].result
    for ro in roWin:
        for re in reWin:   
            #section 1
            #(1) DoubleRollingAggregate
            s_transformed1 = DoubleRollingAggregate(
                agg='mean',
                window=(ro,1),
                center=True,
                diff="l1").transform(s)

            #(2) resample
            s_transformed1 = s_transformed1.resample(re).sum()

            #(3) InterQuartileRangeAD
            iqr_ad = InterQuartileRangeAD(c=(0.5,None))
            anomalies1 = iqr_ad.fit_detect(s_transformed1)

            #section 2
            #(1) DoubleRollingAggregate
            s_transformed2 = DoubleRollingAggregate(
                agg='mean',
                window=(ro,1),
                center=True,
                diff="diff").transform(s)

            #(2) resample
            s_transformed2 = s_transformed2.resample(re).sum()

            #(3) ThresholdAD
            threshold_ad = ThresholdAD(high=float("inf"), low=0.0)
            anomalies2 = threshold_ad.detect(s_transformed2)

            #section 3
            #(1) AndAggregator
            target = pd.concat([anomalies1,anomalies2],axis=1)
            anomaly = AndAggregator().aggregate(target)

            #(2) Saving the result of the anomaly with dataframe
            anoDF = anomaly.to_frame()
            anoDF = anoDF.rename(columns = {0:'anomaly'})
            anoDF.to_pickle(f'./data/{targetList[idx]}_DRA_{ro}_Resample_{re}.pkl')

            #(3) Visualizing a graph and saving it
            graph = s.rolling(window=ro).mean().resample(re).sum()
            fig = plt.figure(figsize=(16,8)) 
            if idx==0: #A 데이터
                newgraph = plot(graph.rename("A"), anomaly=anomaly, anomaly_color="red", anomaly_tag="span")     
                plt.plot(newgraph[0].plot())
                plt.axvline(datetime.strptime("2021-04-22 00:00:00", "%Y-%m-%d %H:%M:%S"),
                                0, 1, color='orange', linestyle='--', linewidth=2)
            else: #B 데이터
                newgraph = plot(graph.rename("B"), anomaly=anomaly, anomaly_color="red", anomaly_tag="span")     
                plt.plot(newgraph[0].plot())
                plt.axvline(datetime.strptime("2021-04-15 00:00:00", "%Y-%m-%d %H:%M:%S"),
                                0, 1, color='green', linestyle='--', linewidth=2)
                
            plt.xticks(fontsize=16,rotation=15)
            plt.yticks(fontsize=16)
            plt.title(f'{targetList[idx]}_DRA_{ro}_Resample_{re}',fontsize=25, fontweight='bold', loc='left')
            plt.savefig(f'./graph/{targetList[idx]}_DRA_{ro}_Resample_{re}.png')
            plt.clf()
```

### Next
- I will select the best result upon testing these anomaly dataframes.
- 이상치 데이터 프레임을 가지고 평가하여 최선의 결과를 선택한다.


```python

```

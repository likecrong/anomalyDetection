# Making DataFrames and the Data Processing
- The video data used in the previous version cannot be used as a security problem.
- 기존 영상데이터는 보안상의 문제로 활용할 수 없다.
    - Unable to re-collect data
    - 데이터 재수집 불가능
- I used some dataframes in the previous version.
- 기존 데이터프레임을 활용한다.
    - It was consisted of avi and smi files when I used.
    - 기존 영상데이터는 avi와 smi 파일 형식으로 구성되어 있었다.
    - The smi file contained the date and time when the video was filmed.
    - smi파일에 영상이 촬영된 날짜와 시간에 대한 내용이 담겨있었다.

## 1. Making DataFrames

### Variables
- variables: Date, result
- 데이터 프레임 변수: Date, result
- Date: The date and time are included.
- Date: 날짜와 시간이 담겼다.
    - 재작성한 예시 버전의 'increase_width'변수의 값은 2이지만, 기존 버전의 'increase_width'변수의 값은 5이기 때문에 Date의 gap이 5초이다.
- result: The average value of the magnitude.
- result: magnitude의 평균값이다.


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
import os
import pandas as pd
import pickle
import numpy as np
```


```python
fileList = os.listdir(filepath)
#fileList
```


```python
A = pd.DataFrame()

count = 0
for _ in fileList:
    if 'cctv21' in _:
        
        #데이터프레임 불러오기
        data = pd.read_pickle(filepath+_)
        #magnitude 리스트
        target = data.result.tolist()
        
        #magnitude의 평균값을 구하는 과정 (예외처리포함)
        remove_set = {np.inf, -np.inf}
        li = [i for i in target if i not in remove_set]
        m = np.mean(li)
        m = round(m, 2)
        for n,_ in enumerate(target):
            if _ == np.inf or _ == -np.inf:
                target[n] = m
            else:
                target[n] = round(_,2)        
        
        #정제된 magnitude 리스트 설정
        data.result = target
        A=pd.concat([A,data],axis=0)
        
A = A.reset_index(drop=True)
```


```python
B = pd.DataFrame()

count = 0
for _ in fileList:
    if 'cctv23' in _:
        
        #데이터프레임 불러오기
        data = pd.read_pickle(filepath+_)
        #magnitude 리스트
        target = data.result.tolist()
        
        #magnitude의 평균값을 구하는 과정 (예외처리포함)
        remove_set = {np.inf, -np.inf}
        li = [i for i in target if i not in remove_set]
        m = np.mean(li)
        m = round(m, 2)
        for n,_ in enumerate(target):
            if _ == np.inf or _ == -np.inf:
                target[n] = m
            else:
                target[n] = round(_,2)        
        
        #정제된 magnitude 리스트 설정
        data.result = target
        B=pd.concat([B,data],axis=0)
        
B = B.reset_index(drop=True)
```


```python
# 2021.04.10.데이터부터 활용
count = 0
for _ in A.Date.tolist():
    if '2021-04-09' in _:
        count+=1
A = A.iloc[count:,:]
A = A.reset_index(drop=True)
```


```python
# 2021.04.10.데이터부터 활용
count = 0
for _ in B.Date.tolist():
    if '2021-04-09' in _:
        count+=1
B = B.iloc[count:,:]
B = B.reset_index(drop=True)
```


```python
A.to_pickle(filepath+'A.pkl')
B.to_pickle(filepath+'B.pkl')
```

### Result


```python
#A.plot()
```

![aplot](https://user-images.githubusercontent.com/112467598/200161052-454620df-9a66-4488-9221-578583dce6ec.png)



```python
#B.plot()
```

![bplot](https://user-images.githubusercontent.com/112467598/200161063-e51c1d2a-2ca3-4173-be34-cec656a762ea.png)


## 2. The data processing
- The default index was switched to the datatime-type index.
- 기본인덱스로부터 datatime 형 인덱스로 전환한다.
- The 'result' value of excessing the UML was replaced by 0.
- result의 값에 대해 UML 초과 값을 0으로 치환한다.


```python
import matplotlib.pyplot as plt
from datetime import datetime
import statistics as s
```


```python
# 저장한 피클 파일 리스트 불러오기
fileList = os.listdir(filepath)
file_list_pkl = [file for file in fileList if file.endswith(".pkl") and len(file)==5]
print(file_list_pkl)
num = len(file_list_pkl)
print(num)
```

    ['A.pkl', 'B.pkl']
    2
    


```python
# 피클 파일 리스트로부터 데이터프레임 리스트 생성
df = []
for i,name in enumerate(file_list_pkl):
    data = pd.read_pickle(filepath+name)
    #print(data)
    df.append(data)
```


```python
for k,_ in enumerate(df):
    
    # 문자열 -> datetime 변환
    mylist = _.Date.tolist()
    newlist = []
    for string in mylist:
        newlist.append(datetime.strptime(string, '%Y-%m-%d %H:%M:%S'))
    _.Date = newlist
    _ = _[_.Date <= '2021-04-29']
    
    # 3std 이상 0 처리
    mylist = _.result.tolist()
    threshold = np.mean(mylist) + 3*np.std(mylist) #임계값
    mylist = [0 if x>=threshold else x for x in mylist] # UML초과 0처리    
    _.result = mylist
     
    # 인덱스 설정
    _ = _.sort_values('Date').set_index('Date')
    _.to_pickle(filepath+f'/3std_{file_list_pkl[k]}')
    #print(_)
```

    C:\Users\admin\AppData\Local\Temp\ipykernel_124388\2444462454.py:15: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      _.result = mylist
    

### Result


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

    ['3std_A.pkl', '3std_B.pkl']
    2
    


```python
#df[0].plot() # A데이터
```

![au](https://user-images.githubusercontent.com/112467598/200161088-80d762c0-55e8-4daf-9cb2-db7cde3455c6.png)


```python
#df[1].plot() #B데이터
```

![bu](https://user-images.githubusercontent.com/112467598/200161106-9d5e37b1-d938-4fb3-9b03-e41451aadf92.png)


### Next
- I will use ADTK to detect anomalies from these dataframes.
- 정제된 데이터프레임을 가지고 ADTK를 사용해 이상치 탐지를 진행한다.


```python

```

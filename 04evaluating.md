# Evaluating the algorithm with the confusion matrix
- The results for the dataframes are evaluated using the Confusion Matrix.
- 데이터 셋에 대한 결과를 Confusion Matrix를 이용하여 평가한다.
- Tye Evaluation Metric is consisted of the accuracy, the precision, and the recall.
- 평가지표는 정확성, 정밀도, 재현율이다.

## 1. Loading dataframes about the anomaly detection


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
# 저장한 피클 파일 리스트 불러오기
fileList = os.listdir(filepath)
file_list_pkl = [file for file in fileList if file.endswith(".pkl") and len(file)==26]
#print(file_list_pkl)
num = len(file_list_pkl)
print(num)

# 피클 파일 리스트로부터 데이터프레임 리스트 생성
df = []
for i,name in enumerate(file_list_pkl):
    data = pd.read_pickle(filepath+name)
    df.append(data)
```

    72
    

### details
- Half of the 'df' list was the result of the data A, the other was the data B.
- df 리스트의 절반은 A데이터의 결과, 나머지는 B데이터의 결과이다.

## 2. Scoring the dataframe with the confusion matrix

### Background
- Due to the nature of the disease, the symptom of the A case appears within two or three days, and the B case appears within five or six days.
- A데이터는 질병 특성상 증상이 2,3일 이내 나타나고 B데이터는 5,6일 이내 나타난다.


```python
import datetime as dt
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
```


```python
def score(target, order):
    '''
    입력변수: series, A데이터인지 B데이터인지
    출력변수: accuracy, precision, recall
    
    해당 데이터에 대해 정확도, 정밀도, 재현율을 계산하는 함수
    '''
    #(1) 예측 데이터 처리 - 이진 데이터
    y_pred=target.anomaly.tolist()
    y_pred=[1 if _==True else 0 for _ in y_pred]
    
    #(2) 정답 데이터 처리
    y_true=[0 for _ in range(len(y_pred))]
    indexList=target.index.tolist()    
    if order=='A':
        aDate=dt.datetime.strptime("2021-04-22 00:00:00", "%Y-%m-%d %H:%M:%S")
        date = dt.timedelta(days=4)
    else:
        aDate=dt.datetime.strptime("2021-04-15 00:00:00", "%Y-%m-%d %H:%M:%S")
        date = dt.timedelta(days=7)
    for idx in range(len(y_pred)):
        if aDate<=indexList[idx]<=(aDate+date):
            y_true[idx]=1

    #(3) 정확도, 정밀도, 재현율 구하기
    #print(f"accuracy(정확도): {metrics.accuracy_score(y_true,y_pred):.3f}")
    #print(f"precision(정밀도): {metrics.precision_score(y_true,y_pred):.3f}")
    #print(f"recall(재현율): {metrics.recall_score(y_true,y_pred):.3f}")
    
    accuracy=round(metrics.accuracy_score(y_true,y_pred),3)
    precision=round(metrics.precision_score(y_true,y_pred),3)
    recall=round(metrics.recall_score(y_true,y_pred),3)
    
    return accuracy, precision, recall
```


```python
length=num//2 # A와 B 데이터의 길이 변수

# 평가지표를 케이스별로 저장하는 리스트
accuracyA = []
precisionA = []
recallA = []
accuracyB = []
precisionB = []
recallB = []

# 평가지표를 통해 계산하는 구간
for idx in range(length):
    #print(file_list_pkl[idx])
    #print(file_list_pkl[idx+length])
    
    a,p,r=score(df[idx],'A')
    accuracyA.append(a)
    precisionA.append(p)
    recallA.append(r)
    
    a,p,r=score(df[idx+length],'B')
    accuracyB.append(a)
    precisionB.append(p)
    recallB.append(r)

# 계산값에 대해 케이스별로 데이터프레임화
dfA = pd.DataFrame({'name':file_list_pkl[:length],'accuracy':accuracyA, 'precision':precisionA, 'recall':recallA})
dfB = pd.DataFrame({'name':file_list_pkl[length:],'accuracy':accuracyB, 'precision':precisionB, 'recall':recallB})
```

### Why did I prioritize precision and recall?
- The data set was unbalanced because the purpose of the data set is to detect anomalies. The result was difficult to evaluate with accuracy.
- 해당 데이터 셋은 이상치 탐지가 목적이기 때문에 데이터가 불균형하다. 결과를 정확성으로 평가하기에 어렵다.
- In this case, It is better to use the case of the high precision and recall.
- 이러한 케이스에서는 정밀도와 재현율에서 높은 경우를 사용하는 게 낫다. 


```python
tmpA = dfA.sort_values(['precision','recall'],ascending=False).head(10) #상위 10개만 활용
indexA = set(tmpA.index.tolist())
tmpA
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>A_DRA_01H_Resample_06H.pkl</td>
      <td>0.792</td>
      <td>0.667</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A_DRA_03H_Resample_03H.pkl</td>
      <td>0.791</td>
      <td>0.600</td>
      <td>0.091</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A_DRA_06H_Resample_01H.pkl</td>
      <td>0.790</td>
      <td>0.545</td>
      <td>0.062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A_DRA_01H_Resample_12H.pkl</td>
      <td>0.769</td>
      <td>0.500</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A_DRA_03H_Resample_12H.pkl</td>
      <td>0.769</td>
      <td>0.500</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A_DRA_03H_Resample_06H.pkl</td>
      <td>0.779</td>
      <td>0.500</td>
      <td>0.059</td>
    </tr>
    <tr>
      <th>14</th>
      <td>A_DRA_06H_Resample_06H.pkl</td>
      <td>0.779</td>
      <td>0.500</td>
      <td>0.059</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A_DRA_01H_Resample_03H.pkl</td>
      <td>0.778</td>
      <td>0.444</td>
      <td>0.121</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A_DRA_03H_Resample_01H.pkl</td>
      <td>0.783</td>
      <td>0.417</td>
      <td>0.052</td>
    </tr>
    <tr>
      <th>18</th>
      <td>A_DRA_12H_Resample_01H.pkl</td>
      <td>0.781</td>
      <td>0.412</td>
      <td>0.072</td>
    </tr>
  </tbody>
</table>
</div>




```python
tmpB = dfB.sort_values(['precision','recall'],ascending=False).head(10) #상위 10개만 활용
indexB = set(tmpB.index.tolist())
tmpB
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>B_DRA_12H_Resample_03H.pkl</td>
      <td>0.641</td>
      <td>0.667</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B_DRA_01H_Resample_12H.pkl</td>
      <td>0.615</td>
      <td>0.500</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>9</th>
      <td>B_DRA_03H_Resample_12H.pkl</td>
      <td>0.615</td>
      <td>0.500</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>15</th>
      <td>B_DRA_06H_Resample_12H.pkl</td>
      <td>0.615</td>
      <td>0.500</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>21</th>
      <td>B_DRA_12H_Resample_12H.pkl</td>
      <td>0.615</td>
      <td>0.500</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>27</th>
      <td>B_DRA_24H_Resample_12H.pkl</td>
      <td>0.615</td>
      <td>0.500</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>33</th>
      <td>B_DRA_48H_Resample_12H.pkl</td>
      <td>0.615</td>
      <td>0.500</td>
      <td>0.067</td>
    </tr>
    <tr>
      <th>13</th>
      <td>B_DRA_06H_Resample_03H.pkl</td>
      <td>0.621</td>
      <td>0.462</td>
      <td>0.105</td>
    </tr>
    <tr>
      <th>25</th>
      <td>B_DRA_24H_Resample_03H.pkl</td>
      <td>0.614</td>
      <td>0.400</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>20</th>
      <td>B_DRA_12H_Resample_06H.pkl</td>
      <td>0.610</td>
      <td>0.333</td>
      <td>0.034</td>
    </tr>
  </tbody>
</table>
</div>




```python
case = list(indexA&indexB)
#print(case)

for idx in case:
    print(f'Best Case: ',end='')
    print(tmpA.loc[idx]['name'])
    print(f'Best Case: ',end='')
    print(tmpB.loc[idx]['name'])
    print('--------------------------')
```

    Best Case: A_DRA_03H_Resample_12H.pkl
    Best Case: B_DRA_03H_Resample_12H.pkl
    --------------------------
    Best Case: A_DRA_01H_Resample_12H.pkl
    Best Case: B_DRA_01H_Resample_12H.pkl
    --------------------------
    

## 3. The Best Case

### The graph of the 'DRA_03H_Resample_12H'

![A_DRA_03H_Resample_12H](https://user-images.githubusercontent.com/112467598/200159073-6b28c10e-c30f-4b69-adc5-9ba5d153c5c5.png)

![B_DRA_03H_Resample_12H](https://user-images.githubusercontent.com/112467598/200159106-f7cca65e-cecb-459e-a2c3-d3a3e9118e0a.png)


```python
print(tmpA.loc[9])
print(tmpB.loc[9])
```

    name         A_DRA_03H_Resample_12H.pkl
    accuracy                          0.769
    precision                           0.5
    recall                            0.111
    Name: 9, dtype: object
    name         B_DRA_03H_Resample_12H.pkl
    accuracy                          0.615
    precision                           0.5
    recall                            0.067
    Name: 9, dtype: object
    

### The graph of the 'DRA_01H_Resample_12H'

![A_DRA_01H_Resample_12H](https://user-images.githubusercontent.com/112467598/200159130-590a0ac5-dc22-4470-9664-758ac644082c.png)

![B_DRA_01H_Resample_12H](https://user-images.githubusercontent.com/112467598/200159144-29e37ecf-c383-41ad-be5e-328cd0bc26f6.png)



```python
print(tmpA.loc[3])
print(tmpB.loc[3])
```

    name         A_DRA_01H_Resample_12H.pkl
    accuracy                          0.769
    precision                           0.5
    recall                            0.111
    Name: 3, dtype: object
    name         B_DRA_01H_Resample_12H.pkl
    accuracy                          0.615
    precision                           0.5
    recall                            0.067
    Name: 3, dtype: object
    

### details
- By prioritizing the precision and the recall instead of the accuracy, It was made that the window unit of the 'DRA' and the 'resample' that can satisfy both the case A and the case B.
- 정확도 대신 정밀도와 재현율을 우선으로 하여, A 케이스와 B 케이스 모두 만족할 수 있는 DRA의 window 단위, resample의 window 단위를 판단하였다.
- When 1 hour and 3 hours were applied to DRA's window and 12 hours were applied to resample's window, the algorithm performed the best.
- DRA의 window에 1시간, 3시간을 적용하고, resample의 window에 12시간을 적용했을 때, 해당 알고리즘이 가장 좋은 성능을 냈다.

## 4. Discussion
- Among the various experimental methods, the case that I think is the most representative was selected, and the code was written based on this.
- 여러 실험 방법 중 가장 대표한다고 생각하는 케이스를 선정하였고, 이를 기반하여 코드를 작성하였다.
- In addition to the ADTK library, the algorithm has a way to use the Kats library.
- 알고리즘에 ADTK 라이브러리 이외에도 Kats 라이브러리를 사용하는 방법이 있다.
- I would like to know how evaluate the classification data in a different way.
- 분류 데이터에 대해 다른 방법으로 평가할 수 있지 않을까라는 생각을 했다.
- There was a disadvantage that it was difficult to collect the big data because it was the medical data.
- 의료용 데이터이기 때문에 빅데이터를 수집하기 어렵다는 단점이 있다.


```python

```

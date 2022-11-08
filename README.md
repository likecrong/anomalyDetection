# anomalyDetection
## A study of anomaly detection about the symptom animals have on the activity-Based for early
- This study has been rewritten as a security issue.
- Period: November 4, 2022 ~ November 7, 2022
- Number of people: 1 person
- IED: Jupyter Notebook
- skill: python3, ADTK, openCV
  
### The contents of a study
- It was conducted based on pig experiment data by a university.
- Period: April 10, 2021~ April 29, 2021
- Data A: 2021.04.22. (Vaccination) A is injected.
- Data B: 2021.04.15. (Vaccination) B Virus is injected.
- OBJECTIVE: Using the ADTK, I would like to detect the anomaly section of the degraded activity from the data A and B.
- Process
1. Data collection: Data frames are made by extracting pig activity from image data.
2. Data preprocessing: the data exceeding UML for data frames is replaced with zero.
3. Data Analysis and Visualization:
	- (1) modifying the algorithm of the persistAD function of the ADTK.
		- original algorithm : aggregate(DRA) -> anomalydetection
		- fixed algorithm : aggregate(DRA)-> resample -> anomalydetection
	- (2) Applying some numbers like '1h' and '12h' to the window of the 'DRA' and 'resample' functions, and then the detector found anomalies.
	- (3) Framing data for anomalies.
	- (4) Visualizing the data and anomalies.
	- (5) Repeat (3) and (4) for the process of (2).
4. Evaluation of the data analysis result
	- (1) Calculating the accuracy, precision, and recall.
	- (2) With precision and recall rate as a priority, obtaining a total of 20 cases of the top 10 cases of the A and the B.
	- (3) Interlocking each of the 10 cases.
	- (4) Some graphs were presented for the best cases.
  
  
## 이상징후 조기감지를 위한 활동성 기반 가축 이상탐지 연구
- 해당 연구 내용은 보안상 문제로 재작성되었습니다.
- 기간: 2022.11.04.~ 2022.11.07.
- 인원: 1명
- IED: Jupyter Notebook
- skill: python3, ADTK, openCV
  
### 연구 내용
- 모 대학교에서 진행한 돼지 실험데이터를 기반으로 진행한다.
	- 기간: 2021.04.10.~ 2021.04.29.
	- A데이터: 2021.04.22.(접종일) a균을 투입한다.
	- B데이터: 2021.04.15.(접종일) b바이러스를 투입한다.
- 목표: ADTK를 이용하여 A데이터와 B데이터의 활동성 저하 구간을 탐지한다.
- 과정
1. 데이터 수집: 영상데이터로부터 돼지의 활동성을 추출해 데이터프레임을 생성한다.	
2. 데이터 전처리: 데이터프레임에 대해 UML을 초과하는 데이터는 0으로 대체한다.
3. 데이터 분석 및 시각화: 
	- (1) ADTK의 persistAD 함수의 알고리즘을 수정한다.
		- original 알고리즘 : aggregate(DRA) -> anomalydetection
		- fixed 알고리즘 : aggregate(DRA)-> resample -> anomalydetection
	- (2) DRA, resample 함수에 여러 단위를 적용하면서 이상치를 탐지한다.
	- (3) 이상치에 대해 데이터 프레임화 한다.
	- (4) 데이터와 이상치에 대해 시각화한다.
	- (5) (2)의 과정에 대해 (3),(4)를 반복하면서 분석을 진행한다.
4. 데이터 분석 결과 평가
	- (1) 정확도, 정밀도, 재현율을 계산한다.
	- (2) 정밀도, 재현율을 우선순위로 하여 A데이터와 B데이터의 상위 10개의 케이스 총 20개를 구한다.
	- (3) 각 10개의 케이스에 대해 교집합한다.
	- (4) Best Case에 대해 그래프를 제시한다.


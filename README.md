# 결과
----
```
Trainer will use only 1 of 2 GPUs because it is running inside an interactive / notebook environment. You may try to set `Trainer(devices=2)` but please note that multi-GPU inside interactive / notebook environments is considered experimental and unstable. Your mileage may vary.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
Testing DataLoader 0: 100%|██████████| 3000/3000 [03:43<00:00, 13.40it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
         test/f1             0.996999979019165
    test/f1_weighted        0.9969989061355591
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
[{'test/f1': 0.996999979019165, 'test/f1_weighted': 0.9969989061355591}]
```
----
__F1-score__: `0.997` (비공식 1등.. [공식 1등: 0.996]) [pre-trained model 을 사용하면 안된다는 것을 경진대회 도중 issue를 통해서 특정 개인에게만 알려줌^^]

### 각 치아에 대한 충치 판별 시각화 자료.
- 빨강: 실제 충치, 파랑: 충치라고 예측한 치아
![각 이빨에 대한 충치 여부 판별](https://github.com/comeeasy/healthcare-ai-contest/blob/main/result/output.png)
----
# Model Weight
- [Model download link](https://drive.google.com/drive/folders/1Dn8m15I3uQ5nXXqGVlXXb9Tz8j-Bf2Eq?usp=sharing)

# 소감
- 폐쇄망에서 진행된 경진대회인 만큼 운영진이 철저히 준비를 했어야 했는데 그렇지 못했음.
- 하나의 GPU 서버를 서로 다른 팀 인원들끼리 사용하도록하여 지저분한 GPU 점유 경쟁이 일어났고 그로인해 학습을 전혀 못한 인원들도 발생하였음.
- Pre-trained model 을 사용하면 안된다는 것을 대회 도중에 알려줌. (전체 공지도 하지 않음.) 이런 운영진들이 사전학습 모델을 사용했는지 안했는지 판단을 가능한가에 대한 의문이 들었음..
- 그냥 제출했으면 1등 했을거 같지만 pre-trained model을 사용했기 때문에 팀원들끼리 합의하에 중도 포기하였음..

![2023 헬스케어 AI 경진대회(포스터](https://raw.githubusercontent.com/bab-korea/healthcare-ai-contest/main/headlthcare_ai_contest_poster.png)

# 2023 구강이미지 합성데이터 헬스케어 AI 경진대회

구강이미지 합성 데이터 헬스케어 AI경진대회
"2023 NIA 구강이미지 합성데이터셋 구축사업"의 일환으로 추진된 치과 구강이미지 합성 데이터 분야의 헬스케어 AI 경진대회 입니다.<p>
구강이미지 합성데이터를 활용한 충치 치아 분류 AI모델 개발에 도전해보세요!
<p>
본 대회는 NAVER CLOUD PLATFORM의 고성능 클라우드 인프라 상에서 진행됩니다.
<br>

## 대회주제
구강 이미지 합성 데이터를 활용한 충치 치아를 분류하는 AI 모델 개발

## 심사 및 평가방식
- IT/AI분야 전문가로 구성된 전문 평가위원단 구성 심사 진행
   ① 사전 검토 : 평가 적격/비적격 참여자 선별
   ② 정량 평가 : 평가지표를 통한 점수 산정
- f1-score가 1점에 가까울 수록 높은 점수 부여 (f1-score를 점수로 환산하여 순위 선정)

## 결과 제출 형식
* 프로그램 결과물 및 성능지표<br>
  ① 프로그램 결과물(개발된 모델) : GPU서버 內 정답 디렉토리에 업로드  - 작성된 모델의 소스 파일 (사용한 언어 및 라이브러리의 버전이 작성된 파일)<br>
  ② 성능지표 : GPU서버 內 성능지표 디렉토리에 업로드<br>
* Report (기술문서) 제출<br>
 - 심사 후 수상 대상자 제한 제출  ※ 수상대상 참가자에게는 일정 및 진행사항에 대해 개별 안내드립니다.<br>
 - 결과 및 분석 레포트를 간단히 요약하여 PDF 파일 형태로 제출<br>
 - 레포트에 반드시 포함되어야 하는 내용은 제출팀의 이름, 모형에 대한 간단한 설명<br>
 - 보고서 분량은  A4 1page ~ 10page 이내<br>

## 시상 및 혜택
- 총시상팀: 8개팀 / 총상금: 500만원

<table class="tbl_prize">
  <tr>
    <th style="text-align:left;width:50%">시상</th>
    <th style="text-align:center;width:15%">시상 수</th>
        <th style="text-align:left;width:35%">상금</th>
  </tr>
  <tr>
    <td>
      <strong>대상</strong><br>
    </td>
    <td> 1팀 </td>
    <td align=center> 200만원 </td>
  </tr>
 <tr>
    <td>
      <strong>최우수상</strong><br>
    </td>
        <td align=center> 1팀 </td>
       <td style="text-align:center"> 100만원</td>
   </tr>
      <tr>
    <td>
      <strong>우수상</strong><br>
    </td>
        <td align=center> 2팀 </td>
    <td style="text-align:center">각 50만원</td>
   </tr>
   <tr>
    <td>
      <strong>입선</strong><br>
    </td>
        <td align=center> 4팀 </td>
    <td style="text-align:center">각 25만원</td>
   </tr>
</table>

## 경진대회 일정
- 참가신청: 2023년 11월 30일(금) ~ 12월 12일(화)
- 대회참가: 2023년 12월 06일(수) ~ 12월 12일(화) 18:00까지
- 심사기간: 2023년 12월 14일(목) ~ 12월 15일(금)
- 결과발표: 2023년 121월 16일(토)
- 시상식: 2023년 12월 21일(수) 예정

## 추진
- 주관: 서울대학교 치과병원
- 후원: 한국지능정보사회진흥원

## 문의 및 FAQ
ISSUE 페이지에 문의글을 남가시면 담당자가 답변드립니다.

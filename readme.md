# Deep_Knowledge_Tracing

## **Abstract**

- Deep Knowledge Tracing(이하 DKT)란 교육기관에서 시험을 실시하고 성적에 따라 얼마만큼 아는지 평가하는 할 때 개개인에 맞춤화된 피드백을 받기가 어려운 문제점을 해결하는 방법이다.
- DKT를 활용하면 우리는 학생 개개인에게 수학의 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능 따라서 DKT는 교육 AI의 추천이라고 불린다. DKT는 맞춤화된 교육을 제공하기 위해 아주 중요한 역할을 맡는다.
- 필자는 이러한 인공지능 모델을 설계하기 위해 여러 종류의 모델(lightgbm, catboost, lightgcn, SASRec)을 사용하는 Multi model 아키텍처를 설계하였다.
- 최종적으로 public / private 에서 모두 3등을 기록하며 Public AUC기준 0.8255를 달성하였다.

## Introduction
<img width="1000" alt="Untitled" src="https://github.com/Bae-hong-seob/Deep_Knowledge_Tracing/assets/49437396/b2c28c54-13d4-406e-8469-a793635dced7">
<br>

- DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론  
- 시험에 대해 과목을 얼마만큼 이해하고 있는지 측정 및 활용하여 아직 풀지 않은 미래의 문제에 대한 정답 여부를 확인할 수 있다.  
- DKT를 활용하면 우리는 학생 개개인에게 과목에 대한 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능하다.  
- 이번 대회에서는 Iscream 데이터셋을 이용, 모델 구축 및 평가를 진행하였다.  
- 또한 이번 대회는 학생 개개인의 이해도를 나타내는 지식 상태가 아닌 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중한다.  

## Experiment Setting

- HW : V100 GPU  
- Tool : Visual Studio, Wandb  
- SW : Python 3.10.13, torch 2.1.0, transformers 4.35.2, numpy 1.26.0, pandas 2.1.3, lightgbm 3.2.1  

## Dataset

- Input : 총 7,442명의 사용자들과 총 9,454개의 고유 문항간의 Interaction  
    - userID ( 사용자 ID )  
    - assessmentItemID ( 문제 번호 )  
    - testId ( 시험지 번호 )  
    - Timestamp ( 문제를 풀기 시작한 시간 )  
    - KnowledgeTag ( 지식 유형 )  
    - answerCode (정답 여부)  
- output : test_data 사용자들의 마지막 문제의 정답 여부(0 또는 1)  

## Team members

1. 김진용 : LightGCN, UltraGCN 구현  
2. 박치언 : XLNet 모델링 및 구현, SASRec 모델 참조  
3. 배홍섭 : [Team Leader] Feature Engineering 및 Lightgbm 고도화  
4. 안제준 : Catboost, BERT4REC 구현  
5. 윤도현 : 데이터 EDA, TabNet, LastQuery 구현  

## Schedule

- competition : 2024-01-03 ~ 2024-01-25
<img width="1000" alt="스크린샷 2024-01-29 오전 2 07 46" src="https://github.com/Bae-hong-seob/Deep_Knowledge_Tracing/assets/49437396/75f2bafb-3e6d-4bca-b91a-c2ec340eab43"><br>

1. 프로젝트 개발환경 구축 (Server, Github, Google Drive, WandB)
2. EDA를 통해 데이터 분포 파악 및 Feature Engineering에 필요한 Insight 도출
3. 베이스라인 코드에 대한 이해를 바탕으로 모델 개선점 파악
4. Feature Engineering을 통해 데이터 설명력 증진
5. 리서치를 바탕으로 Baseline보다 뛰어난 성능의 모델 구현 및 Fine-Tuning
6. Post-Processing 및 Public Score을 기준으로 고성능 모델 간 Ensemble 진행
7. 최종 제출

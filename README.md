# Diagnose or Scale Up?
### 오류 방향을 진단해 그 경계만 고칠 것인가, 모델과 증강을 일괄로 키울 것인가

cat · dog · not(cat과 dog가 아닌 나머지 이미지) 3클래스 이미지 분류에서 모델을 개선하는 두 방법을 같은 조건에서 비교한 실험입니다.

- **T (Targeted)**: 모델이 어느 방향으로 틀리는지(예: 고양이를 not으로 놓침) 먼저 진단하고, 그 경계를 겨냥해 백본과 증강을 고릅니다.
- **U (Uniform)**: 정확도 같은 집계 지표만 보고 더 큰 모델과 더 강한 증강을 한꺼번에 적용합니다.

결과를 공정하게 판정하기 위해 처방 규칙과 판정 규칙을 학습 전에 **해시**로 잠가 두었고(사전등록),  
차이의 원인까지 확인하기 위해 백본과 증강을 따로 바꿔 보는 **요인 설계**와 **절제 실험**을 함께 넣었습니다.

---

## 1. 실험 질문

| 질문 | 확인 방법 |
|---|---|
| T는 자기가 겨냥한 오류 방향에서 U보다 덜 틀리는가? | H1 T − U (난이도별) · H2 겨냥 방향에서만 차이가 나는가 · H3 다른 클래스를 희생하지 않았는가 · H8 seed별 일관성 |
| 차이가 있다면 원인은 무엇인가? (HARD) | H4 백본 · H5 증강 처방 · H7 혼합 짝 방식 · H9 클래스 가중치 · H10 학습률 |
| 진단 자체는 쓸모가 있었는가? | D1 baseline 대비 개선 · D3 not 임계값 조정만으로 풀리는가 |

- **지표**: 겨냥 방향 오류 = 진단에서 지목한 방향의 오류 건수 ÷ test 이미지 수 × 1,000. 차이는 "T − U"처럼 앞 조건에서 뒤 조건을 뺀 값이라, 음수면 앞 조건이 덜 틀린 것입니다.
- **정확도 대신 방향별 오류를 본 이유**: 정확도는 모든 오류를 한 숫자로 합쳐, 오류가 한 방향에서 다른 방향으로 옮겨가도 잘 드러나지 않습니다. 실제로 HARD에서 혼합 배치의 클래스 가중치를 끄자 not→cat 오류는 seed당 68.0건 줄고 cat→not 오류는 65.4건 늘었지만, 전체 오류는 518.4건에서 505.0건으로 2.6% 줄어드는 데 그쳤습니다(H9).
- 판정 기준(95% 구간 · Holm · margin · TOST)과 그 근거는 [3.6](#36-판정-기준)에 정리했습니다.

---

## 2. 데이터

**CIFAR-10의 cat · dog**와 **CIFAR-100의 not**(cat과 dog가 아닌 나머지 이미지)으로 만들었습니다 [[1]](#ref-1). **not은 cat · dog와의 시각적 유사도에 따라 세 난이도로 나눴고**(CIFAR-100 클래스 기준 EASY 50 · MID 27 · HARD 22개 분량), cat · dog(train 5,000 / test 1,000장씩)는 세 난이도에 같은 파일을 썼습니다. 난이도를 나눈 것은 not이 동물과 얼마나 닮았는지에 따라 진단 처방의 효과가 달라지는지 보기 위해서 입니다.

이미지는 모두 32×32입니다. 사전학습 가중치(torchvision `IMAGENET1K_V1`)가 224×224 입력을 기준으로 만들어져 있어 [[2]](#ref-2) 224×224로 확대해 학습했습니다.

| 난이도 | not (train / test) | 분할 후 train / val / test | test의 not 비율 |
|---|---|---|---:|
| EASY (유사도 낮음) | 25,000 / 5,000 | 27,799 / 6,950 / 7,000 | 71.4% |
| MID (중간) | 13,500 / 2,700 | 18,764 / 4,691 / 4,700 | 57.4% |
| HARD (높음) | 11,000 / 2,200 | 16,767 / 4,192 / 4,200 | 52.4% |

**중복 검사와 분할**

- CIFAR-10 test 이미지의 3.3%, CIFAR-100 test 이미지의 10%는 train에 준중복 이미지가 있습니다 [[3]](#ref-3). 그래서 분할 전에 MD5로 **정확 중복**을, dHash [[4]](#ref-4)로 **준중복**(**64비트 해밍 거리 ≤ 4** 그리고 **256비트 ≤ 32**)을 찾았습니다. 두 해시가 모두 가까울 때만 준중복으로 봐 오탐을 줄였습니다.
- test와 겹친 train 이미지와 그 그룹, 라벨이 섞인 그룹은 train에서 뺐습니다. (EASY 251 · MID 45 · HARD 41장, 난이도별 train 폴더의 0.2~0.7%).
- **test는 한 장도 빼지 않았습니다.** 겹친 쌍 중 한쪽만 빼도 누수를 막을 수 있으므로, 평가 기준인 test 대신 train 쪽을 뺐습니다. 덕분에 원본 test 이미지가 그대로 남았고, 세 난이도가 같은 동물 test 2,000장을 공유해 난이도 간 비교 기준도 유지됐습니다.
- **validation(train의 20%)은 이미지 한 장씩이 아니라 준중복 그룹 단위로 나눴습니다.** 같은 이미지의 사본이 train과 validation에 갈라지면 모델이 학습 때 본 이미지를 validation에서 다시 맞히는 셈이라 validation 성능이 실제보다 높게 나오고 [[5]](#ref-5), validation으로 하는 진단과 epoch · 학습률 선택이 함께 왜곡되기 때문입니다.
- **not 이미지가 cat · dog보다 2배 이상 많아**(HARD train not 8,777장, cat · dog 각각 약 4,000장), 그대로 학습하면 **애매한 이미지를 not으로 판단하는 쪽으로 치우치기 쉽습니다.** 그래서 장수가 적은 클래스일수록 한 장의 손실을 더 크게 반영하는 클래스 가중치 `w_c = N / (K · N_c)`를 썼습니다(scikit-learn 'balanced'와 같은 식 [[6]](#ref-6)). HARD에서는 cat · dog 1.40, not 0.64로, 세 클래스가 손실에서 같은 몫(가중치 × 장수)을 갖습니다.

---

## 3. 실험 과정

### 시스템 아키텍처

<img width="600" alt="fig01_system_architecture" src="https://github.com/user-attachments/assets/6fe83a5f-dcd2-4491-a46a-7530b22ccaf8" />

※ 시스템 아키텍처 시각화에는 Claude 기반 이미지 생성 기능을 활용했습니다.


| 단계별 진행 | 설계 이유 |
|---|---|
| **① 진단**<br>ResNet18 baseline(B, seed 42 · 43 · 44)의 **validation** 예측으로 방향별 오류율을 구하고 규칙 A~E 적용 | test는 확증 판정에만 쓰려고 진단에 쓰지 않았습니다. seed 3개의 예측을 합쳐, 2% 기준(3.1) 근처에서 판정이 seed마다 달라지지 않게 했습니다 |
| **② 처방 · 사전등록**<br>T · U 처방 설계, epoch과 학습률 선택(validation), 가설 · 대비 · 판정 규칙을 SHA-256으로 잠근 뒤 확증 학습 시작 | 결과를 본 뒤 분석 방법을 고르면, 결과에 끼워 맞춘 해석도 미리 예측한 것처럼 보이게 됩니다 [[7]](#ref-7). 잠근 뒤 내용이 바뀌면 해시가 맞지 않아 다음 단계가 멈춥니다 |
| **③ 확증 학습**<br>판정에 쓰는 본 학습입니다. 같은 난이도 · seed의 모든 학습 조건(arm)을 같은 배치 순서 · 같은 증강 난수로 함께 학습(lockstep). seed 42~46 × 60 epoch, 모델 80개 | 배치 순서나 증강 난수만 달라져도 결과가 흔들립니다 [[8]](#ref-8). 같은 난수를 공유하면 두 arm의 결과가 함께 움직이는 만큼 차이의 분산이 줄어듭니다 [[9]](#ref-9). 백본이 다르면 학습 경로가 갈라지므로, 남는 변동은 seed 5개로 추정했습니다 |
| **④ 판정**<br>test를 한 번만 열어 예측을 잠그고, 사전등록 규칙으로 자동 판정 | 같은 test를 보며 여러 번 고치면 test에 과적합됩니다 [[10]](#ref-10). 그래서 test는 한 번만 열 수 있는 보관소(vault)에 잠가 두었습니다 |

학습이 계획대로 됐는지는 학습 검증 250개 항목(arm 간 step 수 · 학습 이미지 수 일치, 모든 epoch의 증강 계획 재계산 일치, 최종 가중치 해시 등)으로 확인했고, 모두 통과했습니다.

### 3.1 진단과 T 처방

진단은 동물↔not 4방향(cat→not · dog→not · not→cat · not→dog)만 봤습니다. cat↔dog 방향은 이 실험이 다루는 동물–not 경계가 아니어서, T가 겨냥하지 않는 대조 방향(H2)으로 남겼습니다.

| 규칙 | 조건 (validation) | 켜지는 처방 | 근거 |
|---|---|---|---|
| target | 오류율 ≥ 2%인 방향 | 겨냥 방향 · 진단 쌍 짝(cat↔not, dog↔not) | 오류가 실제로 몰린 방향만 겨냥하기 위한 문턱. 문턱 근처 판정이 seed에 흔들리지 않도록 seed 3개의 예측을 합쳐 계산 |
| A | 동물→not 오류 중 확신도 ≥ 0.9인 비율 ≤ 10% | Random Erasing p 0.5 | 놓친 동물 이미지를 대부분 낮은 확신도로 틀렸다면 가려지거나 일부만 보인 경우일 수 있다고 보고, 일부를 지운 이미지로 학습해 가림에 강해지게 함. p · 면적(2~40%) · 종횡비(0.3~3.3)는 논문 기본값 [[11]](#ref-11) |
| B | 과적합(train − val 정확도) ≥ 0.03 | CutMix (α 1.0) | 과적합을 줄이는 정규화. α = 1은 논문의 모든 실험에서 쓴 값 [[12]](#ref-12) |
| C | not→동물 방향도 target | MixUp (α 0.4) | 동물과 not을 양쪽 방향으로 헷갈리는 경우로 보고, 두 클래스 이미지를 섞어 그 사이를 매끄럽게 학습시킴. α는 ImageNet에서 이득이 확인된 0.1~0.4 범위의 상한 [[13]](#ref-13) |
| D | 4방향 모두 target이고, not 로짓에 상수를 더해 임계값을 옮겨도 macro-F1 이득 ≤ 0.002 | 후보 중 ImageNet top-1이 가장 높은 백본 | 임계값 조정으로 풀리지 않으면 모델이 두 클래스를 구분할 특징을 충분히 뽑지 못하는 것으로 보고 백본을 바꿈. ImageNet 정확도가 높은 모델일수록 파인튜닝 성능도 높음(CIFAR-10/100 포함 12개 데이터셋, r = 0.96) [[14]](#ref-14) |
| E | val 정확도의 epoch 간 변화량 중앙값 ≥ 0.0025 | EMA (decay 0.999) | 가중치 평균으로 학습 잡음을 줄임 [[15]](#ref-15) |

규칙과 임계값은 모두 진단 전에 프로토콜에 적어 잠갔습니다(`01_protocol_lock`). 진단 결과를 본 뒤 규칙을 바꾸면 해시가 맞지 않아 다음 단계가 멈추고, 같은 진단 수치를 넣으면 항상 같은 처방이 나옵니다.

**진단 결과** (baseline validation, seed 3개 합산 · 굵은 값 = target)

| 난이도 | cat→not | dog→not | not→cat | not→dog | 과적합 | 흔들림 | 켜진 규칙 |
|---|---:|---:|---:|---:|---:|---:|---|
| EASY | **8.0%** | **3.5%** | 1.1% | 0.5% | 0.049 | 0.0027 | A · B · E |
| MID | **11.9%** | **6.3%** | **3.0%** | 1.8% | 0.096 | 0.0034 | A · B · C · E |
| HARD | **18.3%** | **13.8%** | **6.4%** | **3.6%** | 0.156 | 0.0040 | A · B · C · D · E |

- 문턱에 가장 가까운 방향은 MID not→dog(1.8%)였고, 이 방향은 target에서 빠졌습니다.
- 규칙 A는 확신도 0.9 이상 비율이 0~0.3%라 세 난이도 모두에서 켜졌습니다.
- 백본 후보는 ResNet18 · ResNet50 [[16]](#ref-16) · EfficientNet-B0 [[17]](#ref-17)으로 미리 정했습니다. 규칙 D가 켜진 HARD에서만 top-1이 가장 높은 EfficientNet-B0(77.692 > ResNet50 76.130 > ResNet18 69.758 [[2]](#ref-2))로 바뀌었습니다.

### 3.2 U 처방

U는 정확도 · macro-F1 · 클래스별 precision · recall · 과적합 · 흔들림만 입력으로 받습니다. 방향별 오류가 들어오면 `ControlFailure`로 멈추게 해, U가 진단 정보를 쓰지 못하게 막았습니다. baseline validation 정확도가 EASY 0.951 → HARD 0.845로 0.107 떨어지고 과적합 · 흔들림도 기준을 넘어, 아래 세 트리거가 모두 켜졌습니다. 그래서 세 난이도에 같은 처방이 적용됐습니다.

| 트리거 | 조건 | 처방 | 근거 |
|---|---|---|---|
| capacity | EASY→HARD 정확도 하락 ≥ 0.03 또는 macro-F1 < 0.90 | ResNet50 | ResNet18을 같은 계열의 더 깊은 모델로 키우는 것은 가장 흔한 확대 방식 [[17]](#ref-17) (ImageNet 분류기 포함 파라미터 11.7M → 25.6M [[2]](#ref-2)) |
| regularize | 과적합 ≥ 0.03 | RandAugment(2, 9) + 무작위 짝 CutMix · MixUp + Random Erasing 0.25 | ImageNet 학습에서 널리 쓰는 강한 증강 묶음. 구성과 Random Erasing 0.25는 DeiT 레시피 [[18]](#ref-18), RandAugment 값은 논문이 ImageNet ResNet-50에 쓴 N = 2, M = 9 [[19]](#ref-19) |
| stabilize | 흔들림 ≥ 0.0025 | EMA | T의 규칙 E와 같음 [[15]](#ref-15) |

### 3.3 T와 U 처방 정리

| | T (진단 기반) | U (일괄 확대) |
|---|---|---|
| 입력 | 진단 결과 (방향별 오류) | 집계 지표 |
| 백본 | ResNet18 (EASY · MID) / **EfficientNet-B0 (HARD)** | ResNet50 |
| 증강 | Random Erasing 0.5 · 진단 쌍 짝으로 CutMix(EASY), CutMix + MixUp(MID · HARD) | Random Erasing 0.25 · RandAugment · 무작위 짝으로 CutMix + MixUp |
| 공통 | EMA · 같은 데이터 분할 · 같은 학습 레시피 · 같은 난수 | |

CutMix · MixUp은 두 방법 모두 배치의 50%에만 적용했고 α(CutMix 1.0, MixUp 0.4)도 같게 맞춰, 짝을 짓는 방식만 다르게 했습니다. 그래서 HARD에서 두 처방의 실제 차이는 Random Erasing 확률, RandAugment 유무, 짝 방식 세 가지입니다.

### 3.4 같은 조건 맞추기

- **학습 레시피**: AdamW(lr 2e-4, weight decay 0.01), 3 epoch warm-up 후 cosine 감소, label smoothing 0.1, 배치 64 등은 모든 arm에 같게 고정했습니다. 처방 외 조건을 고정하는 것이 목적이라 arm별로 따로 조정하지 않았고(학습률만 아래처럼 따로 점검), 사전등록 직전에 `control_table()`로 T와 U의 고정 조건이 같은지 확인했습니다.
- **epoch 60, 조기 종료 없음**: arm마다 멈추는 시점이 다르면 학습량이 달라져 처방 효과와 학습량 효과가 섞이고, lockstep 학습은 모든 arm이 같은 step을 밟는 것을 전제로 합니다. 그래서 모든 arm을 60 epoch 학습하고 마지막 가중치로 평가했습니다. epoch 수는 학습량이 가장 적은 조건에서도 충분하도록 epoch당 step이 가장 적은 HARD에서 정했습니다. {40, 60, 80}을 학습해 보고, 더 길게 학습해도 겨냥 방향 오류(margin의 절반인 6.43/1,000 이내) · macro-F1(0.005 이내) · T − U 차이가 거의 변하지 않는 가장 짧은 값으로 골랐습니다. 확증 모델 80개 중 75개는 마지막 12 epoch 동안 허용오차 넘게 개선되지 않았습니다(나머지 5개 중 4개는 baseline).
- **학습률**: 공통 학습률 2e-4가 한쪽 방법에만 맞는 값일 수 있어, HARD에서 T와 U 각각 {1e-4, 2e-4, 4e-4} 중 validation macro-F1이 가장 높은 값을 골랐습니다(T 2e-4, U 1e-4). U를 1e-4로 학습한 Ustar와의 비교가 H10입니다.
- **seed 분리**: epoch과 학습률은 확증 seed(42~46)와 다른 seed(40 · 41)로 골라, 선택에 쓴 학습 결과가 확증 비교에 섞이지 않게 했습니다.

### 3.5 원인 분리 (HARD)

T와 U는 백본과 증강이 함께 다르기 때문에 T − U만으로는 차이의 원인을 알 수 없습니다. 원인 분리 가설(H4~H10)은 HARD에서 검정하도록 미리 정했습니다. HARD는 규칙 D로 T의 백본까지 바뀌어 백본과 처방이 가장 크게 얽히고, 4방향이 모두 target이라 진단의 역할도 가장 큰 난이도이기 때문입니다.

- **요인 설계**: T 처방과 U 처방을 백본 3종(ResNet18 · ResNet50 · EfficientNet-B0)에 모두 적용했습니다. `T@R50`은 T 처방을 ResNet50에 적용한 arm입니다.
- **절제 실험**: 구성 요소 하나만 바꾼 arm을 더했습니다. **Trand**는 짝만 무작위로 바꿨고(혼합 짝 중 진단 쌍 비율 89.4% → 49.9%), **Tunw**는 혼합 배치의 클래스 가중치를 껐고, **Ustar**는 U에 validation 최적 학습률(1e-4)을 줬습니다.

```text
H4 백본 주효과 = ½(T + U@EffB0) − ½(T@R50 + U)
H5 처방 주효과 = ½(T + T@R50)   − ½(U@EffB0 + U)
```

### 3.6 판정 기준

| 항목 | 방법 | 근거 |
|---|---|---|
| 95% 구간 | seed 간 분산 + test 이미지 bootstrap(10,000회) 분산을 합친 hybrid 구간, Welch 자유도 | 학습 난수와 평가 데이터 표본이 모두 결과를 흔드는 변동 요인이라 둘 다 반영했습니다 [[8]](#ref-8). seed가 3개면 구간이 지나치게 넓어져 5개로 했습니다(seed 성분 자유도 2 → 4) |
| 다중 비교 | Holm 보정 (family: H1 세 난이도 · H2 세 난이도 · 확증 가설 H4 · H5 · H7 · H8 · H9 · H10) | 가설이 여러 개라, family 안에서 하나라도 우연히 유의하게 나올 확률을 0.05 이하로 묶었습니다. Holm은 Bonferroni와 같은 보장을 주면서 검정력은 같거나 높습니다 [[20]](#ref-20) |
| margin (실질적 차이 기준) | baseline validation 오류의 10%, 최소 1/1,000 (겨냥 방향 EASY 1.65 · MID 5.63 · **HARD 12.86**) | 의미 있게 볼 최소 효과 크기는 데이터를 보기 전에 정해야 해서 [[21]](#ref-21) 프로토콜에 고정했습니다. 겨냥 방향 baseline 오류가 EASY 16.5, HARD 128.6/1,000으로 약 8배 차이가 나서 고정값 대신 비례값으로 잡았습니다. 10%라는 비율 자체는 사전에 정한 선택값입니다(민감도는 6장) |
| 실질적 동등 (TOST) | 90% 구간이 ±margin 안에 들면 동등 | "유의하지 않음"은 "같음"이 아니므로, 90% 구간이 ±margin 안에 들 때만 실질적 동등으로 판정했습니다. 이 조건은 α = 0.05의 두 단측 검정(TOST)을 모두 통과하는 것과 같습니다 [[21]](#ref-21)[[22]](#ref-22) |
| 특이성 (H2) | 겨냥 방향 차이에서 대조 방향(cat↔dog) 차이를 뺀 값(DiD, 차이의 차이)을 직접 검정 | "겨냥 방향은 유의, 대조 방향은 비유의"만으로는 두 방향이 다르다고 말할 수 없습니다 [[23]](#ref-23) |

판정은 네 가지입니다. **지지** = Holm p ≤ 0.05이고 95% 상한 < 0(상한 < −margin이면 "실질적 크기") · **반대 방향** = Holm p ≤ 0.05이고 추정치 > 0 · **실질적 동등** = 90% 구간이 ±margin 안 · **결론 유보** = 어느 쪽에도 해당하지 않음.

---

## 4. 결과

![결과 요약](images/fig02_summary_results.png)

### 4.1 T와 U 비교

| 가설 | 비교 | 결과 (/1,000, 95% 구간) | 판정 |
|---|---|---|---|
| H1 · H8 | HARD T − U | **−18.62** [−25.98, −11.26] | 지지 (seed 5개 모두 음수) |
| H2 | HARD 겨냥 − 대조 방향 (DiD) | −15.81 [−24.88, −6.74] | 겨냥 방향에 특이적, 대조 방향의 동등은 미확립 |
| H3 | HARD 원래 클래스 재현율 · macro-F1 (T − U) | cat +3.52%p · dog +4.48%p · not +0.45%p · macro-F1 +0.024 | 통과 (EASY · MID는 재현율 비열등 미확립) |
| H1 | EASY / MID T − U | +2.46 [+0.68, +4.23] / +4.94 [+0.37, +9.51] | 반대 방향 (겨냥 방향 오류는 U가 더 적음) |
| D1 | HARD baseline 대비 T − B / U − B | −21.24 [−31.30, −11.18] / −2.62 [−9.90, +4.66] | T만 개선, U는 baseline과 실질적 동등 |

**난이도별 성능** (test, seed 5개 평균, B = ResNet18 baseline)

| 난이도 | 겨냥 방향 오류 /1,000 (B / T / U) | macro-F1 (B / T / U) | 정확도 (B / T / U) |
|---|---|---|---|
| EASY | 14.97 / 9.06 / 6.60 | 0.9092 / 0.9093 / 0.9016 | 0.9501 / 0.9496 / 0.9442 |
| MID | 52.94 / 53.74 / 48.81 | 0.8881 / 0.8869 / 0.8923 | 0.9102 / 0.9073 / 0.9123 |
| HARD | 121.14 / 99.90 / 118.52 | 0.8369 / 0.8665 / 0.8427 | 0.8525 / 0.8766 / 0.8551 |

- **HARD에서는 T가 U보다 덜 틀렸습니다.** 차이는 동물을 not으로 놓치는 오류에서 나왔고(cat→not −3.64%p, dog→not −3.18%p), 다른 클래스의 재현율을 희생하지 않았습니다(H3).
- **HARD에서 U는 baseline보다 나아지지 않았습니다.** 공통 학습률(2e-4)에서는 ResNet50과 강한 증강을 쓰고도 겨냥 방향 오류가 baseline(ResNet18)과 실질적으로 같았습니다(D1).
- **EASY · MID에서는 겨냥 방향 오류가 U에서 더 적었습니다.** 다만 EASY 정확도는 T가 0.54%p 높았습니다(95% 구간 [+0.06, +1.02]). U는 겨냥 방향 오류가 적은 대신 다른 방향 오류가 많았습니다. 겨냥 방향은 난이도마다 달라(EASY 2 · MID 3 · HARD 4방향) 겨냥 방향 오류는 같은 난이도 안에서만 비교합니다.
- **not 임계값만 옮겨서는 개선되지 않았습니다.** baseline의 validation에서 고른 not 로짓 보정값을 test에 적용해도 macro-F1 이득은 세 난이도 모두 0.005 미만이었습니다(D3).

### 4.2 HARD에서 무엇이 차이를 만들었나

| 가설 | 비교 | 결과 (/1,000, 95% 구간) | 판정 |
|---|---|---|---|
| H4 | 백본 주효과 (EffB0 − R50) | **−23.62** [−30.00, −17.23] | 지지 (실질적 크기) |
| H5 | 처방 주효과 (T 처방 − U 처방) | +5.00 [+0.47, +9.53] | 실질적 동등 (Holm p = 0.097, 90% 구간 [+1.26, +8.74]가 ±12.86 안) |
| H7 | 진단 쌍 짝 (T − Trand) | +5.62 [−2.58, +13.82] | 실질적 동등 (90% 구간 [−0.96, +12.20]) |
| H9 | 클래스 가중치 끔 (Tunw − T) | not→cat −3.09%p · cat→not +6.54%p | 지지, 그러나 트레이드오프 |
| H10 | 학습률 맞춤 (T − Ustar) | −6.90 [−14.35, +0.54] | 결론 유보 |

![HARD 요인 설계](images/fig05_hard_factorial.png)

| 겨냥 방향 오류 (/1,000) | ResNet18 | ResNet50 | EfficientNet-B0 |
|---|---:|---:|---:|
| T 처방 | 124.81 | 122.00 | **99.90** (= T) |
| U 처방 | 121.86 | **118.52** (= U) | 93.38 |

- **학습률 2e-4 조건에서 차이는 대부분 백본에서 나왔습니다.** T − U(−18.62)를 T 처방 기준으로 나누면 백본 교체분(T − T@R50)이 −22.10, 처방 차이분(T@R50 − U)이 +3.48입니다. U 처방 기준으로 나눠도 백본 −25.14, 처방 +6.52로 같은 결론입니다. fig02 (b)의 속이 빈 점이 이 분해이며, Holm 보정을 하지 않은 탐색 분석입니다.
- **방향 맞춤 증강의 이득은 확인되지 않았습니다.** 세 백본 모두에서 T 처방이 U 처방보다 수치상 오히려 더 틀렸습니다(+2.95 · +3.48 · +6.52).
- **클래스 가중치를 끄면 오류가 다른 방향으로 옮겨갔습니다.** not→cat 오류가 줄어든 만큼 cat→not 오류가 늘어, 전체 오류는 2.6%만 줄었습니다(H9).
- HARD T는 U보다 파라미터가 약 5.5배 적고(3클래스 모델 기준 4.34M vs 24.03M) GFLOPs는 약 10배 적었지만(0.39 vs 4.09), 확률 보정 오차(ECE, 15구간 [[24]](#ref-24))는 0.167로 U(0.065)보다 컸습니다.

---

## 5. 결론

> 진단 기반 처방(T)은 HARD에서 일괄 확대(U)보다 겨냥 방향 오류를 1,000장당 18.62건 줄였습니다. 그러나 학습률을 2e-4로 고정한 조건에서 그 이득은 대부분 진단 규칙이 고른 백본(EfficientNet-B0)에서 왔고, 같은 백본 위에서 방향 맞춤 증강의 이득은 확인되지 않았습니다. EASY · MID에서는 겨냥 방향 오류가 U에서 더 적었고, U에도 최적 학습률을 주면 HARD에서도 T가 낫다고 결론 내릴 수 없었습니다.

T와 U만 비교했다면 이 결과를 "방향 맞춤 증강이 효과가 있다"로 잘못 읽을 수 있었습니다. 요인 설계를 미리 넣어 둔 덕분에 실제 주원인이 백본이라는 것을 확인할 수 있었습니다.

- **말할 수 있는 것**: HARD에서 T가 U보다 덜 틀렸고, seed를 바꿔도 같은 방향이었습니다. 학습률 2e-4 조건에서 그 주원인은 백본입니다. 혼합 배치의 클래스 가중치를 끄면 오류가 다른 방향으로 옮겨갑니다. 진단된 경계는 not 임계값 조정만으로는 풀리지 않습니다.
- **말할 수 없는 것**: 진단 기반 처방이 일반적으로 낫거나 쓸모없다는 것. 처방 효과가 아예 없다는 것(1,000장당 12.86건보다 작은 효과는 이 실험으로 가려낼 수 없습니다). 다른 데이터 · 해상도로의 일반화.

---

## 6. 한계

- **데이터**: 32×32 이미지를 224로 확대한 단일 데이터셋입니다. 난이도마다 not 비율도 함께 바뀌어, 난이도 효과에 유사도 외 요인이 섞여 있습니다.
- **원인 분리 범위**: 요인 설계는 HARD에만 했습니다. EASY · MID에서 U가 나은 이유(T는 ResNet18, U는 ResNet50)는 분리하지 못했습니다.
- **학습률**: U의 최적 학습률 1e-4가 탐색 범위의 하한이었고, 낮출수록 성능이 계속 올랐습니다(4e-4 0.8134 → 2e-4 0.8371 → 1e-4 0.8488). 요인 설계 arm은 모두 2e-4로 고정했는데, U의 학습률만 1e-4로 낮춰도 test 겨냥 방향 오류가 11.71 줄었습니다(118.52 → 106.81). ResNet50 두 칸이 모두 그만큼 준다고 가정하면 H4는 약 −11.9로 margin 안에 들어오므로, 백본 효과의 크기는 학습률에 민감합니다. H10도 U의 최적값이 더 낮다면 T에 유리하게 치우쳤을 수 있습니다.
- **특이성 (H2)**: DiD는 절대 건수 차이라 오류가 많은 겨냥 방향에서 커지기 쉽습니다. 상대 감소율은 겨냥 방향 −15.7%, 대조 방향 −10.7%였고, 모든 방향이 같은 비율로 줄었다고 가정해도 DiD는 약 −14.5입니다.
- **margin과 검정력**: 동등 판정은 margin에 달려 있습니다. H7은 margin이 baseline 오류의 9.5% 이상, H5는 6.8% 이상일 때만 동등입니다. seed가 5개라 그보다 작은 효과는 배제하지 못합니다.
- **확률 보정**: T 처방은 같은 백본에서도 ECE가 높았지만(ResNet18 0.117 vs 0.063), ECE는 사전등록 지표가 아니어서 검정하지 않았습니다.
- **사전등록과 test**: 사전등록은 외부 등록처가 아닌 로컬 해시 잠금입니다. test는 공개된 CIFAR test 이미지라, 규칙을 정하기 전에 이 이미지들을 전혀 보지 않았다고 보장할 수는 없습니다. vault 개방은 두 번째 시도에서 완료됐습니다(7.4).
- **재현성**: 난이도를 나눈 유사도 기준과 난이도별 CIFAR-100 클래스 목록, 데이터 생성 스크립트, 모델 가중치는 저장소에 넣지 않았습니다. GPU 연산의 비트 단위 결정성도 켜지 않았습니다.

---

## 7. 트러블슈팅

**7.1 준중복 판정이 연쇄로 묶여 라벨이 섞인 103장 그룹 발생**
- 문제: EASY 중복 검사에서 라벨 충돌 103장이 모두 한 그룹에 묶였고(not 98 · dog 4 · cat 1), test 한 장이 train 48장과 준중복으로 잡혔습니다.
- 원인: Union-Find는 A–B, B–C가 준중복이면 A와 C도 한 그룹으로 묶습니다. 32×32 이미지에서는 구도가 단순한 이미지끼리 dHash가 같아지기 쉬워(그룹 중 13장이 같은 해시) 이런 연결이 길게 이어졌습니다.
- 해결: 누수를 놓치지 않는 쪽을 택해 해당 그룹을 train에서 통째로 뺐습니다. 제외는 train의 0.2~0.7%에 그쳤습니다.

**7.2 Colab 세션 끊김 속에서 lockstep 학습 이어가기**
- 문제: 진단 · epoch 선택 · 학습률 선택 · 확증 학습을 합친 run 32개(run = 같은 난이도 · seed의 arm을 함께 학습하는 단위) 중 21개가 한 세션에 끝나지 못해, 64개 세션에 걸쳐 학습했습니다. HARD 확증 run 하나는 arm 10개를 함께 학습해 연산만 6.6~6.9시간이 걸렸습니다.
- 어려운 점: 가중치만 불러와 이어 학습하면 재개 전후로 난수 흐름이 달라져, 모든 arm이 같은 난수를 받는다는 lockstep 조건이 깨집니다.
- 해결: 난수를 (난이도, seed, epoch, step) 키로 매번 새로 만들어 재개해도 같은 값이 나오게 했고, 재개 전에 설정 해시와 직전 epoch 증강 계획 해시가 다르면 멈추게 했습니다. 다시 계산한 증강 계획이 모든 epoch에서 일치해 학습 검증 250개 항목을 모두 통과했습니다.

**7.3 덮어쓴 체크포인트가 Drive 휴지통에 쌓이는 문제**
- 문제: Drive에서 덮어쓰거나 지운 파일은 휴지통에 남습니다. HARD run 체크포인트는 약 1.93GB라, 5 epoch마다 저장하고 직전 파일을 지우더라도 휴지통을 비우지 않으면 run당 약 23GB가 쌓이게 됩니다.
- 해결: 매 epoch 체크포인트는 Colab 로컬에만 저장하고, Drive에는 5 epoch마다 한 벌만 두었습니다. Drive API로 휴지통의 `cdnckpt_*` 파일만 영구 삭제했습니다.
- 결과: 재개 32회 중 31회가 이 Drive 체크포인트에서 이어졌습니다.

**7.4 test vault 첫 개방 시도 중단**
- 문제: 접근 로그에 test 개방 시도가 2번, 완료 · 잠금이 1번 기록됐습니다. 첫 시도는 예측을 저장하기 전에 멈췄습니다.
- 확인: 두 시도의 기대 해시(사전등록 · 학습 검증 · 가중치 80개)가 같아 그 사이 모델과 규칙은 바뀌지 않았습니다. 예측 파일 3개는 모두 두 번째 시도에서 저장됐고, 판정은 이 한 번의 개방에서만 나왔습니다.
- 해결: test를 열기 전에 이미지 캐시와 최종 가중치 80개가 모두 있는지 점검하는 셀(04-0)을 두어, 하나라도 없으면 test를 열지 않게 했습니다.

---

## 8. 실행 방법

- **환경**: Google Colab (Tesla T4) · Python 3.13 · PyTorch 2.11 · torchvision 0.26
- **경로**: `dataset.zip`을 `/content/drive/MyDrive/DiagnoseOrScaleUp/`에 두면 결과가 같은 폴더의 `experiments_full/`과 `report_bundle.zip`에 저장됩니다.
- **순서**: `SMOKE = True`로 01 → 04를 빠르게 점검한 뒤, `SMOKE = False`로 01 → 02 → 03 → 04를 실행합니다. 세션이 끊기면 같은 노트북을 다시 실행하면 체크포인트에서 이어집니다. 01 실행 뒤에는 공통 엔진 셀을 수정하지 않습니다(잠금 해시가 맞지 않아 멈춤).

```text
.
├── 01_protocol_data_diagnosis.ipynb        # 중복 검사 · 분할 · 진단
├── 02_prescriptions_preregistration.ipynb  # T/U 처방 · epoch · 학습률 · 사전등록
├── 03_confirmatory_training.ipynb          # lockstep 확증 학습 · 학습 검증
├── 04_test_and_hypotheses.ipynb            # test 1회 평가 · 가설 판정
├── images/                                 # README 그림
└── experiments_full/                       # 결과 묶음 (잠금 파일 · 대비 CSV · 최종 보고서)
```

---

## 참고 문헌
1. <a id="ref-1"></a>Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10/100 기술 보고서). [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
2. <a id="ref-2"></a>PyTorch. *torchvision — Models and pre-trained weights* (ImageNet top-1 · 파라미터 수 · GFLOPs · 입력 전처리). [docs.pytorch.org/vision/stable/models.html](https://docs.pytorch.org/vision/stable/models.html)
3. <a id="ref-3"></a>Barz, B., Denzler, J. (2020). Do We Train on Test Data? Purging CIFAR of Near-Duplicates. *Journal of Imaging*, 6(6), 41. [doi:10.3390/jimaging6060041](https://doi.org/10.3390/jimaging6060041)
4. <a id="ref-4"></a>Krawetz, N. (2013). Kind of Like That (dHash). *The Hacker Factor Blog*. [hackerfactor.com](https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)
5. <a id="ref-5"></a>Kapoor, S., Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. *Patterns*, 4(9), 100804. [doi:10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)
6. <a id="ref-6"></a>scikit-learn. `sklearn.utils.class_weight.compute_class_weight` (`'balanced'`). [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
7. <a id="ref-7"></a>Nosek, B. A., Ebersole, C. R., DeHaven, A. C., Mellor, D. T. (2018). The Preregistration Revolution. *PNAS*, 115(11), 2600–2606. [doi:10.1073/pnas.1708274114](https://doi.org/10.1073/pnas.1708274114)
8. <a id="ref-8"></a>Bouthillier, X. et al. (2021). Accounting for Variance in Machine Learning Benchmarks. *MLSys*. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
9. <a id="ref-9"></a>Glasserman, P., Yao, D. D. (1992). Some Guidelines and Guarantees for Common Random Numbers. *Management Science*, 38(6), 884–908. [doi:10.1287/mnsc.38.6.884](https://doi.org/10.1287/mnsc.38.6.884)
10. <a id="ref-10"></a>Dwork, C. et al. (2015). The Reusable Holdout: Preserving Validity in Adaptive Data Analysis. *Science*, 349(6248), 636. [doi:10.1126/science.aaa9375](https://doi.org/10.1126/science.aaa9375)
11. <a id="ref-11"></a>Zhong, Z. et al. (2020). Random Erasing Data Augmentation. *AAAI*. [arXiv:1708.04896](https://arxiv.org/abs/1708.04896)
12. <a id="ref-12"></a>Yun, S. et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
13. <a id="ref-13"></a>Zhang, H., Cisse, M., Dauphin, Y. N., Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
14. <a id="ref-14"></a>Kornblith, S., Shlens, J., Le, Q. V. (2019). Do Better ImageNet Models Transfer Better? *CVPR*. [arXiv:1805.08974](https://arxiv.org/abs/1805.08974)
15. <a id="ref-15"></a>Morales-Brotons, D., Vogels, T., Hendrikx, H. (2024). Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits. *TMLR*. [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)
16. <a id="ref-16"></a>He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
17. <a id="ref-17"></a>Tan, M., Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
18. <a id="ref-18"></a>Touvron, H. et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention. *ICML* (Table 9). [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
19. <a id="ref-19"></a>Cubuk, E. D., Zoph, B., Shlens, J., Le, Q. V. (2020). RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. *NeurIPS*. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
20. <a id="ref-20"></a>Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. [JSTOR 4615733](https://www.jstor.org/stable/4615733)
21. <a id="ref-21"></a>Lakens, D., Scheel, A. M., Isager, P. M. (2018). Equivalence Testing for Psychological Research: A Tutorial. *AMPPS*, 1(2). [doi:10.1177/2515245918770963](https://doi.org/10.1177/2515245918770963)
22. <a id="ref-22"></a>Schuirmann, D. J. (1987). A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability. *J. Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680. [doi:10.1007/BF01068419](https://doi.org/10.1007/BF01068419)
23. <a id="ref-23"></a>Gelman, A., Stern, H. (2006). The Difference Between "Significant" and "Not Significant" is not Itself Statistically Significant. *The American Statistician*, 60(4), 328–331. [doi:10.1198/000313006X152649](https://doi.org/10.1198/000313006X152649)
24. <a id="ref-24"></a>Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

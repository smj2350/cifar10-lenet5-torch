## 📚 라이브러리
- **PyTorch / TorchVision**: 모델·데이터셋·학습 루프  
- **matplotlib**: 학습 Loss·Accuracy 시각화  

---

## 🏗️ 모델 구조

| 단계 | 연산 | 출력 크기 |
|------|------|-----------|
| Input | ­­- | **3 × 32 × 32** |
| 1 | `Conv2d(3→6, 5×5)` → **Sigmoid** → `AvgPool(2)` | 6 × 14 × 14 |
| 2 | `Conv2d(6→16, 5×5)` → **Sigmoid** → `AvgPool(2)` | 16 × 5 × 5 |
| 3 | `Conv2d(16→120, 5×5)` → **Sigmoid** | 120 × 1 × 1 |
| 4 | **Flatten** → `Linear(120→84)` → **Sigmoid** | 84 |
| 5 | `Linear(84→10)` → **LogSoftmax** | 10 (log-prob) |

- **가중치 초기화**: PyTorch 기본값 (필요 시 He 초기화 권장)  
- **Bias**: Conv/Linear 레이어 기본 포함  
- **손실 함수**: `NLLLoss` (= CrossEntropy + log 확률)  

---

## ⚙️ 하이퍼파라미터
| 항목 | 값 |
|------|----|
| Optimizer | **SGD** (momentum = 0.9) |
| Learning Rate | **0.001** |
| Epochs | **20** |
| Batch Size | **256** |
| Loss | **NLLLoss** |

---

## 🔄 파이프라인
- **Transform** :  
  `ToTensor()` → `Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))`
- **DataLoader** :  
  `trainloader` shuffle = True (IID)  
  `testloader` shuffle = False
- **평가 단계** :  
  ```python
  model.eval()
  with torch.no_grad():
- 평균 손실 = Σ batch loss / len(loader)
- 정확도 = 맞힌 샘플 / 전체 샘플

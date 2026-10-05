# Lab 1 - Supervised Learning

## Mô tả

Bài lab hoàn chỉnh về Machine Learning với California Housing Dataset, bao gồm:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Regression Models (Linear, SGD, Polynomial, Ridge, Lasso)
- Classification Models (KNN, Naive Bayes, Decision Tree, Random Forest, SVM)
- Model Evaluation với RMSE, MAE, R², Accuracy, Precision, Recall, AUC-ROC

## Cấu trúc thư mục

```
MDS301_TRANTUANHUNG_FPTU/
├── Lab_1_Supervised_Learning.md    # Đề bài lab
├── lab1_notebook.ipynb             # Bài làm chính (Jupyter Notebook)
├── requirements.txt                # Thư viện cần thiết
├── README.md                       # Hướng dẫn sử dụng
├── create_notebook.py              # Script tạo notebook
└── plots/                          # Thư mục chứa biểu đồ PNG
    ├── eda_plots.png
    ├── learning_curves.png
    ├── knn_accuracy_vs_k.png
    └── rf_feature_importance.png
```

## Cách cài đặt môi trường

### 1. Tạo virtual environment (khuyến nghị)

**Trên Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

**Trên macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Kiểm tra môi trường đã cài đặt

```bash
python -c "import numpy; import pandas; import sklearn; print('All packages OK')"
```

## Cách chạy Notebook

### Cách 1: Chạy trong Jupyter Lab (khuyến nghị)

```bash
jupyter lab lab1_notebook.ipynb
```

### Cách 2: Chạy với Jupyter Notebook

```bash
jupyter notebook lab1_notebook.ipynb
```

### Cách 3: Chạy tất cả cells tự động

```bash
jupyter nbconvert --to notebook --execute --inplace lab1_notebook.ipynb
```

## Yêu cầu hệ thống

- Python 3.8+
- RAM: 8GB+ (đủ để chạy Random Forest và SVM)
- Ổ cứng: ~500MB free space

## Ghi chú quan trọng

1. **Không thay đổi nội dung file đề** (`Lab_1_Supervised_Learning.md`)
2. **Điền thông tin sinh viên** vào notebook trước khi nộp
3. **Chạy notebook từ đầu đến cuối** (Kernel > Restart & Run All)
4. **Kiểm tra output** trước khi nộp

## Các bước hoàn thành Lab

| Phần | Nội dung | Điểm |
|------|----------|------|
| Part 1 | Setup Environment | 10 pts |
| Part 2 | EDA | 15 pts |
| Part 3 | Preprocessing Pipeline | 20 pts |
| Part 4 | Regression Models | 25 pts |
| Part 5 | Classification Models | 20 pts |
| Part 6 | Report & Interpretation | 10 pts |
| Part 7 | Git (Bonus) | 5 pts |
| **Tổng** | | **100 pts** |

## Troubleshooting

### Lỗi "Module not found"
```bash
pip install <module-name>
```

### Lỗi Memory khi chạy SVM
Đã có cơ chế tự động dùng subset 5000 samples cho SVM nếu cần.

### Lỗi OneDrive sync
Nếu gặp vấn đề với OneDrive, copy project ra thư mục khác và chạy.

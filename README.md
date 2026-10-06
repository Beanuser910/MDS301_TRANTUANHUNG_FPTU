# Lab 1 - Supervised Learning

## Mô tả

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



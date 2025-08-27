# 📧 Email/SMS Spam Classifier

A lightweight **Streamlit app** that detects **spam vs. ham** messages using **TF‑IDF features** and a **Multinomial Naive Bayes** model trained on the classic **SMS Spam Collection** dataset.

---

## 🔍 Overview
- **App UI:** `Streamlit` (`app.py`)  
- **Model pipeline:** `NLTK` preprocessing → `TfidfVectorizer` (max_features=3000) → `MultinomialNB`  
- **Artifacts used by the app:** `vectorizer.pkl` (TF‑IDF) and `model.pkl` (MultinomialNB) generated from the notebook `sms-spam-detection.ipynb`.  
- **Dataset:** 5572 messages with labels `['ham', 'spam']` (distribution: {'ham': 4825, 'spam': 747}).

---

## 🧠 How it Works
**Text preprocessing (`transform_text`):**
1) lowercase → 2) tokenize (`nltk.word_tokenize`) → 3) keep only alphanumerics  
4) remove `stopwords` & punctuation → 5) `PorterStemmer` stemming → 6) join back to string  

**Vectorization & Model:**
- `TfidfVectorizer(max_features=3000)` to convert text to features  
- `MultinomialNB` for fast and robust spam classification  
- Models are saved with `pickle` as `vectorizer.pkl` and `model.pkl` (loaded by the app).

> The notebook also experiments with additional models (SVM, RandomForest, ExtraTrees, Gradient Boosting, XGBoost) and ensembles (Voting, Stacking). The **VotingClassifier** achieved the best recorded metrics in the notebook (≈ **98.16% accuracy**, **99.17% precision** on the test split), while the **deployed app** keeps **MultinomialNB** for simplicity and speed.

---

## 📁 Project Structure
```
├── app.py                      # Streamlit UI: loads vectorizer.pkl & model.pkl
├── sms-spam-detection.ipynb    # Training & evaluation; saves pkl artifacts
├── spam.csv                    # SMS Spam Collection dataset (UCI)
├── vectorizer.pkl              # TF‑IDF vocabulary & weights (created by notebook)
├── model.pkl                   # Trained MultinomialNB classifier (created by notebook)
└── README.md
```

---

## ⚙️ Setup

### 1) Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate       # Windows PowerShell
```

### 2) Install dependencies
```bash
pip install streamlit scikit-learn nltk numpy pandas
# Optional for experiments in the notebook
pip install matplotlib seaborn xgboost
```

### 3) Download NLTK resources (first run only)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

> If your `model.pkl`/`vectorizer.pkl` were trained on an older scikit‑learn (e.g., 0.24.1) and you see warnings when loading them, either:  
> a) `pip install scikit-learn==0.24.1` **or** b) re‑run the notebook to retrain & resave with your current version.

---

## ▶️ Run the App
From the project root:
```bash
streamlit run app.py
```
Then open the URL that Streamlit prints (usually `http://localhost:8501`).

- Type/paste a message in the textbox and click **Predict**  
- Output: **“Spam, beware of the message”** or **“Not Spam”**

> The app supports an optional background image (`background.png`) via base64 CSS; remove or replace as needed.

---

## 📊 Reproducing Training
Open `sms-spam-detection.ipynb` and run all cells to:
- Clean & preprocess text with `NLTK`
- Fit `TfidfVectorizer(max_features=3000)`
- Train **MultinomialNB** and evaluate metrics
- Save artifacts:
  ```python
  import pickle
  pickle.dump(tfidf, open('vectorizer.pkl','wb'))
  pickle.dump(mnb,   open('model.pkl','wb'))
  ```

---

## 🧪 Dataset
- **Name:** SMS Spam Collection (UCI)  
- **Rows/Columns:** 5572 × 5  
- **Label distribution:** {'ham': 4825, 'spam': 747}  
- **Columns used:** `v1` = label (`spam`/`ham`), `v2` = message text (extra unnamed columns are dropped)

> Attribution: SMS Spam Collection Dataset, UCI Machine Learning Repository.

---

## 🧰 Troubleshooting
- **`UnicodeDecodeError` when reading CSV:** use `encoding="latin1"` or `encoding="cp1252"`  
- **`LookupError: stopwords/punkt`**: run the NLTK download snippet above  
- **Pickle warnings**: ensure scikit‑learn version matches the artifacts or retrain  
- **Streamlit doesn’t open**: always run with `streamlit run app.py` (not `python app.py`)

---

## 📄 License
MIT License (suggested). Add a `LICENSE` file or choose one via GitHub → *Add file → Create new file → “Choose a license template”*.

---

## 👤 Author
**Himanshu Gupta**  
GitHub: https://github.com/himanshugupta00235

---

⭐ If this project helps you, please **star** the repo!

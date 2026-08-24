SPAMGUARD AI - Email Spam Classifier
======================================

FILES:
- train_model.py       -> Full training pipeline (preprocessing, TF-IDF, model comparison)
- app.py                -> Streamlit web app (signup/login, persistent history, dark/light
                            mode, English/Urdu language toggle, notifications)
- spam_model.pkl        -> Saved best model (Linear SVM)
- tfidf_vectorizer.pkl  -> Saved TF-IDF vectorizer
- model_results.csv     -> Accuracy/Precision/Recall/F1 for all models tried
- confusion_matrix.png  -> Confusion matrix of best model
- model_comparison.png  -> Bar chart comparing all models
- spam.csv              -> Original training dataset
- app_data.json          -> Auto-created on first run; stores accounts and each user's
                            history/notifications/settings permanently between sessions

HOW TO RUN:
1. Install requirements:
   python -m pip install streamlit streamlit-option-menu scikit-learn pandas joblib matplotlib seaborn

2. Launch the app:
   python -m streamlit run app.py

3. Create an account (any details work), then sign in. A demo account also
   exists: username "demo", password "demo123".

APP FEATURES:
- Create Account + Sign In flow
- Sidebar navigation: Home, Classify Message, Model Performance, History, Notifications, How It Works, Settings
- History and notifications are saved to app_data.json, so they persist even
  after closing and reopening the app
- Home page shows live stats, recent activity feed, and spam-spotting tips
- Settings page: Light/Dark theme toggle, English/Urdu (native script) language toggle
- Model performance table ranked starting at 1

RESULTS SUMMARY:
Model                     Accuracy   Precision   Recall   F1-Score
Linear SVM (BEST)         98.07%     96.64%      87.79%   92.00%
Random Forest             97.20%     99.04%      78.63%   87.66%
Multinomial Naive Bayes   96.62%     98.98%      74.05%   84.72%
Logistic Regression       95.55%     97.75%      66.41%   79.09%
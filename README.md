# ⚡ Energy Consumption Prediction using LightGBM

This project predicts energy consumption based on various household features such as temperature, humidity, appliance usage, light usage, and occupancy. It uses LightGBM for model training and Streamlit for interactive visualization.

---

## 📂 Project Structure

```
energy-lightgbm-project/
├── data/
│   └── energy_data.csv                # Raw dataset
├── models/
│   ├── lightgbm_model.pkl             # Initial trained model
│   └── best_lightgbm_model.pkl        # Best model from hyperparameter tuning
├── output/
│   ├── cross_validation_results.txt   # RMSE scores from CV
│   └── grid_search_results.txt        # Best hyperparameters and scores
├── scripts/
│   ├── preprocess.py                  # Data loading & preprocessing
│   ├── train.py                       # LightGBM training pipeline
│   ├── evaluate.py                    # Streamlit UI for prediction
│   ├── cross_validate.py              # Cross-validation with LightGBM
│   └── tune_hyperparameters.py        # Grid search for LightGBM
├── notebooks/
│   └── energy_modeling.ipynb          # Jupyter notebook for experimentation
└── requirements.txt                   # Project dependencies
```

---

## 🚀 Features

- 📊 End-to-end ML pipeline from preprocessing to evaluation
- 🔍 Cross-validation with RMSE scoring
- 🧪 Grid search for hyperparameter tuning
- 🌐 Interactive prediction dashboard using Streamlit
- 🧠 Model explainability support via feature importance

---

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/energy-lightgbm-project.git
   cd energy-lightgbm-project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python scripts/train.py
   ```

5. **Run cross-validation**
   ```bash
   python scripts/cross_validate.py
   ```

6. **Tune hyperparameters**
   ```bash
   python scripts/tune_hyperparameters.py
   ```

7. **Launch Streamlit app**
   ```bash
   streamlit run scripts/evaluate.py
   ```

---

## 📈 Sample Output

- RMSE: `51.31`
- R² Score: `0.78`
- Cross-validation RMSE: `~2666.34`

---

## 📊 Input Features

- Temperature (°C)
- Humidity (%)
- Appliance Usage (kWh)
- Light Usage (kWh)
- Occupancy (count)

---

## ✅ TODOs

- Add SHAP-based explainability
- Add forecasting with time series methods
- Dockerize the app for deployment

---

## 🤝 Contribution

Feel free to fork the repo and submit pull requests. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Maintainer

Developed by [Your Name](https://www.linkedin.com/in/yourprofile)  
📧 Email: your.email@example.com
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Загрузка данных
supply = pd.read_csv('supply-3.csv')
X = supply[['free_sh_share', 'avg_eta_hours', 'city_area', 'hour', 'day_of_week', 'month'] + 
       [col for col in supply.columns if col.startswith('tariff_zone_')]]
y = supply['online']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Лучшие параметры из RandomizedSearch
best_params = {
    'bootstrap': False,
    'max_depth': 11,
    'max_features': 'log2',
    'min_samples_leaf': 1,
    'min_samples_split': 5,
    'n_estimators': 178,
    'random_state': 42,
    'n_jobs': -1
}

# Обучение модели
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, 'rf_courier_forecast_model.pkl')
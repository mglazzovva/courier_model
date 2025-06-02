import streamlit as st
import joblib
import numpy as np

# Загрузка модели
model = joblib.load("rf_courier_forecast_model.pkl")

# Названия признаков
feature_columns = model.feature_names_in_
tariff_zone_cols = [col for col in feature_columns if col.startswith("tariff_zone_")]
tariff_zones = [col.replace("tariff_zone_", "") for col in tariff_zone_cols]

st.title("🧮 Прогноз количества курьеров")

# Ввод признаков
free_sh_share = st.slider("Free SH share", 0.0, 1.0, 0.2, step=0.01)
ETA = st.slider("ETA", 5.0, 50.0, 1.0, step=0.1)

# 3-часовые окна
three_hour_windows = [(start, start+2) for start in range(0, 24, 3)]
window_strs = [f"{start}:00-{end}:59" for start, end in three_hour_windows]
window_idx = st.selectbox("Time interval", range(len(three_hour_windows)), format_func=lambda x: window_strs[x])
hour_range = range(three_hour_windows[window_idx][0], three_hour_windows[window_idx][1]+1)

# Селектор дня недели
day_map = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}
day_str = st.selectbox("Day of week", list(day_map.keys()))
day_of_week = day_map[day_str]

selected_zone = st.selectbox("Tariff_zone", tariff_zones)
city_area = st.selectbox("Zone area", [650, 400, 600])

# Формируем входные векторы для каждого часа в окне
def build_features(hour):
    features = {
        'free_sh_share': free_sh_share,
        'avg_eta_hours': (50/60 - ETA / 60) / 3,
        'city_area': city_area,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': 5
    }
    for full_colname in tariff_zone_cols:
        zone = full_colname.replace("tariff_zone_", "")
        features[full_colname] = 1 if zone == selected_zone else 0
    return np.array([features[col] for col in model.feature_names_in_])

# Прогноз для окна
if st.button("🔍 Predict"):
    preds = []
    for h in hour_range:
        input_vector = build_features(h).reshape(1, -1)
        pred = model.predict(input_vector)[0]
        preds.append(pred)
    mean_pred = np.mean(preds)
    st.success(f"📦 You need: **{round(mean_pred)}**")

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo 

def load_data():
    """Функция для автоматической загрузки данных"""
    local_path = 'data/ai4i2020.csv'
    
    
    if os.path.exists(local_path):
        st.success(f" Данные загружены из локального файла: {local_path}")
        return pd.read_csv(local_path)
    
    
    try:
        st.info(" Локальный файл не найден. Загружаю датасет из репозитория UCI...")
        ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601) 
        X = ai4i_2020_predictive_maintenance_dataset.data.features 
        y = ai4i_2020_predictive_maintenance_dataset.data.targets 
        df = pd.concat([X, y], axis=1)
        return df
    except Exception as e:
        st.error(f" Не удалось загрузить данные автоматически: {e}")
        return None

def analysis_and_model_page():
    st.title(" Анализ данных и модель")

    
    data = load_data()

    
    if data is None:
        uploaded_file = st.file_uploader("Или загрузите датасет (CSV) вручную", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)

    if data is not None:
        st.write("### Обзор данных", data.head())

       
        target_col = 'Machine failure' if 'Machine failure' in data.columns else 'Target'
        
        
        cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        existing_drops = [c for c in cols_to_drop if c in data.columns]
        data_clean = data.drop(columns=existing_drops)
        
        
        le = LabelEncoder()
        data_clean['Type'] = le.fit_transform(data_clean['Type'])
        
        
        X = data_clean.drop(columns=[target_col])
        y = data_clean[target_col]

        
        num_cols = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        
        st.header("Метрики модели")
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        col1.metric("Точность (Accuracy)", f"{accuracy_score(y_test, y_pred):.2%}")
        
        with col1:
            st.subheader("Матрица ошибок")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Отчет")
            st.text(classification_report(y_test, y_pred))

        
        st.divider()
        st.header(" Прогноз состояния оборудования")
        
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                in_type = st.selectbox("Тип (L/M/H)", ["L", "M", "H"])
                in_air = st.slider("Температура воздуха [K]", 295.0, 305.0, 300.0)
                in_proc = st.slider("Температура процесса [K]", 305.0, 315.0, 310.0)
            with c2:
                in_rpm = st.number_input("Обороты [rpm]", 1200, 2800, 1500)
                in_torque = st.number_input("Момент [Nm]", 3.0, 76.0, 40.0)
                in_tool = st.number_input("Износ инструмента [min]", 0, 250, 10)
            
            submit = st.form_submit_button("Выполнить диагностику")

        if submit:
            
            input_data = pd.DataFrame([[in_type, in_air, in_proc, in_rpm, in_torque, in_tool]], 
                                     columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 
                                              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
            # Обработка
            input_data['Type'] = le.transform(input_data['Type'])
            input_data[num_cols] = scaler.transform(input_data[num_cols])
            
            res = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            if res == 1:
                st.error(f"Высокий риск поломки! Вероятность: {prob:.1%}")
            else:
                st.success(f" Оборудование работает штатно. Риск поломки: {prob:.1%}")


if __name__ == "__main__":
    analysis_and_model_page()
#!/usr/bin/env python3
"""
Script para treinar e salvar o modelo XGBoost para predição de preços do Airbnb.
Baseado na análise dos notebooks 01_Analise, 02_EDA e 03_Modelagem.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Carrega os dados processados."""
    data_path = "data/processed/listings_model_baseline.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Dados carregados: {df.shape}")
    return df

def prepare_features(df):
    """Prepara as features para o modelo."""
    # Aplicar transformação logarítmica no preço
    df["price_log"] = np.log1p(df["price"])
    
    # Definir features
    TARGET = "price_log"
    
    features_num = [
        "accommodates", "bedrooms", "bathrooms", "beds",
        "number_of_reviews", "reviews_per_month", "review_scores_rating",
        "minimum_nights", "maximum_nights", "availability_365",
        "latitude", "longitude"
    ]
    
    features_cat = ["room_type", "property_type"]
    
    # One-hot encoding para variáveis categóricas
    df_encoded = pd.get_dummies(df[features_num + features_cat + [TARGET]], drop_first=True)
    
    X = df_encoded.drop(columns=[TARGET])
    y = df_encoded[TARGET]
    
    logger.info(f"Features preparadas: {X.shape[1]} features, {len(y)} amostras")
    return X, y, df_encoded.columns.tolist()

def train_model(X, y):
    """Treina o modelo XGBoost."""
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Configurar modelo XGBoost (parâmetros do notebook)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Treinar modelo
    logger.info("Iniciando treinamento do modelo...")
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Performance do modelo:")
    logger.info(f"  MAE: {mae:.3f}")
    logger.info(f"  RMSE: {rmse:.3f}")
    logger.info(f"  R²: {r2:.3f}")
    
    return model, X.columns.tolist()

def save_model_and_metadata(model, feature_names):
    """Salva o modelo e metadados necessários para predição."""
    # Criar diretório models se não existir
    os.makedirs("models", exist_ok=True)
    
    # Salvar modelo
    model_path = "models/xgboost_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Modelo salvo em: {model_path}")
    
    # Salvar nomes das features
    features_path = "models/feature_names.pkl"
    joblib.dump(feature_names, features_path)
    logger.info(f"Nomes das features salvos em: {features_path}")
    
    # Salvar metadados adicionais
    metadata = {
        "model_type": "XGBRegressor",
        "target_transformation": "log1p",
        "features_count": len(feature_names),
        "training_date": pd.Timestamp.now().isoformat()
    }
    
    metadata_path = "models/metadata.pkl"
    joblib.dump(metadata, metadata_path)
    logger.info(f"Metadados salvos em: {metadata_path}")

def main():
    """Função principal."""
    try:
        # Carregar dados
        df = load_data()
        
        # Preparar features
        X, y, all_columns = prepare_features(df)
        
        # Treinar modelo
        model, feature_names = train_model(X, y)
        
        # Salvar modelo e metadados
        save_model_and_metadata(model, feature_names)
        
        logger.info("Treinamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        raise

if __name__ == "__main__":
    main()

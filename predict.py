#!/usr/bin/env python3
"""
Script para fazer predições de preços do Airbnb usando o modelo XGBoost treinado.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Union

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AirbnbPricePredictor:
    """Classe para fazer predições de preços do Airbnb."""
    
    def __init__(self, model_path="models/xgboost_model.pkl", 
                 features_path="models/feature_names.pkl",
                 metadata_path="models/metadata.pkl"):
        """
        Inicializa o preditor carregando o modelo e metadados.
        
        Args:
            model_path: Caminho para o arquivo do modelo
            features_path: Caminho para o arquivo com nomes das features
            metadata_path: Caminho para o arquivo de metadados
        """
        self.model = None
        self.feature_names = None
        self.metadata = None
        
        self._load_model(model_path, features_path, metadata_path)
    
    def _load_model(self, model_path, features_path, metadata_path):
        """Carrega o modelo e metadados."""
        try:
            # Verificar se os arquivos existem
            for path in [model_path, features_path, metadata_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Arquivo não encontrado: {path}")
            
            # Carregar modelo
            self.model = joblib.load(model_path)
            logger.info(f"Modelo carregado de: {model_path}")
            
            # Carregar nomes das features
            self.feature_names = joblib.load(features_path)
            logger.info(f"Features carregadas: {len(self.feature_names)} features")
            
            # Carregar metadados
            self.metadata = joblib.load(metadata_path)
            logger.info(f"Metadados carregados: {self.metadata}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def _prepare_input_data(self, input_data: Dict) -> pd.DataFrame:
        """
        Prepara os dados de entrada para predição.
        
        Args:
            input_data: Dicionário com os dados de entrada
            
        Returns:
            DataFrame preparado para predição
        """
        # Criar DataFrame com os dados de entrada
        df = pd.DataFrame([input_data])
        
        # Aplicar one-hot encoding para variáveis categóricas
        # room_type
        if 'room_type' in df.columns:
            df = pd.get_dummies(df, columns=['room_type'], drop_first=True)
        
        # property_type  
        if 'property_type' in df.columns:
            df = pd.get_dummies(df, columns=['property_type'], drop_first=True)
        
        # Garantir que todas as features esperadas estejam presentes
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Valor padrão para features ausentes
        
        # Reordenar colunas para corresponder à ordem esperada pelo modelo
        df = df[self.feature_names]
        
        return df
    
    def predict_price(self, input_data: Dict) -> Dict:
        """
        Faz predição do preço baseado nos dados de entrada.
        
        Args:
            input_data: Dicionário com os dados de entrada contendo:
                - accommodates: número de hóspedes
                - bedrooms: número de quartos
                - bathrooms: número de banheiros
                - beds: número de camas
                - number_of_reviews: número de avaliações
                - reviews_per_month: avaliações por mês
                - review_scores_rating: nota média
                - minimum_nights: noites mínimas
                - maximum_nights: noites máximas
                - availability_365: disponibilidade no ano
                - latitude: latitude
                - longitude: longitude
                - room_type: tipo de quarto
                - property_type: tipo de propriedade
        
        Returns:
            Dicionário com predição e informações adicionais
        """
        try:
            # Preparar dados
            X = self._prepare_input_data(input_data)
            
            # Fazer predição (retorna log(price+1))
            price_log_pred = self.model.predict(X)[0]
            
            # Converter de volta para preço original: exp(price_log) - 1
            price_pred = np.expm1(price_log_pred)
            
            # Calcular intervalo de confiança aproximado (baseado no RMSE do treinamento)
            # RMSE ≈ 0.42 do notebook, convertido para escala original
            rmse_original = np.expm1(0.42)  # Aproximação
            confidence_interval = 1.96 * rmse_original  # 95% de confiança
            
            # Converter todos os valores numpy para tipos Python nativos para serialização JSON
            result = {
                "predicted_price": float(round(price_pred, 2)),
                "price_log": float(round(price_log_pred, 3)),
                "confidence_interval": {
                    "lower": float(round(max(0, price_pred - confidence_interval), 2)),
                    "upper": float(round(price_pred + confidence_interval, 2))
                },
                "model_info": {
                    "model_type": str(self.metadata.get("model_type", "XGBRegressor")),
                    "features_used": int(len(self.feature_names)),
                    "training_date": str(self.metadata.get("training_date", "N/A"))
                }
            }
            
            logger.info(f"Predição realizada: ${price_pred:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            raise
    
    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """
        Retorna a importância das features do modelo.
        
        Args:
            top_n: Número de features mais importantes para retornar
            
        Returns:
            Dicionário com features e suas importâncias
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = dict(zip(self.feature_names, importances))
                
                # Ordenar por importância e converter para tipos Python nativos
                sorted_features = sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)
                
                # Converter valores numpy para float Python
                top_features = [(name, float(importance)) for name, importance in sorted_features[:top_n]]
                
                return {
                    "top_features": top_features,
                    "total_features": int(len(self.feature_names))
                }
            else:
                return {"error": "Modelo não suporta feature importance"}
                
        except Exception as e:
            logger.error(f"Erro ao obter importância das features: {str(e)}")
            return {"error": str(e)}

def main():
    """Função principal para teste do preditor."""
    # Exemplo de uso
    predictor = AirbnbPricePredictor()
    
    # Dados de exemplo
    sample_data = {
        "accommodates": 4,
        "bedrooms": 2,
        "bathrooms": 1.0,
        "beds": 2,
        "number_of_reviews": 25,
        "reviews_per_month": 2.5,
        "review_scores_rating": 4.5,
        "minimum_nights": 1,
        "maximum_nights": 30,
        "availability_365": 200,
        "latitude": 40.7128,
        "longitude": -74.0060,
        "room_type": "Entire home/apt",
        "property_type": "Apartment"
    }
    
    # Fazer predição
    result = predictor.predict_price(sample_data)
    print("Resultado da predição:")
    print(f"Preço predito: ${result['predicted_price']}")
    print(f"Intervalo de confiança: ${result['confidence_interval']['lower']} - ${result['confidence_interval']['upper']}")
    
    # Mostrar importância das features
    importance = predictor.get_feature_importance()
    print("\nTop 5 features mais importantes:")
    for feature, imp in importance['top_features'][:5]:
        print(f"  {feature}: {imp:.3f}")

if __name__ == "__main__":
    main()

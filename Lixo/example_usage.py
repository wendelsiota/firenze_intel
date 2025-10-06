#!/usr/bin/env python3
"""
Exemplo de uso da API de predição de preços Airbnb.
Demonstra como usar a classe AirbnbPricePredictor e fazer requisições HTTP.
"""

import requests
import json
from predict import AirbnbPricePredictor

def example_direct_prediction():
    """Exemplo de predição direta usando a classe."""
    print("=== Exemplo de Predição Direta ===")
    
    try:
        # Inicializar preditor
        predictor = AirbnbPricePredictor()
        
        # Dados de exemplo - apartamento em Nova York
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
        
        print(f"Preço predito: ${result['predicted_price']}")
        print(f"Intervalo de confiança: ${result['confidence_interval']['lower']} - ${result['confidence_interval']['upper']}")
        print(f"Modelo: {result['model_info']['model_type']}")
        
        # Mostrar importância das features
        importance = predictor.get_feature_importance(top_n=5)
        print("\nTop 5 features mais importantes:")
        for feature, imp in importance['top_features']:
            print(f"  {feature}: {imp:.3f}")
            
    except Exception as e:
        print(f"Erro na predição direta: {e}")

def example_api_request():
    """Exemplo de requisição HTTP para a API."""
    print("\n=== Exemplo de Requisição HTTP ===")
    
    # URL da API (assumindo que está rodando localmente)
    api_url = "http://localhost:5000"
    
    # Dados de exemplo - quarto privado em São Paulo
    sample_data = {
        "accommodates": 2,
        "bedrooms": 1,
        "bathrooms": 1.0,
        "beds": 1,
        "number_of_reviews": 15,
        "reviews_per_month": 1.8,
        "review_scores_rating": 4.2,
        "minimum_nights": 2,
        "maximum_nights": 14,
        "availability_365": 150,
        "latitude": -23.5505,
        "longitude": -46.6333,
        "room_type": "Private room",
        "property_type": "House"
    }
    
    try:
        # Fazer requisição POST
        response = requests.post(f"{api_url}/predict", json=sample_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Preço predito: ${result['predicted_price']}")
            print(f"Intervalo de confiança: ${result['confidence_interval']['lower']} - ${result['confidence_interval']['upper']}")
        else:
            print(f"Erro na API: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Erro: Não foi possível conectar à API")
        print("Certifique-se de que a aplicação está rodando (python app.py)")
    except Exception as e:
        print(f"Erro na requisição HTTP: {e}")

def example_multiple_predictions():
    """Exemplo de múltiplas predições."""
    print("\n=== Exemplo de Múltiplas Predições ===")
    
    # Diferentes cenários
    scenarios = [
        {
            "name": "Apartamento de luxo em Manhattan",
            "data": {
                "accommodates": 6, "bedrooms": 3, "bathrooms": 2.0, "beds": 3,
                "number_of_reviews": 50, "reviews_per_month": 3.0, "review_scores_rating": 4.8,
                "minimum_nights": 3, "maximum_nights": 30, "availability_365": 180,
                "latitude": 40.7589, "longitude": -73.9851,
                "room_type": "Entire home/apt", "property_type": "Apartment"
            }
        },
        {
            "name": "Quarto compartilhado econômico",
            "data": {
                "accommodates": 1, "bedrooms": 0, "bathrooms": 0.5, "beds": 1,
                "number_of_reviews": 5, "reviews_per_month": 0.5, "review_scores_rating": 3.8,
                "minimum_nights": 1, "maximum_nights": 7, "availability_365": 300,
                "latitude": 40.7128, "longitude": -74.0060,
                "room_type": "Shared room", "property_type": "House"
            }
        },
        {
            "name": "Casa de família no subúrbio",
            "data": {
                "accommodates": 8, "bedrooms": 4, "bathrooms": 3.0, "beds": 5,
                "number_of_reviews": 30, "reviews_per_month": 2.0, "review_scores_rating": 4.5,
                "minimum_nights": 2, "maximum_nights": 14, "availability_365": 120,
                "latitude": 40.7505, "longitude": -73.9934,
                "room_type": "Entire home/apt", "property_type": "House"
            }
        }
    ]
    
    try:
        predictor = AirbnbPricePredictor()
        
        for scenario in scenarios:
            result = predictor.predict_price(scenario["data"])
            print(f"\n{scenario['name']}:")
            print(f"  Preço: ${result['predicted_price']}")
            print(f"  Intervalo: ${result['confidence_interval']['lower']} - ${result['confidence_interval']['upper']}")
            
    except Exception as e:
        print(f"Erro nas predições múltiplas: {e}")

def main():
    """Função principal."""
    print("🏠 Exemplos de Uso - Predição de Preços Airbnb")
    print("=" * 50)
    
    # Exemplo 1: Predição direta
    example_direct_prediction()
    
    # Exemplo 2: Requisição HTTP
    example_api_request()
    
    # Exemplo 3: Múltiplas predições
    example_multiple_predictions()
    
    print("\n" + "=" * 50)
    print("✅ Exemplos concluídos!")
    print("\nPara usar a interface web:")
    print("1. Execute: python run_app.py")
    print("2. Acesse: http://localhost:5000")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script para testar se a correção do erro de serialização JSON funcionou.
"""

import json
import sys
from predict import AirbnbPricePredictor

def test_json_serialization():
    """Testa se os dados são serializáveis para JSON."""
    print("🧪 Testando serialização JSON...")
    
    try:
        # Inicializar preditor
        predictor = AirbnbPricePredictor()
        print("✅ Preditor inicializado")
        
        # Dados de teste
        test_data = {
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
            "latitude": 43.7696,  # Florença
            "longitude": 11.2558,
            "room_type": "Entire home/apt",
            "property_type": "Apartment"
        }
        
        # Fazer predição
        result = predictor.predict_price(test_data)
        print("✅ Predição realizada")
        
        # Testar serialização JSON
        json_str = json.dumps(result, indent=2)
        print("✅ Serialização JSON bem-sucedida")
        
        # Mostrar resultado
        print("\n📊 RESULTADO DA PREDIÇÃO:")
        print(f"💰 Preço predito: €{result['predicted_price']}")
        print(f"📈 Intervalo: €{result['confidence_interval']['lower']} - €{result['confidence_interval']['upper']}")
        print(f"🏠 Modelo: {result['model_info']['model_type']}")
        
        # Testar feature importance
        importance = predictor.get_feature_importance(top_n=5)
        json_importance = json.dumps(importance, indent=2)
        print("✅ Feature importance serializada com sucesso")
        
        print("\n🎯 TOP 5 FEATURES MAIS IMPORTANTES:")
        for feature, imp in importance['top_features']:
            print(f"   {feature}: {imp:.3f}")
        
        print("\n✅ TODOS OS TESTES PASSARAM!")
        print("🚀 A aplicação deve funcionar sem erros de serialização JSON")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {str(e)}")
        print(f"🔍 Tipo do erro: {type(e).__name__}")
        return False

def main():
    """Função principal."""
    print("🔧 TESTE DE CORREÇÃO - Erro de Serialização JSON")
    print("=" * 60)
    
    success = test_json_serialization()
    
    if success:
        print("\n🎉 CORREÇÃO APLICADA COM SUCESSO!")
        print("💡 Agora você pode executar a aplicação sem erros:")
        print("   python run_app.py")
    else:
        print("\n❌ AINDA HÁ PROBLEMAS")
        print("💡 Verifique se o modelo foi treinado: python train_model.py")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

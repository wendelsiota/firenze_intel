#!/usr/bin/env python3
"""
Demonstração da aplicação de predição de preços Airbnb para Florença, Itália.
Este script mostra exemplos de uso com localizações reais em Florença.
"""

from predict import AirbnbPricePredictor
import json

def demo_florence_locations():
    """Demonstra predições para diferentes localizações em Florença."""
    
    print("🏛️  DEMONSTRAÇÃO - Predição de Preços Airbnb Florença, Itália")
    print("=" * 70)
    
    try:
        # Inicializar preditor
        predictor = AirbnbPricePredictor()
        print("✅ Modelo carregado com sucesso!")
        print()
        
        # Localizações famosas em Florença
        florence_locations = [
            {
                "name": "Centro Histórico (Duomo)",
                "description": "Próximo ao Duomo, coração de Florença",
                "data": {
                    "accommodates": 4,
                    "bedrooms": 2,
                    "bathrooms": 1.0,
                    "beds": 2,
                    "number_of_reviews": 45,
                    "reviews_per_month": 3.2,
                    "review_scores_rating": 4.7,
                    "minimum_nights": 2,
                    "maximum_nights": 14,
                    "availability_365": 180,
                    "latitude": 43.7731,  # Duomo
                    "longitude": 11.2560,
                    "room_type": "Entire home/apt",
                    "property_type": "Apartment"
                }
            },
            {
                "name": "Ponte Vecchio",
                "description": "Apartamento com vista para o Arno",
                "data": {
                    "accommodates": 2,
                    "bedrooms": 1,
                    "bathrooms": 1.0,
                    "beds": 1,
                    "number_of_reviews": 28,
                    "reviews_per_month": 2.1,
                    "review_scores_rating": 4.5,
                    "minimum_nights": 3,
                    "maximum_nights": 30,
                    "availability_365": 200,
                    "latitude": 43.7679,  # Ponte Vecchio
                    "longitude": 11.2529,
                    "room_type": "Entire home/apt",
                    "property_type": "Apartment"
                }
            },
            {
                "name": "Oltrarno (Bairro Artesão)",
                "description": "Casa tradicional no bairro artesão",
                "data": {
                    "accommodates": 6,
                    "bedrooms": 3,
                    "bathrooms": 2.0,
                    "beds": 4,
                    "number_of_reviews": 35,
                    "reviews_per_month": 2.8,
                    "review_scores_rating": 4.6,
                    "minimum_nights": 2,
                    "maximum_nights": 21,
                    "availability_365": 150,
                    "latitude": 43.7650,  # Oltrarno
                    "longitude": 11.2480,
                    "room_type": "Entire home/apt",
                    "property_type": "House"
                }
            },
            {
                "name": "Santa Maria Novella",
                "description": "Quarto privado próximo à estação",
                "data": {
                    "accommodates": 2,
                    "bedrooms": 1,
                    "bathrooms": 1.0,
                    "beds": 1,
                    "number_of_reviews": 15,
                    "reviews_per_month": 1.5,
                    "review_scores_rating": 4.2,
                    "minimum_nights": 1,
                    "maximum_nights": 7,
                    "availability_365": 250,
                    "latitude": 43.7744,  # Santa Maria Novella
                    "longitude": 11.2494,
                    "room_type": "Private room",
                    "property_type": "House"
                }
            },
            {
                "name": "Piazzale Michelangelo",
                "description": "Villa com vista panorâmica da cidade",
                "data": {
                    "accommodates": 8,
                    "bedrooms": 4,
                    "bathrooms": 3.0,
                    "beds": 6,
                    "number_of_reviews": 22,
                    "reviews_per_month": 1.8,
                    "review_scores_rating": 4.8,
                    "minimum_nights": 4,
                    "maximum_nights": 30,
                    "availability_365": 120,
                    "latitude": 43.7629,  # Piazzale Michelangelo
                    "longitude": 11.2654,
                    "room_type": "Entire home/apt",
                    "property_type": "Villa"
                }
            }
        ]
        
        print("📍 LOCALIZAÇÕES EM FLORENÇA:")
        print("-" * 50)
        
        for i, location in enumerate(florence_locations, 1):
            print(f"\n{i}. {location['name']}")
            print(f"   {location['description']}")
            print(f"   📍 Coordenadas: {location['data']['latitude']:.4f}, {location['data']['longitude']:.4f}")
            
            # Fazer predição
            result = predictor.predict_price(location['data'])
            
            print(f"   💰 Preço predito: €{result['predicted_price']:.2f}/noite")
            print(f"   📊 Intervalo: €{result['confidence_interval']['lower']:.2f} - €{result['confidence_interval']['upper']:.2f}")
            print(f"   🏠 Tipo: {location['data']['room_type']} | {location['data']['property_type']}")
            print(f"   👥 Capacidade: {location['data']['accommodates']} hóspedes")
        
        print("\n" + "=" * 70)
        print("🎯 ANÁLISE COMPARATIVA:")
        print("-" * 30)
        
        # Análise dos resultados
        results = []
        for location in florence_locations:
            result = predictor.predict_price(location['data'])
            results.append({
                'name': location['name'],
                'price': result['predicted_price'],
                'type': location['data']['room_type'],
                'capacity': location['data']['accommodates']
            })
        
        # Ordenar por preço
        results.sort(key=lambda x: x['price'])
        
        print("💰 Preços ordenados (menor para maior):")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['name']}: €{result['price']:.2f} ({result['type']}, {result['capacity']} pessoas)")
        
        # Estatísticas
        prices = [r['price'] for r in results]
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        
        print(f"\n📈 ESTATÍSTICAS:")
        print(f"   💵 Preço médio: €{avg_price:.2f}")
        print(f"   💸 Menor preço: €{min_price:.2f}")
        print(f"   💎 Maior preço: €{max_price:.2f}")
        print(f"   📊 Variação: €{max_price - min_price:.2f}")
        
        print(f"\n🏆 RECOMENDAÇÕES:")
        print(f"   💰 Melhor custo-benefício: {results[0]['name']} (€{results[0]['price']:.2f})")
        print(f"   🌟 Mais luxuoso: {results[-1]['name']} (€{results[-1]['price']:.2f})")
        
    except Exception as e:
        print(f"❌ Erro na demonstração: {str(e)}")
        print("💡 Certifique-se de que o modelo foi treinado: python train_model.py")

def demo_map_features():
    """Demonstra as funcionalidades do mapa interativo."""
    print("\n🗺️  FUNCIONALIDADES DO MAPA INTERATIVO:")
    print("-" * 50)
    print("✅ Mapa centrado em Florença (Duomo)")
    print("✅ Marcador arrastável para seleção precisa")
    print("✅ Clique no mapa para alterar localização")
    print("✅ Pontos de interesse marcados:")
    print("   🏛️  Duomo (Catedral)")
    print("   🌉 Ponte Vecchio")
    print("   🏢 Palazzo Vecchio")
    print("   🎨 Uffizi (Museu)")
    print("   ⛪ Santa Maria Novella")
    print("✅ Coordenadas atualizadas automaticamente")
    print("✅ Zoom e navegação intuitivos")

def main():
    """Função principal da demonstração."""
    demo_florence_locations()
    demo_map_features()
    
    print("\n" + "=" * 70)
    print("🚀 PARA USAR A APLICAÇÃO WEB:")
    print("   1. Execute: python run_app.py")
    print("   2. Acesse: http://localhost:5000")
    print("   3. Use o mapa interativo para selecionar sua localização")
    print("   4. Preencha as características da propriedade")
    print("   5. Obtenha a predição de preço em euros!")
    print("\n🇮🇹 Buona fortuna con la tua proprietà a Firenze!")

if __name__ == "__main__":
    main()

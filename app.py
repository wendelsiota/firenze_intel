#!/usr/bin/env python3
"""
Aplicação web Flask para predição de preços do Airbnb.
Interface web para consumir o modelo XGBoost treinado.
"""

from flask import Flask, render_template, request, jsonify
import os
import logging
import numpy as np
from predict import AirbnbPricePredictor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar Flask app
app = Flask(__name__)

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Inicializar preditor
try:
    predictor = AirbnbPricePredictor()
    logger.info("Preditor inicializado com sucesso")
except Exception as e:
    logger.error(f"Erro ao inicializar preditor: {str(e)}")
    predictor = None

@app.route('/')
def index():
    """Página principal com formulário de predição."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para fazer predições via API."""
    try:
        if predictor is None:
            return jsonify({"error": "Modelo não disponível"}), 500
        
        # Obter dados do formulário
        data = request.get_json() if request.is_json else request.form
        
        # Validar dados obrigatórios
        required_fields = [
            'accommodates', 'bedrooms', 'bathrooms', 'beds',
            'number_of_reviews', 'reviews_per_month', 'review_scores_rating',
            'minimum_nights', 'maximum_nights', 'availability_365',
            'latitude', 'longitude', 'room_type', 'property_type'
        ]
        
        # Converter dados para formato esperado
        input_data = {}
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo obrigatório ausente: {field}"}), 400
            
            # Converter para tipos apropriados
            if field in ['accommodates', 'bedrooms', 'beds', 'number_of_reviews', 
                        'minimum_nights', 'maximum_nights', 'availability_365']:
                input_data[field] = int(data[field])
            elif field in ['bathrooms', 'reviews_per_month', 'review_scores_rating', 
                          'latitude', 'longitude']:
                input_data[field] = float(data[field])
            else:
                input_data[field] = str(data[field])
        
        # Fazer predição
        result = predictor.predict_price(input_data)
        
        # Converter tipos numpy para tipos Python nativos
        result = convert_numpy_types(result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

@app.route('/model_info')
def model_info():
    """Endpoint para informações sobre o modelo."""
    try:
        if predictor is None:
            return jsonify({"error": "Modelo não disponível"}), 500
        
        # Obter informações do modelo
        importance = predictor.get_feature_importance(top_n=10)
        
        result = {
            "model_info": predictor.metadata,
            "feature_importance": importance,
            "total_features": len(predictor.feature_names)
        }
        
        # Converter tipos numpy para tipos Python nativos
        result = convert_numpy_types(result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

@app.route('/health')
def health():
    """Endpoint de health check."""
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor is not None
    })

if __name__ == '__main__':
    # Criar diretório templates se não existir
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

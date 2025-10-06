"""
Arquivo de configuração de exemplo para a aplicação de predição Airbnb.
Copie este arquivo para config.py e ajuste os valores conforme necessário.
"""

import os

class Config:
    """Configurações base da aplicação."""
    
    # Configurações do Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Configurações do modelo
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/xgboost_model.pkl')
    FEATURES_PATH = os.environ.get('FEATURES_PATH', 'models/feature_names.pkl')
    METADATA_PATH = os.environ.get('METADATA_PATH', 'models/metadata.pkl')
    
    # Configurações de dados
    DATA_PATH = os.environ.get('DATA_PATH', 'data/processed/listings_model_baseline.csv')
    
    # Configurações de logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

class DevelopmentConfig(Config):
    """Configurações para desenvolvimento."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Configurações para produção."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key-must-be-set'

class TestingConfig(Config):
    """Configurações para testes."""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Mapeamento de configurações
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

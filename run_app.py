#!/usr/bin/env python3
"""
Script para executar a aplicação web de predição de preços Airbnb.
Verifica se o modelo está treinado e inicia a aplicação.
"""

import os
import sys
import subprocess
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Verifica se as dependências estão instaladas."""
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        import xgboost
        import joblib
        logger.info("Todas as dependências estão instaladas")
        return True
    except ImportError as e:
        logger.error(f"Dependência não encontrada: {e}")
        logger.error("Execute: pip install -r requirements.txt")
        return False

def check_model():
    """Verifica se o modelo está treinado."""
    model_files = [
        "models/xgboost_model.pkl",
        "models/feature_names.pkl", 
        "models/metadata.pkl"
    ]
    
    for file in model_files:
        if not os.path.exists(file):
            logger.error(f"Modelo não encontrado: {file}")
            logger.error("Execute primeiro: python train_model.py")
            return False
    
    logger.info("Modelo encontrado e pronto para uso")
    return True

def check_data():
    """Verifica se os dados estão disponíveis."""
    data_file = "data/processed/listings_model_baseline.csv"
    if not os.path.exists(data_file):
        logger.error(f"Dados não encontrados: {data_file}")
        logger.error("Certifique-se de que os dados processados estão disponíveis")
        return False
    
    logger.info("Dados encontrados")
    return True

def main():
    """Função principal."""
    logger.info("=== Verificando aplicação de predição Airbnb ===")
    
    # Verificar dependências
    if not check_requirements():
        sys.exit(1)
    
    # Verificar dados
    if not check_data():
        logger.warning("Dados não encontrados, mas continuando...")
    
    # Verificar modelo
    if not check_model():
        logger.info("Modelo não encontrado. Deseja treinar agora? (s/n)")
        response = input().lower().strip()
        
        if response in ['s', 'sim', 'y', 'yes']:
            logger.info("Iniciando treinamento do modelo...")
            try:
                subprocess.run([sys.executable, "train_model.py"], check=True)
                logger.info("Treinamento concluído com sucesso!")
            except subprocess.CalledProcessError as e:
                logger.error(f"Erro durante o treinamento: {e}")
                sys.exit(1)
        else:
            logger.info("Execute 'python train_model.py' para treinar o modelo")
            sys.exit(1)
    
    # Iniciar aplicação
    logger.info("Iniciando aplicação web...")
    logger.info("Acesse: http://localhost:5000")
    logger.info("Pressione Ctrl+C para parar")
    
    try:
        from app import app
    except ImportError:
        logger.error("Erro ao importar app.py")
        sys.exit(1)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Aplicação encerrada pelo usuário")
    except Exception as e:
        logger.error(f"Erro ao executar aplicação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

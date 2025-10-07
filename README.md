# 🏠 Predição de Preços Airbnb - Aplicação Web

Esta aplicação web utiliza um modelo de machine learning (XGBoost) para prever preços de propriedades no Airbnb baseado em características específicas. O modelo foi desenvolvido através de análise exploratória de dados e modelagem em notebooks Jupyter.

## 📊 Sobre o Modelo

- **Algoritmo**: XGBoost Regressor
- **Performance**: RMSE = 0.42, R² = 0.54
- **Target**: Preço transformado logaritmicamente (log(price+1))
- **Features**: 12 numéricas + 2 categóricas (após one-hot encoding)

### Features Utilizadas:
- **Características básicas**: accommodates, bedrooms, bathrooms, beds
- **Localização**: latitude, longitude
- **Avaliações**: number_of_reviews, reviews_per_month, review_scores_rating
- **Disponibilidade**: minimum_nights, maximum_nights, availability_365
- **Tipo**: room_type, property_type

## 🚀 Como Usar

### 0. Download e Ambiente
Clone o repositorio e ative um ambiente virtual em sua estação de trabalho

```bash
git clone https://github.com/wendelsiota/firenze_intel.git
cd firenze_inte
python3 -m venv .venv
source .venv/bin/activate
```


### 1. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o Modelo

Primeiro, você precisa treinar o modelo com os dados:

```bash
python train_model.py
```

Este comando irá:
- Carregar os dados de `data/processed/listings_model_baseline.csv`
- Treinar o modelo XGBoost
- Salvar o modelo em `models/xgboost_model.pkl`
- Salvar metadados em `models/`

### 3. Executar a Aplicação Web

**Opção 1 - Script Automático (Recomendado):**
```bash
python run_app.py
```

Este script verifica automaticamente:
- Se as dependências estão instaladas
- Se o modelo está treinado
- Se os dados estão disponíveis
- Oferece treinar o modelo se necessário

**Opção 2 - Execução Direta:**
```bash
python app.py
```

A aplicação estará disponível em: `http://localhost:5000`

### 4. Usar a Interface Web

1. Acesse `http://localhost:5000` no seu navegador
2. Preencha o formulário com as características da propriedade
3. Clique em "Prever Preço" para obter a predição
4. Visualize o preço predito e o intervalo de confiança

## 📁 Estrutura do Projeto

```
firenze_intel/
├── notebooks/                 # Notebooks de análise exploratória
│   ├── 01_Analise.ipynb      # Análise inicial dos dados
│   ├── 02_EDA.ipynb          # Análise exploratória detalhada
│   └── 03_Modelagem.ipynb    # Desenvolvimento e validação do modelo
├── data/
│   ├── raw/                  # Dados originais (não versionados)
│   └── processed/
│       └── listings_model_baseline.csv  # Dados processados para treinamento
├── models/                   # Modelos treinados e metadados
│   ├── xgboost_model.pkl     # Modelo XGBoost treinado
│   ├── feature_names.pkl     # Nomes das features utilizadas
│   └── metadata.pkl          # Metadados do modelo (tipo, data, etc.)
├── templates/
│   └── index.html            # Interface web HTML
├── train_model.py            # Script para treinar o modelo
├── predict.py               # Classe AirbnbPricePredictor
├── app.py                   # Aplicação Flask principal
├── run_app.py              # Script auxiliar com verificações automáticas
├── requirements.txt         # Dependências Python
└── README.md               # Esta documentação
```

## 📋 Descrição dos Arquivos

### Scripts Principais

#### `app.py`
Aplicação Flask principal que fornece a interface web e API REST para predição de preços.
- **Funcionalidades**: Interface web, endpoints de API, validação de dados
- **Endpoints**: `/`, `/predict`, `/model_info`, `/health`
- **Dependências**: Flask, numpy, predict.py

#### `run_app.py`
Script auxiliar que verifica automaticamente o ambiente antes de iniciar a aplicação.
- **Verificações**: Dependências instaladas, modelo treinado, dados disponíveis
- **Funcionalidades**: Treinamento automático do modelo se necessário
- **Uso recomendado**: Script principal para iniciar a aplicação

#### `train_model.py`
Script para treinar e salvar o modelo XGBoost baseado nos dados processados.
- **Entrada**: `data/processed/listings_model_baseline.csv`
- **Saída**: Modelo e metadados em `models/`
- **Parâmetros**: Configuração otimizada do XGBoost

#### `predict.py`
Classe `AirbnbPricePredictor` para fazer predições usando o modelo treinado.
- **Métodos**: `predict_price()`, `get_feature_importance()`
- **Funcionalidades**: Preparação de dados, one-hot encoding, conversão de tipos

### Dados e Modelos

#### `data/processed/listings_model_baseline.csv`
Dataset processado contendo as features necessárias para treinamento.
- **Formato**: CSV com features numéricas e categóricas
- **Target**: Preço original (será transformado em log durante treinamento)

#### `models/`
Diretório contendo o modelo treinado e metadados:
- **`xgboost_model.pkl`**: Modelo XGBoost serializado
- **`feature_names.pkl`**: Lista com nomes das features na ordem correta
- **`metadata.pkl`**: Informações sobre o modelo (tipo, data de treinamento, etc.)

### Interface Web

#### `templates/index.html`
Interface HTML para interação com o usuário.
- **Tecnologias**: HTML5, Bootstrap 5, JavaScript
- **Funcionalidades**: Formulário de entrada, visualização de resultados

### Notebooks de Análise

#### `notebooks/01_Analise.ipynb`
Análise inicial dos dados e limpeza.
- **Conteúdo**: Carregamento, exploração básica, tratamento de outliers

#### `notebooks/02_EDA.ipynb`
Análise exploratória de dados detalhada.
- **Conteúdo**: Visualizações, correlações, distribuições

#### `notebooks/03_Modelagem.ipynb`
Desenvolvimento, treinamento e validação do modelo.
- **Conteúdo**: Feature engineering, comparação de algoritmos, otimização

### Configuração

#### `requirements.txt`
Lista de dependências Python necessárias para o projeto.
- **Principais**: Flask, pandas, numpy, scikit-learn, xgboost, joblib
- **Versões**: Especificadas para compatibilidade

## 🔧 API Endpoints

### POST `/predict`
Faz predição de preço baseado nas características fornecidas.

**Parâmetros obrigatórios:**
```json
{
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
```

**Resposta:**
```json
{
    "predicted_price": 150.25,
    "price_log": 5.012,
    "confidence_interval": {
        "lower": 120.50,
        "upper": 180.00
    },
    "model_info": {
        "model_type": "XGBRegressor",
        "features_used": 15,
        "training_date": "2024-01-15T10:30:00"
    }
}
```

### GET `/model_info`
Retorna informações sobre o modelo e importância das features.

### GET `/health`
Health check da aplicação.

## 🧪 Testando o Modelo

Você pode testar o modelo diretamente via Python:

```python
from predict import AirbnbPricePredictor

# Inicializar preditor
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
print(f"Preço predito: ${result['predicted_price']}")
```

## 📈 Performance do Modelo

O modelo XGBoost foi escolhido por apresentar a melhor performance:

| Métrica | Random Forest | XGBoost |
|---------|---------------|---------|
| MAE     | 0.29          | 0.29    |
| RMSE    | 0.43          | 0.42    |
| R²      | 0.52          | 0.54    |

## 🎯 Características da Interface Web

- **Design responsivo** com Bootstrap 5
- **Formulário intuitivo** com validação
- **Visualização clara** dos resultados
- **Intervalo de confiança** para as predições
- **Loading states** para melhor UX

## 🔍 Análise dos Dados

O modelo foi desenvolvido baseado em análise exploratória que revelou:

- **Distribuição assimétrica** dos preços (cauda longa à direita)
- **Correlação forte** entre capacidade e preço
- **Impacto significativo** do tipo de quarto no preço
- **Necessidade de transformação logarítmica** para melhor ajuste


## 📝 Notas Importantes

- O modelo foi treinado com dados específicos e pode precisar de retreinamento com novos dados
- As predições são baseadas em padrões históricos e podem não refletir condições de mercado atuais
- O intervalo de confiança é uma aproximação baseada no RMSE do treinamento
- Use `python run_app.py` para uma experiência mais robusta com verificações automáticas
- Certifique-se de que os dados estão em `data/processed/listings_model_baseline.csv` antes do treinamento

## 🛠️ Solução de Problemas

### Modelo não encontrado
Se receber erro "Modelo não disponível":
1. Execute `python train_model.py` para treinar o modelo
2. Ou use `python run_app.py` que oferece treinamento automático

### Dados não encontrados
Se receber erro sobre dados ausentes:
1. Verifique se `data/processed/listings_model_baseline.csv` existe
2. Certifique-se de que o arquivo contém as colunas necessárias

### Dependências ausentes
Se houver erros de importação:
1. Execute `pip install -r requirements.txt`
2. Use um ambiente virtual Python recomendado

## 🧪 Testando o Modelo

### Via API
Use ferramentas como Insomnia ou Postman para testar os endpoints:
- **POST** `/predict` - Fazer predições
- **GET** `/model_info` - Informações do modelo
- **GET** `/health` - Status da aplicação

### Via Python
```python
from predict import AirbnbPricePredictor

# Inicializar preditor
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
print(f"Preço predito: ${result['predicted_price']}")
```

## 📄 Licença

Este projeto é para fins educacionais e de demonstração.

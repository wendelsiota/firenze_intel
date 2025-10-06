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
├── notebooks/                 # Notebooks de análise
│   ├── 01_Analise.ipynb
│   ├── 02_EDA.ipynb
│   └── 03_Modelagem.ipynb
├── data/
│   └── processed/
│       └── listings_model_baseline.csv
├── models/                    # Modelos treinados (criado após treinamento)
│   ├── xgboost_model.pkl
│   ├── feature_names.pkl
│   └── metadata.pkl
├── templates/
│   └── index.html            # Interface web
├── train_model.py            # Script de treinamento
├── predict.py               # Classe de predição
├── app.py                   # Aplicação Flask
├── requirements.txt         # Dependências
└── README.md               # Esta documentação
```

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
- Para melhor performance, considere retreinar o modelo periodicamente

## 🤝 Contribuição

Para contribuir com o projeto:

1. Faça fork do repositório
2. Crie uma branch para sua feature
3. Implemente suas mudanças
4. Teste adequadamente
5. Submeta um pull request

## 📄 Licença

Este projeto é para fins educacionais e de demonstração.

# ozon-hack-2023
Поиск одинаковых товаров на маркетплейсе

* [CatBoost_model](golden_boost.ipynb) — обучение градиентного бустинга на исходных данных даёт 0.42 на LB. (64 vcpu, 128 ram)
* [Make_embeds](make_embeds.py) — получение векторов от предобученного LaBSE.
* [LaBSE_finetune](LaBSE.ipynb) — дообучение LaBSE. (V100)
* [MultiModal](MultiModal+LaBSE+norm.ipynb) — обучение мультимодальной сети с Contrastive loss. (V100)
* [Ensemble](ensemble.ipynb) — обучение ансамбля моделей. (64 vcpu, 128 ram)
* [Test_inference](inference.ipynb) — полный инференс для тестовых данных. 
* Дополнительные данные и веса: [ozon_models](https://drive.google.com/drive/folders/1P0UPs-qN1H0OZXaKU5e4I4AtrNZqbbEO?usp=sharing) (а ещё уже посчитанные ембеддинги нашими сетями)

В репозитории также есть промежуточные нотбуки разработанные на протяжении соревнования.

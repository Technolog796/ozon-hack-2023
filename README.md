# ozon-hack-2023
Поиск одинаковых товаров на маркетплейсе

* [pairs_neuro](pairs_neuro.ipynb) - Contrastive loss baseline, AUC: 0.878 -> 0.879 (extra positive pairs)
* [EDA](EDA.ipynb) - Catboost baseline, AUC: 0.905

color embeds + biLstm (reverse+forward), dists AUC: 0.869 -> 0.875:
* [pairs_neuro_with_color](experiments/pairs_neuro_with_color.ipynb) - padding, cb AUC: 0.879791577700001
* [pairs_neuro_with_color_pack](experiments/pairs_neuro_with_color_pack.ipynb) - padding&packing, cb AUC: 0.8797019040150235

# ozon-hack-2023
Поиск одинаковых товаров на маркетплейсе

* [train_val_split](train_val_split.ipynb) - разделение на train/val с уменьшением числа общих товаров
* [pairs_neuro](pairs_neuro.ipynb) - Contrastive loss baseline, cb AUC: 0.878 -> 0.879 (extra positive pairs), dists AUC: 0.869 -> 0.870
* [EDA](EDA.ipynb) - Catboost baseline, cb AUC: 0.905
* [base_cb_model](base_cb_model.ipynb) - base Catboost mode, PRAUC: 0.894

color embeds + biLstm (reverse+forward), dists AUC: 0.869 -> 0.875:
* [pairs_neuro_with_color](experiments/pairs_neuro_with_color.ipynb) - padding, cb AUC: 0.879791577700001
* [pairs_neuro_with_color_pack](experiments/pairs_neuro_with_color_pack.ipynb) - padding&packing, cb AUC: 0.8797019040150235

Experiments:
* [pairs_siamneuro](experiments/pairs_siamneuro.ipynb) - siamese net with BCE loss
* [pairs_siamneuro_new_loss](experiments/pairs_siamneuro_new_loss.ipynb) - siamese net with Radmir's loss

Сердечно-сосудистые заболевания (ССЗ) являются самой частой причиной смерти по всему земному шару. 
Наиболее эффективной стратегией борьбы с ССЗ является заблаговременное выявление проблем (диагностика) и своевременное начало лечения.

Задача: обучить модель машинного обучения, которая по табличным данным предсказывает вероятность сердечных заболеваний, проходящих регулярную диспансеризацию.

Показателем оценки является показатель ROC AUC.

Источник данных: https://www.kaggle.com/competitions/yap15-heart-diseases-predictions/data

Содержание:

1. Предобработка и исследовательский анализ данных
2. Разработка модели ML
3. Тестирование модели

Результат:

Модель случайного леса показала лучшее значение ROC AUC, поэтому тестирование проводилось на этой модели.

Используемые библиотеки

pandas
numpy
matplotlib
seaborn
scikit-learn
pickle
csv
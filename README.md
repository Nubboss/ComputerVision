# Прототип для тестового задания Описание тестового задания в test.md

## Установить зависимости:

```bash
pip install -r requirements.txt
```
## Запустить:

```bash
python main.py --video video1.mp4
```
## Выход:

    output.mp4 — видео с ROI и детекциями
    report.txt — средние задержки и хронология событий

## Выбор видео и столика
    Видео: передаёте через --video (пример: video1.mp4)
    Столик: выбирается вручную в первом кадре через интерактивный cv2.selectROI

## Логика детекции

    Модель: Ultralytics YOLO (yolov8n) — учитываются только объекты класса person
    ROI: проверка пересечения бокса человека с выбранным ROI (AABB)
    Состояния:
        table_empty — нет людей в ROI
        table_occupied — есть ≥1 человек в ROI
        approach — переход empty → occupied, если пустота длилась ≥ 2.0 с
    Дебаунс смены состояния: 0.5 с
    События записываются в pandas.DataFrame с временными метками

## Результат (пример)
mean_empty_to_occupied:  
mean_empty_to_approach:  

## Проблемные кадры

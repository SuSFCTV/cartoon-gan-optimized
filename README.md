# CartoonGAN: ускорение на CPU (INT8) и перенос на Android

Учебный проект по дисциплине **«Эффективные ИИ-модели»**.

**Цели работы:**

- Ускорить генератор **CartoonGAN** на CPU с помощью **INT8-квантования**.
- Сравнить производительность:
  - FP32 на GPU (RTX 4070) и CPU;
  - INT8 (частичное PTQ) на CPU.
- Перенести модель на **Android** с помощью **TorchScript (FP32)** и продемонстрировать работу на мобильном устройстве / эмуляторе.
- Пояснить, почему:
  - не удалось полноценно заквантовать всю модель в INT8;
  - не удалось перенести INT8-модель на мобилку.

Базовая модель:  
[FilipAndersson245/cartoon-gan](https://github.com/FilipAndersson245/cartoon-gan)


---

## 1. Установка и подготовка окружения

### 1.1. Python-окружение (Windows 11, RTX 4070)

1. Создать и активировать виртуальное окружение:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Обновить `pip`:

```bash
pip install --upgrade pip
```

3. Установить зависимости:

```bash
pip install torch torchvision
pip install opencv-python Pillow numpy tqdm
```

### 2.2. Веса генератора

1. Скачать `trained_netG.pth` из репозитория [cartoon-gan](https://github.com/FilipAndersson245/cartoon-gan).
2. Положить файл в папку `checkpoints/`:

```text
checkpoints/trained_netG.pth
```

---

## 3. FP32 инференс и бенчмарки

### 3.1. Генерация изображения (FP32 GPU / CPU)

**GPU FP32:**

```bash
python -m src.fp32_inference \
  -i data/test/2.png \
  -o artifacts/output_fp32_gpu.jpg \
  -d cuda
```

**CPU FP32:**

```bash
python -m src.fp32_inference \
  -i data/test/2.png \
  -o artifacts/output_fp32_cpu.jpg \
  -d cpu
```

Оба скрипта сохраняют cartoon-версию входного изображения (256×256) и выводят время инференса.

### 3.2. Бенчмаркинг FP32

**Полученные результаты:**

```text
[FP32 Benchmark] device=cuda, img_size=256, runs=50, warmup=10
[FP32 cuda] mean=7.29 ms, p50=7.14 ms, p90=7.71 ms

[FP32 Benchmark] device=cpu, img_size=256, runs=50, warmup=10
[FP32 cpu] mean=75.24 ms, p50=73.37 ms, p90=83.87 ms
```

---

## 4. INT8 Post-Training Quantization (PTQ) на CPU

### 4.1. Калибровочный набор

В `data/calib/` нужно положить набор изображений (десятки/сотни), похожих на реальные входы (фото, которые затем будут «очеловечены» в мультяшный стиль).

Пример:

```text
data/calib/01.jpg
data/calib/02.jpg
...
data/calib/200.jpg
```


### 4.2. Квантование и инференс (частичная INT8-модель)

Запуск:

```bash
python -m src.int8_ptq_inference \
  --calib_dir data/calib \
  -i data/test/2.png \
  -o artifacts/output_int8_cpu.jpg
```

Что делает `int8_ptq_inference.py`:

1. Создаёт обёртку над `Generator` с `QuantStub/DeQuantStub`.
2. Квантует **только энкодер (`gen.down`)**:

   * свёрточные слои → INT8;
   * сложные блоки (res-блоки, decoder) оставляет в FP32.
3. Проводит калибровку на изображениях из `data/calib/`.
4. Конвертирует модель в гибридный INT8/FP32-вариант.
5. Выполняет инференс на CPU и сохраняет `artifacts/output_int8_cpu.jpg`.

**Результат по времени (INT8 на CPU):**

```text
[INT8] Inference time (CPU INT8) = 66.7 ms
```

### 4.3. Сравнение производительности

| Режим      | Устройство | Тип модели              | Средняя латентность (ms) | p50 (ms) | p90 (ms) | Размер весов |
| ---------- | ---------- | ----------------------- | ------------------------ | -------- | -------- |--------------|
| FP32       | GPU        | Полный FP32             | 7.29                     | 7.14     | 7.71     | 43МБ         |
| FP32       | CPU        | Полный FP32             | 75.24                    | 73.37    | 83.87    | 43МБ         |
| INT8 (PTQ) | CPU        | Частично INT8 (encoder) | **66.7**                 | —        | —        | 30МБ         |

**Вывод:**

* Ускорение **CPU INT8 vs CPU FP32** ≈ **1.13×** (около 12 % уменьшения латентности).

Визуально `output_int8_cpu.jpg` очень близок к `output_fp32_cpu.jpg`.

---

## 5. Сохранение и проверка INT8-весов

### 5.1. Сохранение INT8-весов и параметров

После успешной калибровки можно сохранить INT8-веса и параметры квантизации:

```bash
python -m src.dump_int8_weights
```

Скрипт создаёт:

* `artifacts/cartoon_gan_int8_state_dict.pth` — INT8-веса (state_dict);
* `artifacts/cartoon_gan_int8_quant_params.pth` — параметры квантизации (scale, zero_point и др.).


**Итог:**
INT8-веса корректно восстанавливаются и дают тот же результат, что и модель сразу после PTQ.

---

## 6. Экспорт FP32-модели в TorchScript для Android

### 6.1. Экспорт (только CPU)

```bash
python -m src.export_mobile_fp32
```

Скрипт перевод модель в TorchScript:

```text
artifacts/cartoon_gan_mobile_fp32_cpu.pt
```

### 6.2. Проверка TorchScript FP32 на десктопе

```bash
python -m src.test_mobile_fp32_ts \
  -i data/test/2.png \
  -o artifacts/output_ts_fp32_cpu.jpg \
  --model artifacts/cartoon_gan_mobile_fp32_cpu.pt
```


---

### Сравнение оригинальной картинки и сгенерированной

| Оригинал | FP32                                                   | INT8 |
|---------|--------------------------------------------------------|------|
| <img src="data/test/2.png" width="256"/> | <img src="artifacts/output_fp32_cpu.jpg" width="256"/> |<img src="artifacts/output_int8_cpu.jpg" width="256"/>      |


## 7. Ограничения и принятые решения

### 7.1. Почему не получилось заквантовать всю модель в INT8

* Архитектура CartoonGAN включает:

  * `Conv2d + BatchNorm2d + ReLU` (encoder),
  * 8 residual-блоков,
  * `UpBlock` с `ConvTranspose2d`, блюром и `BatchNorm2d` (decoder).
* На Windows использовался quantized backend: `onednn`.
* При попытках полной INT8-квантизации через `torch.ao.quantization` возникали проблемы:

  * `ConvTranspose2d` не поддерживался с per-channel observer:

    * `AssertionError: Per channel weight observer is not supported yet for ConvTranspose{n}d`.
  * quantized-цепочка с BatchNorm вызывала:

    * `NotImplementedError: Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU' backend`.

**Вывод:** полностью INT8-версию CartoonGAN (encoder + res + decoder) собрать корректно не удалось из-за ограничений текущего стека PyTorch + onednn.

### 7.2. Почему не перенесли INT8-модель на мобильное устройство

* INT8-модель реализована как Python-обёртка с использованием `torch.ao.quantization` (QuantStub/DeQuantStub, кастомные qconfig и т.п.).
* Попытки экспортировать её в TorchScript приводили к:

  * ошибкам сериализации/загрузки:

    * `RuntimeError: required keyword attribute 'value' is undefined`;
  * runtime-ошибкам quantized-операторов (особенно вокруг `ops.quantized.batch_norm2d` и `native_batch_norm`).
* PyTorch Mobile:

  * не умеет выполнять `torch.ao.quantization` на устройстве;
  * ожидает статический TorchScript-граф с поддерживаемыми quantized-ops;
  * не поддерживает весь набор quantized-операторов, нужных для данной архитектуры.w

### 7.3. Итоговые инженерные решения

1. **FP32 baseline:**

   * реализован инференс на GPU (RTX 4070) и CPU;
   * проведены бенчмарки и зафиксировано качество (референсные изображения).

2. **INT8 ускорение на CPU:**

   * выполнено **post-training quantization только для энкодера** (свёрточные слои);
   * получено ускорение на CPU;
   * качество визуально почти не отличается от FP32;
   * INT8-веса сохранены и повторно загружаются из файлов.

3. **Мобильный сценарий:**

   * на Android перенесена **FP32 TorchScript-модель**, экспортированная на CPU;
   * модель успешно запускается на эмуляторе/устройстве и выдаёт cartoon-изображения;
   * INT8-модель на мобилку не переносилась по причине ограничений PyTorch Mobile.

**Краткий вывод по работе:**

* Цель ускорения CartoonGAN на CPU посредством INT8 и демонстрации работы на мобильном устройстве **достигнута частично**:

  * реализовано и проверено INT8-ускорение на CPU (частичное квантование);
  * сохранены и протестированы INT8-веса;
  * FP32-модель успешно перенесена на Android (TorchScript).
* Полная INT8-модель и её перенос на мобилку не реализованы осознанно, с техническим обоснованием:

  * ограничения backend’а `onednn` и `torch.ao.quantization`;
  * ограничения набора quantized-операторов в PyTorch Mobile для архитектуры CartoonGAN.



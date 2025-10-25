import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr
import os

# Определение классов (из ноутбука: bear, panda, raccoon)
class_names = ['bear', 'panda', 'raccoon']

# Трансформации для валидации/теста (из ноутбука), адаптированные для ONNX (numpy output)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Абсолютный путь к ONNX модели (из ноутбука)
base_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем директорию скрипта
onnx_path = os.path.join(base_dir, '..', 'notebooks', 'models', 'resnet18_bs16_lr0.0001.onnx')

# Загрузка ONNX модели для инференса на CPU
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# Функция предсказания с ONNX
def predict(image):
    # Преобразование изображения
    img = Image.fromarray(image)  # Gradio передает numpy array
    img = test_transforms(img).numpy()  # Преобразуем в numpy для ONNX
    img = np.expand_dims(img, axis=0)  # Добавляем batch dimension
    
    # Предсказание с ONNX
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_outs[0], axis=1)[0]
    
    return class_names[pred]

# Gradio интерфейс
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),  # Убрал shape, оставил type для numpy
    outputs="text",  # Выход: текст с классом
    title="Классификатор: Медведь, Енот или Панда",
    description="Загрузите изображение, и модель определит, что на нём: bear (медведь), panda (панда) или raccoon (енот). Инференс выполняется с помощью ONNX на CPU."
)

# Запуск приложения
iface.launch()
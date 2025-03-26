# API de Predicción de Precios de Vivienda 🏡

Esta API en Flask permite predecir el precio de una vivienda en base a los parámetros ingresados.

## 🚀 Instalación y Configuración

1. **Clonar el repositorio**  
```bash
git clone https://github.com/hugomoreno-tajamar/api_red_neuronal.git
cd api_red_neuronal
```

2. **Crear un entorno virtual y activarlo**

```bash
python -m venv venv
source venv/bin/activate  # En Mac/Linux
venv\Scripts\activate     # En Windows
```

3. **Instalar las dependencias**

```bash
pip install -r requirements.txt
```

4. **Ejecutar la API**

```bash
python app.py
```

La API correrá en http://127.0.0.1:5000.

---

## 🛠️ Uso de la API
### **Hacer una predicción con ```curl```**

Linux:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
    "rooms": 3,
    "baths": 2,
    "sqft": 100,
    "typology": "Pisos",
    "neighborhood": "Malasaña"
}'
```

CMD de Windows:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"rooms\":3, \"baths\":2, \"sqft\":100, \"typology\":\"Pisos\", \"neighborhood\":\"Malasaña\"}"
```
### **Hacer una predicción con ```Postman```**

1. Abrir Postman.

2. Seleccionar ```POST```.

3. Ingresar la URL: ```http://127.0.0.1:5000/predict```.

4. Ir a la pestaña **Body**, elegir ```raw``` y seleccionar ```JSON```.

5. Insertar el siguiente JSON:

```json
{
   "rooms": 3,
   "baths": 2,
   "sqft": 100,
   "typology": "Pisos",
   "neighborhood": "Malasaña"
}
```
6. Hacer clic en Send y recibir la predicción.

---

## 📌 Notas
- Asegúrate de que el modelo ```.keras``` y los escaladores están en el directorio correcto.

- Si hay problemas con dependencias, ejecuta:

```bash
pip install -r requirements.txt
```
¡Listo! Ahora puedes hacer predicciones con la API. 🚀

import os
from uuid import uuid4
import datetime
import requests
import re



#------------------------------------------------------------------------------------------------------------
TEMP_DIR = "temp_images"

async def save_upload_file(upload_file):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    file_ext = os.path.splitext(upload_file.filename)[-1]
    file_name = f"{uuid4()}{file_ext}"
    file_path = os.path.join(TEMP_DIR, file_name)

    print(f" Saving file to: {file_path}")  # Debug log

    with open(file_path, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)

    return file_path





#------------------------------------------------------------------------------------------------------------


# OpenWeather API Key
openweather_api_key = "896b1063f9040c0f98c1281f34a4aad2"

def get_time():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")

def get_weather(city=""):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweather_api_key}&units=metric&lang=ar"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        temp = data['main']['temp']
        description = data['weather'][0]['description']
        return f"درجة الحرارة في {city} هي {temp}°C وحالة الطقس: {description}."
    else:
        return "لم أتمكن من جلب معلومات الطقس حالياً."

def extract_city(question):
    match = re.search(r"(?:in|في)\s+([a-zA-Z\u0600-\u06FF\s]+)", question, re.IGNORECASE)
    if match:
        city = match.group(1).strip()
        city = re.sub(r"[^\w\s\u0600-\u06FF]", "", city)
        return city
    else:
        return "syria"


#-----------------------------------------------------------------------------------------------------------------
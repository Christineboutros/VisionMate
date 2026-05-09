import cv2
import numpy as np
import easyocr
import pytesseract
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from app.utils import get_time, get_weather, extract_city
import google.generativeai as genai
from ultralytics import YOLO
import torch


#-----------------------------------------------------------------------------------------------------------------
# Load EasyOCR Reader for English and Arabic

reader = easyocr.Reader(['en', 'ar'], gpu=True)

def detect_text_color(image, bbox):
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[2]
    text_region = image[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray)
    return "light" if avg_intensity > 150 else "dark"

def preprocess_image(image_path, text_color="light"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if text_color == "light":
        _, processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 12)
        processed = cv2.bitwise_not(processed)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    return processed

def detect_text(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f" Failed to read image from: {image_path}")
        return ["Image read failed"]

    print(f" Running EasyOCR on image: {image_path}")
    easyocr_results = reader.readtext(image, detail=1, paragraph=True)

    print(f" EasyOCR returned: {easyocr_results}")

    easy_texts = []
    text_color = "light"

    if easyocr_results:
        print(f" EasyOCR found {len(easyocr_results)} blocks")

        for result in easyocr_results:
            try:
                if len(result) == 3:
                    bbox, text, prob = result
                else:
                    bbox, text = result[0], result[1]  # fallback if no prob
                    prob = 1.0  # assume it's okay

                # Optional: skip very short or obviously wrong text
                if len(text.strip()) > 0 and prob > 0.4:
                    text_color = detect_text_color(image, bbox)
                    easy_texts.append((bbox[0][1], bbox[0][0], text))  # (y, x, text)

            except Exception as e:
                print(f" Error parsing OCR result: {e}")

        # Sort and extract only the text part
        easy_texts.sort(key=lambda b: (b[0], b[1]))  # Sort by y, then x
        final_text = [text for (_, _, text) in easy_texts]

    else:
        print(" EasyOCR returned no results, falling back to Tesseract")
        processed = preprocess_image(image_path, text_color)
        custom_config = "--psm 6 --oem 3"
        tesseract_text = pytesseract.image_to_string(processed, config=custom_config, lang='eng+ara')
        final_text = tesseract_text.split("\n")

    final_text = list(dict.fromkeys([line.strip() for line in final_text if line.strip()]))
    os.remove(image_path)  # Clean up after OCR
    return final_text



#--------------------------------------------------------------------------------------------------------------



model_path = os.path.join("app", "models", "Final_currency_model.h5")
banknote_model = tf.keras.models.load_model(model_path)

banknote_classes = [
    '1000_old_syp', '1000_syp', '100_usd', '10_usd', '1_usd', '2000_syp',
    '20_usd', '2_usd', '5000_syp', '500_old_syp', '500_syp', '50_usd', '5_usd'
]

def preprocess_image_for_banknote(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_banknote_class(img_path):
    print(f"🪙 Running banknote classifier on image: {img_path}")
    img_array = preprocess_image_for_banknote(img_path)
    predictions = banknote_model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = banknote_classes[predicted_index]
    confidence = float(predictions[0][predicted_index])
    print(f" Predicted: {predicted_class} ({confidence:.4f})")
    os.remove(img_path)
    return {"class": predicted_class, "confidence": round(confidence, 4)}


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

# Gemini API Key
#genai.configure(api_key="AIzaSyCc4rVFBftgnWBJCKt168vmCIUPTjvbc-g")


# Configure Gemini with your API key
genai.configure(api_key="AIzaSyDCx-m_1f1qqNS0_3tx2spjlVncY7mDK5U")  # use your working key

def ask_gemini(question: str) -> str:
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")  # <-- use full model name
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "حدث خطأ أثناء معالجة السؤال."

def assistant_response(question: str) -> str:
    q_lower = question.lower()
    time_keywords = ["time", "clock", "الوقت", "الساعة" , "الساعه"]
    weather_keywords = ["weather", "temperature", "الطقس", "الحرارة"]

    if any(word in q_lower for word in time_keywords):
        return f"الوقت الحالي هو: {get_time()}"
    elif any(word in q_lower for word in weather_keywords):
        city = extract_city(question)
        return get_weather(city)
    else:
        return ask_gemini(question)


# ------------------- YOLO Object Detection ----------------------





# Load YOLOv8 model
device = torch.device("cpu")
model = YOLO("app/models/yolov8n-oiv7.pt")  # Adjust model path as needed

# Constants
MIN_BOX_AREA_REL = 0.01
MAX_DISTANCE = 8.0  # in meters


relevant_classes = ["Alarm clock", "Ambulance", "Animal", "Apple", "Backpack", "Bagel", "Baked goods", "Ball", "Balloon", "Banana",
                    "Band-aid", "Barrel", "Baseball bat", "Bathroom accessory", "Bathroom cabinet", "Bathtub", "Bed", "Beer", "Bell pepper",
                    "Belt", "Bench", "Bicycle", "Bidet", "Billiard table", "Binoculars", "Blender", "Book", "Boot", "Bottle", "Bowl", "Box", "Boy",
                    "Bread", "Briefcase", "Broccoli", "Burrito", "Bus", "Cabbage", "Cake", "Calculator","Camera", "Can opener", "Candle", "Canary",
                    "Candy","Canoe", "Cantaloupe", "Car", "Carrot", "Cart", "Cassette deck", "Cat", "Cattle", "Cello", "Chainsaw", "Chair", "Cheese",
                    "Chest of drawers", "Chicken", "Christmas tree", "Clock", "Closet", "Cocktail", "Coffee", "Coffee table", "Coffeemaker", "Coin",
                    "Computer monitor", "Container", "Cookie", "Couch", "Countertop", "Croissant", "Cucumber", "Cupboard", "Cutting board", "Desk",
                    "Digital clock", "Dishwasher", "Dog", "Doll", "Drawer", "Drink", "Dumbbell", "Flag", "Flower", "Football", "Fountain", "French fries",
                    "Fruit", "Frying pan", "Furniture", "Gas stove", "Girl", "Glasses", "Guitar", "Hamburger", "Headphones", "Kitchen & dining room table",
                    "Kitchen knife", "Knife", "Ladder", "Lamp", "Light switch", "Luggage and bags", "Microwave oven", "Mobile phone", "Motorcycle",
                    "Nightstand", "Orange", "Oven", "Palm tree", "Pancake", "Paper towel", "Person", "Piano", "Pillow", "Pizza", "Plant", "Plastic bag",
                    "Potato", "Pressure cooker", "Refrigerator", "Remote control", "Salt and pepper shakers", "Sandwich", "Sink", "Sofa bed", "Spoon", "Stairs",
                    "Stop sign", "Stool", "Street light", "Suitcase", "Swimming pool", "Teapot", "Telephone", "Television", "Toilet", "Toilet paper", "Towel",
                    "Traffic light", "Traffic sign", "Laptop", "Train", "Tree", "Truck", "Van", "Wardrobe", "Woman"]


class_name_translations = {   "Alarm clock": "ساعة منبّه",
    "Ambulance": "سيارة إسعاف",
    "Animal": "حيوان",
    "Apple": "تفاحة",
    "Backpack": "حقيبة ظهر",
    "Bagel": "خبز البيغل",
    "Baked goods": "مخبوزات",
    "Ball": "كرة",
    "Balloon": "بالون",
    "Banana": "موزة",
    "Band-aid": "لاصق جروح",
    "Barrel": "برميل",
    "Baseball bat": "مضرب بيسبول",
    "Bathroom accessory": "إكسسوار حمام",
    "Bathroom cabinet": "خزانة حمام",
    "Bathtub": "حوض استحمام",
    "Bed": "سرير",
    "Beer": "بيرة",
    "Bell pepper": "فلفل رومي",
    "Belt": "حزام",
    "Bench": "مقعد",
    "Bicycle": "دراجة هوائية",
    "Bidet": "شطّاف",
    "Billiard table": "طاولة بلياردو",
    "Binoculars": "منظار",
    "Blender": "خلاط",
    "Book": "كتاب",
    "Boot": "جزمة",
    "Bottle": "زجاجة",
    "Bowl": "وعاء",
    "Box": "صندوق",
    "Boy": "ولد",
    "Bread": "خبز",
    "Briefcase": "حقيبة أوراق",
    "Broccoli": "بروكلي",
    "Burrito": "بوريتو",
    "Bus": "حافلة",
    "Cabbage": "ملفوف",
    "Cake": "كعكة",
    "Calculator": "آلة حاسبة",
    "Camera": "كاميرا",
    "Can opener": "فتاحة علب",
    "Candle": "شمعة",
    "Canary": "كناري",
    "Candy": "حلوى",
    "Canoe": "زورق",
    "Cantaloupe": "شمّام",
    "Car": "سيارة",
    "Carrot": "جزر",
    "Cart": "عربة",
    "Cassette deck": "مسجل كاسيت",
    "Cat": "قطة",
    "Cattle": "ماشية",
    "Cello": "تشيلو",
    "Chainsaw": "منشار كهربائي",
    "Chair": "كرسي",
    "Cheese": "جبن",
    "Chest of drawers": "خزانة أدراج",
    "Chicken": "دجاج",
    "Christmas tree": "شجرة عيد الميلاد",
    "Clock": "ساعة",
    "Closet": "خزانة ملابس",
    "Cocktail": "كوكتيل",
    "Coffee": "قهوة",
    "Coffee table": "طاولة قهوة",
    "Coffeemaker": "آلة قهوة",
    "Coin": "عملة معدنية",
    "Computer monitor": "شاشة كمبيوتر",
    "Container": "حاوية",
    "Cookie": "بسكويت",
    "Couch": "كنبة",
    "Countertop": "سطح مطبخ",
    "Croissant": "كرواسون",
    "Cucumber": "خيار",
    "Cupboard": "خزانة",
    "Cutting board": "لوح تقطيع",
    "Desk": "مكتب",
    "Digital clock": "ساعة رقمية",
    "Dishwasher": "غسالة صحون",
    "Dog": "كلب",
    "Doll": "دمية",
    "Drawer": "درج",
    "Drink": "مشروب",
    "Dumbbell": "دمبل",
    "Flag": "علم",
    "Flower": "زهرة",
    "Football": "كرة قدم",
    "Fountain": "نافورة",
    "French fries": "بطاطا مقلية",
    "Fruit": "فاكهة",
    "Frying pan": "مقلاة",
    "Furniture": "أثاث",
    "Gas stove": "موقد غاز",
    "Girl": "بنت",
    "Glasses": "نظارات",
    "Guitar": "قيثارة",
    "Hamburger": "همبرغر",
    "Headphones": "سماعات رأس",
    "Kitchen & dining room table": "طاولة مطبخ وطعام",
    "Kitchen knife": "سكين مطبخ",
    "Knife": "سكين",
    "Ladder": "سلم",
    "Lamp": "مصباح",
    "Light switch": "مفتاح كهرباء",
    "Luggage and bags": "حقائب سفر",
    "Microwave oven": "ميكروويف",
    "Mobile phone": "هاتف محمول",
    "Motorcycle": "دراجة نارية",
    "Nightstand": "طاولة جانبية",
    "Orange": "برتقالة",
    "Oven": "فرن",
    "Palm tree": "نخلة",
    "Pancake": "فطيرة",
    "Paper towel": "منشفة ورقية",
    "Person": "شخص",
    "Piano": "بيانو",
    "Pillow": "وسادة",
    "Pizza": "بيتزا",
    "Plant": "نبتة",
    "Plastic bag": "كيس بلاستيك",
    "Potato": "بطاطا",
    "Pressure cooker": "طنجرة ضغط",
    "Refrigerator": "ثلاجة",
    "Remote control": "جهاز تحكم",
    "Salt and pepper shakers": "مملحة وفلفل",
    "Sandwich": "شطيرة",
    "Sink": "مغسلة",
    "Sofa bed": "كنبة سرير",
    "Spoon": "ملعقة",
    "Stairs": "درج",
    "Stop sign": "علامة توقف",
    "Stool": "كرسي مرتفع",
    "Street light": "ضوء شارع",
    "Suitcase": "حقيبة سفر",
    "Swimming pool": "مسبح",
    "Teapot": "إبريق شاي",
    "Telephone": "هاتف",
    "Television": "تلفاز",
    "Toilet": "مرحاض",
    "Toilet paper": "ورق حمام",
    "Towel": "منشفة",
    "Traffic light": "إشارة مرور",
    "Traffic sign": "لافتة مرور",
    "Train": "قطار",
    "Tree": "شجرة",
    "Truck": "شاحنة",
    "Van": "فان",
    "Wardrobe": "خزانة ملابس",
    "Woman": "امرأة",
    "Laptop":"لابتوب"
}

class_name_plurals = {  "شخص": "أشخاص",
    "امرأة": "نساء",
    "رجل": "رجال",
    "ولد": "أولاد",
    "بنت": "بنات",
    "سيارة": "سيارات",
    "شاحنة": "شاحنات",
    "حافلة": "حافلات",
    "فان": "فانات",
    "دراجة نارية": "دراجات نارية",
    "دراجة هوائية": "دراجات هوائية",
    "قطة": "قطط",
    "كلب": "كلاب",
    "نبتة": "نباتات",
    "شجرة": "أشجار",
    "زهرة": "زهور",
    "كتاب": "كتب",
    "كرسي": "كراسي",
    "كنبة": "كنبات",
    "حقيبة ظهر": "حقائب ظهر",
    "حقيبة أوراق": "حقائب أوراق",
    "زجاجة": "زجاجات",
    "وعاء": "أوعية",
    "صندوق": "صناديق",
    "بيتزا": "بيتزات",
    "همبرغر": "همبرغر",
    "كعكة": "كعكات",
    "موزة": "موز",
    "تفاحة": "تفاح",
    "برتقالة": "برتقال",
    "بطاطا": "بطاطا",
    "بسكويت": "بسكويتات",
    "كرة": "كرات",
    "بالون": "بالونات",
    "شمعة": "شمعات",
    "دمية": "دمى",
    "مصباح": "مصابيح",
    "مقلاة": "مقالي",
    "كيس بلاستيك": "أكياس بلاستيك",
    "عملة معدنية": "عملات معدنية",
    "حذاء": "أحذية",
    "قهوة": "قهاوي",
    "سكين": "سكاكين",
    "ملعقة": "ملاعق",
    "مغسلة": "مغاسل",
    "دُمية": "دمى",
    "آلة حاسبة": "آلات حاسبة",
    "كاميرا": "كاميرات",
}

real_heights = {  "Person": 1.7,
    "Man": 1.8,
    "Woman": 1.65,
    "Dog": 0.4,
    "Boy": 1.4,
    "Girl": 1.4,
    "Car": 1.5,
    "Truck": 3.5,
    "Bus": 3.0,
    "Van": 2.0,
    "Bicycle": 1.2,
    "Motorcycle": 1.1,
    "Street light": 7.0,
    "Traffic light": 3.5,
    "Stop sign": 2.1,
    "Tree": 5.0,
    "Bench": 0.8,
    "Chair": 0.9,
    "Bed": 0.7,
    "Sofa bed": 0.9,
    "Table": 0.75,           
    "Piano": 1.0,
    "Wardrobe": 2.0,
    "Refrigerator": 1.8,
    "Bathtub": 0.6,
    "Sink": 0.8,
    "Lamp": 1.5,
    "Kitchen & dining room table": 0.75,
    "Nightstand": 0.6,
    "Traffic sign": 2.0,
    "Swimming pool": 1.5,    
    "Mobile phone": 0.18,
}

direction_arabic = {"left": "على اليسار", "center": "أمامك", "right": "على اليمين"}

# Helper to classify direction
def get_position(x_center, img_w):
    left_threshold = img_w * 0.3
    right_threshold = img_w * 0.7

    if x_center < left_threshold:
        return "left"
    elif x_center < right_threshold:
        return "center"
    else:
        return "right"

# Main function to call from FastAPI
def detect_objects_with_yolo(image_path):
    results = model(image_path, device=device)[0]
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    direction_summary = {
        "left": {},
        "center": {},
        "right": {}
    }

    FOCAL_LENGTH = img_w * (4.3 / 5.7)

    center_outputs = []

    for box in results.boxes:
        x, y, w, h = box.xywh[0]
        area_rel = (w * h) / (img_w * img_h)

        if area_rel < MIN_BOX_AREA_REL:
            continue

        cls_id = int(box.cls.item())
        label = model.names[cls_id]

        if label not in relevant_classes:
            continue

        pos = get_position(x.item(), img_w)

        if label in real_heights:
            real_height = real_heights[label]
            bbox_height_pixels = h.item()
            if bbox_height_pixels == 0:
                continue
            distance = (FOCAL_LENGTH * real_height) / bbox_height_pixels

            if distance > MAX_DISTANCE:
                continue

            if pos == "center":
                center_outputs.append(
                    f"{class_name_translations.get(label, label)} أمامك على بُعد تقريبي: {distance:.1f} متر"
                )
        else:
            continue

        direction_summary[pos][label] = direction_summary[pos].get(label, 0) + 1

    final_output = []

    # Center objects with distance
    final_output.extend(center_outputs)

    # Left/Right summary
    for direction in ["left", "right"]:
        items = direction_summary[direction]
        if items:
            desc = ", ".join([
                f"{v} {class_name_plurals.get(class_name_translations.get(k, k), class_name_translations.get(k, k) + 'ات') if v > 1 else class_name_translations.get(k, k)}"
                for k, v in items.items()
            ])
            final_output.append(f"- {desc} {direction_arabic[direction]}")

    return {"detections": final_output}

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib
import string
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import random 
import json
import difflib
from difflib import get_close_matches
 
import base64 
import cv2   

from PIL import Image

from flask_cors import CORS

app = Flask(__name__) 
CORS(app)
# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values to be between 0 and 1
    return img_array

# Load and preprocess chatbot data
file_path ='C:\\Users\\kavya\\MiniProj\\Two_one\\content.json'
data1 = {}
2
try:
    with open(file_path, encoding='utf-8') as content:
        data1 = json.load(content)
    print("Keys in the loaded JSON data:", data1.keys())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' could not be found.")
except json.JSONDecodeError:
    print(f"Error: There was an issue decoding JSON in '{file_path}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

tags = []
inputs = []
responses = {}
for intent in data1['intent']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])

data = pd.DataFrame({"inputs": inputs, "tags": tags})
data = data.sample(frac=1)

data['inputs'] = data['inputs'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
# print(input_shape)

vocabulary = len(tokenizer.word_index)
# print("number of unique words : ", vocabulary)
output_length = le.classes_.shape[0]
# print("output length: ", output_length)

i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model1 = Model(i, x)

model1.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

train=model1.fit(x_train, y_train, epochs=500)

model1.save("chatbot_model.keras")

np.save('x_train.npy', x_train)
joblib.dump(le, 'label_encoder.pkl')

tokenizer_config = tokenizer.get_config()
with open('tokenizer_config.json', 'w') as f:
    json.dump(tokenizer_config, f)

plt.plot(train.history['accuracy'], label='training set accuracy')
plt.plot(train.history['loss'], label='training set loss')
plt.legend()


names = ["sad","cancer", "treat option", "type of cancer","types of cancer", "cancer prevention","causes of cancer","cancer genetics","types of treat option",
         "cancer prevention strategies","cancer clinical trails","supportive care treatment","cancer and mental health", "cancer statistics", "pain management","screening","ovarian cancer",
         "cervical cancer", "leukemia","liver cancer","thyroid cancer","esophageal cancer","head and neck cancer","multiple myeloma","kidney cancer","gastric cancer","bone cancer","brain cancer",  
                      
                   "cancer in children","immunotherapy","prostate cancer","pancreatic cancer","uterine cancer","myeloma","diagnosed","staging","gets cancer","cancer start","contagious","vaccine",
                   "cure","stages","drugs","test","people get cancer","epidemic","eat","chemical","smoking","prevented","diagnosis delayed","universal test","breast cancer causes","awareness day",
                   "skin or tissue","silent killer","connective tissue","mesothelioma","tobacco","hpv hbv","processed meats","risk factor","injuries","emotion","stress","sugar","alcohol","lung diagnosed",
                   "lung treatment","lung risk","liver prevention","liver detect","fatty liver","pbc","liver treatment","medicines for liver cancer","liver cyst","liver backpain",
                   "genetics of liver","emotion","lung diagnosed","genetics of liver","death 2020","palliative care","world cancer day","who report","history of cancer day","cases in 2022",
                   "cases in 2020","cancer occur","cancer spread","prevent","tumour","oncology","oncologist",
                   
                   "treatment breastcancer","treatment liver cancer","treatment thyroid cancer","treatment bladder cancer","treatment brain cancer","treatment esophageal cancer",
                   "treatment head and neck cancer","treatment multiple myeloma","treatment rectal cancer","treatment Testicular cancer","treatment kidney cancer","treatment prostrate cancer",
                   "treatment gastric cancer","treatment uterine cancer","treatment colateral cancer", 
                   
                   
                "grow and spread","treatment","symptoms","causes","prevent cancer","cancer types","cancer diagnosis","cancer hereditary","symptoms of lung cancer","lungcancer","symptoms of skin cancer","symptoms of prostrate cancer","symptoms of colon cancer",
                "symptoms of ovarian cancer","symptoms of pancreatic cancer","symptoms of cervical cancer","symptoms of leukemia","symptoms of liver cancer","symptoms of thyroid cancer","symptoms of bladder cancer","symptoms of brain cancer","symptoms of esophageal cancer",
                "symptoms multiple myeloma","symptoms of rectal cancer","symptoms of testicular cancer","symptoms of kidney cancer","symptoms of prostrate cancer","symptoms of gastric cance","symptoms of uterine cancer","symptoms of colateral cancer",
                "symptoms of bone cancer","treatment breastcancer","treatment lungcancer","treatment coloncancer","head cancer","neck cancer"
                
                "aiims","cancer and excersice","cancer nutrition","symptoms of leukemia","onco father","cancer oncology","cancer survivorship","clinical terms of cancer","good bye","pancreatic cancer","symptoms of cervical cancer","symptoms of esophageal cancer",
                "symptoms of pancreatic cancer","symptoms of ovarian cancer","symptoms of liver cancer","tamilnadu","types of oncologists","were are you","who are you"
                ]

image_paths = { "sad":"sad1.jpg","cancer": "cancer.jpg",
    "treat option": "treatoption1.png","type of cancer": "typeofcancer.jpeg","types of cancer":"typesofcancer.jpg",
    "cancer prevention":"CancerPrevention.jpg","causes of cancer":"causes.png",
    "cancer genetics":"genetics.jpg", "types of treat option":"typesoftreatoption.png", 
     "cancer prevention strategies":"cancer prevention.png","cancer clinical trails":"clinical trail.jpg",
    "supportive care treatment"  :"supportive.jpg","cancer and mental health":"cancer and mentalhealth.jpg","cancer statistics"  :"cancer statitics.jpg",
   "pain management"  :"pain management.jpg","screening" :"screening.jpg",
   "ovarian cancer" :"ovarian cancer.jpg","cervical cancer" :"cervicalcancer.jpg", "leukemia"   :"leukemia.jpg", "liver cancer"  :"liver cancer.png","thyroid cancer" :"thyroid cancer.jpg","esophageal cancer":"esophagus-cancer.jpg",
   "head and neck cancer"  :"headneck.jpg",  "multiple myeloma"  :"multiple myeloma.jpg", "kidney cancer":"kidney.jpg",
  "gastric cancer":"gastric.jpg","bone cancer":"bone1.jpg", 

  "brain cancer":"brain.jpg",  
  "cancer in children":"children cancer.jpeg","immunotherapy":"mmunotherapy.jpg","prostate cancer":"prostrate cancer.jpg","pancreatic cancer":"pancreatic cancer1.jpg"
  ,"uterine cancer":"uterine cancer.jpeg","myeloma":"myeloma.jpg","diagnosed":"diagnosed.jpg","staging":"stages.png","gets cancer":"gets cancer.jpg",
  "cancer start":"cancer start.jpg","contagious":"contagious cancer.jpg","vaccine":"vaccine1.jpg","cure":"cancer cure.jpg","stages":"stage 0 to 4.png",
  "drugs":"drugs cancer.jpeg","test":"cancer test.jpeg","people get cancer":"people get cancer.jpeg","epidemic":"epidemic.jpeg",
  "eat":"eat.jpeg","chemical":"chemical cancer.jpg","smoking":"smoking cancer.jpg","prevented":"prevented.jpeg","diagnosis delayed":"diagnosis cancer.jpg",
  "universal test":"universal test.jpeg","breast cancer causes":"breast cancer causes.jpeg","awareness day":"awareness day.jpg","skin or tissue":"skin or tissue.jpeg",
  "silent killer":"silent killer.jpg","connective tissue":"sarcoma.jpeg","mesothelioma":"mesothelioma.jpg","tobacco":"death by cancer.jpg","hpv hbv":"hpv hbv.jpg",
  "processed meats":"cancer foods.jpg","risk factor":"risk factor.jpg","injuries":"injuries cancer.jpg","emotion":"emotion cancer.jpg",
  "stress":"stress cancer.jpeg","sugar":"sugar feed cancer.jpg","alcohol":"alcohol cancer.jpeg","lung diagnosed":"lung diagnosed cancer.jpg","lung treatment":"lung treatment.jpg",
  "lung risk":"lung risk.png","liver prevention":"liver.jpeg","liver detect":"liver detect.jpg","fatty liver":"fatty liver.jpeg","pbc":"pbc cancer.jpg",
  "liver treatment":"liver treatment.jpg","medicines for liver cancer":"medicines for liver cancer.jpeg","liver cyst":"liver cyst.jpg",
  "liver backpain":"liver backpain.jpg","genetics of liver":"genetics of liver.jpg","death 2020":"death 2020.png",
  "palliative care":"palliative care.jpg","world cancer day":"world cancer day.jpeg","who report":"who report.png","history of cancer day":"history of cancer day.jpeg",
  "cases in 2022":"cases in 2022.png","cases in 2020":"cases in2020.png","cancer occur":"cancer body.jpg","cancer spread":"cancer spread.jpg","prevent":"prevent.jpg",
  "tumour":"tumour.jpg","oncology":"oncology.jpg","oncologist":"oncologist.jpg", 
  
  "treatment breastcancer":"treatment breastcancer.jpeg","treatment liver cancer":"treatment liver cancer.jpg","treatment thyroid cancer":"treatment thyroid cancer.jpeg",
  "treatment bladder cancer":"treatment bladder cancer.jpg","treatment brain cancer":"treatment brain cancer.jpg","treatment esophageal cancer":"treatment esophageal cancer.jpg",
  "treatment head and neck cancer":"treatment head and neck cancer.png","treatment multiple myeloma":"treatment multiple myeloma.jpg","treatment rectal cancer":"treatment rectal cancer.jpg",
  "treatment Testicular cancer":"treatment Testicular cancer.jpeg","treatment kidney cancer":"treatment kidney cancer.jpg","treatment prostrate cancer":"treatment prostrate cancer.png",
  "treatment gastric cancer":"treatment gastric cancer.jpeg","treatment uterine cancer":"treatment uterine cancer.jpg","treatment colateral cancer":"treatment colateral cancer.jpeg",
   
   
   
   "grow and spread":"grow.jpg","treatment":"treatment.jpg","symptoms":"symptoms.jpg","causes":"causes.jpg","prevent cancer":"prevention.jpg","cancer types":"steps.jpg","cancer diagnosis medical test to diagnoise cancer":"diagnosis.jpg",
   "cancer hereditary cancer inheritence":"hereditary.png","symptoms of lung cancer":"symptoms of lung cancer.jpg","lungcancer":"lung.jpg","symptoms of skin cancer":"symptoms of skin.jpg","symptoms of prostrate cancer":"prostaate.jpg","symptoms of colon cancer":"symptoms of colon.png",
   "symptoms of ovarian cancer":"ovarian cancer.png","symptoms of pancreatic cancer":"pancreatic cancer.jpg","symptoms of cervical cancer":"cervical cancer.jpg","symptoms of leukemia":"leukemia.jpg","symptoms of liver cancer":"liver.jpg","symptoms of thyroid cancer":"thyroid cancer.jpg",
   "symptoms of bladder cancer":"bladder cancer.jpg","symptoms of brain cancer":"symptoms of brain cancer.jpg","symptoms of esophageal cancer":"esophageal.jpeg","symptoms multiple myeloma":"symptoms multiple myeloma.jpg","symptoms of rectal cancer":"rectal cancer.png","symptoms of testicular cancer":"symptoms of testicular cancer.jpg",
   "symptoms of kidney cancer":"kidney cancer.jpg","symptoms of prostrate cancer":"prostate cancer.jpeg","symptoms of gastric cance":"gastric cancer.jpg","symptoms of uterine cancer":"symptoms of uterine cancer.jpg","symptoms of colateral cancer":"symptoms of colateral cancer.jpg","symptoms of bone cancer":"symptomsofbonecancer.jpg",
   "treatment breastcancer":"breast cancer.jpg","treatment lungcancer":"treatment of lung cancer.jpg","treatment coloncancer":"treatment of clone cancer.jpg","head cancer":"head cancer.jpg","neck cancer":"neck cancer.jpeg",
   
   "aiims":"aiims.jpeg","cancer and excersice":"cancer and excercise.jpeg","cancer nutrition":"cancer nutrition.jpeg","cancer oncology":"cancer oncology.jpeg","cancer survivorship":"cancer survivor ship.jpeg","clinical terms of cancer":"clinical terms of cancer.jpeg","good bye":"good bye.jpeg",
   "pancreatic cancer":"pancreatic cancer.jpeg","symptoms of cervical cancer":"symptoms of cervical cancer.jpeg","symptoms of esophageal cancer":"symptoms of esophageal cancer.jpeg","symptoms of liver cancer":"symptoms of liver cancer.jpeg","symptoms of pancreatic cancer":"symptoms of pancreatic cancer.jpeg",
   "symptoms of ovarian cancer":"symtoms of ovarian cancer.jpeg","tamilnadu":"tamilnadu.jpeg","types of oncologists":"types of oncologists.jpeg","were are you":"were are you.jpeg","who are you":"who are you.jpeg","symptoms of leukemia":"leukemia.jpg","onco father":"onco father.jpeg"
                    
    }



 
default_sad_image_path = "sad1.jpg"
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
      while True:
   
          
          img_format="" 
          
          texts_p=[]
          data = request.get_json()
          # user_input = data['user_input']
          prediction_input = data['user_input']
          
        # Preprocess user input for NLP model
          prediction_input = ''.join([letters.lower() for letters in prediction_input if letters not in string.punctuation])
          texts_p.append(prediction_input)

          prediction_input = tokenizer.texts_to_sequences(texts_p)
          prediction_input = np.array(prediction_input).reshape(-1)
          prediction_input = pad_sequences([prediction_input], input_shape)

        # Predict using NLP model
          output = model1.predict(prediction_input) 
          # output=output.argmax() 
          predicted_confidence = np.max(output)
          predicted_tag = le.inverse_transform(np.argmax(output, axis=1))[0]

          print("Shape of output:", output.shape)
          confidence_threshold = 0.5
          # Reshape the output array to 1D
          output_reshaped = output.reshape(-1)
        # Check if the predicted confidence is above the threshold
          if predicted_confidence > confidence_threshold and predicted_tag in responses:
            response = random.choice(responses[predicted_tag]) 
            image_path = image_paths[predicted_tag.lower()]  # Use the predicted tag for the image
          else:
            response = "I'm sorry, but I don't have sufficient information on that topic, so I can't provide a confident answer."
            image_path = image_paths["sad"] 
            predicted_tag="sad"
             
          predicted_image_path = ""
          try:
        # Predict image using image classification model
            loaded_model = tf.keras.models.load_model('name_to_image_model.keras')
            predicted_label = loaded_model.predict(np.expand_dims(load_and_preprocess_image(image_paths[predicted_tag.lower()]), axis=0))
            predicted_index = np.argmax(predicted_label)
            predicted_name = names[predicted_index]

        # Display the predicted image
            predicted_image_path = image_paths[predicted_tag.lower()]
            predicted_image = Image.open(predicted_image_path)
            image_array = np.array(predicted_image)

            if predicted_image_path.lower().endswith(('.png', '.jpeg', '.jpg')):
              img_format = 'jpeg'  # You want the format, not the file extension
            elif predicted_image_path.lower().endswith('.png'):
              img_format = 'png'
            else:
              img_format = 'jpeg'  # Default to jpeg

            _, img_buffer = cv2.imencode(f'.{img_format}', image_array)  # Convert image array to specified format
            img_str = base64.b64encode(img_buffer).decode('utf-8')
            
            
            
            print("Length of base64 string:", len(img_str)) 
            print("Image format: ",img_format)
            print("Image Content:", img_str[:100])  # Print the first 100 characters of the image content
            
       
          finally:
            print("Predicted Tag:", predicted_tag)
            print("Confidence Score:", predicted_confidence)
            print("Response:", response)
            print("Image Path:", image_path)
            print("Predicted Image Path:", predicted_image_path)
  
        # Return response and base64-encoded image
          return jsonify({'response': response, 'predicted_image_base64': img_str,
                          'image_format': img_format})

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)

    
    







# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.utils import to_categorical
# import numpy as np

# import matplotlib.pyplot as plt

names = ["sad","cancer", "treat option", "type of cancer","types of cancer", "cancer prevention","causes of cancer","cancer genetics","types of treat option",
         "cancer prevention strategies","cancer clinical trails","supportive care treatment","cancer and mental health", "cancer statistics", "pain management","screening","ovarian cancer",
         "cervical cancer", "leukemia","liver cancer","thyroid cancer","esophageal cancer","head and neck cancer","multiple myeloma","kidney cancer","gastric cancer","bone cancer","brain cancer",  
                      
                   "cancer in children","immunotherapy","prostate cancer","pancreatic cancer","uterine cancer","myeloma","diagnosed","staging","gets cancer","cancer start","contagious","vaccine",
                   "cure","stages","drugs","test","people get cancer","epidemic","eat","chemical","smoking","prevented+","diagnosis delayed","universal test","breast cancer causes","awareness day",
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

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split  # Add this import

import matplotlib.pyplot as plt



# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values to be between 0 and 1
    return img_array

# Create X (image data) and Y (label data)
X = []
Y = []

for name, image_path in image_paths.items():
     if name in names:
        X.append(load_and_preprocess_image(image_path))
        Y.append(name)

X = np.array(X)
Y_numeric = np.array([names.index(name) for name in Y])

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(names), activation='softmax'))  # Adjusted for the number of classes

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_numeric, test_size=0.2, random_state=42)

# One-hot encode the labels
Y_train_encoded = to_categorical(Y_train, num_classes=len(names))
Y_val_encoded = to_categorical(Y_val, num_classes=len(names))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train_encoded, epochs=10, validation_data=(X_val, Y_val_encoded))

# Save the model
model.save('name_to_image_model.keras')

# Load the trained model
loaded_model = tf.keras.models.load_model('name_to_image_model.keras')

# Example usage: Predict the image for a new name
input_name = input("You:")
predicted_label = loaded_model.predict(np.expand_dims(load_and_preprocess_image(image_paths[input_name.lower()]), axis=0))
predicted_index = np.argmax(predicted_label)
predicted_name = names[predicted_index]

# Display the predicted image
predicted_image_path = image_paths[input_name]
predicted_image = load_img(predicted_image_path)
plt.imshow(predicted_image)
plt.title(f"Predicted Image for {input_name}")
plt.show()






















# # Function to load and preprocess an image
# def load_and_preprocess_image(image_path):
#     img = load_img(image_path, target_size=(64, 64))
#     img_array = img_to_array(img)
#     img_array /= 255.0  # Normalize pixel values to be between 0 and 1
#     return img_array

# # Create X (image data) and Y (label data)
# X = []
# Y = []

# for name, image_path in image_paths.items():
#      if name in names:
#         X.append(load_and_preprocess_image(image_path))
#         Y.append(name)

# X = np.array(X)
# Y_numeric = np.array([names.index(name) for name in Y])
# # Y = np.array(Y)

# # Define the model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# # model.add(layers.Dense(num_classes, activation='softmax'))
# model.add(layers.Dense(len(names), activation='softmax'))  # Adjusted for the number of classes 


# # Split the data into training and validation sets
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y_numeric, test_size=0.2, random_state=42)

# # One-hot encode the labels
# Y_train_encoded = to_categorical(Y_train, num_classes=len(names))
# Y_val_encoded = to_categorical(Y_val, num_classes=len(names))




# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # # Train the model
# # model.fit(X, np.array([names.index(name) for name in Y]), epochs=10, validation_split=0.2)

# # Train the model
# model.fit(X_train, Y_train_encoded, epochs=10, validation_data=(X_val, Y_val_encoded))
# # model.fit(X, np.array([names.index(name) for name in Y]), epochs=10)



# # Save the model
# model.save('name_to_image_model.keras')

# # Load the trained model
# loaded_model = tf.keras.models.load_model('name_to_image_model.keras')

# # Example usage: Predict the image for a new name
# input_name = input("You:")
# predicted_label = loaded_model.predict(np.expand_dims(load_and_preprocess_image(image_paths[input_name.lower()]), axis=0))
# predicted_index = np.argmax(predicted_label)
# predicted_name = names[predicted_index]



# # Display the predicted image
# predicted_image_path = image_paths[input_name]
# predicted_image = load_img(predicted_image_path)
# plt.imshow(predicted_image)
# plt.title(f"Predicted Image for {input_name}")
# plt.show()

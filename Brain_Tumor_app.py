from tensorflow.keras.models import load_model
import cv2
import tempfile
import hashlib
#
import tensorflow as tf
import os
import random
import numpy as np 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

import math
from PIL import Image
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
import seaborn as sb
from tensorflow.keras.layers import Input, Conv2D,AveragePooling2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout, Lambda, GlobalAveragePooling2D, GaussianNoise, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Layer
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Input, MaxPooling2D, Multiply, Concatenate, UpSampling2D, Conv2D,Conv2DTranspose
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
#
import numpy as np
from PyPDF2 import PdfReader # to read the pdf file
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter # Recursively breaks larger text into small chunk, one after another
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS # Vector database
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=ChatGoogleGenerativeAI(model= 'gemini-2.0-flash-exp',temperature=0.3)# temperature the more low we give the less productive it becomes and try to be obvious for the next word generation 

def get_pdf_text(pdf_docs):
    text=''
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf) # contains in pages
        for page in pdf_reader.pages:
            text+=page.extract_text() # extracting text from the page and storing it in text variable
    return text
def break_text_into_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000) # it's the class instance I have created, which will first created this empty object named text_splitter and implicitly passing the object in the constructor as a first argument and initialise it in specific memory particular for that instance
    chunks=text_splitter.split_text(text) # passing the text for chunking into the methodz
    return chunks
def converting_to_embedding():
    embedding=GoogleGenerativeAIEmbeddings(model='models/embedding-001') # Creating 'embedding' instance of this embedding model(embedding-001) class(GoogleGenerativeAIEmbeddings) which will convert chunks into a high dimensional vectors or embedding so we can store in vector database for later retrieval of most relevant chunks according to query
    return embedding
def storing_to_vectordatabase(text_chunks):
    vectorDB=FAISS.from_texts(texts=text_chunks,embedding=converting_to_embedding()) # converting each text_chunks with embedding and storing in vectorDB, direct memory mean in RAM , lost after if cancel it 
    vectorDB.save_local('FAISS_INDEX') # saving locally as it gets lost if stored in Memory
def get_conversational_chain():
    prompt_templates="""
    Your are a medical expert and you are given a context from a medical document.
    Answer the question as detailed as possible from the provided context ,make sure to provide all the details,if the answer is not present in the context just say 'Answer is not present in the context',Don't make up things
    and if asked "Who developed this model?" then answer "This model is developed by Amartya Ghosh and Manish"
    Context:\n{context}?\n
    Question:\n{question}\n
    
    Answer:
    
    """
    
    prompt=PromptTemplate(template=prompt_templates,input_variables=['context','question'])
    chain=load_qa_chain(model,chain_type='stuff',prompt=prompt) # returning configured instance 
    return chain
    # Check below
@st.cache_resource
def load_model_once():
    return load_model('Brain Tumor\WDFF-NET_files\model.h5')
def prediction(image):
    model_brain = load_model_once()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(image.read())
        temp_path = temp.name
    img = cv2.imread(temp_path,cv2.IMREAD_COLOR)
    img = cv2.resize(img, (352, 352))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img,dtype=np.uint8)
    img = (img/255.0).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    y_pred=model_brain.predict(img)
    y_pred=np.argmax(y_pred)
    classes_name=['glioma',
                    'meningioma',
                    'no',
                    'pituitary']
    return classes_name[y_pred]
import base64

def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: 35% 65%;
        background-position: bottom right;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
if 'response_history' not in st.session_state:
    st.session_state.response_history = []

def user_input(Query):
    storedDB = FAISS.load_local(
        'FAISS_INDEX',
        embeddings=converting_to_embedding(),
        allow_dangerous_deserialization=True
    )
    docs = storedDB.similarity_search(Query)
    chain = get_conversational_chain()
    response = chain(
        {'input_documents': docs, 'question': Query},
        return_only_outputs=True
    )

    # Append to session_state to persist
    st.session_state.response_history.append(response['output_text'])

    # Display responses (latest on top)
    st.write("## 🧑‍⚕️ AI Assistant Responses")
    for i, resp in enumerate(reversed(st.session_state.response_history), 1):
        st.write(f"""
                    <div style="background-color: rgba(255, 255, 255, 0.8);
                    color: black;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 8px;
                    border: 3px solid cyan;
                    font-family: Arial;
                    text-align: left;">
            <b>Response {len(st.session_state.response_history) - i + 1}:</b><br>{resp}
        </div>
        """, unsafe_allow_html=True)


# def display_response_box(response_text):
#     html_code = f"""
#     <div style="
#         background-color: rgba(212, 175, 55, 0.2);  /* Transparent dark golden */
#         color: black;
#         padding: 20px;
#         border: 2px solid black;                  /* Black border */
#         border-radius: 10px;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#         margin-top: 20px;
#         font-size: 16px;
#         font-family: 'Arial', sans-serif;
#         max-width: 90%;
#     ">
#         {response_text['output_text']}
#     </div>
#     """
#     st.markdown(html_code, unsafe_allow_html=True)


def get_image_hash(image_file):
    return hashlib.md5(image_file.getvalue()).hexdigest()

def main():
    set_background("tumor.png")
    st.title('🧑‍⚕️ AI Medical Assistant ')

    with st.sidebar:
        st.title('Upload Section')

        image_file = st.file_uploader("Upload Brain MRI Image for Prediction", type=["jpg", "png", "jpeg"])

        if image_file is not None:
            image_hash = get_image_hash(image_file)
            st.image(image_file, caption="Uploaded Brain MRI", use_container_width=True)

            # Only predict if the uploaded image is new
            if st.session_state.get('image_hash') != image_hash:
                with st.spinner("Predicting Tumor Type..."):
                    tumor_name = prediction(image_file)
                    st.session_state['image_hash'] = image_hash
                    st.session_state['tumor_name'] = tumor_name
                    st.success(f'Prediction: {tumor_name.upper()} Tumor ✅')
            else:
                tumor_name = st.session_state['tumor_name']

        else:
            tumor_name = None

        # pdf_docs = st.file_uploader('Upload PDF Files for Context', type=['pdf'], accept_multiple_files=True)
        # if st.button('Submit & Process PDFs'):
        #     with st.spinner("Processing PDFs..."):
        #         raw_text = get_pdf_text(pdf_docs)
        #         chunks = break_text_into_chunks(raw_text)
        #         storing_to_vectordatabase(chunks)
        #         st.success('PDF processing done!')

    # st.image(image_file, caption="Uploaded Brain MRI", use_container_width=True)
    # st.success(f"Prediction: {st.session_state['tumor_name'].upper()} Tumor ✅")
    # set_background("Tumor.png")

    query = st.text_input('Ask your question:')
    if query:
        st.write("Query submitted ✅")
        if tumor_name:
            query = f"Tumor type: {tumor_name}. Question: {query}"
        user_input(query)
    
    


if __name__ == '__main__':
    main()

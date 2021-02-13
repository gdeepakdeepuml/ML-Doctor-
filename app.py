import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as components 
import time
from PIL import Image
import joblib
import torch 
import random 
import json
import torch 

from model import NeuralNet
from nltk_util import tokenization,stemm,bag_of_word

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('intentscovid.json','r') as f:
    intents = json.load(f)

File = 'datacovid.pth'
data = torch.load(File)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
model_state = data['model_state']
all_words = data['all_words']
tags = data ['tags']

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = 'covidbot'
#print("do you help to protect from covid!! let's chat! type 'quit' to get out !! ")

def chatpost(msg):
#while True:
    #sentence = input("You: ")
    print(msg)
    sentence = msg
    if sentence.lower() == "quit":
        print("its work's") 

    sentence = tokenization(sentence)
    x = bag_of_word(sentence,all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.65:
        for intent in intents['intents']:
            if tag == intent['tag']:
                responses = intent['responses']
                
        return (random.choice(responses))
    else:
        answer="I don't understand can you be more clear please!"
        return (answer)



st.set_page_config(page_title="ML-Doctor",page_icon='⚕️',layout="centered",initial_sidebar_state="collapsed",)

def predict_function(to_predict_list, size):
    changed_list = []
    for i in to_predict_list:
        changed_list.append(int(float(i)))
    to_predict = np.array(changed_list).reshape(1,size)
    if size == 10:
        loaded_model = joblib.load("modelpik/model4")
        result = loaded_model.predict(to_predict)
        if result[0] == 1:
            return "sorry ! You are suffering from Liver damage "
        else:
            return "stay Happy! you have no Liver damage"
    elif size == 8:
        pickle_in = open("modelpik/model_diabetes.pkl","rb")
        model=pickle.load(pickle_in)
        prediction_diabetes=model.predict(to_predict)
        if prediction_diabetes[0] == 1:
            return "sorry ! You aresuffering from Diabetes"
        else:
            return "stay Happy! you have no Diabetes"
    elif size == 5:
        loaded_model = joblib.load("modelpik/model_cancer")
        prediction_cancer = loaded_model.predict(to_predict)
        if prediction_cancer[0] == 1:
            return "sorry ! You are suffering from Breast cancer"
        else:
            return "stay Happy! you have no Breast cancer "
    elif size == 11:
        loaded_model = joblib.load("modelpik/mode_heart")
        prediction_heart = loaded_model.predict(to_predict)
        if prediction_heart[0] == 1:
            return "sorry ! You are suffering from Heart Disease"
        else:
            return "stay Happy! you have no Heart Disease"
    elif size == 13:
        mymodel = open("modelpik/atlest_now.pkl", "rb")
        loaded_model = pickle.load(mymodel)
        prediction_covid = loaded_model.predict(to_predict)
        if prediction_covid[0] == 1:
            return "sorry ! You are suffering from Covid-19"
        else:
            return "stay Happy !you have no Covid-19 "
    elif size == 12:
        loaded_model = joblib.load("modelpik/model_kidney")
        prediction_kidney = loaded_model.predict(to_predict)
        if prediction_kidney[0] == 1:
            return "sorry ! You are suffering from Kidney Disease"
        else:
            return "stay Happy !you have no Kidney Disease"
        
# here coms the maln code of API  
## this for Covid-19 Bot            
def get_text():
    input_text = st.text_input("You Question :")
    return input_text 

def main():
    heading = """
    <h1 style="text-align: center;"> ML-Doctor for Easy Diagnosis<h1>
    """
    st.markdown(heading,unsafe_allow_html=True)
    mian_image = Image.open("images/AI-Medicine.png")
    st.image(mian_image,use_column_width=True)
    st.subheader("Details")
    st.write("Welcome to the ML-Doctor a machine learning web app which can make the diagnosis of human diseases easy.")
    st.write("This is mainly based on as the application of the machine learning,meant to be employed in the remote and the downtrodden area.")
    st.subheader("Overview")
    st.write("In todays time we see a lot of the shortage of the doctors in the world especially in India.A lot of people are suffering a lot without the help of the proper medical checkup.Also most of the cases many cases arise leading to dealth due to lack of timely medical checkup")
    st.write("So to cope up with all of those problems this app is designed which would prove its benefits upto much extent.")
    #st.markdown(html_temp_main,unsafe_allow_html=True)
    st.write("-"*10)
    html_main = """
    <h3>Application:</h3>
        <ul style="list-style-type:square"> 
            <li>To remove the dependencies on the doctors</li>
            <li>To help out the poor and helpless people with the normal medical checkup</li>
            <li>To help people avoid paying huge amount to the doctors unnecessarily</li>
            <li>To extend the role of the technology in the medical field</li>
        </ul>
    """
    st.markdown(html_main,unsafe_allow_html=True)
    st.write("-"*10)

    #option = st.radio("choose your test..",["Heart","Diabetes","Breast cancer","liver","kidney","Malaria","pneumonia"])
    st.write("Click on the Disease name to know about it and the process of diagnosis")
    with st.beta_expander("COVID-19 Diagnose"):
        html_temp_liver = """
        <hr style="width:100%;text-align:left;margin-left:0"><hr>
        """
        st.markdown(html_temp_liver,unsafe_allow_html=True)
        
        heading_liver = """
        <h1 style="text-align: center;">COVID-19 Prediction<h1>
        """
        st.markdown(heading_liver,unsafe_allow_html=True)
        
        image = Image.open("images/covid.jpeg")
        
        st.image(image,use_column_width=True)
        
        st.write('Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.')
        st.write('The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette (for example, by coughing into a flexed elbow).')
        st.write('Two kinds of tests are available for COVID-19: viral tests and antibody tests.')
        html_covid = """
        <ul> 
            <li>A <a href="https://www.cdc.gov/coronavirus/2019-ncov/testing/diagnostic-testing.html">viral test</a> tells you if you have a current infection.</li>
            <li>An <a href="https://www.cdc.gov/coronavirus/2019-ncov/testing/serology-overview.html">antibody test</a> might tell you if you had a past infection.</li>
        </ul>
        """
        st.markdown(html_covid,unsafe_allow_html=True)
        
        st.subheader("symptoms:")
        html_covid_sysm = """
        <h5>Most common symptoms:</h5>
        <ul style="list-style-type:square"> 
            <li>Fever</li>
            <li>Dry cough</li>
            <li>Tiredness</li>
        </ul>
        <h5>Serious symptoms:</h5>
        <ul style="list-style-type:square"> 
            <li>Difficulty breathing or shortness of breath</li>
            <li>Chest pain or pressure</li>
            <li>Loss of speech or movement</li>
        </ul>
        """
        st.markdown(html_covid_sysm,unsafe_allow_html=True)
        
        st.write("we have Three types of Test here. Covid Test with normal blood test report need blood test report and full the values in the given feilds and Covid Test with Scanings  needs the X-ray Scan of the Lung Or Scan ")
        st.write("But out of all we will go with Covid Test with normal blood test report need blood test report")

        st.subheader("fill the follwing fileds for the testing process..")
        Age_covid = st.text_input('Your Age')
        WBC_count = st.text_input('WBC (Leukocyte Count)')
        Neutrophils = st.text_input('Neutrophils')
        Lymphocytes = st.text_input('Lymphocytes')
        Monocytes = st.text_input('Monocytes')
        Eosinophils = st.text_input('Eosinophils')
        Basophils = st.text_input('Basophils')
        Platelets = st.text_input('Platelets')
        ALT = st.text_input('ALT (Alanine Amino Transferase)')
        AST= st.text_input('AST (Aspartate Aminotransferase)')
        Proteina_C = st.text_input('Proteina C reativa mg/dL')
        LDH = st.text_input('LDH (Lactate Dehydrogenase)')
        GGT = st.text_input('GGT (Gamma-Glutamyl Transferase)')
        pre_list_covid = [Age_covid, WBC_count , Neutrophils , Lymphocytes, Monocytes, Eosinophils, Basophils, Platelets,ALT ,AST ,Proteina_C,LDH ,GGT]
        result = ""
        #print(pre_list_covid)
        flage = True
        if st.button("Predict Covid"):
            for i in range(len(pre_list_covid)):
                if pre_list_covid[i] == '':
                    st.subheader("Fill the form correctly")
                    flage = False 
                    break
            if flage :    
                result = predict_function(pre_list_covid,len(pre_list_covid))    
        st.subheader(result)
        st.markdown(html_temp_liver,unsafe_allow_html=True)
            
            
    with st.beta_expander("Liver Disease"):
        html_temp_liver = """
        <hr style="width:100%;text-align:left;margin-left:0"><hr>
        """
        st.markdown(html_temp_liver,unsafe_allow_html=True)

        heading_liver = """
        <h1 style="text-align: center;">Liver Disease Prediction<h1>
        """
        st.markdown(heading_liver,unsafe_allow_html=True)
        
        image = Image.open("images/fatty-liver-disease-1200.jpg")
        
        st.image(image,use_column_width=True)
        
        st.write("Liver disease is any disturbance of liver function that causes illness. The liver is responsible for many critical functions within the body and should it become diseased or injured, the loss of those functions can cause significant damage to the body. Liver disease is also referred to as hepatic disease.Liver disease is a broad term that covers all the potential problems that cause the liver to fail to perform its designated functions. Usually, more than 75% or three quarters of liver tissue needs to be affected before a decrease in function occurs.")
        
        st.subheader("Symptoms")
        st.write("Classic symptoms of liver disease include:")
        html_temp_liver = """
	    <h4>Classic symptoms of liver disease include:</h4> 
        <ul> 
            <li>nausea</li>
            <li>vomiting</li>
            <li>right upper quadrant abdominal pain, and</li>
            <li>aundice (a yellow discoloration of the skin due to elevated bilirubin concentrations in the bloodstream).</li>
        </ul>
	    """ 
        st.markdown(html_temp_liver,unsafe_allow_html=True)
        st.subheader("fill the follwing filed for the test process ..!")
        Age_liver = st.text_input("Age")
        Gender = st.text_input("Gender (0 of female and 1 if male)")
        Total_Bilirubin = st.text_input("Total_Bilirubin")
        Direct_Bilirubin = st.text_input("Direct_Bilirubin")
        Alkaline_Phosphotase = st.text_input("Alkaline_Phosphotase")
        Alamine_Aminotransferase = st.text_input("Alamine_Aminotransferase")
        Aspartate_Aminotransferase = st.text_input("Aspartate_Aminotransferase")
        Total_Protiens = st.text_input("Total_Protiens")
        Albumin = st.text_input("Albumin")
        Albumin_and_Globulin_Ratio = st.text_input("Albumin_and_Globulin_Ratio")
       
        pre_list = [Age_liver,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]

        result = ""
        flage = True 
        if st.button("Predict"):
            for i in range(len(pre_list)):
                if pre_list[i] == '':
                    st.subheader("Fill the form correctly")
                    flage = False 
                    break
            if flage:
                result = predict_function(pre_list,len(pre_list))    
            
        st.subheader(result)
        st.markdown(html_temp_liver,unsafe_allow_html=True)
        
    with st.beta_expander("Diabetes Disease"):
        
        html_temp_liver = """
        <hr style="width:100%;text-align:left;margin-left:0"><hr>
        """
        st.markdown(html_temp_liver,unsafe_allow_html=True)

        heading_liver = """
        <h1 style="text-align: center;">Diabetes Disease Prediction<h1>
        """
        st.markdown(heading_liver,unsafe_allow_html=True)
        
             
        image = Image.open("images/Diabetes2.jpg")
        
        st.image(image, use_column_width=True)
        st.write("Diabetes is a serious condition that causes higher than normal blood sugar levels. Diabetes occurs when your body cannot make or effectively use its own insulin, a hormone made by special cells in the pancreas called islets (eye-lets). Insulin serves as a “key” to open your cells, to allow the sugar (glucose) from the food you eat to enter. Then, your body uses that glucose for energy. ")
        st.write("Diabetes causes vary depending on your genetic makeup, family history, ethnicity, health and environmental factors. There is no common diabetes cause that fits every type of diabetes as the causes of diabetes vary depending on the individual and the type.")        
        st.subheader("What health problems can people with diabetes develop?")
        
        html_temp_diaetes = """
	    <h4>Over time, high blood glucose leads to problems such as</h4> 
        <ul> 
            <li>heart disease</li>
            <li>stroke</li>
            <li>kidney disease</li>
            <li>eye problems</li>
            <li>dental disease</li>
            <li>nerve damage</li>
            <li>foot problems</li>
        </ul>
	    """    
        st.markdown(html_temp_diaetes,unsafe_allow_html=True)
        
        st.subheader("fill the form for the status of Diabetes...!!!")
        
        pregnancies = st.text_input("pregnancies")
        Glucose = st.text_input("Glucose")
        BloodPressure = st.text_input("BloodPressure")
        skinThickness = st.text_input("skinThickness")
        insulen = st.text_input("insulen")
        bmi = st.text_input("BMI")
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction")
        age = st.text_input("age")
        pre_list_diabetes = [pregnancies,Glucose,BloodPressure,skinThickness,insulen,bmi,DiabetesPedigreeFunction,age]
        result=""
        # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
        # after hitting the button the prediction process will go on and then we print the success message by "st.success()"
        flage = True
        if st.button("Predict diabetes"):
            for i in range(len(pre_list_diabetes)):
                if pre_list_diabetes[i] == '':
                    st.subheader("Fill the form correctly")
                    flage = False 
                    break
            if flage:
                result=predict_function(pre_list_diabetes,len(pre_list_diabetes))
        
        st.subheader(result)
        st.markdown(html_temp_liver,unsafe_allow_html=True)
        
        
    with st.beta_expander("Breast Cancer Disease"):
        
        html_temp_liver = """
        <hr style="width:100%;text-align:left;margin-left:0"><hr>
        """
        st.markdown(html_temp_liver,unsafe_allow_html=True)

        heading_liver = """
        <h1 style="text-align: center;">Breast Cancer Prediction<h1>
        """
        st.markdown(heading_liver,unsafe_allow_html=True)
             
        image = Image.open("images/cancer.jpeg")
        
        st.image(image, use_column_width=True)
        st.write("Breast cancer is cancer that develops from breast tissue. Signs of breast cancer may include a lump in the breast, a change in breast shape, dimpling of the skin, fluid coming from the nipple, a newly inverted nipple, or a red or scaly patch of skin.")
        st.subheader("Symptoms")
        html_temp_cancer= """
	    <h4>Signs and symptoms of breast cancer may include:</h4> 
        <ul> 
            <li>A breast lump or thickening that feels different from the surrounding tissue</li>
            <li>Change in the size, shape or appearance of a breast</li>
            <li>Changes to the skin over the breast, such as dimpling</li>
            <li>A newly inverted nipple</li>
            <li>Peeling, scaling, crusting or flaking of the pigmented area of skin surrounding the nipple (areola) or breast skin</li>
            <li>Redness or pitting of the skin over your breast, like the skin of an orange</li>
        </ul>
	    """    
        st.markdown(html_temp_cancer,unsafe_allow_html=True)
        
        st.subheader("fill the form for the status of Breast cancer ...!!!")
        
        mean_radius = st.text_input("mean_radius")
        mean_texture = st.text_input("mean_texture")
        mean_perimeter = st.text_input("mean_perimeter")
        mean_area = st.text_input("mean_area")
        mean_smoothness = st.text_input("mean_smoothness")

        pre_list_cancer = [mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]
        result=""
        # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
        # after hitting the button the prediction process will go on and then we print the success message by "st.success()"
        flage = True
        if st.button("Predict Breast Cancer"):
            for i in range(len(pre_list_cancer)):
                if pre_list_cancer[i] == '':
                    st.subheader("Fill the form correctly")
                    flage = False 
                    break
            if flage:
                result=predict_function(pre_list_cancer,len(pre_list_cancer))
        
        st.subheader(result)
        st.markdown(html_temp_liver,unsafe_allow_html=True)
        
            
    with st.beta_expander("Heart Disease"):
        html_temp_liver = """
            <hr style="width:100%;text-align:left;margin-left:0"><hr>
            """
        st.markdown(html_temp_liver,unsafe_allow_html=True)

        heading_liver = """
            <h1 style="text-align: center;">Heart Disease Prediction<h1>
            """
        st.markdown(heading_liver,unsafe_allow_html=True)
                
        image = Image.open("images/Heart-diseases.jpg")
            
        st.image(image, use_column_width=True)
        st.write("Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects), among others.")
        st.write("The term 'heart disease' is often used interchangeably with the term 'cardiovascular disease.' Cardiovascular disease generally refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack, chest pain (angina) or stroke. Other heart conditions, such as those that affect your heart's muscle, valves or rhythm, also are considered forms of heart disease.")
        st.write("Many forms of heart disease can be prevented or treated with healthy lifestyle choices.")
        st.subheader("Symptoms")
        html_temp_cancer= """ 
            <ul> 
                <li>Chest pain, chest tightness, chest pressure and chest discomfort (angina)</li>
                <li>Shortness of breath</li>
                <li>Pain, numbness, weakness or coldness in your legs or arms if the blood vessels in those parts of your body are narrowed</li>
                <li>Pain in the neck, jaw, throat, upper abdomen or back</li>
            </ul>
            """    
        st.markdown(html_temp_cancer,unsafe_allow_html=True)
            
        st.subheader("fill the form for the status of Breast cancer ...!!!")
            
        age_heart = st.text_input("Age:")
        Gender = st.text_input("Sex(1:Male,0:Female):")
        cp = st.text_input("chest pain type  (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)")
        Trestbps = st.text_input("resting blood pressure")
        chol = st.text_input("serum cholestoral in mg/dl")
        restecg = st.text_input("resting electrocardiographic results (values 0,1,2)")
        thalach = st.text_input("maximum heart rate achieved")
        exang = st.text_input("exercise induced angina(1 = yes; 0 = no)")
        oldpeak = st.text_input("oldpeak = ST depression induced by exercise relative to rest")
        slop = st.text_input("the slope of the peak exercise ST segment(Value 1: upsloping, Value 2: flat, Value 3: downsloping)")
        thal = st.text_input("thal: 1 = normal; 2 = fixed defect; 3 = reversable defect",)

        pre_list_cancer = [age_heart,Gender,cp,Trestbps,chol,restecg,thalach,exang,oldpeak,slop,thal]
        result=""
            # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
            # after hitting the button the prediction process will go on and then we print the success message by "st.success()"
        flage = True
        if st.button("Predict Heart Disease "):
            for i in range(len(pre_list_cancer)):
                if pre_list_cancer[i] == '':
                    st.subheader("Fill the form correctly")
                    flage = False 
                    break
                if flage:
                    result=predict_function(pre_list_cancer,len(pre_list_cancer))
            
        st.subheader(result)
        st.markdown(html_temp_liver,unsafe_allow_html=True)
        
    
    with st.beta_expander("Kidney Disease"):
        html_temp_liver = """
            <hr style="width:100%;text-align:left;margin-left:0"><hr>
            """
        st.markdown(html_temp_liver,unsafe_allow_html=True)

        heading_liver = """
            <h1 style="text-align: center;">Kidney Disease Prediction<h1>
            """
        st.markdown(heading_liver,unsafe_allow_html=True)
                
        image = Image.open("images/kidney.jpg")
        st.image(image,use_column_width=True)
        
        st.write("The kidneys play key roles in body function, not only by filtering the blood and getting rid of waste products, but also by balancing the electrolyte levels in the body, controlling blood pressure, and stimulating the production of red blood cells.")
        st.write("The kidneys are located in the abdomen toward the back, normally one on each side of the spine. They get their blood supply through the renal arteries directly from the aorta and send blood back to the heart via the renal veins to the vena cava. (The term 'renal' is derived from the Latin name for kidney.)")
        st.subheader("Symptoms:")
        html_temp_kidney= """ 
            <ul> 
                <li>Lethargy</li>
                <li>Weakness</li>
                <li>Shortness of breath</li>
                <li>Generalized swelling (edema)</li>
                <li>Generalized weakness due to anemia</li>
                <li>Loss of appetite</li>
                <li>Lethargy</li>
                <li>Fatigue</li>
                <li>Congestive heart failure</li>
                <li>Metabolic acidosis</li>
                <li>High blood potassium (hyperkalemia)</li>
                <li>Fatal heart rhythm disturbances (arrhythmias) including ventricular tachycardia and ventricular fibrillation</li>
                <li>Rising urea levels in the blood (uremia) may lead to brain encephalopathy, pericarditis (inflammation of the heart lining), or low calcium blood levels (hypocalcemia)</li>
            </ul>
            """    
        st.markdown(html_temp_kidney,unsafe_allow_html=True)
        
        st.markdown(html_temp_liver,unsafe_allow_html=True)
        st.subheader("Fill the form for the status of Kidney Disease..")
        
        age_kind = st.text_input("Your age:")
        B_p = st.text_input("BP - blood pressure")
        a_l = st.text_input("AL - albumin")
        pcc = st.text_input("PCC - pus cell clumps")
        bgr = st.text_input("BGR - blood glucose random")
        bu = st.text_input("BU - blood urea")
        su = st.text_input("SU - serum creatinine")
        hemo = st.text_input("Hemo - hemoglobin")
        pcv = st.text_input("pcv - packed cell volume")
        htn = st.text_input("htn - hypertension")
        dm = st.text_input("dm - diabetes mellitus")
        appet = st.text_input("appet - appetite")
        
        pre_list_kidney = [age_kind,B_p,a_l,pcc,bgr,bu,su,hemo,pcv,htn,dm,appet]
        result=""
            # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
            # after hitting the button the prediction process will go on and then we print the success message by "st.success()"
        flage = True
        if st.button("Predict Kidney Disease "):
            for i in range(len(pre_list_kidney)):
                if pre_list_kidney[i] == '':
                    st.subheader("Fill the form correctly")
                    flage = False 
                    break
            if flage:
                result=predict_function(pre_list_kidney,len(pre_list_kidney))
            
        st.subheader(result)
        st.markdown(html_temp_liver,unsafe_allow_html=True)
    
    st.subheader("The Chat-Bot for COVID-19")
    st.write("COVID-19 affects different people in different ways. Most infected people will develop mild to moderate illness and recover without hospitalization. here the CovidBot which can help to answer any question about COVID-19")
    
    user_input = get_text()
    result = chatpost(user_input)

    #user_input_list.append(user_input)
    #bible_ans.append(Initialize botchatpost(user_input))
    if st.button("send"):
        st.text_area("Covid-19 Bot Say's :",value=result,
                    height=40,max_chars=None,key=None)
    else:
        st.text_area("Covid-19 Bot Say's :",value="Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. i can help You to be safe and answer your question",
                    height=30,max_chars=None,key=None)
    
    #####################
    st.subheader("Creator:")
    st.write('G Deepak')
    st.write('Aspiring Data Scientist')

        
if __name__=='__main__':
    main()

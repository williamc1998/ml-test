import datetime
import pickle
from pathlib import Path
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import json
import ssl,os,urllib
import altair as alt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


st.set_page_config(page_title='API call', page_icon=':rocket:')

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottieurl('https://assets5.lottiefiles.com/private_files/lf30_jyndijva.json')

with st.container():
    image = Image.open('/Users/william/Downloads/Agilisys-Logo-Black-RGB.png')
    st.image(image)
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('ML backend demonstration')
        st.title('Azure dependant scoring')
        st.write('Cloud deployed logistic regression API called on this page to predict JSON data')
    with right_column:
        st_lottie(lottie_coding, height=300, key='coding')
        


st.write("---")

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('Input JSON file of correct format')
        uploaded_file = st.file_uploader("Submit file")
        if uploaded_file is not None:
            with left_column:
                st.write('File recieved, predicting samples....')
                print('acceptedfile')
                def allowSelfSignedHttps(allowed):# bypass the server certificate verification on client side
                    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
                        ssl._create_default_https_context = ssl._create_unverified_context
                allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.
                        # Request data goes here
                        # The example below assumes JSON formatting which may be updated
                        # depending on the format your endpoint expects.
                        # More information can be found here:
                        # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
                data = uploaded_file

                body = data


                url = 'http://d30c0598-3fa6-4e82-874b-4862c5bf3a3c.uksouth.azurecontainer.io/score'


                headers = {'Content-Type':'application/json'}

                req = urllib.request.Request(url, body, headers)

                try:
                    response = urllib.request.urlopen(req)

                    result = response.read()
                    print(result)
                    diabetic_count = 0
                    result_dict = json.loads(result.decode('utf-8'))
                    with st.expander("See breakdown"):
                        for i,j in enumerate(result_dict['predict_proba']):
                            val = result_dict['predict_proba'][i][1]
                            if val>=0.5:
                                diabetic_count+=1
                                st.write(f'sample {i+1} predicted diabetic with probability {val}')
                            else:
                                st.write(f'sample {i+1} predicted non-diabetic with probability {val}')
                except urllib.error.HTTPError as error:
                    print("The request failed with status code: " + str(error.code))

                            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
                    print(error.info())
                    print(error.read().decode("utf8", 'ignore'))
                        
            with right_column:
                percentage = ((diabetic_count/len(result_dict['predict_proba'])*100))
                dicto = {'Count':[diabetic_count,len(result_dict['predict_proba'])]}
                chart_data = np.array([diabetic_count,len(result_dict['predict_proba'])])
                st.markdown('''
<style>
/*center metric label*/
[data-testid="stMetricLabel"] > div:nth-child(1) {
    justify-content: center;
}

/*center metric value*/
[data-testid="stMetricValue"] > div:nth-child(1) {
    justify-content: center;
}
</style>
''', unsafe_allow_html=True)
                st.metric(label='predicted diabetic',value=f'{percentage}%')
                    
                fig,ax = plt.subplots()
                ax.pie(chart_data,explode=(0.1,0.1),labels=['Diabetic','Non-diabetic'],
                       shadow=False,startangle=90)
                st.pyplot(fig)

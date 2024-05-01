from langchain.llms import Clarifai
from clarifai_utils.modules.css import ClarifaiStreamlitCSS

import streamlit as st
from PIL import Image
import pandas as pd
import base64
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc

stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())

st.title("Classifier Demo")

st.header("Step 1: Enter an API key")
key = st.text_input("API Key")

if key == '':
    st.warning("An API Key has not been entered.")
    st.stop()
else:
    st.write("API Key has been uploaded.")
    
file_data = st.file_uploader("Upload Image")
if file_data == None:
    st.warning("File needs to be uploaded.")
    st.stop()
else:
    image = Image.open(file_data)
    st.image(image)
    

from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

# Create an application on Clarifai and put its ID here.
YOUR_APPLICATION_ID = "4136e8da26f0434b809473f237b0daf7"

# This is how you authenticate.
metadata = (("authorization", f"Key {key}"),)

request = service_pb2.PostModelOutputsRequest(
    # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
    model_id="aaa03c23b3724a16a56b629203edc62c",
    user_app_id=resources_pb2.UserAppIDSet(app_id=YOUR_APPLICATION_ID),
    inputs=[
        resources_pb2.Input(
            data=resources_pb2.Data(image=resources_pb2.Image(base64=file_data.getvalue()))
        )
    ],
)
response = stub.PostModelOutputs(request, metadata=metadata)

if response.status.code != status_code_pb2.SUCCESS:
    print(response)
    raise Exception(f"Request failed, status code: {response.status}")
names = []
confidences = []
for concept in response.outputs[0].data.concepts:
    names.append(concept.name)
    confidences.append(concept.value)
    # st.write( "%12s: %.2f" % (concept.name, concept.value))

df = pd.DataFrame({
    "Concept Name" : names,
    "Concept Confidences" : confidences
})

st.dataframe(df)

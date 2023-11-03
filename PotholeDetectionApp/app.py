# Python In-built packages
from pathlib import Path
import PIL
from PIL import Image
# External packages
import streamlit as st
import base64

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_icon="🚧",
    page_title="Pothole Detection App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.markdown("<h1 style='text-align: center;'> Pothole Detection App </h1>", unsafe_allow_html=True)
st.write("An innovative pothole detection app using YOLOv8, deep learning, capable of analysing pothole severity, and integrating it into a user-friendly website for road assessment.")


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Sidebar
st.sidebar.header("ML Model Config")


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 0, 100, 50)) / 100


model=None
# loading weight file
weight_file="C:/Users/miman/potholetry1/PotholeDetectionApp/weights/best.pt"
#weight_file = st.sidebar.file_uploader("Upload Model Weight File", type=("pt"))
set_background('bgimages/background1.jpg')
# loading weight file and creat file
if weight_file:
    model_path = Path(weight_file)
    try:
        model = helper.load_model(model_path)
        st.success("Model successfully loaded.")
    except Exception as ex:
        st.error(f"Unable to load model. Error: {ex}")


st.sidebar.header("Image/Video Config")
source_radio = st.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    helper.show_model_not_loaded_warning(model)
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                set_background('bgimages/background1.jpg')
            else:
                uploaded_image = PIL.Image.open(source_img)
                
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    
    with col2:
        if source_img is None:
            pass
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                
                # try:
                
                with st.expander("From this angle: "):
                    i=0
                    for box in boxes:
                        i += 1
                        data = box.data  # Access the 'data' attribute for the current box
                        x1 = data[0, 0]  # x1 coordinate
                        y1 = data[0, 1]  # y1 coordinate
                        x2 = data[0, 2]  # x2 coordinate
                        y2 = data[0, 3]  # y2 coordinate
                        width= x2-x1
                        height=y2-y1
                        area= width*height
                        perimeter=2*(width+height)
                        
                        st.write(f"Area of detected pothole {i}: {area:.2f}")
                        st.write(f"Perimeter of detected pothole {i}: {perimeter:.2f}")
                    st.write("The number of potholes detected is:", i)
                # except Exception as ex:
                #     # st.write(ex)
                #     st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    
    helper.show_model_not_loaded_warning(model)
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")

import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


def set_custom_theme(primary_color):
    # Custom CSS to set the theme color
    custom_css = f"""
    <style>
    .reportview-container .main .block-container {{
        max-width: 1000px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    .reportview-container .main {{
        color: black;
        background-color: white;
    }}
    .css-1v3fvcr {{
        color: {primary_color};
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# set_custom_theme("#ff6347")
tf.executing_eagerly()
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title="Neural Style Transfer", 
    layout="wide",
)



st.header("NEURAL STYLE TRANSFER")

st.sidebar.markdown("SEMESTER 8 PROJECT")
st.sidebar.markdown("1. Pankaj Singh Kanyal _ 20BCS6668")
st.sidebar.markdown("2. Sheikh Hussain _ 20BCS6628")

def load_image(image_buffer, image_size=(512, 256)):
    img = plt.imread(image_buffer).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
      img = img / 255.
    if len(img.shape) == 3:
      img = tf.stack([img, img, img], axis=-1)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def export_image(tf_img):
	pil_image = Image.fromarray(np.squeeze(tf_img*255).astype(np.uint8))
	buffer = BytesIO()
	pil_image.save(buffer, format="PNG")
	byte_image = buffer.getvalue()
	return byte_image

def st_ui():
    image_upload1 = st.sidebar.file_uploader("Content Image",type=["jpeg", "png", "jpg"], accept_multiple_files=False, key=None, help="Upload the image whom you want to style")
    image_upload2 = st.sidebar.file_uploader("Style Image",type=["jpeg", "png", "jpg"], accept_multiple_files=False, key=None, help="Upload the image whose style you want")
    col1,col2,col3= st.columns(3)
    
    st.sidebar.title("Transfer Style")
    st.sidebar.markdown("Your personal neural style transfer")

    with st.spinner("Loading content image.."):
        if image_upload1 is not None:
            col1.header("Content Image")
            col1.image(image_upload1,use_column_width=True)
            original_image = load_image(image_upload1)
        else:
            original_image = load_image("1.jpg")
    
    with st.spinner("Loading style image.."):
        if image_upload2 is not None:
            col2.header("Style Image")
            col2.image(image_upload2,use_column_width=True)
            style_image = load_image(image_upload2)
            # style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
            style_image = tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=(1,1),padding="VALID")(style_image)
        else:
            style_image = load_image("2.jpg")
            # style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
            style_image = tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=(1,1),padding="VALID")(style_image)

    

    if st.sidebar.button(label="Start Styling"):
        if image_upload2 and image_upload1:
            with st.spinner('Generating Stylized image ...'):

                # Load image stylization module.
                stylize_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

                results = stylize_model(tf.constant(original_image), tf.constant(style_image))
                stylized_photo = results[0]
                col3.header("Final Image")
                col3.image(np.array(stylized_photo))
                st.download_button(label="Download Final Image", data=export_image(stylized_photo), file_name="stylized_image.png", mime="image/png")

        else:
            st.sidebar.markdown("Please upload images...")
            
            
if __name__ == "__main__":
    model_path = r'D:\Sem 8\Capstone Project\Neural Style Transfer\model'
    
    st_ui()


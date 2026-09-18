import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from pathlib import Path
import csv
import io
import streamlit as st
from PIL import Image, ImageOps
from inference import TomatoSystem

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Tomato Lens', page_icon='🍅', layout='wide')

@st.cache_resource
def load_system():
    return TomatoSystem(ROOT / 'models')

st.title('🍅 Tomato Lens')
st.write('Upload a leaf photo to check its similarity to known tomato images and view the disease prediction.')
try:
    with st.spinner('Loading saved models…'):
        system = load_system()
except Exception:
    st.error('The saved models could not load. Check the deployment logs and confirm all model files were uploaded.')
    raise

left, right = st.columns(2)
with left:
    photo = st.file_uploader('Choose a leaf image', type=['jpg', 'jpeg', 'png', 'webp'])
    st.caption('Up to 10 MB. Images are processed in memory and are not saved by this app.')
    valid = False
    if photo is not None:
        data = photo.getvalue()
        if len(data) > 10 * 1024 * 1024:
            st.error('Please choose an image smaller than 10 MB.')
        else:
            try:
                with Image.open(io.BytesIO(data)) as image:
                    if image.width * image.height > 20_000_000:
                        raise ValueError('Please choose an image smaller than 20 megapixels.')
                    st.image(ImageOps.exif_transpose(image).convert('RGB'), caption=photo.name)
                valid = True
            except (OSError, ValueError, Image.DecompressionBombError):
                st.error('This image could not be opened or is too large. Try another JPG or PNG.')
    analyze = st.button('Analyze leaf', type='primary', disabled=not valid)

with right:
    st.subheader('Screening and prediction')
    if analyze:
        try:
            with st.spinner('Checking your image…'):
                result = system.predict(data)
            a, b = st.columns(2)
            a.metric('Feature distance', f"{result['distance']:.4f}")
            b.metric('Acceptance limit', f"{result['threshold']:.4f}")
            st.caption('Smaller distance means greater similarity. Distance is not a percentage.')
            if result['accepted']:
                st.success('Passed similarity screening')
                st.subheader(result['disease'])
                st.metric('Disease confidence', f"{result['confidence']:.1%}")
                for item in result['probabilities']:
                    st.write(f"{item['label']}: **{item['score']:.1%}**")
                    st.progress(max(0., min(1., item['score'])))
                st.info('Confidence is the model’s score for this image, not overall accuracy or a guarantee that the diagnosis is correct.')
            else:
                st.warning('Outside the learned tomato feature range. No disease prediction was made. Genuine tomato leaves can also be rejected.')
        except ValueError as error:
            st.error(str(error))
    else:
        st.write('Choose an image and select **Analyze leaf** to see its result.')

st.divider()
st.subheader('How to read these results')
st.write('The first stage checks feature similarity. Only accepted images reach the three-class disease classifier. Other plants or objects can pass this check, and genuine tomato leaves can fail it.')
with st.expander('Saved test results and limitations'):
    rows = list(csv.DictReader((ROOT / 'models/known_acceptance.csv').open()))
    st.table([{'Group': r['group'].replace('_', ' '),
               'Accepted / total': f"{r['accepted']} / {r['total']}",
               'Acceptance': f"{float(r['acceptance_rate']):.1%}"}
              for r in rows if r['split'] == 'test'])
    st.write('These are known-tomato acceptance rates, not disease accuracy or rejection rates for cars and faces. Late Blight acceptance was 61/75 (81.3%).')
    st.write('The previously reported 88.44% disease accuracy is a historical notebook result, not independently verified by this bundle. The 95% calibration target does not guarantee 95% acceptance on new images.')
st.caption('Three-class research demonstration · Saved models only · No retraining')

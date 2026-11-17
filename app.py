import os, re, json
import numpy as np, pandas as pd, cv2, torch
import streamlit as st
from PIL import Image
from rapidfuzz import process, fuzz
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = 'Avni2028/trocr-prescription-bd'
THRESH = 60
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_all():
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(device).eval()
    drugs = pd.read_csv(os.path.join(HERE, 'drug_list.csv'))['medicine_name'].astype(str).tolist()
    info = json.load(open(os.path.join(HERE, 'drug_info.json')))
    return processor, model, drugs, info

def deskew(gray, max_angle=8, step=0.5):
    inv = 255 - cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    h, w = inv.shape
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(inv, M, (w, h), flags=cv2.INTER_NEAREST)
        s = float(np.var(r.sum(axis=1)))
        if s > best_score:
            best, best_score = a, s
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def clean(pil):
    gray = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return deskew(gray)

def best_match(pred, drugs):
    m = process.extractOne(pred.strip(), drugs, scorer=fuzz.ratio, processor=lambda s: s.lower())
    return m[0], float(m[1])

def answer(q, med, info):
    q = q.lower()
    if med is None:
        return 'I am not sure which medicine this is, so please upload a clearer image first.'
    d = info.get(med, {})
    if re.search(r'\b(generic|salt|composition)\b', q):
        key, label = 'generic_name', 'Generic name'
    elif re.search(r'\b(used for|use|uses|purpose|treat\w*)\b', q):
        key, label = 'used_for', 'Used for'
    elif re.search(r'\b(form|tablet|syrup|capsule)\b', q):
        key, label = 'form', 'Form'
    elif re.search(r'\b(what is|about|describe|description|tell me|information)\b', q):
        key, label = 'description', 'About'
    else:
        return ('I can answer: what it is, what it is used for, its form, and its generic name. '
                'For anything else, please ask your doctor or pharmacist.')
    v = str(d.get(key, '')).strip()
    if not v or v.startswith('['):
        return 'I do not have that information in my drug file. Please ask your pharmacist.'
    return f'{label}: {v}'

st.set_page_config(page_title='Prescription Chatbot')
st.title('Handwritten Prescription Recognition Chatbot')
st.warning('Demo only. Always confirm the result with a doctor or pharmacist. No dosage advice is given.')

processor, model, drugs, info = load_all()

file = st.file_uploader('Upload an image of a handwritten medicine name', type=['jpg', 'jpeg', 'png'])
if file is not None:
    key = (file.name, file.size)
    if st.session_state.get('file_key') != key:
        pil = Image.open(file)
        gray = clean(pil)
        pv = processor(images=Image.fromarray(gray).convert('RGB'), return_tensors='pt').pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=24)
        raw = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        name, score = best_match(raw, drugs)
        st.session_state.update(file_key=key, pil=pil, gray=gray, raw=raw, name=name,
                                score=score, med=(name if score >= THRESH else None),
                                messages=[])
    c1, c2 = st.columns(2)
    c1.image(st.session_state['pil'], caption='Uploaded image', width=260)
    c2.image(st.session_state['gray'], caption='After preprocessing', width=260)
    st.write('**Raw prediction:**', st.session_state['raw'])
    if st.session_state['med']:
        st.success(f"Recognised medicine: **{st.session_state['med']}**  (confidence {st.session_state['score']:.0f}%)")
    else:
        st.error(f"Not sure (best match: {st.session_state['name']}, confidence {st.session_state['score']:.0f}%). Please try a clearer image.")

    st.subheader('Ask about this medicine')
    for m in st.session_state['messages']:
        with st.chat_message(m['role']):
            st.write(m['text'])
    q = st.chat_input('Ask: what is it? what is it used for? what form is it?')
    if q:
        a = answer(q, st.session_state['med'], info)
        st.session_state['messages'] += [{'role': 'user', 'text': q}, {'role': 'assistant', 'text': a}]
        st.rerun()

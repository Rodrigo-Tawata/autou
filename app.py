import streamlit as st
import pdfplumber
import joblib
import os
from openai import OpenAI

st.set_page_config(page_title="Classificador de Emails - AutoU Case", layout="centered")
st.title("Classificador de Emails — AutoU (MVP)")

uploaded = st.file_uploader("Envie .pdf ou .txt (ou cole o email abaixo)", type=["pdf","txt"])
text_input = st.text_area("Ou cole o texto do email aqui")

# -------- EXTRAÇÃO DE TEXTO --------
def extract_text_from_file(f):
    if f is None:
        return ""
    if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(f) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
        except:
            return ""
    else:
        try:
            return f.read().decode("utf-8")
        except:
            return str(f.read())

# -------- GERAR RESPOSTA COM IA --------
def gerar_resposta_ai(email_text, categoria):
    """
    Gera resposta automática usando OpenAI GPT, contextualizando ao conteúdo do email.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Use variável de ambiente
        if not openai.api_key:
            st.warning("OPENAI_API_KEY não encontrada! Usando template padrão.")
            return None

        prompt = f"""
        Você é um assistente que responde emails de forma clara, cordial e objetiva.
        Classifique este email como {categoria} e gere uma resposta adequada, levando em consideração o conteúdo real do email.
        
        Email: "{email_text}"
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.5,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Erro ao gerar resposta AI: {e}")
        return None

# -------- OBTER TEXTO --------
if uploaded:
    raw_text = extract_text_from_file(uploaded)
elif text_input.strip():
    raw_text = text_input
else:
    raw_text = ""

# -------- PROCESSAR EMAIL --------
if st.button("Processar") and raw_text.strip():
    st.subheader("Texto extraído")
    st.write(raw_text[:5000])

    model_path = "models/classifier.joblib"
    if not os.path.exists(model_path):
        st.error("Modelo não encontrado. Rode train.py primeiro.")
    else:
        pipeline = joblib.load(model_path)
        pred = pipeline.predict([raw_text])[0]
        probs = pipeline.predict_proba([raw_text])[0]
        conf = max(probs)

        # Exibir categoria
        if pred == "Produtivo":
            st.success(f"**Categoria:** {pred}  —  Confiança: {conf:.2f}")
        else:
            st.info(f"**Categoria:** {pred}  —  Confiança: {conf:.2f}")

        # Gerar resposta
        response_ai = gerar_resposta_ai(raw_text, pred)
        if response_ai:
            response = response_ai
        else:
            # fallback template
            if pred == "Produtivo":
                if "não consegui" in raw_text.lower():
                    response = "Olá, entendemos que houve um impedimento. Avise-nos se precisar de ajuda ou suporte."
                else:
                    response = "Olá, recebemos sua solicitação e estamos verificando. Retornaremos em breve."
            else:
                response = "Olá, obrigado pela mensagem! Se precisar de algo mais específico, nos avise."

        st.subheader("Resposta sugerida")
        st.text_area("Resposta automática", value=response, height=160)

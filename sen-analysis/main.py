from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import streamlit as st

# Few-shot öğrenme için örnekler
examples = [
    {
        "text": "The product exceeded my expectations and works flawlessly!",
        "sentiment": "Pozitif",
        "explanation": "The text expresses satisfaction and enthusiasm with terms like 'exceeded my expectations' and 'flawlessly', indicating a positive sentiment."
    },
    {
        "text": "Hizmet berbattı ve personel çok kabaydı.",
        "sentiment": "Olumsuz",
        "explanation": "'Korkunç' ve 'kaba' gibi kelimelerin kullanılması açıkça memnuniyetsizliği belirtir, bu da olumsuz bir duygudur."
    },
    {
        "text": "Bulasik makinelerinin vazgeçilmezi Butun bulasiklar tertemiz ve parlak olarak cikiyor Icim rahat.",
        "sentiment": "Pozitif",
        "explanation": "üründen memnun kaldığını,gönül rahatlığıyla kullandığını belirtiyor."
    },

]
# Örnekler için şablon
example_template = """
Text: {text}
Sentiment: {sentiment}
Explanation: {explanation}
"""

# Bu şablonla prompt oluşturuluyor
example_prompt = PromptTemplate(
    input_variables=["text", "sentiment", "explanation"],
    template=example_template
)

# FewShotPromptTemplate oluşturma
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix = "You are an expert sentiment analysis model. I want you to output the model in the language used for user input. Analyze the sentiment of the text provided. Let the sentiment in the model depend on the model's judgment." ,
    suffix="\nNow analyze the following text.\n\nText: {text}\nSentiment:\nExplanation:",
    input_variables=["text"]  # 'text' değişkeni burada kullanılacak
)

# LLMChain ve OpenAI bağlantısı
def llm_chain(text_to_analyze):
    llm = OpenAI(temperature=0.4)
    chain = LLMChain(llm=llm, prompt=few_shot_prompt)
    
    
    result = ''
    for chunk in chain.stream(text_to_analyze):  # stream() generator’dür.
        print(chunk)  
        if isinstance(chunk, dict) and "text" in chunk:
            result += chunk["text"]  
        else:
            result += str(chunk) 

    return result  

# Streamlit arayüzü
def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
    
    # Sidebar için dil seçeneği ekleniyor
    language = st.sidebar.selectbox("Select Language", ("English", "Türkçe"))
    
    # Dil tercihini session_state'de saklıyoruz
    if "language" not in st.session_state:
        st.session_state.language = "English"  # Varsayılan dil İngilizce
    else:
        st.session_state.language = language

    # Sidebar'da kullanıcının sorduğu soruları kaydetmek
    if "queries" not in st.session_state:
        st.session_state.queries = []

    # Sidebar'da sorduğu soruları listele
    if st.session_state.language == "Türkçe":
        st.sidebar.write("Sorularınız:")
    else:
        st.sidebar.write("Your Queries:")

    for query in st.session_state.queries:
        st.sidebar.write(query)

    # Kullanıcıya soruları silme seçeneği sunuluyor.
    if st.sidebar.button("Clear All Questions"):
        st.session_state.queries = []  # Tüm soruları siler

    # Başlık
    if st.session_state.language == "Türkçe":
        st.title("Duygu Analizi")
        st.markdown("""
        Verilen bir metnin duygusunu analiz edin.
        """)
    else:
        st.title("Sentiment Analysis")
        st.markdown("""
        Analyze the sentiment of a given text.
        """)

    # Kullanıcıdan metin girişi al
    text_to_analyze = st.text_input("Enter text to analyze:" if st.session_state.language == "English" else "Analiz edilecek metni girin:")

    # Analiz butonuna tıklanması durumunda
    if st.button("Analyze" if st.session_state.language == "English" else "Analiz Et"):
        if text_to_analyze:
            # Kullanıcıyı kaydet
            st.session_state.queries.append(text_to_analyze)

            with st.spinner("Analyzing..." if st.session_state.language == "English" else "Analiz ediliyor..."):
                result = llm_chain(text_to_analyze)

            # Dinamik çıktı formatı
            if st.session_state.language == "Türkçe":
                result_display = f"""
                **Duygu Durumu**: {result.split('\n')[0]}\n
                **Açıklama**: {result.split('\n')[1]}
                """
            else:
                result_display = f"""
                **Sentiment**: {result.split('\n')[0]}\n
                **Explanation**: {result.split('\n')[1]}
                """

            # Kullanıcı mesajı ve sonuçları ekrana yazdırma
            st.chat_message("user").write(text_to_analyze)
            st.chat_message("assistant").write(result_display)

if __name__ == "__main__":
    main()

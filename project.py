import streamlit as st
import joblib
import pandas as pd

spam = joblib.load("spam_clas.pkl")
news = joblib.load("news_cat.pkl")
lang = joblib.load("lang_det.pkl")
restro = joblib.load("restro_review.pkl")

st.set_page_config(layout="wide")
st.markdown("""
            <style>
            .stApp {background-color:lightgrey 
            </style>
""", unsafe_allow_html=True)

st.markdown("""
            <h1 style='background-color:black; color:orange;font-size:40px ;padding:10px; border-radius:10px; text-align:center; margin-bottom:20px'>
                🔍 LENS eXpert - Multi Domain Text Analysis
            </h1>
""", unsafe_allow_html=True)

#st.title("LENS eXpert - NLP SUits")
tab1,tab2,tab3,tab4 = st.tabs(["📩 Spam Classification","🗣️ Language Detection","📰 News Classification","⭐️ Restaurant Review"])       
with tab1:
    msg1 = st.text_input("Enter any Messages",)
    if st.button("🕵️ Predict"):
        pred = spam.predict([msg1])
        if pred[0]==0:
            #print("Spam")
            st.image("spam.jpg")
            st.success("This is Spam Message")
        else:
            #print("Valid Msg")
            st.image("not_spam.png")
            st.success("This is Valid Message")

    uploaded_file = st.file_uploader("Upload File to predict Messages",type=["txt","csv"])
    if uploaded_file:
        df_spam = pd.read_csv(uploaded_file,header=None,names=['MSG'])
        # df.index = range(1,df.shape[0]+1)
        # st.dataframe(df)

        pred = spam.predict(df_spam.MSG)
        df_spam.index=range(1,df_spam.shape[0]+1)
        df_spam["PREDICTION"]=pred
        df_spam["PREDICTION"]=df_spam["PREDICTION"].map({0:"Spam",1:"Valid"})
        st.dataframe(df_spam)

with tab2:
    msg2 = st.text_input("Enter any Sentence to detect language",key="lang_input")
    if st.button("🕵️ Detect",key="lang_btn"):
        pred = lang.predict([msg2])
        st.success(pred[0])

    uploaded_file = st.file_uploader("Upload a .txt or .csv File to Detect Language",type=["txt","csv"])
    if uploaded_file:
        df_lang = pd.read_csv(uploaded_file,header=None,names=['Sentences'])

        pred = lang.predict(df_lang.Sentences)
        df_lang.index=range(1,df_lang.shape[0]+1)
        df_lang["Language"]=pred
        st.dataframe(df_lang)

with tab3:
    msg3 = st.text_input("Enter any News headline with short description to classify",key="news_input")
    if st.button("🕵️ Classify",key="news_btn"):
        pred = news.predict([msg3])
        st.success(pred[0])

    uploaded_file = st.file_uploader("Upload a .txt or .csv File to classify News",type=["txt","csv"])
    if uploaded_file:
        df_news = pd.read_csv(uploaded_file,header=None,names=['News'])

        pred = news.predict(df_news.News)
        df_news.index=range(1,df_news.shape[0]+1)
        df_news["News_Category"]=pred
        st.dataframe(df_news)

with tab4:
    msg4 = st.text_input("Enter your Restaurant Review",key="restro_input")
    if st.button("🕵️ Review",key='restro_btn'):
        pred = restro.predict([msg4])
        if pred[0]==0:
            st.image("dislike.jpg")
            st.success("Negative Review")
        else:
            st.image("like.jpg")
            st.success("Positive Review")

    uploaded_file = st.file_uploader("Upload a .txt or .csv File to See Restaurant Review",type=["txt","csv"])
    if uploaded_file:
        df_restro = pd.read_csv(uploaded_file,header=None,names=['Review'])
    
        pred = restro.predict(df_restro.Review)
        df_restro.index=range(1,df_restro.shape[0]+1)
        df_restro["Analysis"]=pred
        df_restro["Analysis"]=df_restro["Analysis"].map({0:"👎 Disliked",1:"👍 Liked"})
        st.dataframe(df_restro)

st.markdown("---------------")
st.markdown("""
            <center>
            🤖 This is a ML Project using with Streamlit 🤖
            </center>
""",unsafe_allow_html=True)

st.sidebar.image("logo.jpg")
with st.sidebar.expander("ℹ️ About"):       
    st.write("We are a group of student try to understand the concept")

with st.sidebar.expander("📞 Contact Us"):       
    st.write("77XXXXXX73")
    st.write("testproject@sample.com")

with st.sidebar.expander("🙋‍♂️ Help"):       
    st.write("For any query,Feel free to ask")
    


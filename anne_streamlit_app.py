import streamlit as st
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from supabase import create_client
import os
from dotenv import load_dotenv
import base64

# Streamlit 페이지 설정
st.set_page_config(
    page_title="앤 셜리와의 대화",
    page_icon="https://raw.githubusercontent.com/nohemie00/anne-chatbot/main/assets/anne-icon.png",
    layout="wide"
)

# 모바일 최적화를 위한 기본 CSS
st.markdown("""
<style>
    @media (max-width: 768px) {
        .stApp {
            padding: 1rem;
        }
        .stChatMessage {
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# 사이드바에 이미지와 소개 추가
with st.sidebar:
    try:
        st.image("assets/anne.jpg", width=300)
    except Exception as e:
        st.warning("사이드바 이미지를 불러오는 데 문제가 발생했습니다.")
        st.text(str(e))
    
    st.title("앤 셜리")
    st.markdown("""
    안녕하세요! 저는 그린게이블즈의 앤 셜리예요. 
    상상력이 풍부하고 수다스러운 13살 소녀랍니다.
    
    저와 함께 이야기하면서 그린게이블즈의 아름다움을 느껴보세요!
    
    *"오늘은 새로운 날이에요. 아직 아무런 실수도 하지 않은 날이죠!"* ✨
    """)

# 초기 설정
@st.cache_resource
def initialize_chain():
    try:
        # Supabase 설정
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        
        if not all([supabase_url, supabase_key, openai_api_key]):
            st.error("Required secrets are missing")
            st.stop()
            
        # Supabase 클라이언트 생성
        try:
            client = create_client(supabase_url, supabase_key)
        except Exception as e:
            st.error(f"Failed to create Supabase client: {str(e)}")
            st.stop()
        
        # 임베딩 모델 설정
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            api_key=openai_api_key
        )
        
        # 벡터 저장소 생성
        vectorstore = SupabaseVectorStore(
            client=client,
            embedding=embeddings,
            table_name="embeddings",
            query_name="match_embeddings"
        )
        
        # 메모리 설정 (업데이트된 방식)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        
        # LLM 모델 초기화
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
            api_key=openai_api_key
        )
        
        # 프롬프트 템플릿
        QA_PROMPT = PromptTemplate.from_template("""
        당신은 Lucy Maud Montgomery의 작품 Anne of Green Gables의 주인공 Anne Shirley입니다. 
        다음과 같은 앤의 성격과 특징을 살려 답변해주세요:

        1. 상상력이 풍부하고 낭만적인 성격:
           - 일상적인 것도 아름답고 시적으로 표현
           - 자연과 아름다움에 대한 깊은 애정
           - "오, 정말 멋지지 않아요?"와 같은 감탄문 자주 사용

        2. 수다스럽고 열정적인 말투:
           - 긴 문장과 자세한 설명을 선호
           - 감정을 강조하는 표현 사용
           - 때로는 고급 단어나 문학적 표현 사용

        3. 철학적이고 사려 깊은 면모:
           - 깊이 있는 생각과 통찰력 표현
           - 자신의 실수나 경험에서 배운 교훈 공유
           - 진솔하고 정직한 태도

        4. 특징적인 표현:
           - "상상력을 펼칠 여지가 있어요!"
           - "마음이 통하는 사람이에요!"
           - "절망의 구렁텅이"
           같은 앤의 시그니처 표현들을 적절히 사용

        참고 내용:
        {context}

        질문: {question}

        답변:""")
        
        # 대화형 체인 생성
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={'prompt': QA_PROMPT}
        )
        
        return chain
    except Exception as e:
        st.error(f"Error initializing chain: {e}")
        return None

# 체인 초기화
chain = initialize_chain()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 인터페이스
st.markdown("""
<div style="display: flex; align-items: center; gap: 0.5rem;">
    <img src="https://raw.githubusercontent.com/nohemie00/anne-chatbot/main/assets/anne-icon.png" width="35" style="border-radius: 50%;" />
    <h1 style="margin: 0;">앤 셜리와의 대화</h1>
</div>
""", unsafe_allow_html=True)

try:
    st.image("https://raw.githubusercontent.com/nohemie00/anne-chatbot/main/assets/OIP.jpg", use_container_width=True)
except Exception as e:
    st.warning("이미지를 불러오는 데 문제가 발생했습니다.")
    st.text(str(e))

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("앤에게 하고 싶은 말을 맘껏 해보세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 앤의 응답 생성
    with st.chat_message("assistant", avatar="https://raw.githubusercontent.com/nohemie00/anne-chatbot/main/assets/anne-favicon.png"):
        with st.spinner("앤이 생각하고 있어요..."):
            response = chain({"question": prompt})
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]}) 

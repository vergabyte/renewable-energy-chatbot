import streamlit as st

from main import initialize_chatbot
from src.chatbot import query_chatbot, format_sources


st.title('Renewable Energy Chatbot')


@st.cache_resource
def load_chatbot():
    return initialize_chatbot()


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_chain' not in st.session_state:
    with st.spinner('Loading models...'):
        rag_chain, retriever = load_chatbot()
        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message['role'] == 'assistant':
            if 'chunks' in message:
                with st.expander('Retrieved chunks'):
                    for i, chunk in enumerate(message['chunks'], 1):
                        st.markdown(f'**Chunk {i}:**')
                        st.text(chunk + '...')
                        if i < len(message['chunks']):
                            st.divider()
            if 'sources' in message:
                with st.expander('Sources'):
                    for i, source in enumerate(message['sources'], 1):
                        st.markdown(f'{i}. [{source}]({source})')


if prompt := st.chat_input('Ask about renewable energy'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        answer, source_docs = query_chatbot(
            st.session_state.rag_chain,
            st.session_state.retriever,
            prompt
        )

        sources = format_sources(source_docs)
        chunks = [doc.page_content[:200] for doc in source_docs]

        st.markdown(answer)

        if chunks:
            with st.expander('Retrieved chunks'):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f'**Chunk {i}:**')
                    st.text(chunk + '...')
                    if i < len(chunks):
                        st.divider()

        if sources:
            with st.expander('Sources'):
                for i, source in enumerate(sources, 1):
                    st.markdown(f'{i}. [{source}]({source})')

        st.session_state.messages.append({
            'role': 'assistant',
            'content': answer,
            'sources': sources,
            'chunks': chunks
        })

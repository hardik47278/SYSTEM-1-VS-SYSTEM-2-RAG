import streamlit as st
from pathlib import Path
from pyprojroot import here
from src.pipeline import Pipeline, max_nst_o3m_config
from src.question_processing_copy import QuestionsProcessor

st.set_page_config(page_title="Financial RAG QA", layout="wide")

st.title("📊 Annual Report Question Answering")

# Initialize pipeline
@st.cache_resource
def load_processor():
    root_path = here() / "data" / "test_set"
    pipeline = Pipeline(root_path, run_config=max_nst_o3m_config)

    processor = QuestionsProcessor(
        vector_db_dir=pipeline.paths.vector_db_dir,
        documents_dir=pipeline.paths.documents_dir,
        subset_path=pipeline.paths.subset_path,
        new_challenge_pipeline=True,
        parent_document_retrieval=True,
        llm_reranking=True,
        answering_model="o3-mini-2025-01-31"
    )
    return processor

processor = load_processor()

# User input
question = st.text_input("Ask a question about any company report:")

schema = st.selectbox(
    "Answer Type",
    ["boolean", "number", "name"]
)

if st.button("Get Answer"):
    with st.spinner("Retrieving and reasoning..."):
        try:
            answer = processor.process_question(
                question=question,
                schema=schema
            )

            st.success("Answer Generated")

            st.subheader("Final Answer")
            st.write(answer.get("final_answer"))

            st.subheader("References")
            st.json(answer.get("references", []))

            st.subheader("Relevant Pages")
            st.write(answer.get("relevant_pages", []))

        except Exception as e:
            st.error(str(e))

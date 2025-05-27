import streamlit as st
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity # Not directly used in the final version but good to keep if you expand


# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Advanced Text Analysis and Question Generation")

# --- Model Loading (with caching) ---
@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_name_suggestion_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-small")

@st.cache_resource
def load_qa_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# Load models
sentence_model = load_sentence_transformer_model()
name_suggestion_pipeline = load_name_suggestion_pipeline()
qa_pipeline = load_qa_pipeline()

# --- Streamlit UI ---
st.title("💡 Topic Discovery and Exam Question Generator")

st.markdown("""
This application analyzes your input text to discover underlying topics and then generates exam questions based on those topics.
""")

# Input Text Area
st.header("1. Enter Your Text")
text_input = st.text_area(
    "Paste your content here for analysis:",
    height=300,
    value="""
Electromagnetism, at its core, explores how electric and magnetic fields interact.
Maxwell's equations, a set of four partial differential equations, beautifully
encapsulate this interaction, predicting phenomena from the static attraction
between charges to the dynamic propagation of light. The concept of the
electromagnetic spectrum, ranging from radio waves to gamma rays, demonstrates
the vast applications of these fundamental principles in technology and nature.
Understanding the wave-particle duality inherent in light is also a key aspect,
highlighting the quantum nature of electromagnetic radiation. From a practical
standpoint, electromagnetism is vital for technologies like electric motors,
generators, and wireless communication.

In the realm of mathematics, dynamical systems theory provides a powerful
framework for analyzing systems that evolve over time. This includes everything
from population dynamics to the motion of celestial bodies. Key concepts involve
phase space, attractors (such as fixed points, limit cycles, and strange
attractors), and bifurcations, which describe qualitative changes in a system's
behavior as parameters vary. The famous butterfly effect, a hallmark of chaotic
systems, illustrates extreme sensitivity to initial conditions, where a tiny
change can lead to vastly different long-term outcomes. Understanding the
stability and predictability of these systems is a central goal in various
scientific disciplines.

The psychology of trauma, particularly in its complex forms (C-PTSD), often
involves a profound disruption of an individual's internal world. Traumatic
experiences can lead to fragmented memories, emotional dysregulation, and a
dissociation from one's sense of self. From a psychological perspective, this
can be viewed as a system (the mind) being pushed into a maladaptive dynamic
state, where previously integrated functions become disconnected. Therapeutic
approaches often aim to help individuals re-regulate their emotional responses
and reintegrate dissociated parts of their experience, moving the system towards
more adaptive and coherent "attractors" in their psychological landscape. The
healing process is often non-linear, involving periods of progress and
regression, reflecting the complex dynamics of the human psyche. Moreover, the
intergenerational transmission of trauma is an emerging area of study, showing
how these psychological dynamics can affect entire family systems across generations.
"""
)

# Number of topics and questions per topic
col1, col2 = st.columns(2)
with col1:
    n_clusters = st.slider("Number of Topics to Discover:", min_value=1, max_value=10, value=3)
with col2:
    n_questions = st.slider("Number of Questions per Topic:", min_value=1, max_value=5, value=2)

if st.button("Analyze Text and Generate Questions"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        st.divider()
        st.header("2. Analysis Results")

        # 1. Sentence Segmentation
        with st.spinner("Segmenting text into sentences..."):
            sentences = re.split(r'\.\s*', text_input.replace('\n', ' ').strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            st.success(f"Found {len(sentences)} sentences.")
            with st.expander("View segmented sentences"):
                for i, s in enumerate(sentences):
                    st.write(f"{i}: {s}")

        if not sentences:
            st.error("No sentences found in the input text. Please check your input.")
            st.stop()

        # 2. Generate Sentence Embeddings
        with st.spinner("Generating sentence embeddings..."):
            sentence_embeddings = sentence_model.encode(sentences)
            st.success("Sentence embeddings generated.")

        # 3. Cluster Sentences (Topic Discovery)
        with st.spinner(f"Discovering {n_clusters} topics..."):
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            kmeans.fit(sentence_embeddings)
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            sentences_by_topic_id = {i: [] for i in range(n_clusters)}
            for i, sentence in enumerate(sentences):
                cluster_id = cluster_labels[i]
                sentences_by_topic_id[cluster_id].append(sentence)
            st.success("Topics discovered.")

        # 4. Name Topics
        st.subheader("Discovered Topics and Suggested Names")
        topic_names_map = {}
        final_grouped_sentences = {}

        progress_bar = st.progress(0)
        for cluster_id in range(n_clusters):
            topic_context_for_naming = " ".join(sentences_by_topic_id[cluster_id])

            if not topic_context_for_naming.strip(): # Handle empty clusters
                suggested_name = f"Empty Topic {cluster_id + 1}"
            else:
                prompt_for_name = f"Suggest a concise, 2-4 word topic name for the following text: '{topic_context_for_naming}'"
                suggested_name_output = name_suggestion_pipeline(prompt_for_name, max_new_tokens=10, num_return_sequences=1)
                suggested_name = suggested_name_output[0]['generated_text'].strip()

                # Basic cleanup
                suggested_name = suggested_name.replace("Topic:", "").replace("Name:", "").replace("Category:", "").strip()
                if suggested_name.endswith('.'):
                    suggested_name = suggested_name[:-1]

                # Fallback for empty or duplicate names
                if not suggested_name or suggested_name in topic_names_map.values():
                    suggested_name = f"Topic {cluster_id + 1}"

            topic_names_map[cluster_id] = suggested_name
            final_grouped_sentences[suggested_name] = sentences_by_topic_id[cluster_id]

            st.write(f"**Topic {cluster_id + 1}:** {suggested_name}")
            with st.expander(f"Sentences in '{suggested_name}'"):
                if sentences_by_topic_id[cluster_id]:
                    for s in sentences_by_topic_id[cluster_id]:
                        st.markdown(f"- {s}")
                else:
                    st.info("No sentences in this topic.")
            progress_bar.progress((cluster_id + 1) / n_clusters)
        st.success("Topics named.")


        # 5. Generate Exam Questions
        st.divider()
        st.header("3. Generated Exam Questions")
        questions_generated = False
        
        for topic_name, sentences_list in final_grouped_sentences.items():
            if not sentences_list:
                continue

            topic_context_text = " ".join(sentences_list)
            st.subheader(f"Questions for **{topic_name}**:")

            for i in range(n_questions):
                prompt_variations = [
                    f"Generate a challenging exam question based on the following text about {topic_name}: {topic_context_text}",
                    f"Formulate an analytical question focusing on a key concept from this text regarding {topic_name}: {topic_context_text}",
                    f"What is a key concept discussed in this text regarding {topic_name}? {topic_context_text}",
                    f"Pose a question that requires synthesis of information from the following text on {topic_name}: {topic_context_text}"
                ]
                current_prompt = prompt_variations[i % len(prompt_variations)]

                try:
                    with st.spinner(f"Generating question {i+1} for {topic_name}..."):
                        questions_output = qa_pipeline(current_prompt, max_new_tokens=100, num_return_sequences=1)
                        generated_question = questions_output[0]['generated_text'].strip()

                        if generated_question.endswith('?') and len(generated_question) > 20:
                            st.write(f"- {generated_question}")
                            questions_generated = True
                        elif not generated_question.endswith('?') and len(generated_question) > 20:
                            st.write(f"- {generated_question}?")
                            questions_generated = True
                        else:
                            st.info(f"Could not generate a meaningful question {i+1} for {topic_name}.")

                except Exception as e:
                    st.error(f"Error generating question {i+1} for {topic_name}: {e}")
                    break
            st.markdown("---") # Separator between topics

        if not questions_generated:
            st.info("No questions could be generated. Try adjusting the number of topics or questions, or provide more detailed text.")

st.sidebar.info("This app leverages Hugging Face Transformers and Sentence-Transformers for advanced NLP tasks like text embedding, clustering, and text generation.")

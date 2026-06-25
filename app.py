import streamlit as st
from rag_pipeline import (
    build_rag_pipeline,
    calculate_bmr,
    calculate_tdee,
    DATA_PATH,
    SYSTEM_PROMPT_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS
)


# 1. PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="💪 AI Fitness Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. SESSION STATE
# ============================================

def init_session_state():
    if "rag" not in st.session_state:
        with st.spinner("🚀 Loading AI Fitness Coach..."):
            try:
                st.session_state.rag = build_rag_pipeline()
                st.session_state.rag_ready = True
                st.success("✅ RAG System loaded successfully!")
            except Exception as e:
                st.session_state.rag = None
                st.session_state.rag_ready = False
                st.error(f"❌ Failed to load RAG system: {str(e)}")
                st.info("💡 Make sure you have:")
                st.info("1. GROQ_API_KEY in .env file")
                st.info("2. data/ folder with .txt files")
                st.info("3. prompt.xml file")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "user_data" not in st.session_state:
        st.session_state.user_data = {
            "name": "",
            "age": 25,
            "weight": 70.0,
            "height": 170.0,
            "gender": "male",
            "goal": "weight_loss",
            "level": "beginner",
            "frequency": "3x",
            "equipment": "full_gym",
            "injuries": "",
            "preferences": ""
        }

init_session_state()


# 3. SIDEBAR
# ============================================

with st.sidebar:
    st.title("👤 Profile Settings")
    st.markdown("---")
    
    with st.expander("📝 Personal Data", expanded=True):
        name = st.text_input("Name", value=st.session_state.user_data.get("name", ""))
        gender = st.selectbox("Gender", ["male", "female"], index=0 if st.session_state.user_data.get("gender", "male") == "male" else 1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=st.session_state.user_data.get("age", 25))
        with col2:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=st.session_state.user_data.get("weight", 70.0), step=0.5)
        with col3:
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=st.session_state.user_data.get("height", 170.0), step=0.5)
    
    with st.expander("🎯 Fitness Goals", expanded=True):
        goal = st.selectbox(
            "Main Goal",
            ["weight_loss", "muscle_gain", "maintenance", "performance"],
            format_func=lambda x: {"weight_loss": "🏃 Weight Loss", "muscle_gain": "💪 Muscle Gain", "maintenance": "⚖️ Maintenance", "performance": "🚀 Performance"}.get(x, x)
        )
        level = st.selectbox(
            "Experience Level",
            ["beginner", "intermediate", "advanced"],
            format_func=lambda x: {"beginner": "🌱 Beginner", "intermediate": "📈 Intermediate", "advanced": "🔥 Advanced"}.get(x, x)
        )
        frequency = st.selectbox("Training Frequency", ["1-2x", "3x", "4x", "5x", "6x"])
    
    with st.expander("🏠 Equipment & Preferences", expanded=False):
        equipment = st.selectbox(
            "Equipment Access",
            ["full_gym", "home_gym", "no_equipment"],
            format_func=lambda x: {"full_gym": "🏋️ Full Gym", "home_gym": "🏠 Home Gym", "no_equipment": "🪑 No Equipment"}.get(x, x)
        )
        injuries = st.text_area("Injury History", value=st.session_state.user_data.get("injuries", ""), placeholder="Leave empty if none", height=68)
    
    if st.button("💾 Save Profile", type="primary", use_container_width=True):
        st.session_state.user_data = {
            "name": name, "age": age, "weight": weight, "height": height,
            "gender": gender, "goal": goal, "level": level, "frequency": frequency,
            "equipment": equipment, "injuries": injuries, "preferences": preferences
        }
        st.success("✅ Profile saved successfully!")
        
    
    if weight and height and age:
        st.markdown("---")
        st.subheader("📊 Your Metrics")
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, "moderate")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔥 BMR", f"{bmr:.0f} cal")
        with col2:
            st.metric("⚡ TDEE", f"{tdee:.0f} cal")
        
        if goal == "weight_loss":
            target = tdee - 400
            st.info(f"💡 **Target**: ~{target:.0f} cal/day (Deficit 400)")
        elif goal == "muscle_gain":
            target = tdee + 250
            st.info(f"💡 **Target**: ~{target:.0f} cal/day (Surplus 250)")
        elif goal == "performance":
            target = tdee + 100
            st.info(f"💡 **Target**: ~{target:.0f} cal/day (Small surplus)")
        else:
            target = tdee
            st.info(f"💡 **Target**: ~{target:.0f} cal/day (Maintenance)")
    
    st.markdown("---")
    st.subheader("⚙️ System Info")
    if st.session_state.rag_ready:
        st.success("🟢 Status: Ready")
        if st.session_state.rag:
            st.caption(f"📚 Knowledge: {len(st.session_state.rag.chunks)} chunks")
            st.caption(f"🧠 LLM: {LLM_MODEL}")
            st.caption(f"🧠 Embedding: {EMBEDDING_MODEL}")
            st.caption(f"📏 Chunk Size: {CHUNK_SIZE}")
            st.caption(f"📏 Overlap: {CHUNK_OVERLAP}")
            st.caption(f"🔍 Top K: {TOP_K_RESULTS}")
    else:
        st.error("🔴 Status: Not Ready")
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")


# 4. MAIN CHAT
# ============================================

st.title("PerformAI")
st.caption("AI-Powered Fitness Coach for Personalized Training and Performance")

with st.expander("⚡ Quick Actions", expanded=False):
    cols = st.columns(3)
    quick_questions = [
        ("🏃 Weight Loss", "Create a weight loss program for beginners"),
        ("💪 Muscle Gain", "Create a muscle gain program for intermediate"),
        ("🥗 Meal Plan", "Create a 2000 calorie meal plan"),
        ("🏋️ Squat Form", "How to do squat correctly?"),
        ("💊 Supplements", "What supplements are good for fitness?"),
        ("📊 Progress", "How to track fitness progress?")
    ]
    
    for i, (label, question) in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(label, key=f"qa_{i}", use_container_width=True):
                with st.chat_message("user"):
                    st.markdown(question)
                st.session_state.messages.append({"role": "user", "content": question})
                
                with st.chat_message("assistant"):
                    if st.session_state.rag_ready and st.session_state.rag:
                        with st.spinner("💭 Processing..."):
                            try:
                                response = st.session_state.rag.query(question, st.session_state.user_data)
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            except Exception as e:
                                error_msg = f"❌ Error: {str(e)}"
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        error_msg = "❌ RAG system not ready. Please check:\n1. GROQ_API_KEY in .env\n2. Internet connection\n3. data/ folder exists"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()

st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about workout programs, nutrition, or fitness..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        if st.session_state.rag_ready and st.session_state.rag:
            with st.spinner("💭 Thinking..."):
                try:
                    response = st.session_state.rag.query(prompt, st.session_state.user_data)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            error_msg = "❌ RAG system not ready. Please check:\n1. GROQ_API_KEY in .env\n2. Internet connection\n3. data/ folder exists"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    st.rerun()

st.markdown("---")
st.caption("⚠️ **Disclaimer**: All advice is for educational purposes only. Consult with a healthcare professional for specific medical conditions.")

col1, col2, col3 = st.columns([3, 1, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col3:
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if not st.session_state.messages and st.session_state.rag_ready:
    welcome_msg = """
    👋 **Hello! I'm your AI Fitness Coach!**
    
    I'm here to help you achieve your fitness goals with:
    - 📋 **Personalized workout programs**
    - 🥗 **Science-based nutrition advice**
    - 💪 **Motivation and accountability**
    - 📊 **Effective progress tracking**
    
    **To get started:**
    1. Fill in your profile in the sidebar 👈
    2. Ask me anything about fitness in the chat!
    
    💡 Try the **Quick Actions** above for fast answers.
    
    **What would you like to know today?** 🏋️
    """
    with st.chat_message("assistant"):
        st.markdown(welcome_msg)
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
# LLM Assistant Fitness Coach

An AI-powered fitness coaching assistant built using **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)**. The application delivers personalized workout guidance, nutrition recommendations, and fitness-related information by combining a knowledge base with modern LLM technology.

The project demonstrates an end-to-end LLM pipeline, including document ingestion, vector database indexing, semantic retrieval, and conversational question answering.

---

## 🎯 Project Objectives

This project aims to:

* Build an intelligent fitness coaching assistant using Large Language Models.
* Provide personalized workout and nutrition guidance.
* Answer fitness-related questions using a custom knowledge base.
* Reduce hallucinations through Retrieval-Augmented Generation (RAG).
* Demonstrate an end-to-end LLM application suitable for real-world deployment.

---

## ✨ Features

* 🤖 AI-powered conversational fitness coach
* 📚 Retrieval-Augmented Generation (RAG)
* 🔍 Semantic document search using vector embeddings
* 💬 Context-aware question answering
* 🏋️ Workout recommendations
* 🥗 Nutrition and healthy lifestyle guidance
* 📄 Custom document knowledge base
* 🌐 Interactive web interface with Streamlit

---

## 🛠️ Tech Stack

### Large Language Model

* Groq
* LangChain

### Embedding Model

* HuggingFace Embeddings

### Vector Database

* FAISS

### Framework

* Python
* Streamlit
---


## 🧠 System Architecture

```text
User Input
      │
      ▼
 Streamlit Interface
      │
      ▼
 LangChain Pipeline
      │
      ▼
Retrieve Relevant Documents (FAISS)
      │
      ▼
Retrieved Context
      │
      ▼
Groq LLM
      │
      ▼
Generated Response
```


## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/yourusername/llm_asst_fitness_coach.git
```

Move to the project directory:

```bash
cd llm_asst_fitness_coach
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

Run the application:

```bash
streamlit run app.py
```



## 👤 Author

**Galih**

Sports Science Graduate • Aspiring AI Engineer • Data Scientist • Machine Learning Enthusiast


⭐ If you find this project useful, consider giving it a star!

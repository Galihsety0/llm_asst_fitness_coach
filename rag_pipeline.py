import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document



# 1. KONFIGURASI
# ============================================

DATA_PATH = "data"
SYSTEM_PROMPT_PATH = "prompt.xml"
FAISS_INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 4
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 2048



# 2. SETUP
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Create .env file.")



# 3. DOCUMENT LOADER
# ============================================

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Load semua dokumen dari folder (support .txt dan .pdf)"""
    folder = Path(folder_path)
    documents = []
    
    if not folder.exists():
        logger.warning(f"⚠️ Folder {folder} not found")
        return documents
    
    for file_path in folder.glob("*.txt"):
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.stem
            documents.extend(docs)
            logger.info(f"✅ Loaded: {file_path.name}")
        except Exception as e:
            logger.error(f"❌ Error loading {file_path.name}: {e}")
    
    for file_path in folder.glob("*.pdf"):
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.stem
            documents.extend(docs)
            logger.info(f"✅ Loaded: {file_path.name}")
        except Exception as e:
            logger.error(f"❌ Error loading {file_path.name}: {e}")
    
    logger.info(f"📚 Total documents loaded: {len(documents)}")
    return documents



# 4. CLASS FITNESSRAG
# ============================================

class FitnessRAG:
    def __init__(
        self,
        data_folder: str = DATA_PATH,
        prompt_file: str = SYSTEM_PROMPT_PATH,
        index_path: str = FAISS_INDEX_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K_RESULTS
    ):
        self.data_folder = Path(data_folder)
        self.prompt_file = Path(prompt_file)
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # 1. Init LLM (Groq)
        logger.info(f"🧠 Loading Groq LLM: {self.llm_model}")
        self.llm = ChatGroq(
            model=self.llm_model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            api_key=GROQ_API_KEY
        )
        logger.info("✅ Groq LLM loaded")
        
        # 2. Init Embeddings (HuggingFace)
        logger.info(f"🧠 Loading HuggingFace embeddings: {self.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ HuggingFace embeddings loaded")
        
        # 3. Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # 4. Load & split documents
        logger.info(f"📂 Loading knowledge base from: {self.data_folder}")
        self.documents = load_documents_from_folder(str(self.data_folder))
        self.chunks = self._split_documents()
        
        # 5. Build FAISS vector store
        self.vector_store = self._build_vector_store()
        
        # 6. Setup Retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        # 7. Build RAG chain (pake create_retrieval_chain)
        self.qa_chain = self._build_chain()
        
        logger.info("✅ RAG System Ready!")
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from XML file"""
        if not self.prompt_file.exists():
            logger.warning(f"⚠️ Prompt file {self.prompt_file} not found.")
            return self._get_default_prompt()
        
        try:
            tree = ET.parse(self.prompt_file)
            root = tree.getroot()
            
            prompt_parts = []
            for child in root:
                if child.tag == "role" and child.text and child.text.strip():
                    prompt_parts.append(child.text.strip())
                elif child.tag == "knowledge_boundary" and child.text and child.text.strip():
                    prompt_parts.append(child.text.strip())
                elif child.tag == "safety_guidelines":
                    prompt_parts.append(self._extract_text(child))
                elif child.tag == "response_format":
                    prompt_parts.append(self._extract_text(child))
                elif child.tag == "tone_guidelines":
                    prompt_parts.append(self._extract_text(child))
                elif child.tag == "coaching_principles":
                    prompt_parts.append(self._extract_text(child))
            
            full_prompt = "\n\n".join(prompt_parts) if prompt_parts else self._get_default_prompt()
            logger.info(f"✅ System prompt loaded from {self.prompt_file}")
            return full_prompt
        except Exception as e:
            logger.error(f"❌ Error loading system prompt: {e}")
            return self._get_default_prompt()
    
    def _extract_text(self, element) -> str:
        text_parts = []
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())
        
        for child in element:
            if child.text and child.text.strip():
                text_parts.append(f"  {child.tag.upper()}:\n  {child.text.strip()}")
            if len(list(child)) > 0:
                nested = self._extract_text(child)
                if nested:
                    text_parts.append(nested)
        
        return "\n".join(text_parts) if text_parts else ""
    
    def _get_default_prompt(self) -> str:
        return """
        You are an AI Fitness Coach Professional. Help users achieve their fitness goals.
        
        RULES:
        1. Base answers ONLY on the knowledge base provided
        2. Don't give medical diagnoses
        3. Personalize based on user data
        4. If you don't know, say "I don't have that information"
        5. Always prioritize safety
        
        Use the same language as the user's question (English or Indonesian).
        """
    
    def _split_documents(self) -> List:
        if not self.documents:
            logger.warning("⚠️ No documents to split")
            return []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(self.documents)
        logger.info(f"✂️ Created {len(chunks)} chunks")
        return chunks
    
    def _build_vector_store(self) -> FAISS:
        if self.index_path.exists() and (self.index_path / "index.faiss").exists():
            logger.info(f"🔄 Loading existing FAISS index from {self.index_path}")
            try:
                return FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except TypeError:
                return FAISS.load_local(str(self.index_path), self.embeddings)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing index: {e}. Rebuilding...")
        
        logger.info("🆕 Building new FAISS vector store...")
        vector_store = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )
        
        self.index_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(self.index_path))
        logger.info(f"✅ FAISS index saved to {self.index_path}")
        return vector_store
    
    def _build_chain(self):
        """Build RAG chain menggunakan create_retrieval_chain (bisa kirim user_data)"""
        
        # ✅ Template dengan user_data
        template = f"""
        {self.system_prompt}
        
        Knowledge Base (use ONLY this):
        {{context}}
        
        USER DATA (gunakan data ini untuk personalisasi):
        {{user_data}}
        
        QUESTION: {{input}}
        
        INSTRUKSI PENTING:
        1. Gunakan USER DATA di atas untuk mempersonalisasi jawaban
        2. Jika user sudah memberikan data (usia, berat, goal, level), langsung gunakan
        3. JANGAN tanyakan ulang data yang sudah ada di USER DATA
        4. Berikan rekomendasi yang spesifik berdasarkan data yang ada
        
        ANSWER:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "input", "user_data"]
        )
        
        # Document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        
        # Retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=document_chain
        )
        
        return retrieval_chain
    
    def query(self, question: str, user_data: Dict = None) -> str:
        if user_data is None:
            user_data = {}
        
        try:
            # Format user data sebagai string yang rapi
            user_data_str = json.dumps(user_data, indent=2, ensure_ascii=False)
            
            # Log user data untuk debugging
            logger.info(f"👤 User Data: {user_data_str}")
            
            response = self.qa_chain.invoke({
                "input": question,
                "user_data": user_data_str
            })
            
            return response.get("answer", "No answer found.")
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return f"❌ Error: {str(e)}"



# 5. HELPER FUNCTIONS
# ============================================

def calculate_bmr(weight: float, height: float, age: int, gender: str = "male") -> float:
    if gender.lower() == "male":
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        return (10 * weight) + (6.25 * height) - (5 * age) - 161


def calculate_tdee(bmr: float, activity: str = "moderate") -> float:
    factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "extreme": 1.9
    }
    return bmr * factors.get(activity.lower(), 1.55)



# 6. FACTORY FUNCTION
# ============================================

def build_rag_pipeline():
    return FitnessRAG(
        data_folder=DATA_PATH,
        prompt_file=SYSTEM_PROMPT_PATH,
        index_path=FAISS_INDEX_PATH,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=TOP_K_RESULTS
    )



# 7. TESTING
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing RAG Pipeline with Groq + HuggingFace + FAISS")
    print("=" * 60)
    
    try:
        rag = build_rag_pipeline()
        
        user = {
            "name": "Budi",
            "age": 28,
            "weight": 80,
            "height": 170,
            "gender": "male",
            "goal": "muscle_gain",
            "level": "beginner"
        }
        
        test_questions = [
            "What program should I follow for muscle gain?",
            "How to do squat correctly?",
            "Berapa protein yang saya butuhkan per hari?"
        ]
        
        for q in test_questions:
            print(f"\n❓ {q}")
            print("-" * 40)
            response = rag.query(q, user)
            print(f"🤖 {response}")
            print("-" * 40)
        
        print("\n✅ Testing complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 If still error, try:")
        print("1. Delete faiss_index folder: rm -rf faiss_index")
        print("2. Check GROQ_API_KEY in .env")
        print("3. Run: pip install -r requirements.txt")
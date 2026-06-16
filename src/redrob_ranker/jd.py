"""
jd.py — The released job description, encoded as explicit, machine-readable
requirements.

This is the heart of the system. The challenge JD ("Senior AI Engineer —
Founding Team") is deliberately written so that a keyword/embedding-only
approach fails: it lists explicit *positive* signals, *nice-to-haves*, and
*disqualifiers*, and the organizers planted keyword-stuffer traps and
honeypots in the candidate pool. We therefore translate the prose JD into
weighted feature logic rather than relying on similarity alone.

Every constant here traces back to a specific line in job_description.docx,
so the whole scorer is defensible in the Stage-5 interview.
"""

# --- Title taxonomy --------------------------------------------------------
# The JD's decisive negative: "A candidate who has all the AI keywords listed
# as skills but whose title is 'Marketing Manager' is not a fit, no matter how
# perfect their skill list looks." So title (current + recent career) sets a
# ceiling that a stuffed skills list cannot break through.

CORE_TITLE_TERMS = [
    "machine learning engineer", "ml engineer", "ai engineer",
    "applied scientist", "applied ml", "research engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation", "recsys", "mlops engineer",
    "deep learning engineer", "ml scientist", "machine learning scientist",
]
DATA_SCIENCE_TERMS = ["data scientist", "ml researcher", "research scientist"]
ADJACENT_TECH_TERMS = [
    "data engineer", "analytics engineer", "backend engineer",
    "software engineer", "full stack", "platform engineer",
    "staff engineer", "principal engineer",
]
OTHER_TECH_TERMS = [
    "frontend", "mobile developer", "android", "ios", "devops",
    "cloud engineer", "qa engineer", "java developer", ".net",
    "web developer", "sre", "database administrator",
]
# Everything else (HR Manager, Sales, Accountant, Marketing Manager,
# Civil/Mechanical Engineer, Content Writer, Graphic Designer, ...) is non-tech
# and treated as the trap pool.

TITLE_SCORE = {
    "core": 1.00,
    "data_science": 0.82,
    "adjacent": 0.50,   # rescued only if career history shows real ML/IR work
    "other_tech": 0.22,
    "non_tech": 0.04,
}

# --- Skill concept groups --------------------------------------------------
# "Things you absolutely need" in the JD map to the first three groups; these
# carry the most weight. Matching is substring-based on the lowercased skill
# name so variants ("FAISS", "Pinecone", "sentence-transformers") all hit.

SKILL_GROUPS = {
    # JD: production embeddings-based retrieval — absolutely need
    "retrieval_embeddings": (1.00, [
        "embedding", "sentence-transformer", "sbert", "bge", "e5",
        "retrieval", "rag", "semantic search", "dense retrieval",
    ]),
    # JD: vector DBs / hybrid search infrastructure — absolutely need
    "vectordb_search": (1.00, [
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "bm25", "vector search", "hybrid search", "lucene",
        "solr",
    ]),
    # JD: evaluation frameworks for ranking — absolutely need
    "eval": (0.95, [
        "ndcg", "mrr", "map@", "a/b test", "ab test", "experimentation",
        "ranking metrics", "offline evaluation", "mlflow",
    ]),
    # JD: shipped ranking / search / recommendation systems
    "ranking_recsys": (0.95, [
        "learning to rank", "ltr", "ranking", "recommendation", "recommender",
        "recsys", "collaborative filtering",
    ]),
    # Core modern ML — expected, medium weight
    "ml_core": (0.65, [
        "machine learning", "deep learning", "pytorch", "tensorflow",
        "scikit", "xgboost", "lightgbm", "nlp", "natural language",
        "neural network", "classification", "regression",
    ]),
    # LLM work — JD lists fine-tuning as nice-to-have
    "llm": (0.55, [
        "llm", "large language model", "fine-tuning", "lora", "qlora",
        "peft", "transformers", "hugging face", "huggingface", "bert",
    ]),
    # Supporting data-eng skills — low standalone weight
    "data_eng": (0.30, [
        "spark", "airflow", "kafka", "sql", "etl", "data pipeline",
        "snowflake", "dbt", "databricks",
    ]),
}

# JD: "People whose primary expertise is computer vision, speech, or robotics
# without significant NLP/IR exposure ... you'd be re-learning fundamentals."
DOMAIN_MISMATCH_TERMS = [
    "image classification", "object detection", "segmentation", "gans",
    "computer vision", "opencv", "speech recognition", "tts", "asr",
    "robotics", "slam", "lidar", "pose estimation", "ocr",
]

# JD: "People who have only worked at consulting firms ... in their entire
# career." Mindtree/LTIMindtree/HCL etc. are services firms too.
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree",
    "mphasis", "hexaware", "ibm services", "deloitte", "pwc",
]

# JD positive product-company signal (non-exhaustive; used as a soft boost when
# career descriptions read like real product work at scale).
PRODUCT_SIGNAL_TERMS = [
    "product company", "at scale", "real users", "production",
    "shipped", "deployed", "millions of", "recommendation system",
    "search system", "ranking system",
]

# --- Geography -------------------------------------------------------------
# JD: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR welcome. Outside India
# case-by-case, no visa sponsorship.
PREFERRED_CITIES = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "ncr", "bengaluru", "bangalore",
]

# JD: "AI experience consists primarily of recent (under 12 months) projects
# using LangChain to call OpenAI" — framework-enthusiast / shallow-LLM flag.
FRAMEWORK_ENTHUSIAST_TERMS = ["langchain", "llamaindex", "autogpt", "crewai"]

# JD text used as the document side of the semantic similarity layer.
JD_TEXT = (
    "Senior AI Engineer founding team. Own the intelligence layer: ranking, "
    "retrieval and matching systems deciding what recruiters see when they "
    "search candidates. Production experience with embeddings-based retrieval "
    "(sentence-transformers, BGE, E5), vector databases and hybrid search "
    "(FAISS, Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, OpenSearch), "
    "learning-to-rank, LLM fine-tuning, and rigorous evaluation of ranking "
    "systems with NDCG, MRR, MAP and A/B testing. Shipped end-to-end ranking, "
    "search or recommendation systems to real users at scale at product "
    "companies, not pure research and not pure services. Strong Python and "
    "code quality. Scrappy product-engineering attitude over pure research."
)

# illuminate - Fluxmatic Lighting AI Assistant

**illuminate** is a custom Retrieval-Augmented Generation (RAG) chatbot built for **Fluxmatic Solution LLP** to act as an expert architectural lighting design consultant and customer relations assistant.

---

## 🏢 About Fluxmatic

Fluxmatic Solution LLP is an Indian Limited Liability Partnership (LLP) incorporated on July 7, 2022. The company specializes in importing and supplying high-quality architectural, linear, outdoor, and landscape lighting fixtures.

### Value Proposition
Bridging the gap between architectural vision and technical lighting reality by curating a premium portfolio of luminaires focused on performance, visual comfort, low glare, and durability.

### Partner Brands
- **XAL (Austria)**: Premium linear profiles and architectural workplace ceiling lights.
- **Wever & Ducré (Belgium)**: Trendy and cozy decorative interior fixtures.
- **NEKO (Switzerland)**: Precision modular LED downlights and low-glare spots.
- **Wästberg (Sweden)**: Circadian-rhythm task lights and designer workspace lamps.
- **Unilamp (Thailand)**: Facade and landscape outdoor waterproof projectors.

---

## 🛠️ What We Built

A production-ready, minimalistic lighting consultation chatbot utilizing:

1.  **Streamlit Chat Interface**:
    *   Sleek, light-theme layout inspired by premium minimalist landing pages.
    *   Translucent, glassmorphic chat bubbles (light blue for assistant, light gray for user).
    *   Natively loaded Material Design icons (`:material/lightbulb:`, `:material/phone:`, `:material/mail:`, etc.).
    *   Symmetrical 2x2 grid for prompt starter cards on single-line buttons.
2.  **Same-Tab Dynamic Navigation**:
    *   Clicking the `💡 illuminate` heading link smoothly toggles a detailed **Company Overview** view in the same tab (no new tabs or windows) using Streamlit's native query parameters and a background JavaScript handler.
    *   A clean back-to-chat button navigates visitors back to their active chat history.
3.  **Local Document RAG Pipeline**:
    *   **`ingest.py`**: An ingestion script that splits catalog PDFs recursively, embeds chunks using Hugging Face's `all-MiniLM-L6-v2` encoder, and stores them in a local Chroma vector database.
    *   **`app.py`**: Queries the Chroma vector database to retrieve grounding documents, constructing structured context before asking the Groq Llama 3.1 LLM.
4.  **Typewriter Response Streaming**:
    *   Simulates word-by-word streaming responses (`time.sleep(0.03)`) for a premium user interaction.

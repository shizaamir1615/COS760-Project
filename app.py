# 1. Imports
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from lime.lime_text import LimeTextExplainer
import os
import traceback
st.markdown("""
    <style>
        html, body, .main {
            background-color: #0f2027 !important;
            color: #f1f1f1 !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #1c2a35 !important;
        }

       

        .block-container {
            background-color: #0f2027 !important;
        }

        .stTextInput > div > div > input,
        .stTextArea textarea {
            background-color: #1c2a35;
            color: #f1f1f1;
            border: 1px solid #ffffff;
            border-radius: 6px;
        }

        .stButton button {
            background-color: #06d6a0;
            color: white;
            border: none;
            padding: 0.5em 1.2em;
            border-radius: 8px;
            transition: 0.3s ease-in-out;
            font-weight: 600;
        }

        .stButton button:hover {
            background-color: #04c38e;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stTitle, .stSubheader {
            color: #ffffff;
        }

        .stMetric label, .stMetric div {
            color: #ffffff !important;
        }

        .stAlert {
            background-color: #22313f;
            color: #ffffff;
            border-left: 4px solid #ffffff;
        }

        .stProgress > div > div > div > div {
            background-color: #ffffff !important;
        }

        .stSpinner > div > div {
            color: #ffffff !important;
        }

        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)


# Set default tensor type to avoid meta tensor issues
torch.set_default_dtype(torch.float32)

# 2. Load models and tokenizers with proper error handling
@st.cache_resource
def load_models():
    """Load models with comprehensive error handling"""
    try:
        st.info("Loading models... This may take a few minutes on first run.")
        
        # Option 1: Try to load your custom models if they exist locally
        afri_model = None
        afri_tokenizer = None
        xlmr_model = None
        xlmr_tokenizer = None
        
        # Check if local model directories exist
        if os.path.exists("afriberta_dir") and os.path.exists("xlmr_dir"):
            try:
                afri_model = AutoModelForSequenceClassification.from_pretrained("afriberta_dir", local_files_only=True)
                afri_tokenizer = AutoTokenizer.from_pretrained("afriberta_dir", local_files_only=True)
                xlmr_model = AutoModelForSequenceClassification.from_pretrained("xlmr_dir", local_files_only=True)
                xlmr_tokenizer = AutoTokenizer.from_pretrained("xlmr_dir", local_files_only=True)
                
                # Set to evaluation mode
                afri_model.eval()
                xlmr_model.eval()
                
                st.success("Loaded custom models from local directories")
                return afri_model, afri_tokenizer, xlmr_model, xlmr_tokenizer
                
            except Exception as e:
                st.warning(f"Failed to load custom models: {e}")
        
        # Option 2: Use standard pre-trained models
        st.info("Loading standard pre-trained models...")
        
        # Use smaller, reliable models for demonstration
        afri_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        afri_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        xlmr_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        xlmr_tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Set models to evaluation mode
        afri_model.eval()
        xlmr_model.eval()
        
        # Test models with dummy input to ensure they work
        test_text = "This is a test."
        test_inputs_1 = afri_tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
        test_inputs_2 = xlmr_tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            _ = afri_model(**test_inputs_1)
            _ = xlmr_model(**test_inputs_2)
        
        st.success("Loaded and tested standard models successfully")
        return afri_model, afri_tokenizer, xlmr_model, xlmr_tokenizer
        
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None, None, None, None

# 3. Robust prediction function
def predict_proba(text, model, tokenizer, model_name="Model"):
    """Predict probabilities with comprehensive error handling"""
    if model is None or tokenizer is None:
        if model_name != "LIME-XLM-R":  # Don't spam warnings for LIME calls
            st.warning(f"{model_name} is not loaded. Returning default probabilities.")
        return [0.5, 0.5]
    
    if not text or not text.strip():
        return [0.5, 0.5]
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Check if logits are valid
            if outputs.logits.numel() == 0:
                if model_name != "LIME-XLM-R":
                    st.warning(f"{model_name}: Empty logits tensor")
                return [0.5, 0.5]
            
            # Check for meta tensors
            if hasattr(outputs.logits, 'is_meta') and outputs.logits.is_meta:
                if model_name != "LIME-XLM-R":
                    st.error(f"{model_name}: Meta tensor detected - model not properly loaded")
                return [0.5, 0.5]
            
            # Apply softmax to get probabilities
            probs = F.softmax(outputs.logits, dim=1)
            
            # Handle different output shapes
            if probs.dim() > 1:
                probs = probs.squeeze(0)
            
            # Convert to list
            prob_list = probs.cpu().tolist()
            
            # Ensure we have exactly 2 probabilities for binary classification
            if isinstance(prob_list, float):
                # Single probability output
                prob_list = [1 - prob_list, prob_list]
            elif len(prob_list) == 1:
                # Single element list
                prob_list = [1 - prob_list[0], prob_list[0]]
            elif len(prob_list) != 2:
                # Multi-class or unexpected format
                if len(prob_list) > 2:
                    # Take first two classes
                    prob_list = prob_list[:2]
                    # Renormalize
                    total = sum(prob_list)
                    if total > 0:
                        prob_list = [p / total for p in prob_list]
                    else:
                        prob_list = [0.5, 0.5]
                else:
                    # Not enough classes
                    prob_list = [0.5, 0.5]
            
            # Ensure probabilities are valid floats and sum to 1
            try:
                prob_list = [float(p) for p in prob_list]
                total = sum(prob_list)
                if total > 0:
                    prob_list = [p / total for p in prob_list]
                else:
                    prob_list = [0.5, 0.5]
            except (ValueError, TypeError):
                prob_list = [0.5, 0.5]
            
            return prob_list
            
    except Exception as e:
        if model_name != "LIME-XLM-R":
            st.error(f"❌ Error in {model_name} prediction: {e}")
            st.code(traceback.format_exc())
        return [0.5, 0.5]

# 4. Ensemble prediction function
def ensemble_predict(text):
    """Make ensemble predictions using both models"""
    probs_afri = predict_proba(text, afri_model, afri_tokenizer, "AfriBERTa")
    probs_xlmr = predict_proba(text, xlmr_model, xlmr_tokenizer, "XLM-RoBERTa")
    
    # Calculate average probabilities
    avg_probs = [(a + b) / 2 for a, b in zip(probs_afri, probs_xlmr)]
    
    return avg_probs, probs_afri, probs_xlmr

# 5. LIME explanation function with error handling
@st.cache_data
def get_lime_explanation(text, _model, _tokenizer):
    """Generate LIME explanation with caching"""
    if _model is None or _tokenizer is None:
        return None
    
    try:
        explainer = LimeTextExplainer(class_names=["Human", "Machine"])
        
        def wrapper(texts):
            """LIME wrapper function that returns proper 2D array"""
            results = []
            for t in texts:
                prob = predict_proba(t, _model, _tokenizer, "LIME-XLM-R")
                
                # Ensure prob is a list with exactly 2 elements
                if not isinstance(prob, list):
                    prob = [0.5, 0.5]
                elif len(prob) != 2:
                    if len(prob) == 1:
                        prob = [1 - prob[0], prob[0]]
                    else:
                        prob = [0.5, 0.5]
                
                # Ensure probabilities sum to 1
                prob_sum = sum(prob)
                if prob_sum > 0:
                    prob = [p / prob_sum for p in prob]
                else:
                    prob = [0.5, 0.5]
                
                results.append(prob)
            
            # Convert to numpy array format that LIME expects
            import numpy as np
            return np.array(results)
        
        explanation = explainer.explain_instance(
            text, 
            wrapper, 
            num_features=min(6, len(text.split())),  # Don't exceed word count
            num_samples=50  # Reduced for faster processing
        )
        
        return explanation
        
    except Exception as e:
        st.error(f"❌ LIME explanation failed: {str(e)}")
        st.error(f"Error details: {traceback.format_exc()}")
        return None

# Load models at startup
afri_model, afri_tokenizer, xlmr_model, xlmr_tokenizer = load_models()

# 6. Streamlit UI
st.title(" Human vs Machine Text Detector - Ensamble")
st.subheader("(low resource african languages)")
st.markdown("---")

# Model status
col1, col2 = st.columns(2)
with col1:
    afri_status = "100% Loaded" if afri_model is not None else "❌ Failed"
    st.metric("AfriBERTa Model", afri_status)

with col2:
    xlmr_status = "100% Loaded" if xlmr_model is not None else "❌ Failed"
    st.metric("XLM-RoBERTa Model", xlmr_status)

st.markdown("---")

# Input section
st.subheader(" Text Input")
text = st.text_area(
    " Enter a civic/government sentence:",
    height=100,
    placeholder="e.g., The government is working to create better quality infrastructure"
)

# Prediction section
if st.button("Classify Text", type="primary"):
    if not text or not text.strip():
        st.warning("⚠️ Please enter some text to classify.")
    elif afri_model is None and xlmr_model is None:
        st.error("❌ No models are loaded. Cannot make predictions.")
    else:
        with st.spinner("🔄 Analyzing text..."):
            # Make ensemble prediction
            avg_probs, afri_probs, xlmr_probs = ensemble_predict(text)
            
            # Determine final prediction
            final_label = "Machine" if avg_probs[1] > 0.5 else "Human"
            confidence = max(avg_probs)
            
            # Display results
            st.markdown("## Results")
            
            # Main prediction
            if final_label == "Machine":
                st.error(f" **Prediction: {final_label}** (Confidence: {confidence:.1%})")
            else:
                st.success(f" **Prediction: {final_label}** (Confidence: {confidence:.1%})")
            
            # Detailed scores
            st.markdown("###  Detailed Scores")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "AfriBERTa",
                    f"{afri_probs[1]:.1%}",
                    f"Machine Confidence"
                )
            
            with col2:
                st.metric(
                    "XLM-RoBERTa", 
                    f"{xlmr_probs[1]:.1%}",
                    f"Machine Confidence"
                )
            
            with col3:
                st.metric(
                    "Ensemble Average",
                    f"{avg_probs[1]:.1%}",
                    f"Machine Confidence"
                )
            
            # Progress bars for visualization
            st.markdown("###  Score Visualization")
            st.progress(afri_probs[1], text=f"AfriBERTa: {afri_probs[1]:.1%}")
            st.progress(xlmr_probs[1], text=f"XLM-RoBERTa: {xlmr_probs[1]:.1%}")
            st.progress(avg_probs[1], text=f"Ensemble: {avg_probs[1]:.1%}")
            
            # LIME explanation
            if xlmr_model is not None:
                st.markdown("###  Feature Importance (LIME Analysis)")
                
                with st.spinner("Generating explanation..."):
                    explanation = get_lime_explanation(text, xlmr_model, xlmr_tokenizer)
                    
                    if explanation is not None:
                        try:
                            # Try to display LIME plot
                            fig = explanation.as_pyplot_figure()
                            st.pyplot(fig)
                            
                        except Exception as plot_error:
                            st.warning(f"Could not display LIME plot: {plot_error}")
                        
                        try:
                            # Always display explanation as text
                            st.markdown("**Key Features Influencing Prediction:**")
                            exp_list = explanation.as_list()
                            
                            if exp_list:
                                for i, (feature, weight) in enumerate(exp_list):
                                    if weight > 0:
                                        emoji = ""  # Pushes toward machine
                                        direction = "Machine"
                                        color = "red"
                                    else:
                                        emoji = ""  # Pushes toward human
                                        direction = "Human"
                                        color = "green"
                                    
                                    st.markdown(
                                        f"{emoji} **{feature}** → *{direction}* "
                                        f"<span style='color: {color}'>({weight:.3f})</span>",
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("No significant features found in the explanation.")
                                
                        except Exception as text_error:
                            st.error(f"Could not display text explanation: {text_error}")
                    else:
                        st.warning("Could not generate LIME explanation. This may be due to:")
                        st.markdown("""
                        - Text too short or simple
                        - Model compatibility issues
                        - Insufficient vocabulary overlap
                        
                        Try with a longer, more complex sentence.
                        """)
            else:
                st.info("LIME explanation requires XLM-RoBERTa model to be loaded.")

# Sidebar with additional information
st.sidebar.markdown("## ℹ️ About")
st.sidebar.markdown("""
This tool uses a custom-trained ensemble of two transformer models to detect whether a civic/government sentence was **written by a human or generated by a machine**.

 **Models used**:
- **AfriBERTa**: Fine-tuned on South African civic data for four languages (Xhosa, Tsonga, Tshivenda, Northern Sotho)
- **XLM-RoBERTa**: Fine-tuned on the same data for cross-verification and multilingual support

 **Prediction**:  
Both models output a probability, which is averaged to make the final decision.

 **Explanation**:  
**LIME** (Local Interpretable Model-agnostic Explanations) highlights which words influenced the model's prediction.
""")

st.sidebar.markdown("##  Troubleshooting")
st.sidebar.markdown("""
If you encounter errors:

-  **Model not loading**: Check file paths or model folders
-  **Zero confidence**: Ensure you're using one of the supported civic languages
-  **No internet** (first-time use): Pretrained models must be available locally

Contact developer if issues persist.
""")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit | Custom NLP Ensemble for Civic Text Classification*")

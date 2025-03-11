import streamlit as st
import PyPDF2
import subprocess
import speech_recognition as sr
import re
from transformers import pipeline
# Load the model for question generation and evaluation
question_generator = pipeline("text2text-generation", model="google/flan-t5-large")
evaluation_model = pipeline("text2text-generation", model="google/flan-t5-large")

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# Function to clean DeepSeek's output
def clean_ollama_output(output):
    # Remove unwanted sections using regex
    output = re.sub(r"<.*?>.*?</.*?>", "", output, flags=re.DOTALL)  # Remove XML-like tags
    output = re.sub(r"\[.*?\]", "", output)  # Remove any metadata inside brackets
    output = re.sub(r"failed to get console mode.*?\n", "", output, flags=re.IGNORECASE)  # Remove error messages
    output = re.sub(r"^.*?<think>.*?</think>", "", output, flags=re.DOTALL)  # Remove unwanted thoughts

    # Extract only questions (assuming they are numbered)
    questions = re.split(r"\d+\.\s*", output)
    questions = [q.strip() for q in questions if q.strip()]
    return questions

# Generate questions using DeepSeek
def generate_questions(text, num_questions=10):
    questions = []
    
    # Prepare the prompt for question generation
    prompt = f"Generate a programming-related question from the following text:\n\n{text}"

    try:
        # Use beam search to allow multiple outputs
        results = question_generator(
            prompt,
            max_length=256,               # Limit the length of generated text
            num_return_sequences=num_questions,  # Number of questions to generate
            num_beams=num_questions,      # Use beam search for multiple sequences
            early_stopping=True           # Stop early if output becomes repetitive
        )

        # Extract the generated questions
        questions = [result["generated_text"] for result in results]
        return questions
    
    except Exception as e:
        print(f"Error during question generation: {e}")
        return None

# Speech recognition to capture user answers
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for your answer...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition unavailable"

# Evaluate user responses using DeepSeek
def evaluate_response(question, answer):
    prompt = (
        f"Evaluate the following answer:\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Provide the output in this exact format:\n"
        f"Score: X\nFeedback: ...\n"
        f"Example:\n"
        f"Score: 7\nFeedback: Good understanding of the concept.\n"
    )

    try:
        result = evaluation_model(
            prompt,
            max_length=256,
            num_return_sequences=1
        )
        
        response = result[0]["generated_text"].strip()
        print(f"🔍 Debug: Full Response from Model:\n{response}")

        if not response:
            return None, "Model did not return a valid response."

        # ✅ More flexible regex for extracting score
        match = re.search(r"score[:=\s]*(\d+)", response, re.IGNORECASE)
        feedback_match = re.search(r"feedback[:=\s]*(.*)", response, re.IGNORECASE)

        if match:
            score = int(match.group(1))
            feedback = feedback_match.group(1).strip() if feedback_match else "No detailed feedback provided"
            return score, feedback
        else:
            print("❌ Failed to extract score. Response format incorrect.")
            return None, "Failed to extract score. Response format incorrect."

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None, "Evaluation failed. Please check the model response."


# Calculate final score
def calculate_final_score(scores):
    valid_scores = [s for s in scores if isinstance(s, int)]  # ✅ Filter out invalid scores
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0


# Streamlit UI
def main():
    st.title("AI & Speech-Based Evaluation System")

    # Upload PDF
    st.header("Upload a PDF")
    pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
        st.subheader("Extracted Text from PDF")
        st.write(text)

        # Generate Questions
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                questions = generate_questions(text)
                if questions:
                    st.session_state["questions"] = questions  # Store questions
                    st.session_state["answers"] = []  # Reset answers
                    st.session_state["scores"] = []  # Reset scores
                    st.success("Questions Generated!")

    # Question-Answer Section
    if "questions" in st.session_state:
        st.subheader("Answer the Questions")

        for i, question in enumerate(st.session_state["questions"]):
            st.write(f"**Q{i+1}: {question}**")

           # Capture voice response
            if st.button(f"Answer Q{i+1} via Speech"):
                with st.spinner("Listening..."):
                    answer = recognize_speech()
                    st.session_state["answers"].append(answer)
                    st.write(f"Your Answer: {answer}")

                    # Evaluate the answer
                    with st.spinner("Evaluating..."):
                        score, feedback = evaluate_response(question, answer)

                        if score is not None:
                            st.session_state["scores"].append(score)  # ✅ Append only the score
                            st.write(f"✅ **Score:** {score}")
                            st.write(f"💬 **Feedback:** {feedback}")
                        else:
                            st.error(feedback)  # ✅ Display only feedback if extraction fails


                # Display Final Score
                if "scores" in st.session_state and st.session_state["scores"]:
                    st.subheader("Final Score")
                    final_score = calculate_final_score(st.session_state["scores"])
                    st.write(f"**Your Current Overall  Score: {final_score:.2f} / 10**")

if __name__ == "__main__":
    main()

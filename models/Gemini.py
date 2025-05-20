import os
import google.generativeai as genai
from dotenv import load_dotenv


#Simply a prompt API call
def prompt(text: str) -> str:
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    try:
        genai.configure(api_key=gemini_api_key)

        generation_config = {
            "temperature": 0.0,  
            "max_output_tokens": 1000, 
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
        )

        response = model.generate_content(text)

        return response.text

    except Exception as e:
        return f"Error calling Gemini API: {e}"

if __name__ == '__main__':
    user_prompt = 'Identify an entity in this sentence: Mary had a little lamb'
    
    print(f"User Prompt: {user_prompt}")
    assistant_response = prompt(user_prompt)
    print(f"Gemini Response: {assistant_response}")

    print("\n" + "="*30 + "\n")

    user_prompt_2 = "What is the capital of France?"
    print(f"User Prompt: {user_prompt_2}")
    assistant_response_2 = prompt(user_prompt_2)
    print(f"Gemini Response: {assistant_response_2}")
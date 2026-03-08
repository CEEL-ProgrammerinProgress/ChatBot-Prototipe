# ChatBot-Prototipe
A simple Chatbot prototipe for RRHH. His name is "Bit".

# how use it:
## 🛠️ How to use it:
1. **Install dependencies:** Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
    ```
-lauch the bot:
  ```
    streamlit run Bit.py
 ```
# How to customize your own chatbot:
You can adapt Bit to any other department or company following these steps:

Modify Knowledge: Edit Parameters.json to update the Questions & Answers (Intents).

Retrain the Brain: Use Bit_Proto_Code.py to adjust training settings or add new data.

Model Saving: The training script generates Bit_Chatbot_RRH.keras. If you change this name, remember to update the reference in the training code:model.save("Your_New_Name.keras")
Update the Interface: If you renamed the model, update the load_model function inside Bit.py so the app can find the new "brain".


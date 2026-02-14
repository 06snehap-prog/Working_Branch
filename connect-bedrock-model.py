import boto3
import json
from botocore.config import Config

# --- CONFIGURATION ---
REGION = "us-east-1"
# For Haiku (Cheaper/Faster for Agents)
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

def connect_to_bedrock():
    # Initialize the Bedrock Runtime client
    # SSL Fix: Set 'verify' to the path of your corporate cert if needed
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION,
         
    )

    # Simple message structure for the Converse API
    messages = [
        {
            "role": "user",
            "content": [{"text": "Can you explain what is SAP."}]
        }
    ]

    try:
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            inferenceConfig={"maxTokens": 512, "temperature": 0.5}
        )

        # Extracting the text response
        output_text = response['output']['message']['content'][0]['text']
        print(f"Agent Response: {output_text}")

    except Exception as e:
        print(f"Connection failed: {str(e)}")

if __name__ == "__main__":
    connect_to_bedrock()
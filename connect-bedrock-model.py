import boto3
import json
from botocore.config import Config

# --- CONFIGURATION ---
REGION = "us-east-1"
# For Haiku (Cheaper/Faster for Agents)
#MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0" 
#MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0" 
MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0" 

def connect_to_bedrock():
    # Initialize the Bedrock Runtime client
    # Uses boto3's default credential chain:
    # 1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    # 2. ~/.aws/credentials file
    # 3. ~/.aws/config file
    # 4. IAM roles (if on EC2/SageMaker)
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION
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
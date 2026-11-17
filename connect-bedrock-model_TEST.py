import boto3

def test_ai_agent_connection():
    # Use your active SSO profile
    sso_profile = "AdministratorAccess-715248855541"
    
    try:
        session = boto3.Session(profile_name=sso_profile)
        client = session.client(service_name='bedrock-runtime', region_name='us-east-1')

        # UPDATED: Using the Cross-Region Inference Profile ID
        #model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        #model_id = "amazon.nova-lite-v1:0"
        model_id = "us.amazon.nova-micro-v1:0"

        print(f"Testing {model_id}...")

        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Can you explain what is SAP."}]}]
        )
        
        answer = response['output']['message']['content'][0]['text']
        print(f"\n✅ CONNECTION SUCCESSFUL!")
        print(f"AI Response: {answer}")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    test_ai_agent_connection()
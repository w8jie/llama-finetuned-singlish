import os
import json
import time
from dotenv import load_dotenv
import pandas as pd
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VER")
)
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

'''   
# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a tagline for an ice cream shop. '
response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
                {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
                {"role": "user", "content": "Do other Azure AI services support this too?"}
                ]
            )
print(response.choices[0].message.content)
'''

NUM_SAMPLE = 100

ACTIVITIES = ['cooking',
'studying',
'sleeping',
'eating',
'working',
'exercising',
'reading',
'cleaning',
'shopping',
'driving',
'walking',
'bathing',
'going to work',
'listening to music',
'watching TV',
'playing video games',
'using a computer',
'texting',
'socializing',
'meditating',
'commuting',
'doing laundry',
'ironing clothes',
'dusting',
'vacuuming',
'painting',
'drawing',
'grocery shopping',
'sewing',
'taking a nap',
'jogging',
'biking',
'swimming',
'playing sports',
'checking emails',
'playing with children',
'watching movies',
'playing board games',
'attending school or classes',
'going to the gym',
'playing a musical instrument',
'singing',
'dancing',
'writing',
'photography',
'traveling',
'visiting friends',
'attending events',
'volunteering',
'attending meetings',
'gardening',
'birdwatching',
'stargazing',
'fishing',
'hiking',
'camping',
'exploring nature trails',
'hosting parties',
'DIY crafts',
'knitting',
'brewing coffee or tea',
'journaling',
'scrapbooking',
'taking care of pets',
'organizing spaces',
'budgeting or financial planning',
'meal prepping',
'online shopping',
'window shopping',
'car maintenance',
'home improvement projects',
'renovating or redecorating',
'learning a new language',
'practicing mindfulness',
'yoga',
'doing puzzles',
'learning to code',
'investing or stock trading',
'outdoor sports',
'road trips',
'studying genealogy',
'learning about astronomy',
'participating in forums or online communities',
'attending virtual events or webinars',
'exploring new cuisines',
'trying out new restaurants',
'baking',
'hosting a book club',
'vlogging or blogging',
'watching documentaries',
'listening to podcasts',
'doing charity work',
'practicing calligraphy',
'exploring photography techniques',
'doing home science experiments',
'practicing martial arts',
'planning travel itineraries',
'exploring historical landmarks',
'trying out new hairstyles or makeup',
'organizing charity drives'
]

dataset = {}
DELAY_BETWEEN_CALLS = 1  # Delay in seconds

for index, activity in enumerate(ACTIVITIES):
    print(f"Processing {index + 1}/{len(ACTIVITIES)}: {activity}")
    
    # Attempt API call with error handling
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "system",
                "content": "You are an expert in answering English questions in Singlish"},
                {"role": "user",
                    "content":  f"Create {NUM_SAMPLE} random conversational English (e) questions and Singlish (s) answers to form Question-Answer pairs in json. Write full questions about {activity}."\
                                f"Don't exaggerate the use of Singlish, and be natural, as how a real Singaporean would speak."\
                                f"Start the keys from {(index*NUM_SAMPLE)+1}. For example,"\
                                "{'X':{'e': 'oh my, what happened to your car?', 's': 'aiyo, don't say already, the car behind hit me, really sibei suay.'}"\
                                "..., 'X+5': {'e': 'how is the weather today?', 's': 'hot and humid lor'} }"}],
            temperature=0.01,
            response_format={"type":"json_object"}
        )

        # Parse and update dataset
        output = response.choices[0].message.content
        output_json = json.loads(output)
        dataset.update(output_json)

        # Save the current state to a file after each activity
        with open('english_singlish_chat_v0.2.json', 'w') as f:
            json.dump(dataset, f, indent=None)

    except Exception as e:
        print(f"Error processing activity '{activity}': {e}")

    # Add delay between API calls
    time.sleep(DELAY_BETWEEN_CALLS)

# Convert to tabular csv
df = pd.read_json("english_singlish_chat_v0.2.json")
df = df.T
df = df.reset_index()
df.columns = ["index", "question", "answer"]
df.to_csv("english_singlish_chat_v0.2.csv", index=False)

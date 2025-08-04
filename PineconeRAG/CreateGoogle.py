# Create Index & Namespace using Gemini Embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai


pc = Pinecone(api_key="API_KEY")

index_name = "my-rag-index"
# Dimension will depend on your embedding model.
# For OpenAI's text-embedding-ada-002: 1536
# For Google's text-embedding-004: 768 (or check specific model documentation)
# For Sentence-Transformers like 'paraphrase-multilingual-mpnet-base-v2': 768
embedding_dimension = 768 # Example for Gemini/Sentence-Transformers

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine", # or 'euclidean', 'dotproduct'
        spec=ServerlessSpec(cloud="aws", region="us-east-1") # Example for serverless
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)
print(f"Connected to index: {index.describe_index_stats()}")


# Example: Loading from a text file
# Replace with your actual document loading
# For a simple text string:
raw_text = """
# **The Last Wave**

The Argo Wind had been billed as a floating retreat for the lost and the elite. Ten people, strangers to one another, boarded it in Bali, each seeking escape — from fame, guilt, failure, or fate. None of them imagined that within a week, they would be battling not boredom, but nature, madness, and each other.

On the third night, the sea rose with fury. The wind howled like a creature wronged. Waves slammed into the yacht, tearing apart its steel heart. Electronics failed. Panic spread faster than fire. Then, silence — as the vessel succumbed to the depths.

When the sun rose again, it found ten people strewn across the shore of an uncharted island, bloodied, bruised, and broken.

---

## **The Castaways**

### **Adrian Patel – The Code King**

At 34, Adrian was Silicon Valley’s golden child, having sold his AI firm for billions. He booked this cruise to escape the venom of boardrooms and betrayal. He believed every system — even survival — could be optimized.

His tailored shorts clung to him, soaked and tattered. The salt had ruined his satellite phone. He cursed himself for not bringing his backup GPS — arrogance had replaced preparedness.

### **Nora Whitman – The Truth Seeker**

War zones had been Nora’s home for over a decade. At 41, she was tired. Her Pulitzer-winning articles exposed regimes and revolutions, but never healed her own inner wounds.

She woke with a shattered femur and a throat dry as bone. Still, her first instinct was to crawl — to find others, to bear witness.

### **Dr. Sanjay Rao – The Quiet Healer**

Once Mumbai’s finest liver transplant surgeon, Sanjay, 50, had lost a young mother on his table three months ago. Her face haunted his dreams.

The ocean spared his hands, though his mind remained clouded. When he saw Nora’s injury, he tore his shirt into bandages, slipping silently into his old role.

### **Lisa Monroe – The Mask**

Twenty-six, flawless, and famous. Lisa’s world was digital, curated by lighting and likes. Her skincare brand had crossed \$10 million in revenue. The trip was meant to be her new rebrand — #OceanSoul.

The ocean had stripped away her extensions, lashes, and influencer persona. Alone and phone-less, she screamed, hoping someone — anyone — was watching.

### **Marcus King – The Redeemed**

A walking contradiction. Marcus, 38, had spent half his life behind bars for a crime he didn’t commit. He didn’t regret protecting his cousin. But he did regret losing his daughter’s childhood.

Now, here on a second chance retreat sponsored by a social justice charity, he wondered: was this redemption or punishment?

### **Elena Ruiz – The Wild Mind**

Elena, 29, could name 150 species of moss but struggled to make eye contact with humans. A botanist from Costa Rica, she joined the cruise to study Pacific flora. Her journals were gone, but her instincts were not.

She saw promise in the vines, warning in the trees, and poison underfoot.

### **Tyler Brooks – The Broken Son**

At 22, Tyler had failed everything he tried: college, relationships, rehab. This trip was a gift from parents who couldn’t help anymore. He had come to find peace. Instead, he found sand in his lungs and dread in his gut.

When he wasn’t laughing maniacally, he stared into the trees as if expecting them to laugh back.

### **Mei Chen – The Water Witch**

Daughter of a Chinese sea captain, Mei, 33, grew up reading stars instead of bedtime stories. She had crossed oceans before learning algebra. She worked aboard The Argo Wind as a consultant.

She’d warned the captain about weather anomalies. Now, she blamed herself for the wreck — and silently vowed to save whoever was left.

### **Khalid Mansoor – The Watcher**

Khalid, 47, had taught philosophy at Oxford before grief swallowed him whole. His daughter’s death silenced him for years. This voyage was his first attempt at “rejoining life.”

As others panicked, he simply watched. Thought. Noted who snapped first, who led, who clung to illusion. In silence, he kept a mental journal: *On Being Human, Vol. 2.*

### **Ava Sinclair – The Spy**

To most, Ava was a travel blogger. In truth, she worked for an intelligence agency, assigned to monitor Adrian Patel for suspected intellectual property leaks.

When the boat sank, her mission became survival. But habits die hard. She observed, tracked, and hid a small pistol in a waterproof pouch under her belt.

---

## **Day 3 – Illusion Breaks**

By now, the group had assembled a crude shelter. Mei led the logistics. Sanjay tended wounds. Adrian began cataloging resources like an engineer. Elena foraged. Lisa sulked.

Lisa drank from a puddle, ignoring Elena’s caution about toxic algae. She died violently, convulsing through the night. No one touched her body for hours.

“We’re not in a movie,” Mei said, burying her with sea stones. “There are no heroes here.”

---

## **Day 5 – Cracks and Crevices**

Adrian suggested rations. Tyler mocked him.

“This is a test. You rich guys probably staged this. Hidden cameras everywhere.”

Marcus snapped. “Say that again, and I’ll rip your jaw off.”

Khalid whispered, “In chaos, truth reveals itself.” No one heard him.

---

## **Day 6 – The Raft and the Saboteur**

Adrian and Mei began building a raft. Elena provided waterproof vines. Ava stayed silent.

That night, Tyler crept to the shoreline and loosened the bindings.

The next morning, the raft crumbled on its test float. Mei screamed. Adrian accused. Tyler laughed.

Marcus struck him across the face. “You play with lives now.”

Only Ava pulled Marcus back. “He’s not worth it,” she said.

---

## **Day 7 – The Journalist's End**

Nora’s wound had festered. Sanjay did all he could, fashioning a crude splint and applying aloe sap. It wasn’t enough.

By dusk, her fever burned through her mind. She spoke names no one recognized. At midnight, she asked Khalid, “Did I do enough?”

“You bore witness,” he replied.

She died before dawn.

---

## **Day 9 – The Rope and the Bark**

Marcus had grown quieter. He stared at the ocean for hours.

That morning, Elena found him hanging from a bent palm. A single sentence carved into the trunk: *“Some prisons don’t need walls.”*

Mei wept silently. Tyler threw up. Adrian didn’t react.

---

## **Day 11 – Madness in the Vines**

Tyler attacked Elena near the banana grove, driven by hunger or hate — no one could tell.

She stabbed him with a sharpened bamboo stalk.

Ava arrived too late. She fired her gun into the sand.

“I said STOP!”

Tyler bled out, eyes wide and unblinking.

Now everyone knew Ava carried power.

---

## **Day 13 – The Botanist’s Misstep**

Elena believed she found a new root: *Tacca integrifolia*, rich in carbs and supposedly safe. They boiled it into a paste.

That night, Adrian, Khalid, Ava, and Sanjay ate it. Hours later, Khalid collapsed. Adrian clawed at his throat. Sanjay writhed in seizures.

Only Ava remained coherent.

“You said it was safe,” she growled at Elena.

“I—I was sure…”

Sanjay, in rage or despair, slit Elena’s throat with a clam shell.

By morning, all but Ava were dead.

---

## **Day 14 – The Flare**

Ava climbed the tallest hill and fired her emergency flare. A red flower bloomed in the sky. But the sea remained still.

She spent the next three days in silence, eating raw crab, staring at the surf.

---

## **Day 18 – Rescue**

A fishing boat spotted the melted flare casing glinting in the sand. They found Ava dehydrated, sunburnt, and speaking only in fragments.

---

## **Epilogue – The Survivor**

Ava declined interviews. She told the Coast Guard it was a tragic accident. Only her agency received the full report:

> “Adrian is dead. No data retrieved. Nine dead. One lives. I don't call it survival. I call it subtraction.
>
> I watched people become animals. I became one.
>
> Recommend terminating future surveillance via maritime routes.”

Ava vanished a month later. No credit cards, no signal. Some say she’s in Patagonia. Others say she’s dead.

But those who’ve seen her swear her eyes are different now — sharper. Like she’s still on the island.

Because maybe, in some ways, she always will be.
"""

# For a file:
# loader = TextLoader("your_document.txt")
# documents = loader.load()
# raw_text = documents[0].page_content # Assuming one document for simplicity

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)
# print(f"Split {len(texts)} chunks.")

genai.configure(api_key="API_KEY")

# Note: 'models/embedding-001' or 'models/text-embedding-004' are common.
# Check current Gemini embedding model dimensions.
gemini_embedding_model = "models/embedding-001"
# 'models/embedding-001' typically has 768 dimensions.

vectors_to_upsert = []
for i, chunk in enumerate(texts):
    response = genai.embed_content(model=gemini_embedding_model, content=chunk)
    embedding = response['embedding'] # Access the embedding from the dictionary
    vectors_to_upsert.append({
        "id": f"doc_{i}",
        "values": embedding,
        "metadata": {"text": chunk}
    })

# Upsert to Pinecone
index.upsert(vectors=vectors_to_upsert,namespace="Naamspace")
# print(f"Upserted {len(vectors_to_upsert)} vectors using Gemini embeddings.")






# from openai import OpenAI
# openai_client = OpenAI(api_key="OPENAI_API_KEY")
# openai_embedding_model = "text-embedding-ada-002" # or "text-embedding-3-small", "text-embedding-3-large"
# # Ensure dimension matches the index: ada-002 is 1536, text-embedding-3-small is 1536, text-embedding-3-large is 3072

# vectors_to_upsert = []
# for i, chunk in enumerate(texts):
#     response = openai_client.embeddings.create(input=chunk, model=openai_embedding_model)
#     embedding = response.data[0].embedding
#     vectors_to_upsert.append({
#         "id": f"doc_{i}",
#         "values": embedding,
#         "metadata": {"text": chunk} # Store original text as metadata
#     })

# # Upsert to Pinecone
# index.upsert(vectors=vectors_to_upsert)
# print(f"Upserted {len(vectors_to_upsert)} vectors using OpenAI embeddings.")